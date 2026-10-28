"""Shared motion pieces for the microwave tasks: planning, the door model, files, output.

Used by `open_door.py` and `demo_replay.py`.

* `release_and_back_off` -- records the current grasp (pose + joints) to
  `LAST_GRASP_FILE`, opens the gripper, then backs straight off the handle along the
  gripper's approach axis in ~2 cm Cartesian steps.
* `DoorFrame` / `add_door_model` -- the door's geometry (from `DOOR_FILE`, which the grasp and
  swing write) and its collision model in the sim.

cuRobo (RAMMP's planner, through the driver's go_to_ee_pose / go_to_joint_config) does the
free-space moves: to park, to the regrasp back-off point, to the open-door view -- as
rammp_button_press does. Everything on or along the door (straight on/off the handle, the
swings, the pushes) stays on joint paths planned and checked here: the planner takes a goal
only and can't follow a line or an arc. `check_curobo_goal` gates a cuRobo goal.

All gripper motion goes through `execute_command(Open/CloseGripperCommand())` --
`ArmInterfaceClient` has no `open_gripper()` method.

The arm sim (make_sim, solve_ik, clearance, the IK guards) is capability_engine.sim, shared with
self.arm: one PyBullet copy per process, built from the driver's own model. The guards are ROS
parameters (arm.max_ik_err, arm.j4_guard_deg, ...), so they are read as sim.X at call time.
"""

import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pybullet as p
from capability_engine import sim
from capability_engine.cornell_commands import (
    CartesianTrajectoryCommand,
    JointCommand,
    OpenGripperCommand,
)
from capability_engine.sim import (  # noqa: F401 -- re-exported for the task modules
    ARM,
    Pose,
    clearance,
    continuous_ok,
    keep_out_message,
    make_sim,
    multiply_poses,
    short_delta,
    solve_ik,
    wrap_joints,
)
from scipy.spatial.transform import Rotation as R


class Refused(Exception):
    """A check failed before (or instead of) a motion: the arm holds where it is. The message says why."""


# ---------------------------------------------------------------------------
# Progress output: stdout on the bench, the node's logger under the engine (set_output)
# ---------------------------------------------------------------------------
_output = None
_quiet = 0


def say(*parts, sep=" ", end="\n", flush=False):
    """A progress line, with print()'s signature (ruff bans print in capability code)."""
    if _quiet:
        return
    text = sep.join(map(str, parts))
    if _output is not None:
        for line in text.split("\n"):
            if line.strip():
                _output(line.rstrip())
        return
    sys.stdout.write(text + end)
    if flush:
        sys.stdout.flush()


def set_output(fn):
    """Send say() to `fn(line)` (e.g. the node's get_logger().info); None restores stdout."""
    global _output
    _output = fn


@contextmanager
def quiet():
    """say() prints nothing inside (planning tries whose output only clutters the log)."""
    global _quiet
    _quiet += 1
    try:
        yield
    finally:
        _quiet -= 1


# ---------------------------------------------------------------------------
# Where a run's files go. The open saves the door geometry and the grasp; the close (a later
# run, maybe another process) reads them back -- so this must be persistent (the node's
# data_dir parameter, a host volume in the fragment).
# ---------------------------------------------------------------------------
DATA_DIR = Path("/tmp/microwave")
LAST_GRASP_FILE = DATA_DIR / "last_grasp.json"
DOOR_FILE = DATA_DIR / "door.json"
RUNS_DIR = DATA_DIR / "runs"
DETECT_FRAMES_DIR = DATA_DIR / "detections"


def configure_paths(data_dir):
    """Point every file above into `data_dir` (created). Modules read them as mc.X at call time."""
    global DATA_DIR, LAST_GRASP_FILE, DOOR_FILE, RUNS_DIR, DETECT_FRAMES_DIR
    DATA_DIR = Path(data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LAST_GRASP_FILE = DATA_DIR / "last_grasp.json"
    DOOR_FILE = DATA_DIR / "door.json"
    RUNS_DIR = DATA_DIR / "runs"
    DETECT_FRAMES_DIR = DATA_DIR / "detections"


STEP_M = 0.02
# Park ("end") pose for the open and close tasks, joints in rad. Hand-placed by teleop on the Jetson
# rig, 2026-10-07 (was park_pose.json v4). HARDCODED FOR NOW: per-robot, so this can change -- likely
# to a fragment param or one of the driver's presets (go_to_preset).
PARK_JOINTS = [
    0.6047337092622715,
    0.6574053764067224,
    -2.7310591810191354,
    -2.245420837282717,
    1.0414940588270571,
    1.366535599611345,
    1.2981626670490867,
]


def door_arc_waypoints(
    start_pose, hinge_position, arc_length_m, waypoint_spacing_m, direction=1, rotate_orientation=True
):
    """Waypoints along a door arc about a vertical hinge axis, starting from (not including)
    `start_pose`, at constant height. Copied from `PerceptionInterface._generate_door_arc_waypoints`
    so these scripts don't import the lab's whole perception stack for one function."""
    if arc_length_m <= 0:
        return []
    if waypoint_spacing_m <= 0:
        raise ValueError("waypoint_spacing_m must be > 0")
    if direction not in (-1, 1):
        raise ValueError("direction must be either +1 or -1")
    hx, hy, hz = start_pose.position
    cx, cy, _ = hinge_position
    radius_vec = np.array([hx - cx, hy - cy], dtype=float)
    radius = np.linalg.norm(radius_vec)
    if radius < 1e-8:
        raise ValueError("Handle pose is too close to hinge pose; radius is ~0.")
    start_theta = np.arctan2(radius_vec[1], radius_vec[0])
    total_angle = arc_length_m / radius
    num_segments = max(1, int(np.ceil(arc_length_m / waypoint_spacing_m)))
    start_rot = R.from_quat(start_pose.orientation)
    waypoints = []
    for i in range(1, num_segments + 1):
        delta_angle = direction * i / num_segments * total_angle
        theta = start_theta + delta_angle
        orientation = (
            (R.from_euler("z", delta_angle) * start_rot).as_quat() if rotate_orientation else start_pose.orientation
        )
        waypoints.append(
            Pose(position=(cx + radius * np.cos(theta), cy + radius * np.sin(theta), hz), orientation=orientation)
        )
    return waypoints


def wait_for_joints(ai, q, tol_deg=1.0, timeout_s=6.0):
    """Wait until the arm is within `tol_deg` of the COMMANDED joints (Kortex can
    silently drop a chained command, so velocity ~ 0 is not proof of arrival)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        cur = np.array(ai.get_state()["position"], dtype=float)
        if np.degrees(np.max(np.abs((cur - q + np.pi) % (2 * np.pi) - np.pi))) < tol_deg:
            return True
        time.sleep(0.05)
    return False


def fit_circle_xy(xy):
    """Least-squares circle through 2-D points (Kasa fit). Returns (centre xy, radius, rms residual m)."""
    xy = np.asarray(xy, dtype=float)
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy**2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = float(np.sqrt(c + cx**2 + cy**2))
    resid = np.linalg.norm(xy - [cx, cy], axis=1) - r
    return np.array([cx, cy]), r, float(np.sqrt(np.mean(resid**2)))


def run_cartesian_trajectory(ai, traj, tol_m=0.01, settle_s=2.0, timeout_s=60.0, impedance=None, speed_scale=1.0):
    """Send a CartesianTrajectoryCommand and judge it by where the arm ACTUALLY ends up, not
    the RPC return: the server's wait can be tripped by a stale END/ABORT notification and
    return False at once while Kortex runs the whole trajectory anyway (09-28: returned False
    with the arm still at the start, then reached every waypoint). Waits until the EE is within
    `tol_m` of the last waypoint, or has not moved for `settle_s` (after at least `settle_s`
    had passed, so a not-yet-started motion isn't mistaken for a stop). Returns (reached, err_m, rpc_ok).
    `impedance`: run it compliant, e.g. the door swing (rammp_arm.IMPEDANCE).
    `speed_scale`: e.g. rammp_arm.contact_speed_scale() for a move that pushes on the door."""
    if (impedance is not None or speed_scale != 1.0) and hasattr(ai, "move_cartesian"):
        # sheppy: compliant (joint impedance) and/or slowed execution -- see rammp_arm.IMPEDANCE
        rpc_ok = ai.move_cartesian(
            [(list(pp), list(qq)) for pp, qq in traj], impedance=impedance, speed_scale=speed_scale
        )
    else:
        rpc_ok = ai.execute_command(CartesianTrajectoryCommand([(list(pp), list(qq)) for pp, qq in traj]))
    goal = np.asarray(traj[-1][0], dtype=float)
    t0 = time.time()
    last = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
    last_move = t0
    while time.time() - t0 < timeout_s:
        cur = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
        err = float(np.linalg.norm(cur - goal))
        if err < tol_m:
            time.sleep(0.3)  # let the final taper finish
            cur = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
            return True, float(np.linalg.norm(cur - goal)), rpc_ok
        if np.linalg.norm(cur - last) > 0.002:
            last, last_move = cur, time.time()
        elif time.time() - last_move > settle_s and time.time() - t0 > settle_s:
            return False, err, rpc_ok
        time.sleep(0.1)
    cur = np.asarray(ai.get_state()["ee_pos"][:3], dtype=float)
    return False, float(np.linalg.norm(cur - goal)), rpc_ok


def plan_straight_line(scene, rb, start_pos, quat, direction, dist, seed_joints, label):
    """Joint solutions for a straight Cartesian line, ~2 cm per step, fixed orientation.
    Returns the list of joint vectors, or None if any step fails a gate."""
    n = max(1, int(np.ceil(dist / STEP_M)))
    q, plan = np.asarray(seed_joints, dtype=float), []
    for k in range(1, n + 1):
        tgt = np.asarray(start_pos) + np.asarray(direction) * dist * k / n
        nq, err = solve_ik(scene, rb, tgt, quat, q)
        jump = float(np.degrees(np.max(np.abs(short_delta(q, nq)))))
        j6, j7 = float(np.degrees(nq[5])), float(np.degrees(nq[6]))
        say(
            f"  {label} {k}/{n} -> {np.round(tgt, 3)}  IK err {err * 100:.2f}cm  jump {jump:.1f}deg  "
            f"J6 {j6:.1f}deg  J7 {j7:.1f}deg"
        )
        if (
            err > sim.MAX_IK_ERR
            or jump > sim.MAX_STEP_JUMP_DEG
            or abs(j6) > sim.J6_GUARD_DEG
            or abs(np.degrees(nq[3])) > sim.J4_GUARD_DEG
            or not continuous_ok(q, nq)
        ):
            say(f"  {label}: step {k} fails the IK/jump/J6/wrap gate.")
            return None
        plan.append(nq)
        q = nq
    return plan


BIG_MOVE_TIMEOUT_S = 40.0


def plan_cartesian(scene, rb, p0, q0_quat, p1, q1_quat, seed_joints, label, bodies=None, min_clear=None, closed=False):
    """Straight line p0 -> p1 with the orientation slerped q0 -> q1, ~2 cm / <= 5 deg per step,
    each step IK-solved from the previous one. Gates: IK, per-step jump, J6, wrap and (if
    `bodies`) >= MIN_CLEAR from the door model, checked along each step too."""
    from scipy.spatial.transform import Slerp

    rots = R.from_quat([q0_quat, q1_quat])
    ang = float(np.degrees((rots[0].inv() * rots[1]).magnitude()))
    n = max(1, int(np.ceil(max(np.linalg.norm(np.asarray(p1) - p0) / STEP_M, ang / 5.0))))
    slerp = Slerp([0, 1], rots)
    q, plan, worst = np.asarray(seed_joints, float), [], (9.0, "")
    for k in range(1, n + 1):
        f = k / n
        tgt = np.asarray(p0) + (np.asarray(p1) - p0) * f
        nq, err = solve_ik(scene, rb, tgt, slerp([f]).as_quat()[0], q)
        # one PyBullet IK pass sometimes stalls (joints barely move, ~2.3 cm error), mostly on
        # rotate-in-place steps; re-solving from its own result converges (09-28 sim check)
        for _ in range(3):
            if err <= sim.MAX_IK_ERR:
                break
            nq, err = solve_ik(scene, rb, tgt, slerp([f]).as_quat()[0], nq)
        # wrapped: a free joint going 179.8 -> -179.8 moved 0.4 deg, not 359.6 (Kortex takes the short way)
        jump = float(np.degrees(np.max(np.abs((nq - q + np.pi) % (2 * np.pi) - np.pi))))
        if bodies is not None:
            worst = min(
                [worst] + [clearance(rb, q + short_delta(q, nq) * s_ / 3, bodies, closed) for s_ in (1, 2, 3)],
                key=lambda w: w[0],
            )
        bad = (
            err > sim.MAX_IK_ERR
            or jump > sim.MAX_STEP_JUMP_DEG
            or abs(np.degrees(nq[5])) > sim.J6_GUARD_DEG
            or abs(np.degrees(nq[3])) > sim.J4_GUARD_DEG
            or not continuous_ok(q, nq)
            or (bodies is not None and worst[0] < (min_clear or MIN_CLEAR))
        )
        if bad or k in (1, n) or k % 5 == 0:
            say(
                f"  {label} {k}/{n} -> {np.round(tgt, 3)}  IK err {err * 100:.2f}cm  jump {jump:.1f}deg  "
                f"J4 {np.degrees(nq[3]):.1f}  J6 {np.degrees(nq[5]):.1f}  clear {worst[0] * 100:.1f}cm ({worst[1]})"
                f"{'' if continuous_ok(q, nq) else '  WRAP'}"
            )
        if bad:
            say(f"  {label}: step {k} fails a gate.")
            return None
        plan.append(nq)
        q = nq
    say(f"  {label}: {n} steps, orientation change {ang:.0f} deg, worst clearance {worst[0] * 100:.1f} cm")
    return plan


def ee_pose(rb):
    """(position, orientation) of the sim's end-effector link, world frame (the Cornell robot's
    get_end_effector_pose(), which the shared sim's robot handle doesn't have)."""
    ls = p.getLinkState(rb.robot_id, rb.end_effector_id, physicsClientId=rb.physics_client_id)
    return Pose(tuple(ls[4]), tuple(ls[5]))


def check_curobo_goal(scene, rb, label, q=None, pos=None, quat=None, seed=None, bodies=None):
    """Gate the GOAL of a cuRobo move (self.arm.go_to_*): cuRobo plans the path in the driver, which can't be
    previewed, so only where it ends is checked here -- joints `q`, or (pos, quat) IK'd from `seed`: IK, reach,
    J4/J6 guards, the keep-out zone (fridge) and, with `bodies`, >= MIN_CLEAR from the door model. For a pose goal
    cuRobo may end on another joint branch than this IK; the checked branch is the one nearest the seed.
    Returns the checked goal joints, or None (refused)."""
    if q is None:
        if np.linalg.norm(pos) > MAX_REACH_M:
            say(
                f"  {label}: goal {np.round(pos, 3)} is {np.linalg.norm(pos):.3f} m out (max {MAX_REACH_M})"
                " -- refusing."
            )
            return None
        q, err = solve_ik(scene, rb, pos, quat, seed)
        for _ in range(3):
            if err <= sim.MAX_IK_ERR:
                break
            q, err = solve_ik(scene, rb, pos, quat, q)
        if err > sim.MAX_IK_ERR:
            say(f"  {label}: goal {np.round(pos, 3)} unreachable (IK err {err * 100:.1f} cm) -- refusing.")
            return None
    q = np.asarray(q, float)
    j4, j6 = abs(np.degrees(q[3])), abs(np.degrees(q[5]))
    if j4 > sim.J4_GUARD_DEG or j6 > sim.J6_GUARD_DEG:
        say(f"  {label}: goal joints J4 {j4:.1f} / J6 {j6:.1f} deg past the guards -- refusing.")
        return None
    ko = keep_out_message(rb, q)
    if ko is not None:
        say(f"  {label}: goal is in the keep-out zone ({ko}) -- refusing.")
        return None
    if bodies is not None:
        c = clearance(rb, q, bodies)
        if c[0] < MIN_CLEAR:
            say(f"  {label}: goal {c[0] * 100:.1f} cm from the door model ({c[1]}) -- refusing.")
            return None
        say(f"  {label}: goal ok, {c[0] * 100:.1f} cm from the door model; cuRobo plans the path (not previewed)")
    else:
        say(f"  {label}: goal ok; cuRobo plans the path (not previewed)")
    return q


MAX_REACH_M = 0.90


def execute_joint_plan(ai, plan, label):
    """One JointCommand per step, waiting for convergence to the COMMANDED joints. Kortex
    sometimes rejects a command (ROBOT_MOVEMENT_IN_PROGRESS) and the arm just stays put --
    re-send that step once before giving up (kortex-dropped-substep-bug).
    With self.arm (move_joints) the whole plan is ONE smooth trajectory through every step instead:
    one JointCommand per ~2 cm step became a stop-and-go per step there (loud, 10-07)."""
    if hasattr(ai, "move_joints"):
        ok = ai.move_joints(list(plan))
        if not (ok and wait_for_joints(ai, plan[-1])):
            say(f"  {label}: not at the last step's joints -- stopping here.")
            return False
        return True
    for k, q in enumerate(plan, 1):
        for attempt in (1, 2):
            ai.execute_command(JointCommand(pos=q.tolist()))
            if wait_for_joints(ai, q):
                break
            if attempt == 1:
                say(f"  {label} step {k}: not at the commanded joints after 6 s -- re-sending once.")
                time.sleep(0.5)
        else:
            say(f"  {label} step {k}: still not there after a re-send -- stopping here.")
            return False
    return True


def load_park():
    return wrap_joints(PARK_JOINTS)


def release_and_back_off(ai, back_off_m, execute, save_path=None, park=True, pre_park_x=None):
    """Record the current grasp, open the gripper, back straight off along -approach, then
    (park=True) cuRobo (go_to_joint_config) to the hand-placed park joints -- the end
    position of both the open and the close task. `pre_park_x`: before the park move, go
    straight along world x (toward the base) to this x, same height and orientation -- on
    the open task this clears the open door's free edge, which sticks out toward the arm.
    The straight legs are planned and door-checked, and the park goal checked, before
    anything moves; a failed check means nothing is commanded. cuRobo's path to park is
    not previewed and its world has no door in it -- the straight legs get the hand out
    of the door's way first."""
    save_path = Path(save_path or LAST_GRASP_FILE)
    st = ai.get_state()
    ee = list(st["ee_pos"])
    grasp_pos, quat = np.array(ee[:3]), tuple(ee[3:7])
    approach = R.from_quat(quat).as_matrix()[:, 2]
    q0 = np.array(st["position"], dtype=float)
    say(f"current EE {np.round(grasp_pos, 4)}  gripper {float(st.get('gripper_pos')):.4f}")
    scene, rb = make_sim()
    plan = plan_straight_line(scene, rb, grasp_pos, quat, -approach, back_off_m, q0, "back-off")
    if plan is None:
        say("back-off plan fails a gate -- NOT releasing, arm untouched.")
        return False
    q_park = load_park() if park else None
    legs = [("back-off", plan)]
    if park:
        if not DOOR_FILE.exists():
            say(f"Need {DOOR_FILE} to plan the move to park -- NOT releasing.")
            return False
        bodies, info = add_door_model(scene, rb, json.loads(DOOR_FILE.read_text()), grasp_pos)
        say(f"door model: open {abs(info['open_deg']):.1f} deg, free edge at {np.round(info['free_edge'], 3)}")
        worst = min((clearance(rb, q, bodies) for q in plan), key=lambda w: w[0])
        say(f"  back-off worst clearance {worst[0] * 100:.1f} cm ({worst[1]})")
        q_end = plan[-1]
        if pre_park_x is not None:
            start = grasp_pos - approach * back_off_m
            dx = pre_park_x - start[0]
            if dx < 0:
                leg = plan_straight_line(scene, rb, start, quat, np.array([-1.0, 0, 0]), -dx, q_end, "toward base")
                if leg is None:
                    say("toward-base leg fails a gate -- NOT releasing, arm untouched.")
                    return False
                worst = min((clearance(rb, q, bodies) for q in leg), key=lambda w: w[0])
                say(f"  toward-base leg worst clearance {worst[0] * 100:.1f} cm ({worst[1]})")
                if worst[0] < MIN_CLEAR:
                    say("toward-base leg too close to the door -- NOT releasing, arm untouched.")
                    return False
                legs.append(("toward base", leg))
        if check_curobo_goal(scene, rb, "to park", q=q_park, bodies=bodies) is None:
            say("park goal fails a check -- NOT releasing, arm untouched.")
            return False
    if not execute:
        say(f"\nDRY RUN (release + back-off{' + park' if park else ''}) -- nothing commanded.")
        return True

    back_off_pos = grasp_pos - approach * back_off_m
    Path(save_path).write_text(
        json.dumps(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "grasp_pos": grasp_pos.tolist(),
                "quat": list(quat),
                "grasp_joints": q0.tolist(),
                "back_off_pos": back_off_pos.tolist(),
                "back_off_joints": plan[-1].tolist(),
                "back_off_m": back_off_m,
            },
            indent=1,
        )
    )
    say(f"grasp recorded to {save_path} (for a later --phase regrasp)")

    say("Releasing ...")
    ai.execute_command(OpenGripperCommand())
    time.sleep(1.0)
    ok = True
    for label, leg in legs:
        ok = ok and execute_joint_plan(ai, leg, label)
    if ok and park:
        ok = ai.go_to_joint_config(q_park, "to park")
        say(f"  park: {'reached' if ok else 'did NOT reach the park joints'}")
    final = ai.get_state()
    say(
        f"\nRELEASE {'DONE' if ok else 'STOPPED EARLY'}. final EE:",
        np.round(final["ee_pos"][:3], 4),
        "gripper:",
        final.get("gripper_pos"),
    )
    return ok


# ---------------------------------------------------------------------------
# Door model (for collision-checking moves near the OPEN door)
# ---------------------------------------------------------------------------
# tool_frame -> fingertip distance along the approach axis, measured twice on 09-23
# (6.3 / 6.1 cm); the closed door face sits just past the fingertips at grasp time.
FINGERTIP_PAST_TOOL = 0.062
DOOR_T = 0.04  # door thickness
DOOR_PAST_HANDLE = 0.06  # free edge beyond the handle. Was 0.122 (detector span 47.2 - 35.0 cm), but that
# span includes the control panel: on 09-25 the hand passed the 90-deg door's edge
# at x ~0.27, which only fits a door ending <= ~6 cm past the handle.
DOOR_Z = (0.25, 0.80)  # unmeasured -> deliberately generous (fallback: no closed_handle in the door file)
DOOR_Z_HANDLE_Z = 0.537  # closed-handle z when DOOR_Z was set (09-23); the model keeps DOOR_Z's extent
# relative to the handle, so a lower/higher microwave (09-28: handle ~0.26) is covered
DOOR_Z_MARGIN = 0.02  # padding on the detected door_z range (door file), when the grasp saved one
BODY_DEPTH = 0.40
MIN_CLEAR = 0.03  # every arm/gripper link must stay >= 3 cm from door/body


def door_dims(door):
    """(face behind the closed grasp, door thickness, free edge past the grasp, body depth) for the door model. A door
    file may carry its own (the fridge's, written by its open); the microwave's doesn't -> the constants above."""
    return (
        float(door.get("face_behind_grasp", FINGERTIP_PAST_TOOL + 0.005)),
        float(door.get("door_t", DOOR_T)),
        float(door.get("door_past_handle", DOOR_PAST_HANDLE)),
        float(door.get("body_depth", BODY_DEPTH)),
    )


def save_door_geometry(door_file=None, **kw):
    """Merge door facts into `door_file` (default DOOR_FILE): closed_normal / closed_grasp_pos (grasp), hinge
    (swing)."""
    door_file = door_file or DOOR_FILE
    d = json.loads(door_file.read_text()) if door_file.exists() else {}
    d.update({k: (np.asarray(v).tolist() if not isinstance(v, str) else v) for k, v in kw.items()})
    d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    door_file.write_text(json.dumps(d, indent=1))


def _yaw_about(center, xy, ang):
    d = np.asarray(xy) - center[:2]
    c, s_ = np.cos(ang), np.sin(ang)
    return center[:2] + np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])


def add_door_model(scene, rb, door, open_grasp_pos):
    """Add the door slab (rotated to its current opening) and the fixed microwave body to
    the sim as collision-only boxes. Returns (bodies, info) -- bodies is [(id, name)]."""
    c = rb.physics_client_id
    n = np.asarray(door["closed_normal"], float)
    n[2] = 0
    n /= np.linalg.norm(n)  # out of the door
    g0, hinge = np.asarray(door["closed_grasp_pos"], float), np.asarray(door["hinge"], float)
    t = np.cross([0, 0, 1.0], n)  # along the door
    if np.dot(t, hinge - g0) < 0:
        t = -t  # t points to the hinge
    face_off, door_t, past_handle, body_depth = door_dims(door)
    face = g0[:2] - n[:2] * face_off  # a point on the closed face
    lo = float(np.dot(g0[:2] - face, t[:2])) - past_handle  # free edge (along t)
    hi = float(np.dot(hinge[:2] - face, t[:2]))  # hinge edge
    ang = np.arctan2(*(np.asarray(open_grasp_pos[:2]) - hinge[:2])[::-1]) - np.arctan2(*(g0[:2] - hinge[:2])[::-1])
    ang = (ang + np.pi) % (2 * np.pi) - np.pi
    if door.get("door_z"):
        # the detector's door-face height range (1st/99th pct of the plane fit), saved by the grasp
        z_lo, z_hi = door["door_z"][0] - DOOR_Z_MARGIN, door["door_z"][1] + DOOR_Z_MARGIN
    else:
        dz = float(door["closed_handle"][2]) - DOOR_Z_HANDLE_Z if "closed_handle" in door else 0.0
        z_lo, z_hi = DOOR_Z[0] + dz, DOOR_Z[1] + dz
    zc, zh = (z_lo + z_hi) / 2, (z_hi - z_lo) / 2
    base = np.asarray(scene.robot_base_pose.position)  # identity rotation
    yaw_closed = np.arctan2(n[1], n[0])

    def box(center_xy, half, yaw):
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c)
        return p.createMultiBody(
            0,
            cs,
            basePosition=(base + [center_xy[0], center_xy[1], zc]).tolist(),
            baseOrientation=R.from_euler("z", yaw).as_quat().tolist(),
            physicsClientId=c,
        )

    mid = (lo + hi) / 2
    door_c_closed = face + t[:2] * mid - n[:2] * door_t / 2
    door_id = box(_yaw_about(hinge, door_c_closed, ang), [door_t / 2, (hi - lo) / 2, zh], yaw_closed + ang)
    body_c = face + t[:2] * mid - n[:2] * (door_t + body_depth / 2)
    body_id = box(body_c, [body_depth / 2, (hi - lo) / 2, zh], yaw_closed)
    free_edge = _yaw_about(hinge, face + t[:2] * lo, ang)
    return [(door_id, "door"), (body_id, "microwave body")], {
        "open_deg": float(np.degrees(ang)),
        "free_edge": free_edge,
    }


class DoorFrame:
    """Door geometry from DOOR_FILE. Door-frame coordinates (u, v) at opening angle `deg`:
    u along the door from the hinge toward the free edge, v along the outward normal
    (toward the arm when closed) measured from the slab's mid-plane."""

    def __init__(self, door, sign):
        self.door, self.sign = door, float(sign)
        self.hinge = np.asarray(door["hinge"], float)
        self.g0 = np.asarray(door["closed_grasp_pos"], float)
        n = np.asarray(door["closed_normal"], float)
        n[2] = 0
        self.n = n[:2] / np.linalg.norm(n[:2])
        e = self.g0[:2] - self.hinge[:2]
        e -= self.n * np.dot(e, self.n)
        self.e = e / np.linalg.norm(e)  # hinge -> free edge
        face_off, self.t, past_handle, _ = door_dims(door)  # as add_door_model
        face = self.g0[:2] - self.n * face_off
        self.mid = float(np.dot(face - self.hinge[:2], self.n)) - self.t / 2
        self.u_handle = float(np.dot(self.g0[:2] - self.hinge[:2], self.e))
        self.length = self.u_handle + past_handle

    def _rot(self, deg):
        a = self.sign * np.radians(deg)
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def angle_of(self, xy):
        """Opening angle (deg, positive = open) of a point that was on the grasp line when closed."""
        d = np.asarray(xy[:2]) - self.hinge[:2]
        g = self.g0[:2] - self.hinge[:2]
        a = np.arctan2(g[0] * d[1] - g[1] * d[0], g @ d)
        return float(np.degrees(a) * self.sign)

    def pos(self, u, v, deg, z):
        xy = self.hinge[:2] + self._rot(deg) @ (self.e * u + self.n * (self.mid + v))
        return np.array([xy[0], xy[1], z])

    def quat(self, deg, y_up, yaw_deg=0.0, roll_deg=0.0):
        """Approach (tool z) horizontal, along the door toward the hinge; tool y vertical
        (`y_up` = +-1, keep whatever the wrist has now). `yaw_deg` turns the approach off the
        door line, `roll_deg` rolls the hand about it (which face of the gripper meets the door)."""
        z2 = self._rot(deg + yaw_deg * self.sign) @ -self.e
        zc = np.array([z2[0], z2[1], 0.0])
        yc = np.array([0.0, 0.0, y_up])
        m = np.column_stack([np.cross(yc, zc), yc, zc])
        return (R.from_matrix(m) * R.from_euler("z", roll_deg, degrees=True)).as_quat()

    def bodies_point(self, deg):
        """The closed grasp point swung to `deg` -- what add_door_model reads the angle from."""
        return np.array([*(self.hinge[:2] + self._rot(deg) @ (self.g0[:2] - self.hinge[:2])), self.g0[2]])

    def bodies(self, scene, rb, deg):
        """Door slab at `deg` + microwave body, as [(id, name)] -- remove with `drop`."""
        return add_door_model(scene, rb, self.door, self.bodies_point(deg))[0]


def record_door_angle(ee, door_file=None):
    """After a pull (hand still on the handle): save the opening sense and angle to `door_file`
    (default DOOR_FILE), which the close reads."""
    door = json.loads((door_file or DOOR_FILE).read_text())
    a = DoorFrame(door, 1.0).angle_of(ee)
    save_door_geometry(door_open_deg=abs(a), open_sign=float(np.sign(a) or 1.0), door_file=door_file)
    say(f"door file: open ~{abs(a):.1f} deg (sense {int(np.sign(a) or 1):+d})")


def drop(rb, bodies):
    for bid, _ in bodies:
        p.removeBody(bid, physicsClientId=rb.physics_client_id)


# ---------------------------------------------------------------------------
# Wrist-camera frames for the run log (and the open task's detection frames)
# ---------------------------------------------------------------------------
_camera = None  # perception.WristCamera: set by the node (set_camera), or made on the bench


def set_camera(camera):
    global _camera
    _camera = camera


def camera():
    """The wrist camera. Under the engine the node set it; on the bench (rammp_arm.RammpArm) one is made
    on that arm's node with the default topics."""
    if _camera is None:
        raise Refused("no wrist camera (the node sets one with set_camera)")
    return _camera


def grab_frames(n=1, gap_s=0.3):
    """`n` RGB-D frames from the wrist camera, each with K and base_T_cam (tf2 arm base <- camera).
    Reads topics only -- nothing is commanded."""
    frames = []
    for i in range(n):
        if i:
            time.sleep(gap_s)
        f = camera().grab()
        if f["base_T_cam"] is None:
            raise RuntimeError("no arm base <- camera transform")
        frames.append(f)
    return frames


def save_frame(frame, tag, out_dir, result=None, arm_state=None):
    """<time>_<tag>: _rgb.jpg + .npz (uint16 depth mm, K, base_T_cam, arm joints/tool pose if given) +
    _result.json. ~0.3 MB a frame. Returns the path stem (None if it could not save)."""
    try:
        import cv2

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(frame.get('time', time.time())))}_{tag}"
        if frame.get("rgb") is not None:
            cv2.imwrite(str(stem) + "_rgb.jpg", frame["rgb"], [cv2.IMWRITE_JPEG_QUALITY, 90])
        depth = np.nan_to_num(np.asarray(frame["depth_mm"], dtype=np.float32), nan=0.0)
        extra = {}
        if arm_state is not None:
            extra = {"joints": np.asarray(arm_state["position"], float), "ee": np.asarray(arm_state["ee_pos"], float)}
        if frame.get("base_T_cam") is not None:
            extra["base_T_cam"] = frame["base_T_cam"]
        np.savez_compressed(
            str(stem) + ".npz", depth_mm=np.clip(depth, 0, 65535).astype(np.uint16), K=frame["K"], **extra
        )
        if result is not None:
            Path(str(stem) + "_result.json").write_text(json.dumps(_jsonable(result), indent=1))
        return stem
    except Exception as e:  # never let debugging output stop a run
        say(f"  (could not save the frame: {e})")
        return None


# ---------------------------------------------------------------------------
# Run log: one directory per run
# ---------------------------------------------------------------------------
STREAM_HZ = 25.0  # arm state sampling while a run is on
CMD_EVERY = 5  # keep every 5th commanded trajectory sample (20 ms -> 100 ms)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), 5).tolist()
    if isinstance(x, np.floating | np.integer | np.bool_):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def arm_snapshot(st):
    return {
        "joints_deg": np.degrees(np.asarray(st["position"], float)),
        "ee": np.asarray(st["ee_pos"], float),
        "gripper": float(st["gripper_pos"]),
        "effort": np.asarray(st.get("effort", []), float),
    }


class RunLog:
    """Everything a run leaves behind, in RUNS_DIR/<time>_<what>/:

    run.json  -- args and a timestamped event list: arm state at start/end, the door used
                 for the close (source + values) and its decoded keypoints, every trajectory goal sent
                 (commanded joints every CMD_EVERY samples, ok or not, measured end joints), every gripper
                 command, step outcomes. Rewritten after each event, so a crash keeps what came before.
    arm.npz   -- the arm sampled at STREAM_HZ for the whole run: t, joints, velocity, effort (torques),
                 tool pose, gripper (float32).
    frames    -- camera frames (save_frame: real depth, K, camera pose, arm joints): `frame(tag)`.
    Read-only towards the arm: it only wraps the arm's send/gripper calls to note them."""

    def __init__(self, what, args, ai):
        self.ai, self.t0 = ai, time.time()
        self.dir = RUNS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{what}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            "what": what,
            "args": args,
            "start": time.strftime("%Y-%m-%d %H:%M:%S"),
            "events": [],
        }
        self._samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self._wrap(ai)
        self.event("start", arm=arm_snapshot(ai.get_state()))
        say(f"run log: {self.dir}")

    def event(self, kind, **kw):
        self.data["events"].append({"t": round(time.time() - self.t0, 3), "kind": kind, **_jsonable(kw)})
        try:
            (self.dir / "run.json").write_text(json.dumps(self.data, indent=1))
        except Exception as e:
            say(f"  (run log: could not write run.json: {e})")

    def frame(self, tag):
        """One RGB-D frame from the wrist camera, saved here with the arm's state (camera read only)."""
        try:
            f = grab_frames(1)[0]
            stem = save_frame(f, tag, self.dir, arm_state=self.ai.get_state())
            self.event("frame", tag=tag, path=stem)
        except Exception as e:  # a missing camera must not stop the run
            say(f"  (run log: no '{tag}' frame: {e})")
            self.event("frame_failed", tag=tag, error=str(e))

    def _sample(self):
        while not self._stop.is_set():
            try:
                st = self.ai.get_state()
                self._samples.append(
                    (
                        time.time() - self.t0,
                        st["position"],
                        st.get("velocity", np.zeros(7)),
                        st.get("effort", np.zeros(7)),
                        st["ee_pos"],
                        st["gripper_pos"],
                    )
                )
            except Exception:
                pass
            time.sleep(1.0 / STREAM_HZ)

    def _wrap(self, ai):
        """Note every trajectory goal and gripper command the arm sends (the calls themselves unchanged).
        The wrappers are instance attributes over the class's methods; close() removes them, since under
        the engine the same self.arm serves every run."""
        self._wrapped = []
        if hasattr(ai, "_send"):
            send = ai._send

            def logged_send(t, q, qd, *a, **kw):
                start = round(time.time() - self.t0, 3)
                ok = send(t, q, qd, *a, **kw)
                q = np.asarray(q)
                self.event(
                    "trajectory",
                    start=start,
                    ok=ok,
                    planned_s=float(t[-1]),
                    n_points=len(t),
                    impedance=kw.get("impedance", a[0] if a else None),
                    q_cmd_deg=np.degrees(np.vstack([q[::CMD_EVERY], q[-1:]])),
                    end=arm_snapshot(ai.get_state()),
                )
                return ok

            ai._send = logged_send
            self._wrapped.append("_send")
        if hasattr(ai, "gripper"):
            grip = ai.gripper

            def logged_gripper(position, *a, **kw):
                r = grip(position, *a, **kw)
                self.event(
                    "gripper", cmd=position, force=kw.get("force"), measured=float(ai.get_state()["gripper_pos"])
                )
                return r

            ai.gripper = logged_gripper
            self._wrapped.append("gripper")

    def close(self, ok):
        for name in self._wrapped:
            self.ai.__dict__.pop(name, None)
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.event("end", ok=bool(ok), arm=arm_snapshot(self.ai.get_state()))
        except Exception as e:
            self.event("end", ok=bool(ok), error=str(e))
        if self._samples:
            t, q, dq, tau, ee, g = zip(*self._samples, strict=False)
            np.savez_compressed(
                self.dir / "arm.npz",
                t=np.asarray(t, np.float32),
                joints=np.asarray(q, np.float32),
                velocity=np.asarray(dq, np.float32),
                effort=np.asarray(tau, np.float32),
                ee=np.asarray(ee, np.float32),
                gripper=np.asarray(g, np.float32),
            )
        say(f"run log: {self.dir} ({len(self._samples)} arm samples)")
