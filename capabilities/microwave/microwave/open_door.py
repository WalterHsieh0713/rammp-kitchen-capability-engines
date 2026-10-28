"""Open the microwave on `rchi-cpu-5`: detect the handle, grasp it, swing the door open, let
go and push it further open -- one run.

    python3 -m microwave.open_door             # dry run
    python3 -m microwave.open_door --execute   # grasp, 50-deg swing, push-open
    python3 -m microwave.open_door --no-push-open --execute   # stop after the swing
    python3 -m microwave.open_door --release --execute

1. Detect (`perception.HandleDetector`): YOLO box around the microwave, depth -> door plane, the
   vertical cluster sticking out of it = handle, the door edge farthest from it = hinge. Looks
   until two agree within 3 cm (handle and hinge).
2. Plan everything before any motion: the grasp (squared to the door face: cuRobo from the view pose to a
   line-up point 25 cm out (or less) -- only that goal is checked, cuRobo's path is not previewed -- then
   straight in along the door normal), and the largest swing (75 down to 45 deg) whose whole arc passes the
   sim gates (IK, reach, joint jump, J4/J6 guards). cuRobo can't follow a line or an arc, so the straight-in
   leg and the swing stay on joint paths planned and checked here.
3. Execute: cuRobo to the line-up point, straight in, close, one blended swing. The hand stays on the
   handle at the end.

4. Push-open (default; `--no-push-open` to stop after the swing): let go of the handle and replay the
   user's push-open demo relative to the door (`demo_replay.py`, demos/push_open_2026-10-07.door.json):
   up, over to the right of the microwave, down, into the opening, push it further open. The swing
   is then PUSH_OPEN_SWING_DEG (the demo starts from a ~50 deg swing). Planned and sim-gated from
   where the swing actually ends, before anything moves -- so a dry run only lists its keypoints.

`--release` (separate run, after the swing; instead of the push-open): open the gripper, back straight off the
handle, go toward the base to clear the door's free edge, joint move to `PARK_JOINTS` (microwave_common).

`--fridge` (fridge_constants.py): the same detect + grasp + swing on a fridge, then let go and back out of the handle.
`--fridge --push-open`: the microwave's way instead -- a PUSH_OPEN_PULL_DEG pull, then let go and push the door
further open, round its free edge instead of over it (fridge_push.push_open).

Under the engine: the node's `open` action (node.py -> `run`, which logs the run with RunLog). On the bench:
the commands above (rammp_arm.RammpArm). Refusals raise microwave_common.Refused: the arm holds where it is.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
from capability_engine import sim
from capability_engine.cornell_commands import CloseGripperCommand, OpenGripperCommand
from scipy.spatial.transform import Rotation as R

from . import microwave_common as mc
from .microwave_common import (
    ARM,
    DoorFrame,
    Pose,
    Refused,
    RunLog,
    check_curobo_goal,
    door_arc_waypoints,
    ee_pose,
    fit_circle_xy,
    keep_out_message,
    make_sim,
    multiply_poses,
    plan_cartesian,
    quiet,
    record_door_angle,
    release_and_back_off,
    run_cartesian_trajectory,
    save_door_geometry,
    save_frame,
    say,
)
from .perception import YOLO_MODEL, HandleDetector, WristCamera

# --- detection ---
MAX_LOOKS = 5  # keep looking until two detections agree within DETECT_AGREE
MAX_EMPTY_RETRIES = 8  # per look: retries on an empty read (viewpoint-dependent YOLO misses)
DETECT_AGREE = 0.03
HINGE_AGREE = 0.03  # the two agreeing looks' hinges must agree this well too (horizontal)
PLAUSIBLE_X = (0.30, 0.85)
PLAUSIBLE_Y = (-0.50, 0.20)
PLAUSIBLE_Z = (
    0.05,
    0.65,
)  # was 0.15 (rchi-cpu-5); Jetson+sheppy 10-07: handle z 0.142, microwave ~4 cm above the arm base
MAX_FACE_YAW_DEG = 70.0  # beyond this the plane fit is more likely wrong than the door

# --- grasp (tuned on the arm, see NOTES.md) ---
GRIP_EXT = 0.088  # along the approach axis (negative = past the detected handle). Was -0.010 (rchi-cpu-5);
# Jetson+sheppy+Capybara calib 10-07 touch check: planned grasp 9.8 cm too deep
VERTICAL_CORR = 0.0  # world z. 0.025 (rchi-cpu-5) -> 0.046 (10-07 touch) -> 0: the detector now returns the
# MIDDLE of the bar, not a point near its bottom end
GRASP_X_ADJUST = 0.040  # along the approach axis
PREGRASP_STANDOFFS = (0.25, 0.20, 0.15, 0.10)  # line up this far straight out from the grasp (door normal, level),
# already squared, then go straight in along the normal; the largest that passes the gates
# (10 cm still looked diagonal on 10-21: most of the approach was the view -> line-up leg)
GRASP_PITCHES_DEG = (14, 10, 17, 25, 30, 35, 0)  # TEMPORARY 14 first (10-21): the arm's base is tilted a little on
# this rig, so the user's hand-placed grasp "perpendicular to the door" was 14 deg nose-down
# (demos/user_grasp_pose_2026-10-21.json). REAL DEPLOYMENT: the base will be level -> go back
# to (0, 10, 17, 25, 30, 35) (level first).
# Nose-down tilts tried about the finger axis, IN ORDER (fingers stay
# horizontal across the vertical bar); the smallest reaching the largest swing wins. A low
# handle needs one: 10-07, handle z ~0.14, level -> swing max 25-35 deg (J6 hits its guard),
# 17 -> 70, 25 -> 75 with J6 ending at 92 (sim); the user's hand-placed touches were ~17.
MAX_REACH, MIN_Z, MAX_Z = 0.915, 0.15, 0.75
GRASP_TRACK_ABORT = 0.03

# --- swing ---
SWING_WAYPOINT_SPACING_M = 0.02  # sim-checked spacing
SWING_SEND_EVERY = 4  # Kortex slows at every waypoint: send every 4th (8 cm chords) + the last
SWING_DIRECTION = -1
SWING_MAX_TRY_DEG = 75  # tries this down to 45 in 5-deg steps; 80 dragged the microwave (09-29)
SWING_MIN_DEG = 45
SWING_MAX_JUMP_DEG = 25.0
SWING_MAX_REACH = 0.90
SWING_IK_PROBE_M = 0.01  # 200-iteration IK must get within 1 cm (else Kortex hits SINGULARITY_REGION)
J6_GUARD_DEG = 115.0  # J6 hard limit is +-119.7
GRASP_FORCE = 0.25  # gripper current ceiling on the handle (self.arm default 0.5): a looser hold lets the
# bar turn/shift in the fingers instead of tipping the microwave forward (10-07)
HINGE_BEHIND_FACE = 0.016  # real pivot vs the detected hinge (the door's visible edge): this far behind the door
HINGE_PAST_EDGE = 0.021  # face and this far beyond the edge, away from the handle. Measured 10-07 from the
# hand's path in a stiff 50-deg swing (circle fit rms 0.6 mm; r 36.0 vs 33.7 cm)
SWING_FIT_MIN_DEG = 15  # fit a hinge to the measured path only over at least this much arc
HINGE_RADIUS_RANGE = (0.25, 0.45)

# --- release ---
BACK_OFF_M = 0.20
PUSH_OPEN = Path(__file__).resolve().parent / "demos" / "push_open_2026-10-07.door.json"
PUSH_OPEN_SWING_DEG = 50  # the push-open demo starts from a ~50 deg swing
PUSH_OPEN_TRUST_FROM = 3  # the 50-deg door model contradicts the user's real push-open poses from leg 3 on
PRE_PARK_X = 0.216

# --- --fridge: fridge_constants.OVERRIDES replaces the constants above of the same name (see main) ---
FRIDGE = False
DOOR_SAVE_FILE = None  # None = microwave_common.DOOR_FILE
FRIDGE_HINGE_OFFSET_FROM_GRASP = None  # fridge: measured hinge offset from the closed-door grasp point
#   (door frame: into the door, along it toward the hinge, up)
FRIDGE_LAST_GRASP_FILE = None  # fridge: the release's grasp record (the microwave's stays microwave_common's)
FRIDGE_HANDLE_DETECTOR = (
    None  # fridge: "depth" (this file's detect_handle, fridge_detect.FridgeDepthDetector) or "sam3"
)
FRIDGE_DOOR_MODEL = None  # fridge: door-model geometry written into its door file (microwave_common.door_dims)
FRIDGE_DOOR_Z_FROM_HANDLE = (
    None  # fridge: door bottom / top relative to the handle (its door is taller than any detection)
)
FRIDGE_AFTER_SWING = None  # fridge: "release" or "push_open" -- what follows the swing, planned with it (see main)


def _swing_waypoints(start_pose, hinge, deg):
    """Door-arc waypoints [x, y, z, qx, qy, qz, qw] from `start_pose` about `hinge`."""
    radius = float(np.linalg.norm(np.asarray(start_pose.position)[:2] - np.asarray(hinge)[:2]))
    wps = door_arc_waypoints(
        start_pose=start_pose,
        hinge_position=tuple(hinge),
        arc_length_m=radius * np.radians(deg),
        waypoint_spacing_m=SWING_WAYPOINT_SPACING_M,
        direction=SWING_DIRECTION,
        rotate_orientation=True,
    )
    return [list(w.position) + list(w.orientation) for w in wps]


class _SimState:
    """An arm state for planning from a simulated pose (the after-swing check)."""

    def __init__(self, ee, q, gripper):
        self.st = {"ee_pos": list(ee), "position": list(q), "gripper_pos": gripper}

    def get_state(self):
        return self.st


def _fridge_after_swing(scene, rb, end_wp, q_end, door):
    """Fridge: plan what follows the swing from its end pose `end_wp` and joints `q_end`, against `door` (the door dict
    as it will be after the swing): the release, back-off and route back round the door (fridge_push.
    plan_release_around), or with --push-open the push-open. Returns (ok, message). Prints nothing."""
    import fridge_push

    if FRIDGE_AFTER_SWING == "push_open":
        with quiet():
            ok = fridge_push.plan_push_open(door, _SimState(end_wp, q_end, 0.8).get_state())[0] is not None
        return ok, ("push-open OK" if ok else "push-open: no hand placement behind the door passes every gate")
    pts, msg = fridge_push.plan_release_around(scene, rb, door, end_wp, q_end)
    return pts is not None, msg


def _fridge_door_after(n, g_p, h, hinge, deg, end_pos):
    """The fridge door dict as the open will have saved it once the swing ends (for planning the after-swing step)."""
    door = dict(
        closed_normal=list(n),
        closed_grasp_pos=list(g_p),
        closed_handle=list(h),
        door_z=list(h[2] + np.asarray(FRIDGE_DOOR_Z_FROM_HANDLE)),
        hinge=list(hinge),
        door_open_deg=float(deg),
        **FRIDGE_DOOR_MODEL,
    )
    door["open_sign"] = float(np.sign(DoorFrame(door, 1.0).angle_of(end_pos)) or 1.0)
    return door


def _check_arc(scene, rb, wps, q0):
    """Chained sim IK over every swing waypoint from joints `q0`. Returns (ok, q_end, message)."""
    q = np.asarray(q0, dtype=float)
    for i, w in enumerate(wps):
        pos, quat = w[:3], w[3:]
        for j, jj in enumerate(ARM):
            p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=rb.physics_client_id)
        wpose = multiply_poses(scene.robot_base_pose, Pose(tuple(pos), tuple(quat)))
        sol = p.calculateInverseKinematics(
            rb.robot_id,
            rb.end_effector_id,
            list(wpose.position),
            list(wpose.orientation),
            physicsClientId=rb.physics_client_id,
            maxNumIterations=200,
        )
        nq = np.array(sol[:7])
        for j, jj in enumerate(ARM):
            p.resetJointState(rb.robot_id, jj, float(nq[j]), physicsClientId=rb.physics_client_id)
        ikerr = np.linalg.norm(np.array(ee_pose(rb).position) - np.array(wpose.position))
        jump = float(np.max(np.degrees(np.abs(nq - q))))
        j6 = float(np.degrees(nq[5]))
        if (
            ikerr > SWING_IK_PROBE_M
            or np.linalg.norm(pos) > SWING_MAX_REACH
            or jump > SWING_MAX_JUMP_DEG
            or abs(j6) > J6_GUARD_DEG
            or abs(np.degrees(nq[3])) > sim.J4_GUARD_DEG
        ):
            return (
                False,
                q,
                (
                    f"waypoint {i + 1}: ik_err {ikerr * 100:.2f}cm, reach {np.linalg.norm(pos):.3f}, "
                    f"jump {jump:.1f}deg, J6 {j6:.1f}deg"
                ),
            )
        ko = keep_out_message(rb, nq)  # fridge runs only (None for the microwave)
        if ko is not None:
            return False, q, f"waypoint {i + 1}: {ko}"
        q = nq
    return True, q, f"all {len(wps)} waypoints OK, final J6 {np.degrees(q[5]):.1f}deg"


HANDLE_DETECTOR_MODEL = YOLO_MODEL  # the node sets its yolo_model parameter here
FRAME_ARM = None  # fridge runs: the arm, so each saved detection frame also keeps its joints + tool pose


def _save_frame(frame, tag, attempt, det):
    """Keep every detection's colour + depth frame (and the result) for looking at misses afterwards: real
    depth (uint16 mm), K and the camera pose (microwave_common.save_frame). Until 10-21 the depth was written
    as a float32 PNG, which saturated to 0/255 -- those older frames have no usable depth."""
    try:
        arm_state = None
        if FRAME_ARM is not None:
            try:
                arm_state = FRAME_ARM.get_state()
            except Exception as e:
                say(f"  (could not read the arm for the detection frame: {e})")
        save_frame(frame, f"look{tag}_try{attempt}", mc.DETECT_FRAMES_DIR, result=det, arm_state=arm_state)
    except Exception as e:  # never let debugging output stop a run
        say(f"  (could not save the detection frame: {e})")


def _rounded(x, decimals):
    return "-" if x is None else np.round(x, decimals)


def detect_handle():
    """Two agreeing looks. Returns (handle position, handle orientation, door normal, door_z, door_mid, hinge)."""
    cam = mc.camera()
    if not cam.wait_for_frames(30.0):
        raise Refused("No RGB-D frames")
    if FRIDGE_HANDLE_DETECTOR is not None:
        from fridge_detect import DETECTORS  # the fridge capability's (not in this package)

        detector = DETECTORS[FRIDGE_HANDLE_DETECTOR]()
    else:
        detector = HandleDetector(HANDLE_DETECTOR_MODEL)

    def look(tag):
        for attempt in range(1, MAX_EMPTY_RETRIES + 1):
            f = cam.grab()
            det = detector.detect(f["rgb"], f["K"], f["depth_mm"], f["base_T_cam"])
            _save_frame(f, tag, attempt, det)
            if det is not None:
                break
            say(f"  no detection on look {tag} (retry {attempt}/{MAX_EMPTY_RETRIES})")
            time.sleep(0.5)
        else:
            raise Refused(f"NO DETECTION ({tag}) after {MAX_EMPTY_RETRIES} retries")
        say(
            f"  look {tag}: handle {np.round(det['handle'], 4)}  hinge {_rounded(det['hinge'], 4)}  "
            f"door normal {np.round(det['normal'], 3)}  door z {_rounded(det['door_z'], 3)}"
        )
        return det["handle"], det["quat"], det["normal"], det["door_z"], det["door_mid"], det["hinge"]

    looks, pair = [], None
    for i in range(1, MAX_LOOKS + 1):
        looks.append(look(str(i)))
        pair = next(
            (looks[j] for j in range(len(looks) - 1) if np.linalg.norm(looks[j][0] - looks[-1][0]) <= DETECT_AGREE),
            None,
        )
        if pair is not None:
            break
    if pair is None:
        raise Refused(f"No two of {len(looks)} detections agreed within {DETECT_AGREE * 100:.0f} cm -- refusing.")
    (ha, _, na, zra, mida, hga), (hb, orient, nb, zrb, midb, hgb) = pair, looks[-1]
    say(
        f"detections agree to {np.linalg.norm(ha - hb) * 100:.1f} cm after {len(looks)} look(s); "
        + (f"hinges to {np.linalg.norm(hga[:2] - hgb[:2]) * 100:.1f} cm" if hga is not None else "no hinge detected")
    )
    if hga is not None and np.linalg.norm(hga[:2] - hgb[:2]) > HINGE_AGREE:
        raise Refused(f"the two hinge detections differ by more than {HINGE_AGREE * 100:.0f} cm -- refusing.")
    h = (ha + hb) / 2.0
    hinge = None if hga is None else (hga + hgb) / 2.0  # None: the detector sees no door (fridge SAM 3)
    n = (na + nb) / np.linalg.norm(na + nb)
    door_z = None if zra is None else np.mean([zra, zrb], axis=0)
    door_mid = None if mida is None else np.mean([mida, midb], axis=0)
    if door_z is not None and not (door_z[0] < h[2] < door_z[1] and 0.15 <= door_z[1] - door_z[0] <= 0.60):
        say(f"door z range {np.round(door_z, 3)} doesn't fit a door around the handle -- not saving it")
        door_z = None
    for nm, v, (lo, hi) in (("x", h[0], PLAUSIBLE_X), ("y", h[1], PLAUSIBLE_Y), ("z", h[2], PLAUSIBLE_Z)):
        if not lo <= v <= hi:
            raise Refused(f"handle {nm}={v:.3f} outside plausible {(lo, hi)} -- refusing.")
    say(f"handle (mean) {np.round(h, 4)}  hinge (mean) {_rounded(hinge, 4)}")
    return h, orient, n, door_z, door_mid, hinge


def grasp_pose(h, orient, n, pitch_deg=0.0):
    """Grasp pose for handle position `h` (vertical-corrected here), squared to the door face, tilted
    `pitch_deg` nose-down about the finger axis. The tool point doesn't depend on the tilt."""
    h = h + np.array([0.0, 0.0, VERTICAL_CORR])
    quat = R.from_quat(orient) * R.from_euler("y", np.pi)
    # The detector's orientation is a fixed constant (right only for a door facing the arm
    # head-on); yaw it about world z so the approach axis points into the face (-normal).
    # Yaw only, so the roll -- fingers closing across the vertical bar -- is kept.
    z0, into = quat.as_matrix()[:, 2], -n
    yaw = float(np.arctan2(z0[0] * into[1] - z0[1] * into[0], z0[0] * into[0] + z0[1] * into[1]))
    say(f"door normal {np.round(n, 3)} -> yaw {np.degrees(yaw):+.1f} deg to face the door square")
    if abs(np.degrees(yaw)) > MAX_FACE_YAW_DEG:
        raise Refused(
            f"Face yaw {np.degrees(yaw):.1f} deg > {MAX_FACE_YAW_DEG} deg -- implausible plane fit? Refusing."
        )
    quat = (R.from_euler("z", yaw) * quat * R.from_euler("x", pitch_deg, degrees=True)).as_quat()
    approach = R.from_quat(quat).as_matrix()[:, 2]
    # the tool point is placed along the LEVEL approach -- where the touch check put it -- whatever the tilt
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    pos = h + approach * (GRASP_X_ADJUST - GRIP_EXT)
    say(f"grasp {np.round(pos, 4)}  range {np.linalg.norm(pos):.3f}")
    if np.linalg.norm(pos) > MAX_REACH or not MIN_Z <= pos[2] <= MAX_Z:
        raise Refused(
            f"GATE FAILED at grasp: range {np.linalg.norm(pos):.3f} (max {MAX_REACH}), "
            f"z {pos[2]:.3f} (range {MIN_Z}-{MAX_Z})"
        )
    return h, Pose(tuple(pos), tuple(quat))


def pregrasp_point(grasp, standoff):
    """`standoff` straight back from the grasp along its level approach axis (= into the door face)."""
    approach = R.from_quat(grasp.orientation).as_matrix()[:, 2]
    approach[2] = 0.0
    approach /= np.linalg.norm(approach)
    return np.asarray(grasp.position) - approach * standoff


def plan_approach(scene, rb, grasp, q0):
    """View pose -> line-up point (cuRobo, already squared to the door; only the goal is checked here) -> straight in
    to the grasp, for the largest PREGRASP_STANDOFFS that passes. Returns (joint plan of the straight-in leg, line-up
    point), or (None, None). The straight-in leg is planned from this IK's joints at the line-up point; cuRobo may end
    on another branch, so the real run re-plans it from the measured joints (and re-checks the swing from the grasp)."""
    g_p, g_q = np.asarray(grasp.position), np.asarray(grasp.orientation)
    for standoff in PREGRASP_STANDOFFS:
        pre_p = pregrasp_point(grasp, standoff)
        if not MIN_Z <= pre_p[2] <= MAX_Z:
            continue
        q_pre = check_curobo_goal(scene, rb, f"to line-up {standoff * 100:.0f} cm out", pos=pre_p, quat=g_q, seed=q0)
        if q_pre is None:
            continue
        leg = plan_cartesian(scene, rb, pre_p, g_q, g_p, g_q, q_pre, "straight in")
        if leg is not None:
            return leg, pre_p
    return None, None


def open_task(ai, execute, max_swing=SWING_MAX_TRY_DEG, min_swing=SWING_MIN_DEG):
    """Detect, plan, grasp, swing. Returns True when the swing finished (or the dry run passed); raises Refused
    on any refusal."""
    scene, rb = make_sim()

    st = ai.get_state()
    q0 = np.asarray(st["position"], dtype=float)
    if float(st["gripper_pos"]) > 0.2:
        say("gripper is closed -- " + ("opening it first" if execute else "the real run opens it first"))
        if execute:
            ai.execute_command(OpenGripperCommand())
            time.sleep(1.5)
            if float(ai.get_state()["gripper_pos"]) > 0.2:
                raise Refused("Gripper still closed after opening -- refusing to grasp.")

    global FRAME_ARM
    if FRIDGE:
        FRAME_ARM = ai
    h_raw, orient, n, door_z, door_mid, hinge_seen = detect_handle()
    advice = ""
    if FRIDGE:
        import fridge_push

        advice = "\n" + fridge_push.placement_advice(h_raw, n, q0)
        say(advice[1:])
        # the measured hinge offset from the closed-door grasp point, in the door's frame: into it, along it, up
        try:
            h, grasp = grasp_pose(h_raw, orient, n)
        except Refused as e:
            raise Refused(f"{e}{advice}") from e
        into, along = -np.r_[n[:2], 0.0] / np.linalg.norm(n[:2]), np.cross(np.r_[n[:2], 0.0], [0.0, 0.0, 1.0])
        along /= np.linalg.norm(along)
        o = FRIDGE_HINGE_OFFSET_FROM_GRASP
        if hinge_seen is not None:
            say(f"detected door edge {np.round(hinge_seen, 4)} (not used: measured hinge)")
        hinge = np.asarray(grasp.position) + into * o[0] + along * o[1] + np.array([0.0, 0.0, o[2]])
        say(f"hinge: grasp + measured offset {np.round(o, 3)} (door frame) = {np.round(hinge, 4)}")
    else:
        # the real pivot sits behind the face and beyond the visible edge (HINGE_*), expressed in the door's own frame
        away = np.r_[hinge_seen[:2] - h_raw[:2], 0.0]
        away /= np.linalg.norm(away)
        hinge = hinge_seen - np.r_[n[:2], 0.0] / np.linalg.norm(n[:2]) * HINGE_BEHIND_FACE + away * HINGE_PAST_EDGE
        say(
            f"hinge: detected edge {np.round(hinge_seen, 4)} -> corrected pivot {np.round(hinge, 4)} "
            f"({HINGE_BEHIND_FACE * 100:.1f} cm behind the face, {HINGE_PAST_EDGE * 100:.1f} cm past the edge)"
        )
        h, grasp = grasp_pose(h_raw, orient, n)  # runs the grasp gates once, prints the grasp point
    radius = float(np.linalg.norm(np.asarray(grasp.position)[:2] - hinge[:2]))
    say(f"hinge (detected) {np.round(hinge, 4)}  radius {radius * 100:.1f} cm")
    if not HINGE_RADIUS_RANGE[0] <= radius <= HINGE_RADIUS_RANGE[1]:
        raise Refused(f"hinge radius {radius * 100:.1f} cm is implausible (door ~35 cm) -- refusing.{advice}")

    # --- plan everything before any motion: grasp path + largest swing, for each nose-down tilt ---
    best = None  # (swing deg, pitch, grasp, leg)
    for pitch in GRASP_PITCHES_DEG:
        with quiet():
            _, g = grasp_pose(h_raw, orient, n, pitch)
            leg_p, _ = plan_approach(scene, rb, g, q0)
        if leg_p is None:
            say(f"  tilt {pitch:2d} deg: grasp path fails a gate")
            continue
        for d in range(max_swing, min(min_swing, max_swing) - 1, -5):
            wps_d = _swing_waypoints(g, hinge, d)
            ok, q_end, msg = _check_arc(scene, rb, wps_d, leg_p[-1])
            if ok and FRIDGE:
                # the fridge plans what follows the swing too, before any motion (the microwave's push-open is
                # planned after the swing, from where it really ends)
                ok, m2 = _fridge_after_swing(
                    scene, rb, wps_d[-1], q_end, _fridge_door_after(n, g.position, h, hinge, d, wps_d[-1][:3])
                )
                msg = f"{msg}; {m2}"
            if ok:
                break
        else:
            d, ok = None, False
        say(
            f"  tilt {pitch:2d} deg: grasp J4 {np.degrees(leg_p[-1][3]):6.1f} J6 {np.degrees(leg_p[-1][5]):5.1f} -> "
            + (f"swing {d} deg OK ({msg})" if ok else f"no swing >= {min_swing} deg ({msg})")
        )
        if ok and (best is None or d > best[0]):
            best = (d, pitch, g, leg_p)
        if ok and d == max_swing:
            break  # smallest tilt that reaches the largest swing
    if best is None:
        raise Refused(
            f"no tilt in {GRASP_PITCHES_DEG} gives a grasp + swing of >= {min_swing} deg that passes the "
            f"sim gates -- refusing before any motion. Start joints {np.round(np.degrees(q0), 1)}.{advice}"
        )
    deg, pitch, grasp, leg = best
    g_p, g_q = np.asarray(grasp.position), np.asarray(grasp.orientation)
    say(f"chosen: tilt {pitch} deg nose-down, swing {deg} deg; grasp {np.round(g_p, 4)}")
    _, pre_p = plan_approach(scene, rb, grasp, q0)  # re-print the chosen path

    if not execute:
        say(f"\nDRY RUN -- would grasp, then swing {deg} deg. Nothing commanded.")
        return True

    # --- grasp: cuRobo to the line-up point, then straight in (re-planned from the measured joints) ---
    if not ai.go_to_ee_pose(pre_p, g_q, "to line-up"):
        raise Refused("Did not reach the line-up point -- ABORT before touching the handle, gripper untouched.")
    _, e, _ = run_cartesian_trajectory(ai, [(list(g_p), list(g_q))])
    say(f"grasp: tracking {e * 100:.1f} cm")
    if e > GRASP_TRACK_ABORT:
        raise Refused(f"Tracking {e * 100:.1f} cm at the grasp -- ABORT, gripper untouched.")
    # closed-door geometry for the close task (door_z/door_mid written even when None, so stale ones can't linger)
    door_model = {}
    if FRIDGE:
        # the fridge's measured door (fridge_push's plans check against it): the camera sees only part of its height
        door_z, door_model = h[2] + np.asarray(FRIDGE_DOOR_Z_FROM_HANDLE), FRIDGE_DOOR_MODEL
    save_door_geometry(
        closed_normal=n,
        closed_grasp_pos=g_p,
        closed_handle=h,
        door_z=door_z,
        door_mid=door_mid,
        door_file=DOOR_SAVE_FILE,
        **door_model,
    )
    if hasattr(ai, "gripper"):
        ai.gripper(1.0, force=GRASP_FORCE)  # looser hold (sheppy)
    else:
        ai.execute_command(CloseGripperCommand())
    t0, last = time.time(), -1.0
    time.sleep(0.6)
    while time.time() - t0 < 3.5:  # until the fingers stop closing
        g = float(ai.get_state()["gripper_pos"])
        if abs(g - last) < 0.003:
            break
        last = g
        time.sleep(0.2)
    say(f"gripper after close: {g:.4f} (can't confirm a grasp on this rig -- watch it)")
    if g < 0.2:
        raise Refused("Gripper is OPEN -- nothing grasped. Refusing to swing.")

    # --- swing, re-checked from where the arm actually is ---
    st = ai.get_state()
    ee = list(st["ee_pos"])
    wps = _swing_waypoints(Pose(tuple(ee[:3]), tuple(ee[3:7])), hinge, deg)
    ok, q_end, msg = _check_arc(scene, rb, wps, st["position"])
    if not ok:
        raise Refused(f"swing re-check from the real grasp failed ({msg}) -- stopping, still holding the handle.")
    if FRIDGE:
        ok, m2 = _fridge_after_swing(scene, rb, wps[-1], q_end, _fridge_door_after(n, g_p, h, hinge, deg, wps[-1][:3]))
        say(f"after-swing re-check from the real grasp: {m2}")
        if not ok:
            raise Refused(
                "what follows the swing would fail from the real grasp -- stopping BEFORE the swing, still holding "
                "the handle (door closed). Run with --release instead of --swing-deg to let go and back off." + advice
            )
    if FRIDGE:  # no detected hinge: the door model turns about the measured pivot too
        save_door_geometry(hinge=hinge, hinge_pivot=hinge, detected_edge=hinge_seen, door_file=DOOR_SAVE_FILE)
    else:
        save_door_geometry(
            hinge=hinge_seen, hinge_pivot=hinge, door_file=DOOR_SAVE_FILE
        )  # demos + door model use the detected edge; swing uses the pivot
    sent = wps[SWING_SEND_EVERY - 1 :: SWING_SEND_EVERY]
    if sent[-1] is not wps[-1]:
        sent.append(wps[-1])
    say(f"swing {deg} deg: {len(wps)} waypoints checked, sending {len(sent)} ...")
    ok, err, _ = run_cartesian_trajectory(ai, [(w[:3], w[3:]) for w in sent])
    path = np.array([p_[1:] for p_ in getattr(ai, "last_path", [])])
    final = ai.get_state()
    arc_deg = 0.0
    if len(path) > 1:
        a0, a1 = path[0, :2] - hinge[:2], path[-1, :2] - hinge[:2]
        arc_deg = abs(np.degrees(np.arctan2(a0[0] * a1[1] - a0[1] * a1[0], a0 @ a1)))
    if len(path) > 20 and arc_deg >= SWING_FIT_MIN_DEG:
        c, r, rms = fit_circle_xy(path[:, :2])
        cmd = np.array([w[:3] for w in wps])
        dev = np.max(np.linalg.norm(path[:, None, :2] - cmd[None, :, :2], axis=2).min(axis=1))
        say(
            f"  measured hand path: {len(path)} samples, max {dev * 100:.1f} cm off the commanded arc; circle fit "
            f"hinge {np.round(c, 4)} r {r * 100:.1f} cm (rms {rms * 1000:.1f} mm) vs detected hinge "
            f"{np.round(hinge[:2], 4)} r {np.linalg.norm(np.asarray(ee[:2]) - hinge[:2]) * 100:.1f} cm -- "
            f"off by {np.linalg.norm(c - hinge[:2]) * 100:.1f} cm"
        )
    say(
        f"SWING {'DONE' if ok else 'STOPPED SHORT'}: EE {np.round(final['ee_pos'][:3], 4)} "
        f"({err * 100:.1f} cm from the last waypoint)"
    )
    if FRIDGE:
        record_door_angle(final["ee_pos"], door_file=DOOR_SAVE_FILE)
    else:
        record_door_angle(final["ee_pos"])
    return bool(ok)


def push_open(ai, execute):
    """After the swing: release the handle and push the door further open (the user's 10-07 demo, door-relative)."""
    from . import demo_replay

    say("\n===== release + push further open =====")
    if not execute:
        say("  (starts where the swing ends: planned and sim-gated then, before it moves)")
        return True
    return demo_replay.run(PUSH_OPEN, True, PUSH_OPEN_TRUST_FROM, ai=ai)


def fridge_release(ai, execute):
    """After a full fridge pull: open the gripper, back straight out of the handle, then back round the door to its
    inner side (fridge_push.plan_release_around). All of it is planned from where the arm really is before letting go;
    if any of it fails a check, nothing moves and the hand stays on the handle. No park: the park pose is the
    microwave's."""
    import fridge_push

    say("\n===== release, back off, round the door =====")
    if not execute:
        say("  (starts where the swing ends: planned and sim-checked then, before it moves)")
        return True
    st = ai.get_state()
    pts, msg = fridge_push.plan_release_around(
        *fridge_push._sim(), json.loads(DOOR_SAVE_FILE.read_text()), st["ee_pos"], st["position"]
    )
    say(f"  {msg}")
    if pts is None:
        say("  NOT releasing -- the hand stays on the handle. --release lets go and backs off only.")
        return False
    for pp, _, label in pts[1:]:
        say(f"    {label[:40]:40s} -> {np.round(pp, 3)}")
    if not release_and_back_off(ai, BACK_OFF_M, True, save_path=FRIDGE_LAST_GRASP_FILE, park=False):
        return False
    ok, err, _ = run_cartesian_trajectory(ai, [(list(pp), list(qq)) for pp, qq, _ in pts[1:]])
    say(
        f"ROUND THE DOOR {'DONE' if ok else 'STOPPED SHORT'}: hand {np.round(ai.get_state()['ee_pos'][:3], 3)} "
        f"({err * 100:.1f} cm from the last keypoint)"
    )
    return bool(ok)


def run(ai, execute, swing_deg=None, push=True, speed=1.0, log_args=None):
    """The open task as the node runs it (was door_task.py `open`): grasp + swing, then (`push`) let go and
    replay the push-open demo, all logged to RUNS_DIR/<time>_open (RunLog). `speed` (0..1] scales every arm
    motion for this run only. Returns True when it finished (or the dry run passed); raises Refused."""
    import rammp_arm

    saved = (rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE)
    rammp_arm.CART_SPEED *= speed
    rammp_arm.JOINT_SPEED *= speed
    rammp_arm.GO_TO_SPEED_SCALE = max(0.01, rammp_arm.GO_TO_SPEED_SCALE * speed)
    log = RunLog("open", log_args or {}, ai)
    ok = False
    try:
        log.frame("run_start")
        swing = swing_deg or (PUSH_OPEN_SWING_DEG if push else SWING_MAX_TRY_DEG)
        say("\n===== OPEN: grasp + swing =====")
        ok = open_task(ai, execute, swing)
        if not ok:
            say("swing did not finish -- stopping (still holding the handle).")
        elif push:
            ok = push_open(ai, execute)
        log.event("open", ok=ok)
        say(f"\nOPEN {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
        return ok
    except BaseException as e:
        log.event("stopped", error=repr(e))
        raise
    finally:
        log.close(ok)
        rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE = saved


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--execute", action="store_true", help="command the arm; omit for a dry run")
    a.add_argument(
        "--swing-deg",
        type=int,
        default=None,
        help=f"largest swing to try (deg; default {PUSH_OPEN_SWING_DEG} with the push-open, where the demo "
        f"starts, else {SWING_MAX_TRY_DEG})",
    )
    a.add_argument("--no-push-open", action="store_true", help="stop after the swing, still holding the handle")
    a.add_argument(
        "--release",
        action="store_true",
        help="after a --no-push-open swing: open the gripper, back off the handle, park",
    )
    a.add_argument(
        "--fridge",
        action="store_true",
        help="open a fridge instead: depth handle detection over the whole image (or SAM 3, see "
        "fridge_constants.HANDLE_DETECTOR), measured hinge, fridge_constants.py; "
        "grasp, swing, then let go and back out of the handle (no push-open, no park; --release "
        "does just the let-go + back-out). Swings exactly --swing-deg (default fridge_constants.SWING_DEG) "
        "or refuses",
    )
    a.add_argument(
        "--push-open",
        action="store_true",
        help="with --fridge: open it the microwave's way -- pull fridge_constants.PUSH_OPEN_PULL_DEG (or "
        "--swing-deg), let go, go round the door's free edge at hand height and replay the microwave "
        "push-open demo's push (fridge_push.push_open). The microwave always does this unless "
        "--no-push-open",
    )
    args = a.parse_args()
    if args.push_open and not args.fridge:
        sys.exit("--push-open is for --fridge (the microwave pushes open by default)")
    if args.fridge:
        import fridge_constants

        globals().update(fridge_constants.OVERRIDES, FRIDGE=True)
        import fridge_push

        fridge_push.enable_keep_out()  # the wheelchair: every plan and every trajectory sent is checked
        swing = min_swing = args.swing_deg or (
            fridge_constants.PUSH_OPEN_PULL_DEG if args.push_open else fridge_constants.SWING_DEG
        )  # exactly this swing, or refuse
        globals()["FRIDGE_AFTER_SWING"] = "push_open" if args.push_open else "release"
    else:
        swing = args.swing_deg or (SWING_MAX_TRY_DEG if args.no_push_open else PUSH_OPEN_SWING_DEG)
        min_swing = SWING_MIN_DEG
    import rammp_arm

    mc.configure_paths(mc.DATA_DIR)
    ai = rammp_arm.RammpArm()
    mc.set_camera(
        WristCamera(
            ai.node,
            "/wrist_camera/color/image_raw",
            "/wrist_camera/color/camera_info",
            "/wrist_camera/aligned_depth_to_color/image_raw",
            rammp_arm.FRAME,
        )
    )
    try:
        _run(ai, args, swing, min_swing)
    except Refused as e:
        sys.exit(str(e))
    finally:
        ai.close()


def _run(ai, args, swing, min_swing):
    if args.release and args.fridge:
        release_and_back_off(ai, BACK_OFF_M, args.execute, save_path=FRIDGE_LAST_GRASP_FILE, park=False)
    elif args.release:
        release_and_back_off(ai, BACK_OFF_M, args.execute, pre_park_x=PRE_PARK_X)
    elif open_task(ai, args.execute, swing, min_swing):
        if args.fridge and args.push_open:
            import fridge_push

            fridge_push.push_open(ai, args.execute)
        elif args.fridge:
            fridge_release(ai, args.execute)
        elif not args.no_push_open:
            push_open(ai, args.execute)


if __name__ == "__main__":
    main()
