"""Close the microwave door: push it shut with the side of the hand -- no grasp.

The route-over push-close that latched the door on 2026-10-21 (feeding-deployment
`real_gen3_ros2_close_microwave.py --phase push-close`, code `door_push.push_close`). The gripper is never
touched: the hand may be holding something (a container -- `hold_offset_x`, `keep_orientation`).

1. Route to the swing start, `side_past_hinge` past the hinge and `push_radius` from it -- with
   `route_over`: up over the open door's top, left past the swing start, down, right + forward
   (going left at hand height drives J4 into its limit).
2. Swing about the hinge, hand orientation fixed, radius spiralling in to `push_radius_end`, until
   `push_stop_short` from the closed door in the model.
3. The final push: one slow continuous move straight in, stopped by the joint-torque check once the
   door is shut (`push_contact_nm`).
4. Back off the door, then back out up to `retreat_dist`.

Everything is planned and gated (IK, per-step jump, J4/J6, +-180 wrap, clearance to the door model and
microwave body) before anything moves; a failed gate means nothing is commanded. The door model is the
one the open saved (DOOR_FILE); it doesn't include the handle or whatever the gripper holds.

Also what the fridge's push-close runs (fridge_push.close: `door_file`, `door_shift`, `behind`,
`stop_deg`, `max_tilt_deg`).
"""

import json
import time
from types import SimpleNamespace

import numpy as np
import pybullet as p
import rammp_arm
from capability_engine import sim
from capability_engine.cornell_commands import CartesianTrajectoryCommand
from scipy.spatial.transform import Rotation as R

from . import microwave_common as mc
from .microwave_common import (
    ARM,
    FINGERTIP_PAST_TOOL,
    MIN_CLEAR,
    DoorFrame,
    RunLog,
    add_door_model,
    clearance,
    continuous_ok,
    drop,
    ee_pose,
    execute_joint_plan,
    make_sim,
    plan_cartesian,
    save_door_geometry,
    say,
    solve_ik,
)

ARC_STEP_M = 0.02  # tool travel per sweep waypoint
HANDLE_MARGIN = 0.02  # refuse a swing end whose hand comes closer than this to the handle line
GAP = 0.04  # swing start: closest link this far from the open door (>= MIN_CLEAR)
CLOSE_FRONT_MARGIN = 0.06  # the sideways leg passes this far in front of the free edge (past fingertips)
PUSH_STEP_M = 0.005  # final push step
PUSH_SPEED_SCALE = 0.25  # the push: one continuous move at this x the CONTACT speed (~1.2 cm/s average),
# whatever CART_SPEED is (rammp_arm.contact_speed_scale, 10-24)
CLOSE_HAND_HALF = 0.04  # half-width of the hand along the door, for the handle check (the 09-25 hand demo
# ended 27.6 cm out, handle bar at 33 cm, clear)


def _chain(scene, rb, points, q, bodies, label, min_clear=None, closed=False):
    """plan_cartesian through a list of (pos, quat) points; returns (legs, q_end) or (None, q).
    `min_clear`: optional per-leg clearance, {leg index (0-based): metres}, default MIN_CLEAR."""
    legs = []
    for i in range(len(points) - 1):
        (p0, q0), (p1, q1) = points[i], points[i + 1]
        leg = plan_cartesian(
            scene, rb, p0, q0, p1, q1, q, f"{label} {i + 1}", bodies, min_clear=(min_clear or {}).get(i), closed=closed
        )
        if leg is None:
            return None, q
        legs.append((f"{label} {i + 1}", leg))
        q = leg[-1]
    return legs, q


def _facing_forward(quat, fwd):
    """`quat` turned by the smallest rotation that points its approach (tool z) along `fwd` --
    keeps the hand's roll, e.g. a container held upright stays upright."""
    z = R.from_quat(quat).as_matrix()[:, 2]
    axis = np.cross(z, fwd)
    s, c = np.linalg.norm(axis), float(np.dot(z, fwd))
    turn = R.identity() if s < 1e-9 else R.from_rotvec(axis / s * np.arctan2(s, c))
    return (turn * R.from_quat(quat)).as_quat()


def _plan_close(scene, rb, df, st, args, deg0, behind=False, stop_deg=0.0):
    """To `side_past_hinge` beyond the hinge and `push_radius` from it (diagonally, straight
    there, if that clears the door; else sideways in front of the free edge first), then swing
    about the hinge -- hand orientation unchanged the whole way, radius spiralling in to
    `push_radius_end` -- until it is `preload` into the closed door, then straight back out.
    This is the user's hand-demonstrated close (09-25): straight there kept J3 moving AWAY from
    +-180, where sideways-first drove it across.
    `behind` (fridge: a tall door, no going over it): only the route back out in front of the open door's free edge,
    left past it, then forward to the swing start. `stop_deg` > 0 (fridge: it swings shut by itself from there): the
    swing ends where the door, pushed by the hand, is at `stop_deg` instead of `push_stop_short` from closed."""
    fwd = np.r_[-df.n, 0.0]  # into the microwave, closed-door normal
    side = np.r_[-df.e, 0.0]  # along the closed door, toward the hinge
    q0 = np.array(st["position"], float)
    p_now = np.array(st["ee_pos"][:3])
    if args.face_door:
        quat = _facing_forward(st["ee_pos"][3:7], fwd)
    elif args.keep_orientation:
        quat = tuple(st["ee_pos"][3:7])
    else:
        # default (user, 10-07): fingers pointing LEFT -- horizontal, along the closed door toward the hinge --
        # pushing with the side of the hand; the turn happens on the way to the swing start
        quat = _facing_forward(st["ee_pos"][3:7], side)
    z = args.push_z if args.push_z is not None else p_now[2]
    if args.route_over:
        # --route-over: the swing/push run at mid door height (unless --push-z)
        if not df.door.get("door_z"):
            say("--route-over needs door_z in the door file (the grasp/detection writes it) -- refusing.")
            return None
        z_top = float(df.door["door_z"][1])
        if args.push_z is None:
            z = float(np.mean(df.door["door_z"]))
    h = np.r_[df.hinge[:2], z]

    def a(pt):  # + toward/past the hinge
        return float(np.dot(pt - h, side))

    def b(pt):  # + into the microwave (hinge at 0)
        return float(np.dot(pt - h, fwd))

    bodies = df.bodies(scene, rb, deg0)
    free = add_door_model(scene, rb, df.door, df.bodies_point(deg0))
    drop(rb, free[0])
    free_edge = np.r_[free[1]["free_edge"], z]
    # sideways leg must pass in front of the free edge: fingertips + margin short of it
    b_safe = min(b(p_now), b(free_edge) - FINGERTIP_PAST_TOOL - CLOSE_FRONT_MARGIN)

    def pos(aa, bb):
        return h + side * aa + fwd * bb

    # swing start: `side_past_hinge` past the hinge, or further until it clears the door at its
    # actual angle (a door open ~90 deg needs more than one open ~110 deg)
    a_side = args.side_past_hinge
    while True:
        if args.push_radius <= abs(a_side) or a_side > 0.25:
            say("no swing start clears the open door within 25 cm past the hinge -- refusing.")
            drop(rb, bodies)
            return None
        b_start = -np.sqrt(args.push_radius**2 - a_side**2)
        qs, err = solve_ik(scene, rb, pos(a_side, b_start), quat, q0)
        if err < sim.MAX_IK_ERR and clearance(rb, qs, bodies)[0] >= GAP:
            break
        a_side += 0.01
    say(
        f"hinge {np.round(df.hinge[:2], 3)}; open door free edge ~{np.round(free_edge[:2], 3)} "
        f"(door {deg0:.0f} deg); hand now a {a(p_now) * 100:+.0f} cm / b {b(p_now) * 100:+.0f} cm from the hinge"
    )
    say(
        f"plan: to a {a_side * 100:+.0f} cm (past the hinge) / b {b_start * 100:+.0f} cm (radius "
        f"{args.push_radius * 100:.0f} cm), swing shut spiralling in to {args.push_radius_end * 100:.0f} cm, back out "
        f"{args.retreat_dist * 100:.0f} cm; z {z:.3f}, approach "
        f"{np.round(R.from_quat(quat).as_matrix()[:, 2], 3)}"
    )
    try:
        c0 = clearance(rb, q0, bodies)
        say(f"  clearance now: {c0[0] * 100:.1f} cm ({c0[1]})")
        start = (p_now, tuple(st["ee_pos"][3:7]))
        goal = (pos(a_side, b_start), quat)
        # 1. the demo (09-25): left at the current depth, then straight forward. 2. straight there.
        # 3. round the free edge's corner.
        # 4. back to in front of the free edge first, then left, then forward.
        corner = pos(a(free_edge) + CLOSE_FRONT_MARGIN, b(free_edge) - FINGERTIP_PAST_TOOL - CLOSE_FRONT_MARGIN)
        routes = [
            ("left, then forward", [start, (pos(a_side, b(p_now)), quat), goal]),
            ("straight there", [start, goal]),
            (f"round the free edge's corner via {np.round(corner[:2], 3)}", [start, (corner, quat), goal]),
            (
                f"back to b {b_safe * 100:+.0f} cm, left, then forward",
                [start, (pos(a(p_now), b_safe), quat), (pos(a_side, b_safe), quat), goal],
            ),
        ]
        if behind:
            routes = routes[3:]  # back in front of the free edge, left, forward -- the 4th above
        elif args.route_over:
            # 09-28 (user): going left at hand height drives J4 into its limit, so instead go UP over
            # the open door, LEFT past the swing start, DOWN to mid microwave height, then RIGHT and
            # FORWARD into the swing start -- the swing and push after that are unchanged
            z_over = z_top + args.over_above
            up = np.r_[p_now[:2], z_over]
            left_hi = pos(a_side + args.over_past, b(p_now))
            left_hi[2] = z_over
            down = pos(a_side + args.over_past, b(p_now))
            say(
                f"route-over: up to z {z_over:.3f} (door top {z_top:.3f} + {args.over_above * 100:.0f} cm), left to "
                f"a {(a_side + args.over_past) * 100:+.0f} cm, down to z {z:.3f} (mid door), "
                "right + forward to the swing start"
            )
            routes = [
                (
                    "over the door: up, left, down, right + forward",
                    [start, (up, quat), (left_hi, quat), (down, quat), goal],
                )
            ]
        route = None
        for name, pts in routes:
            say(f" route: {name}")
            route, q = _chain(scene, rb, pts, q0, bodies, "approach")
            if route is not None:
                break
    finally:
        drop(rb, bodies)
    if route is None:
        return None

    # the swing: rotate the start point about the hinge, fixed orientation, until the hand is
    # `push_stop_short` from the CLOSED door (model), found step by step then bisected. The
    # model's closed face came out 1-1.5 cm too deep on 09-25, so the swing stops short and the
    # last bit is the torque-watched push (_push_until_shut). With `stop_deg`, the door model at
    # that angle stands in for the closed one: the hand reaching it = the door pushed to there.
    closed = df.bodies(scene, rb, stop_deg)
    stop = args.push_stop_short
    r0 = pos(a_side, b_start) - h
    # radius spirals in from push_radius to push_radius_end over the swing
    # hand's angle from the closed door line at the start; it meets the closed door ~15 deg out
    span = max(float(np.degrees(np.arccos(np.clip(np.dot(r0[:2], df.e) / np.linalg.norm(r0[:2]), -1, 1)))) - 15.0, 10.0)

    def shrink(d):
        return 1.0 - (1.0 - args.push_radius_end / args.push_radius) * min(d / span, 1.0)

    def at(d):
        return h + np.r_[df._rot(-d) @ r0[:2] * shrink(d), 0.0]

    step = np.degrees(ARC_STEP_M / args.push_radius)
    plan, poses, d, qq = [], [], 0.0, q
    door_lbl = "closed door" if not stop_deg else f"{stop_deg:.0f}-deg door"
    try:
        while True:
            d_next = d + step
            nq, err = solve_ik(scene, rb, at(d_next), quat, qq)
            c_door = clearance(rb, nq, [closed[0]])[0]
            if c_door <= stop:
                lo, hi = d, d_next  # last step: land exactly on the preload
                for _ in range(12):
                    mid = (lo + hi) / 2
                    mq, _ = solve_ik(scene, rb, at(mid), quat, qq)
                    if clearance(rb, mq, [closed[0]])[0] <= stop:
                        hi = mid
                    else:
                        lo = mid
                d_next = hi
                nq, err = solve_ik(scene, rb, at(d_next), quat, qq)
                c_door = clearance(rb, nq, [closed[0]])[0]
            jump = float(np.degrees(np.max(np.abs((nq - qq + np.pi) % (2 * np.pi) - np.pi))))
            c_body = clearance(rb, nq, [closed[1]])
            j6 = float(np.degrees(nq[5]))
            last = c_door <= stop + 1e-4
            bad = (
                err > sim.MAX_IK_ERR
                or jump > sim.MAX_STEP_JUMP_DEG
                or abs(j6) > sim.J6_GUARD_DEG
                or abs(np.degrees(nq[3])) > sim.J4_GUARD_DEG
                or not continuous_ok(qq, nq)
                or c_body[0] < MIN_CLEAR
            )
            k = len(plan) + 1
            if bad or last or k == 1 or k % 4 == 0:
                say(
                    f"  swing {k} ({d_next:5.1f} deg round the hinge) -> {np.round(at(d_next), 3)}  IK err "
                    f"{err * 100:.2f}cm  jump {jump:.1f}deg  J4 {np.degrees(nq[3]):.1f}  J6 {j6:.1f}  to {door_lbl} "
                    f"{c_door * 100:+.1f}cm  body {c_body[0] * 100:.1f}cm ({c_body[1]})"
                )
            if bad:
                say(f"  swing: step {k} fails a gate.")
                return None
            plan.append(nq)
            poses.append((at(d_next), quat))
            qq, d = nq, d_next
            if last:
                break
            if d > deg0 + 60:
                say(
                    f"  swing never reached the {'closed' if not stop_deg else f'{stop_deg:.0f}-deg'} door in the "
                    "model -- refusing."
                )
                return None
        end = at(d)
        u_end = float(np.dot(end[:2] - df.hinge[:2], df.e))
        say(
            f"  swing: {len(plan)} steps, {d:.0f} deg round the hinge; ends with the hand {u_end * 100:.0f} cm "
            f"along the door from the hinge (handle at {df.u_handle * 100:.0f} cm)"
        )
        if u_end + CLOSE_HAND_HALF > df.u_handle - HANDLE_MARGIN:
            say("  the hand would end on the handle (not in the door model) -- refusing; lower --push-radius.")
            return None
        # final push: straight in, PUSH_STEP_M at a time, up to push_max -- by default to the detected
        # closed door + the preload (the door is assumed not to have moved since detection) -- stopped at runtime by
        # the torque check once the door is shut)
        push, pq_ = [], qq
        push_max = args.push_max if args.push_max is not None else args.push_stop_short + args.push_preload
        n_push = int(round(push_max / PUSH_STEP_M))
        for k in range(1, n_push + 1):
            tgt = end + fwd * PUSH_STEP_M * k
            nq, err = solve_ik(scene, rb, tgt, quat, pq_)
            jump = float(np.degrees(np.max(np.abs((nq - pq_ + np.pi) % (2 * np.pi) - np.pi))))
            if err > sim.MAX_IK_ERR or jump > sim.MAX_STEP_JUMP_DEG or not continuous_ok(pq_, nq):
                say(f"  push step {k} fails a gate (IK err {err * 100:.2f}cm, jump {jump:.1f}deg) -- push capped there")
                break
            push.append(nq)
            pq_ = nq
        c_max = clearance(rb, pq_, [closed[0]])[0]
        if n_push:
            say(
                f"  push: up to {len(push)} x {PUSH_STEP_M * 1000:.0f} mm straight in, stopping when the joint "
                f"torques say the door is shut (model: last step {c_max * 100:+.1f} cm vs the closed door)"
            )
        else:
            say(f"  no push (--push-max 0): the swing leaves the door at ~{stop_deg:.0f} deg")
        back = None
        for dist in sorted({args.retreat_dist, 0.15, 0.10, 0.06}, reverse=True):
            if dist <= args.retreat_dist:
                back = plan_cartesian(
                    scene, rb, end, quat, end - fwd * dist, quat, qq, f"back out {dist * 100:.0f} cm", None
                )
                if back is not None:
                    break
        if back is None:
            return None
        c = clearance(rb, back[-1], closed)
        say(f"  after backing out: {c[0] * 100:.1f} cm from door/body ({c[1]})")
    finally:
        drop(rb, closed)
    return {"before": route, "sweep": (plan, poses), "push": push, "after": [("back out", back)]}


def _shift_door(door, dxy):
    """Move the door's recorded positions by `dxy` (arm-base x, y) -- a planning copy."""
    for k in ("hinge", "closed_grasp_pos", "closed_handle", "door_mid"):
        if k in door:
            door[k] = [door[k][0] + dxy[0], door[k][1] + dxy[1], *door[k][2:]]


def _max_tilt_deg(scene, rb, qs, quat):
    """Largest TILT (deg) of the tool, over joint configurations `qs`, away from orientation `quat` (arm-base frame):
    roll and pitch only -- turning about vertical (side to side) is free, as the driver's HOLD_LEVEL.
    Measured as how far "up" (base +z) moves in the tool's own frame. The gates check positions only, so this is
    what confirms the hand really stays level."""
    base = R.from_quat(scene.robot_base_pose.orientation)
    up0 = R.from_quat(quat).inv().apply([0.0, 0.0, 1.0])
    worst = 0.0
    for q in qs:
        for j, jj in enumerate(ARM):
            p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=rb.physics_client_id)
        up = (base.inv() * R.from_quat(ee_pose(rb).orientation)).inv().apply([0.0, 0.0, 1.0])
        worst = max(worst, float(np.degrees(np.arccos(np.clip(up @ up0, -1.0, 1.0)))))
    return worst


def push_close(ai, args, door_file=None, door_shift=None, behind=False, stop_deg=0.0, max_tilt_deg=None):
    """Push the open door shut with the hand facing forward throughout. The gripper is never
    touched (it may be holding a container).
    Fridge (`fridge_push.close`): `door_file` = its own door file; `door_shift` (x, y) = how far the recorded door has
    moved in the arm's frame since it was detected (the wheelchair backed up); `behind` / `stop_deg`: see _plan_close
    (route in front of the free edge instead of over the door; swing only until the door is at `stop_deg`, no push);
    `max_tilt_deg`: refuse unless every planned configuration keeps the hand's tilt (roll / pitch; turning about
    vertical is free) within this of where it starts (holding a container: it must not tip)."""
    door_file = door_file or mc.DOOR_FILE
    if not door_file.exists():
        say(f"need {door_file} -- refusing.")
        return False
    door = json.loads(door_file.read_text())
    deg0 = args.door_deg if args.door_deg is not None else door.get("door_open_deg")
    if deg0 is None or "open_sign" not in door:
        say("door file has no door_open_deg/open_sign (the pull and push-open write them) -- refusing.")
        return False
    if door_shift is not None and np.any(door_shift):
        _shift_door(door, door_shift)
        say(
            f"door moved {np.round(np.asarray(door_shift) * 100, 1)} cm (x, y) since it was recorded; hinge for "
            f"the plan {np.round(door['hinge'][:2], 3)}"
        )
    if args.hold_offset_x:
        # holding a container that sticks out past the fingertips: plan against the door/microwave
        # shifted by this much in x (planning copy only, the door file is not changed), so the hand
        # stays that far back and the container fills the gap
        _shift_door(door, (args.hold_offset_x, 0.0))
        say(
            f"hold offset: planning with the door shifted {args.hold_offset_x * 100:+.1f} cm in x "
            f"(container); hinge for the plan {np.round(door['hinge'][:2], 3)}"
        )
    df = DoorFrame(door, door["open_sign"])
    st = ai.get_state()
    say(
        f"door open ~{deg0:.1f} deg ({'--door-deg' if args.door_deg is not None else 'door file'}); "
        f"gripper {float(st['gripper_pos']):.3f} (left as is)"
    )
    scene, rb = make_sim()
    plan = _plan_close(scene, rb, df, st, args, deg0, behind=behind, stop_deg=stop_deg)
    if plan is None:
        say("\nno plan passes every gate -- refusing, arm untouched.")
        return False
    if max_tilt_deg is not None:
        qs = (
            [q for _, leg in plan["before"] for q in leg]
            + list(plan["sweep"][0])
            + list(plan["push"])
            + [q for _, leg in plan["after"] for q in leg]
        )
        tilt = _max_tilt_deg(scene, rb, qs, st["ee_pos"][3:7])
        say(
            f"hand tilt (roll / pitch) over the whole plan: at most {tilt:.1f} deg from where it starts "
            f"(limit {max_tilt_deg:.0f}; turning about vertical is free)"
        )
        if tilt > max_tilt_deg:
            say("the hand would tilt -- refusing, arm untouched.")
            return False
    if not args.execute:
        say(f"\nDRY RUN (push-close from {deg0:.0f} deg) -- nothing commanded.")
        return True
    for label, leg in plan["before"]:
        if not execute_joint_plan(ai, leg, label):
            return False
    if not _run_sweep(ai, *plan["sweep"], args.smooth):
        return False
    if plan["push"]:  # none for the fridge (--push-max 0): it swings shut from stop_deg
        done = _push_until_shut(ai, plan["push"], args.push_contact_nm)
        # back along the push steps actually taken, to the swing's end, then the planned back-out
        back_steps = plan["push"][: max(done - 1, 0)][::-1] + [plan["sweep"][0][-1]]
        if not execute_joint_plan(ai, back_steps, "off the door"):
            return False
    save_door_geometry(door_open_deg=stop_deg, door_file=door_file)
    for label, leg in plan["after"]:
        if not execute_joint_plan(ai, leg, label):
            return False
    st = ai.get_state()
    say(f"\nPUSH-CLOSE DONE. final EE {np.round(st['ee_pos'][:3], 4)}  gripper {float(st['gripper_pos']):.3f}")
    return True


def _push_until_shut(ai, push, contact_nm):
    """The final push: ONE slow continuous move through the planned push joints, with the joint-torque change
    (`effort`, J1-J4 norm, vs a baseline taken before the push) watched live and the move cancelled the moment it
    passes 2 x `contact_nm` -- the door hitting its frame. A single threshold on the torque itself can't do this:
    its zero drifts with pose (kinova-wrench-bias-not-noise). Returns the number of push steps taken."""
    time.sleep(1.0)
    base = np.array(ai.get_state()["effort"], float)
    stop_nm, peak = 2 * contact_nm, [0.0]
    say(
        f"push: baseline torques J1-J4 {np.round(base[:4], 2)} Nm; one continuous push, stop when the change "
        f"> {stop_nm:.1f} Nm"
    )

    def latched():
        sig = float(np.linalg.norm(np.array(ai.get_state()["effort"], float)[:4] - base[:4]))
        peak[0] = max(peak[0], sig)
        return sig > stop_nm

    ai.move_joints(
        list(push),
        cart_lengths=[PUSH_STEP_M] * len(push),
        speed_scale=PUSH_SPEED_SCALE * rammp_arm.contact_speed_scale(),
        monitor=latched,
    )
    time.sleep(0.3)
    q = np.asarray(ai.get_state()["position"], float)
    k = 1 + int(np.argmin([np.max(np.abs((np.asarray(p_) - q + np.pi) % (2 * np.pi) - np.pi)) for p_ in push]))
    if ai.monitor_stopped:
        say(f"  -> door shut at ~+{k * PUSH_STEP_M * 1000:.0f} mm (torque change {peak[0]:.2f} Nm); push stopped")
    else:
        say(
            f"  push: reached the planned maximum (+{len(push) * PUSH_STEP_M * 1000:.0f} mm, peak torque change "
            f"{peak[0]:.2f} Nm) without a clear contact -- check the door by eye"
        )
    return k


def _run_sweep(ai, sweep, poses, smooth):
    if smooth:
        say(f"sweep: one blended Cartesian trajectory over {len(poses)} waypoints ...")
        ok = ai.execute_command(CartesianTrajectoryCommand([(list(pp), list(qq)) for pp, qq in poses]))
        err = float(np.linalg.norm(np.array(ai.get_state()["ee_pos"][:3]) - poses[-1][0]))
        say(f"  sweep {'done' if ok else 'RETURNED FALSE'}, {err * 100:.1f} cm from the last waypoint")
        return bool(ok) and err < 0.02
    return execute_joint_plan(ai, sweep, "sweep")


# The close's settings (the old script's --flags, same names and defaults). The node's `close` action takes any of
# them as goal params. The 10-21 close that latched, holding a container: route_over, keep_orientation, door_deg 90,
# hold_offset_x -0.152, push_radius 0.17, push_radius_end 0.15, over_above 0.18, push_z 0.22, push_contact_nm 6.
CLOSE_DEFAULTS = {
    "door_deg": None,  # current door opening (deg); None: the door file's last value
    "side_past_hinge": 0.08,  # swing start this far past the hinge, sideways (m)
    "push_radius": 0.28,  # swing start this far from the hinge (m)
    "push_radius_end": 0.26,  # the swing spirals in to this radius, clear of the handle (~33 cm out)
    "face_door": False,  # turn the approach to the closed door's normal first
    "keep_orientation": False,  # keep the hand as it is (holding a container); default: fingers pointing left
    "push_stop_short": 0.03,  # end the swing this far short of the model's closed door (m)
    "push_preload": 0.005,  # the push goes this far past the detected closed door (m)
    "push_max": None,  # push at most this far (m); None: push_stop_short + push_preload
    "push_contact_nm": 3.0,  # J1-J4 torque change (Nm) whose double means the door is shut
    "push_z": None,  # height of the swing and push (m); None: where the hand is now
    "route_over": False,  # reach the swing start over the open door's top
    "over_above": 0.08,  # route_over: this far above the detected door top (m)
    "over_past": 0.05,  # route_over: this far further left than the swing start before coming down (m)
    "hold_offset_x": 0.0,  # plan with the door shifted this far in x (m); -0.152 holding a container
    "retreat_dist": 0.20,  # after the push, back straight out up to this far (m)
    "smooth": False,  # the swing as one blended Cartesian trajectory instead of the checked joints
}


def close_args(execute, **params):
    """The settings push_close reads, as the old argparse namespace: CLOSE_DEFAULTS updated by `params`."""
    unknown = set(params) - set(CLOSE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown close settings: {sorted(unknown)}")
    return SimpleNamespace(**{**CLOSE_DEFAULTS, **params}, execute=execute)


def run(ai, execute, speed=1.0, log_args=None, **params):
    """The close as the node runs it: push_close with `params` (CLOSE_DEFAULTS' names), logged to
    RUNS_DIR/<time>_close (RunLog). `speed` (0..1] scales every arm motion for this run only. Returns True when it
    finished (or the dry run passed)."""
    args = close_args(execute, **params)
    saved = (rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE)
    rammp_arm.CART_SPEED *= speed
    rammp_arm.JOINT_SPEED *= speed
    rammp_arm.GO_TO_SPEED_SCALE = max(0.01, rammp_arm.GO_TO_SPEED_SCALE * speed)
    log = RunLog("close", log_args or vars(args), ai)
    ok = False
    try:
        log.frame("close_start")
        ok = push_close(ai, args)
        log.event("close", ok=ok)
        log.frame("close_end")
        say(f"\nCLOSE {'OK' if ok else 'STOPPED'}{'' if execute else ' (dry run)'}")
        return ok
    except BaseException as e:
        log.event("stopped", error=repr(e))
        raise
    finally:
        log.close(ok)
        rammp_arm.CART_SPEED, rammp_arm.JOINT_SPEED, rammp_arm.GO_TO_SPEED_SCALE = saved
