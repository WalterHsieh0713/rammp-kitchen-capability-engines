"""Grab the tupperware by its brim, on self.arm.

Ported from dhyi-svg/feeding-deployment@8f5a3352 (rammp_grasp_tupperware/grasp_tupperware.py and
park.py), which ran on the button press's RammpArm. The grasp geometry is the students'; the arm
calls moved:

    RammpArm(execute, acquire_control) / take_control()   -> self.arm (the orchestrator's token)
    arm.status() problems                                  -> self.arm.not_ready()
    set_gripper(): GripperSetpoint on /setpoint/gripper    -> self.arm.gripper()
    arm.curobo_move(p, R, name, log)                       -> self.arm.go_to_ee_pose(p, quat, name)
    dry run: curobo_move plans + gates                     -> self.arm.plan_to_pose()
    park: min-jerk joint trajectory via arm._send()        -> self.arm.move_joints()
    base_link                                              -> rammp_arm.FRAME (arm_mount)

The grasp point is new. The students took the nearest-depth slice of the brim's mask. On the
flatbot's OAK that is the brim's inner edge (depth on the thin black brim bleeds in from the
container wall), and it wanders 1-4 cm between frames. detect_ring fits a circle to the mask's
outline on the brim's plane instead: the outer edge facing the approach holds to ~2 mm, and the
radius must be the brim's (RING_RADIUS_M), which rejects partial and wrong masks.
"""

import time
from collections import Counter

import cv2
import numpy as np
from capability_engine.arm import Waypoint
from scipy.spatial.transform import Rotation

GRAB_CLASS = "grabber"

# ============================================================
#  CALIBRATE / tune
# ============================================================
# ---- SIDE PINCH of the flat black brim: reach in horizontally, one finger above
#      the brim, one below. Tune these three while watching a dry/slow run. ----
APPROACH_YAW_DEG = 0.0  # which horizontal direction (base frame, about +z) the gripper
#   reaches ALONG. 0 = +x. Rotate if it comes from a blocked side.
APPROACH_TILT_DEG = 75.0  # 0 = straight top-down pinch, 90 = pure horizontal side approach.
#   she wants ~horizontal; back off toward 75-80 for a slight down-tilt.
FINGER_ROLL_DEG = 90.0  # roll the fingers about the approach axis. If they come out stacked
#   the WRONG way (side-by-side instead of top/bottom), set this to 90.
# Fine-nudge the grasp target off the detected point, BASE frame [dx, dy, dz] meters (z up).
GRASP_OFFSET_BASE = np.array([0.0, 0.0, 0.0])
# The tool point along the approach from the brim's outer edge (negative: short of it). On the
# flatbot 2026-10-28, -0.003 grasped cleanly and +0.003 went too deep; on 2026-10-29 -0.003
# still bumped the container's inside.
GRASP_DEPTH_M = -0.008

STANDOFF_M = 0.10
RAISE_M = 0.06

N_FRAMES = 5
AGREE_M = 0.005  # the frames' outer edges agree to this (a still arm: ~2 mm)
CONF = 0.5
DEPTH_MIN = 0.2
DEPTH_MAX = 2.0
MIN_PTS = 30
RING_RADIUS_M = 0.0843  # the brim's outer radius (measured: 8.40-8.46 cm)
RADIUS_TOL_M = 0.003  # a fit off by more is a partial or wrong mask
FIT_RMS_M = 0.004  # the outline's rms distance from its circle (good views: 1.5-2 mm)
DETECT_TIMEOUT_S = 3.0  # N_FRAMES good frames within this, or grab refuses

# gripper
GRIP_OPEN = 0.0
GRIP_CLOSE = 1.0  # CLOSE stalls on the ring at the force ceiling
GRIP_SPEED = 1.0
GRIP_FORCE = 0.5  # force = motor-current ceiling, not a force setpoint
GRIP_SETTLE_S = 2.0
BLEND_M = 0.02  # view -> pre-grasp -> grasp is one motion (go_through): the pre-grasp corner cut
APPROACH_SPEED_SCALE = 0.6  # that approach's speed_scale; the free-space moves run at arm.go_to_speed_scale

# Custom scan pose, captured from /joint_states (already in RADIANS — no np.radians()).
# Camera looks at the box; joints are mid-range (joint 4 off its limit). To change it:
# hand-guide the arm, then `ros2 topic echo /joint_states --once` and paste joint_1..7.
SCAN_JOINTS = np.array(
    [
        -0.06594519491254225,  # joint_1
        -0.3247426675309537,  # joint_2
        3.124004871434801,  # joint_3
        -2.309249351760864,  # joint_4
        -0.02458630318361177,  # joint_5
        -0.39356993537518026,  # joint_6
        1.582662973138126,  # joint_7
    ]
)


class Abort(Exception):
    """A check failed: nothing more is sent and the arm holds where it is."""


def fit_circle(xy):
    """Least-squares circle through 2D points: (centre, radius, rms distance from it)."""
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    (a, b, k), *_ = np.linalg.lstsq(A, (xy**2).sum(axis=1), rcond=None)
    r = np.sqrt(k + a * a + b * b)
    return np.array([a, b]), r, float(np.sqrt(np.mean((np.linalg.norm(xy - [a, b], axis=1) - r) ** 2)))


def ring_in_frame(m, dm, K, R, t):
    """The brim from one frame's ring mask `m` and depth `dm` (metres), the camera's intrinsics K and
    pose (R, t) in the base frame: (centre xy, outer radius, height, fit rms), or a reason string."""
    fx, fy, cx, cy = K
    ys, xs = np.nonzero(m)
    z = dm[ys, xs]
    ok = (z > DEPTH_MIN) & (z < DEPTH_MAX)
    if ok.sum() < MIN_PTS:
        return f"{int(ok.sum())} ring pixels with depth"
    xs, ys, z = xs[ok], ys[ok], z[ok]
    P = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
    h = float(np.median((P @ R.T + t)[:, 2]))  # the brim is flat: one height, robust to its noisy depth
    # The convex hull is the outer edge: the mask's ring is often broken, and its contour then runs
    # round the inner edge too.
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    uv = cv2.convexHull(np.vstack(contours))[:, 0, :].astype(float)
    rays = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))], axis=1) @ R.T
    if np.any(rays[:, 2] >= 0):
        return "outline above the horizon"
    c, r, rms = fit_circle((t + ((h - t[2]) / rays[:, 2])[:, None] * rays)[:, :2])
    if rms > FIT_RMS_M:
        return f"outline {rms * 1000:.0f} mm off a circle"
    if abs(r - RING_RADIUS_M) > RADIUS_TOL_M:
        return f"radius {r * 100:.1f} cm, not {RING_RADIUS_M * 100:.1f}"
    return c, r, h, rms


def detect_ring(camera, model):
    """The brim's outer edge facing the approach (base frame), agreed by N_FRAMES new frames, and the
    numbers behind it. Abort when the ring is not seen, fits badly, or the frames disagree."""
    a = grasp_orientation()[:2, 2]
    a /= np.linalg.norm(a)  # the approach, horizontal
    rings, confs, rejected = [], [], Counter()
    after = None
    deadline = time.monotonic() + DETECT_TIMEOUT_S
    while len(rings) < N_FRAMES and time.monotonic() < deadline:
        frame = camera.frame(deadline - time.monotonic(), after)
        if frame is None:
            break
        bgr, dm = frame
        after = camera.stamp
        r = model.predict(bgr, conf=CONF, retina_masks=True, verbose=False)[0]
        classes = [] if r.masks is None else r.boxes.cls.tolist()
        idx = [i for i, c in enumerate(classes) if r.names[int(c)] == GRAB_CLASS]
        if not idx:
            rejected["no ring detected"] += 1
            continue
        best = max(idx, key=lambda i: float(r.boxes.conf[i]))
        m = r.masks.data[best].cpu().numpy() > 0.5  # retina_masks: the image's size, as is the depth
        ring = ring_in_frame(m, dm, camera.intrinsics(), *camera.pose_in_base(camera.stamp))
        if isinstance(ring, str):
            rejected[ring] += 1
            continue
        rings.append(ring)
        confs.append(float(r.boxes.conf[best]))
    why = "; ".join(f"{k} x{n}" for k, n in rejected.items()) or "none"
    if len(rings) < N_FRAMES:
        raise Abort(f"ring not seen stably ({len(rings)}/{N_FRAMES} good frames; rejected: {why})")
    edges = np.array([[*(c - r * a), h] for c, r, h, _ in rings])
    edge = np.median(edges, axis=0)
    spread = float(np.max(np.linalg.norm(edges - edge, axis=1)))
    if spread > AGREE_M:
        raise Abort(f"frames disagree by {spread * 1000:.0f} mm (> {AGREE_M * 1000:.0f}) -- not grasping")
    return edge, {
        "radius": float(np.median([r for _, r, *_ in rings])),
        "spread_m": spread,
        "fit_rms_m": max(rms for *_, rms in rings),
        "conf": min(confs),
        "rejected": why,
    }


def grasp_orientation():
    """Build the grasp rotation for a SIDE pinch of the flat brim.
    Tool axes: z = approach (into object), y = finger separation, x = the third.
    yaw/tilt aim the approach; roll spins the fingers about it."""
    yaw, tilt, roll = np.radians([APPROACH_YAW_DEG, APPROACH_TILT_DEG, FINGER_ROLL_DEG])
    a = np.array(
        [np.sin(tilt) * np.cos(yaw), np.sin(tilt) * np.sin(yaw), -np.cos(tilt)]
    )  # approach: tilt=0 -> straight down, 90 -> horizontal
    a /= np.linalg.norm(a)
    up = np.array([0.0, 0.0, 1.0])
    s0 = up - np.dot(up, a) * a  # "as vertical as possible" finger-separation axis
    if np.linalg.norm(s0) < 1e-6:  # approach was vertical; pick any perpendicular
        s0 = np.array([1.0, 0.0, 0.0])
    s0 /= np.linalg.norm(s0)
    s = np.cos(roll) * s0 + np.sin(roll) * np.cross(a, s0)  # roll fingers about the approach axis
    s /= np.linalg.norm(s)
    t = np.cross(s, a)
    return np.column_stack([t, s, a])  # columns = world dirs of tool x, y(sep), z(approach)


def grasp_targets(p_detected):
    """(pre-grasp, grasp, raise) tool positions and the grasp rotation, from the detected
    outer edge of the brim (detect_ring) in the base frame."""
    p_base = np.asarray(p_detected, dtype=float) + GRASP_OFFSET_BASE
    R = grasp_orientation()
    approach = R @ np.array([0, 0, 1.0])
    p_base = p_base + GRASP_DEPTH_M * approach
    pre = p_base - STANDOFF_M * approach
    return pre, p_base, p_base + np.array([0, 0, RAISE_M]), R


def level(R):
    """`R` turned about a horizontal axis until its approach (tool z) is horizontal: the raise
    ends with the gripper level with the world (the held container turns with it)."""
    a = R[:, 2]
    k = np.cross(a, [a[0], a[1], 0.0])
    if np.linalg.norm(k) < 1e-9:  # already level
        return R
    angle = np.arcsin(abs(a[2]))
    return Rotation.from_rotvec(k / np.linalg.norm(k) * angle).as_matrix() @ R


def grab(arm, camera, model, execute, log, progress, grasp_hold=0, **motion):
    """Detect the ring, then open, approach (view -> pre-grasp -> grasp as one motion, the
    pre-grasp corner blended), close, and raise with the gripper turned level. Dry run (`execute`
    False): detect and plan the approach and the raise with cuRobo, nothing moves. `motion`: the
    moves' keyword arguments (control_mode, gains). `grasp_hold`: the grasp waypoint's hold
    (HOLD_*), the pre-grasp -> rim leg (default 0, HOLD_NONE). Returns the result payload."""
    why = arm.not_ready()
    if why is not None:
        raise Abort(why)
    progress("detect", 0.0)
    edge, ring = detect_ring(camera, model)
    log(
        f"ring: radius {ring['radius'] * 100:.2f} cm, {N_FRAMES} frames agree to {ring['spread_m'] * 1000:.1f} mm, "
        f"fit rms <= {ring['fit_rms_m'] * 1000:.1f} mm, conf >= {ring['conf']:.2f} (rejected: {ring['rejected']})"
    )
    pre, p_grasp, p_raise, R = grasp_targets(edge)
    quat = Rotation.from_matrix(R).as_quat()
    log(f"grab point base xyz {np.round(p_grasp, 3)}  approach {np.round(R[:, 2], 2)}")
    result = {
        "executed": execute,
        "grasp_point": p_grasp.tolist(),
        "approach": R[:, 2].tolist(),
        "ring": ring,
    }
    quat_raise = Rotation.from_matrix(level(R)).as_quat()
    through = [
        Waypoint(pos=tuple(pre), quat=tuple(quat), blend_radius=BLEND_M),
        Waypoint(pos=tuple(p_grasp), quat=tuple(quat), hold=grasp_hold),
    ]

    if not execute:
        progress("approach", 30.0, "dry run: planning")
        plan = arm.plan_through(through, arm.raw_joints())
        if not plan.success:
            raise Abort(f"approach: cuRobo found no path ({plan.message}) -- arm not commanded")
        log(f"approach         cuRobo {len(plan.t):3d} pts  {plan.t[-1]:.1f} s")
        progress("raise", 70.0, "dry run: planning")
        plan = arm.plan_to_pose(p_raise, quat_raise, plan.q[-1])  # from where the approach ends
        if not plan.success:
            raise Abort(f"raise: cuRobo found no path ({plan.message}) -- arm not commanded")
        log(f"raise            cuRobo {len(plan.t):3d} pts  {plan.t[-1]:.1f} s")
        log("DRY: planned only")
        return result

    progress("grasp", 10.0, "opening")
    arm.gripper(GRIP_OPEN, timeout_s=GRIP_SETTLE_S, force=GRIP_FORCE, speed=GRIP_SPEED)
    progress("approach", 30.0, "view -> pre-grasp -> grasp, one motion")
    if not arm.go_through(through, "approach", APPROACH_SPEED_SCALE, **motion):
        raise Abort(f"approach: {arm.last_error} -- HOLDING HERE")
    progress("grasp", 70.0, "closing")
    arm.gripper(GRIP_CLOSE, timeout_s=GRIP_SETTLE_S, force=GRIP_FORCE, speed=GRIP_SPEED)

    # One gentle raise, then HOLD. No re-detect check and no auto-reopen:
    # the wrist camera moves with the arm, so a re-detect after lifting isn't
    # reliable -- we just lift a little and keep the grip, confirm by eye.
    progress("raise", 85.0)
    result["raised"] = arm.go_to_ee_pose(p_raise, quat_raise, "raise", **motion)
    log("raised + holding" if result["raised"] else f"raise skipped ({arm.last_error}) -- holding in place")
    return result


def park(arm, execute, log):
    """The scan pose by a DIRECT joint move (no cuRobo): works even when cuRobo refuses the start
    state (INVALID_START_STATE). NOT collision-checked: keep the area clear and watch it."""
    q0 = arm.raw_joints()
    log(f"from {np.round(np.degrees(q0), 1)} deg")
    log(f"to   {np.round(np.degrees(SCAN_JOINTS), 1)} deg (scan pose)")
    if not execute:
        log("DRY: not moving")
        return {"executed": False}
    if not arm.move_joints([SCAN_JOINTS]):
        raise Abort(f"park: {arm.last_error} -- HOLDING HERE")
    log("parked at scan pose")
    return {"executed": True}
