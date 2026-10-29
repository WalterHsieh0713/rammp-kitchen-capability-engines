"""Grab the tupperware by the protruding sleeve ring, on self.arm.

Ported from dhyi-svg/feeding-deployment@8f5a3352 (rammp_grasp_tupperware/grasp_tupperware.py and
park.py), which ran on the button press's RammpArm. The detection, the grasp geometry and every
tuned value are copied unchanged; only the arm calls moved:

    RammpArm(execute, acquire_control) / take_control()   -> self.arm (the orchestrator's token)
    arm.status() problems                                  -> self.arm.not_ready()
    set_gripper(): GripperSetpoint on /setpoint/gripper    -> self.arm.gripper()
    arm.curobo_move(p, R, name, log)                       -> self.arm.go_to_ee_pose(p, quat, name)
    dry run: curobo_move plans + gates                     -> self.arm.plan_to_pose()
    park: min-jerk joint trajectory via arm._send()        -> self.arm.move_joints()
    base_link                                              -> rammp_arm.FRAME (arm_mount)

GRASP TARGET: the grabber is the container's flat brim, a RING. The students took the near rim as
the nearest-depth slice of the ring mask. On the flatbot's OAK that slice is the brim's inner edge
(stereo depth on the thin black brim bleeds in from the container wall) and it wanders 2-4 cm
along the rim from frame to frame, so the grasp depth changed with every placement. Instead
(detect_ring): the brim's height from the mask's depth, the mask's outer outline cast onto that
plane, and a circle fit. The outer edge facing the approach is then exact to ~2 mm, and the fitted
radius must be the brim's own (RING_RADIUS_M), which rejects partial or wrong masks.
"""

import time

import numpy as np
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
# The tool point along the approach from the brim's outer edge (negative: short of it). From the
# bags: -0.003 grasped cleanly (20261028_192107), +0.003 went too deep.
GRASP_DEPTH_M = -0.003

STANDOFF_M = 0.10
RAISE_M = 0.06

N_FRAMES = 5
AGREE_M = 0.005  # the frames' outer edges agree to this (a still arm's spread: ~2 mm)
CONF = 0.5
DEPTH_MIN = 0.2
DEPTH_MAX = 2.0
MIN_PTS = 30
RING_RADIUS_M = 0.0843  # the brim's outer radius (the bags' views: 8.40-8.46 cm)
RADIUS_TOL_M = 0.003  # a frame's fit off by more is a partial or wrong mask (good views: +-0.6 mm)
FIT_RMS_M = 0.004  # the outline's rms distance from its circle (good views: 1.5-2 mm)
DETECT_TIMEOUT_S = 3.0  # N_FRAMES good frames within this, or grab refuses

# gripper
GRIP_OPEN = 0.0
GRIP_CLOSE = 1.0  # CLOSE stalls on the ring at the force ceiling
GRIP_SPEED = 1.0
GRIP_FORCE = 0.5  # force = motor-current ceiling, not a force setpoint
GRIP_SETTLE_S = 2.0

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
    import cv2

    fx, fy, cx, cy = K
    ys, xs = np.nonzero(m)
    z = dm[ys, xs]
    ok = (z > DEPTH_MIN) & (z < DEPTH_MAX)
    if ok.sum() < MIN_PTS:
        return f"{int(ok.sum())} ring pixels with depth"
    xs, ys, z = xs[ok], ys[ok], z[ok]
    P = np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1)
    h = float(np.median((P @ R.T + t)[:, 2]))  # the brim is flat: one height, robust to its noisy depth
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    uv = cv2.convexHull(np.vstack(contours))[:, 0, :].astype(float)  # the outer edge only: the mask's ring
    # is usually broken somewhere, and its contour then runs round the inner edge too
    rays = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))], axis=1) @ R.T
    if np.any(rays[:, 2] >= 0):
        return "outline above the horizon"
    c, r, rms = fit_circle((t + ((h - t[2]) / rays[:, 2])[:, None] * rays)[:, :2])
    if rms > FIT_RMS_M:
        return f"outline {rms * 1000:.0f} mm off a circle"
    if abs(r - RING_RADIUS_M) > RADIUS_TOL_M:
        return f"radius {r * 100:.1f} cm, not {RING_RADIUS_M * 100:.1f}"
    return c, r, h, rms


def detect_ring(camera, model, log=None):
    """The brim, agreed by N_FRAMES new frames: its outer edge facing the approach (base frame) and the
    numbers that back it. Raises Abort when the ring is not seen, fits badly, or the frames disagree."""
    a = grasp_orientation()[:, 2]
    a = np.array([a[0], a[1], 0.0]) / np.linalg.norm(a[:2])  # the approach, horizontal
    color = camera.get("color")
    after = camera.stamp_of(color) if color is not None else None  # only frames taken after this call
    rings, confs, why = [], [], {}
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
            why["no ring detected"] = why.get("no ring detected", 0) + 1
            continue
        best = max(idx, key=lambda i: float(r.boxes.conf[i]))
        m = r.masks.data[best].cpu().numpy() > 0.5
        if m.shape != dm.shape:
            import cv2

            m = cv2.resize(m.astype(np.uint8), dm.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        ring = ring_in_frame(m, dm, camera.intrinsics(), *camera.pose_in_base(camera.stamp))
        if isinstance(ring, str):
            why[ring] = why.get(ring, 0) + 1
            continue
        rings.append(ring)
        confs.append(float(r.boxes.conf[best]))
    rejected = "; ".join(f"{k} x{n}" for k, n in why.items()) or "none"
    if len(rings) < N_FRAMES:
        raise Abort(f"ring not seen stably ({len(rings)}/{N_FRAMES} good frames; rejected: {rejected})")
    edges = np.array([[*(c - r * a[:2]), h] for c, r, h, _ in rings])
    edge = np.median(edges, axis=0)
    spread = float(np.max(np.linalg.norm(edges - edge, axis=1)))
    if spread > AGREE_M:
        raise Abort(f"frames disagree by {spread * 1000:.0f} mm (> {AGREE_M * 1000:.0f}) -- not grasping")
    ring = {
        "edge": edge.tolist(),
        "centre": np.median([c for c, *_ in rings], axis=0).tolist(),
        "radius": float(np.median([r for _, r, *_ in rings])),
        "spread_m": spread,
        "fit_rms_m": float(np.max([rms for *_, rms in rings])),
        "conf": float(np.min(confs)),
    }
    if log is not None:
        log(
            f"ring: radius {ring['radius'] * 100:.2f} cm, edge {np.round(edge, 3)}, {len(rings)} frames agree to "
            f"{spread * 1000:.1f} mm, fit rms <= {ring['fit_rms_m'] * 1000:.1f} mm, conf >= {ring['conf']:.2f} "
            f"(rejected: {rejected})"
        )
    return edge, ring


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


def grab(arm, camera, model, execute, log, progress):
    """Detect the ring, then open, pre-grasp, grasp, close and raise. Dry run (`execute` False):
    detect and plan every leg with cuRobo, nothing moves. Returns the result payload."""
    why = arm.not_ready()
    if why is not None:
        raise Abort(why)
    progress("detect", 0.0)
    edge, ring = detect_ring(camera, model, log)
    pre, p_grasp, p_raise, R = grasp_targets(edge)
    quat = Rotation.from_matrix(R).as_quat()
    log(f"grab point base xyz {np.round(p_grasp, 3)}  approach {np.round(R[:, 2], 2)}")
    result = {
        "executed": execute,
        "grasp_point": p_grasp.tolist(),
        "approach": R[:, 2].tolist(),
        "ring": ring,
    }
    legs = [("pre-grasp", pre), ("grasp", p_grasp), ("raise", p_raise)]

    if not execute:
        start = arm.raw_joints()
        for i, (name, p) in enumerate(legs):
            progress(name, 100.0 * (i + 1) / (len(legs) + 1), "dry run: planning")
            plan = arm.plan_to_pose(p, quat, start)
            if not plan.success:
                raise Abort(f"{name}: cuRobo found no path ({plan.message}) -- arm not commanded")
            log(f"{name:16s} cuRobo {len(plan.t):3d} pts  {plan.t[-1]:.1f} s")
            start = plan.q[-1]  # the next leg starts where this one ends
        log("DRY: planned only")
        return result

    progress("grasp", 10.0, "opening")
    arm.gripper(GRIP_OPEN, timeout_s=GRIP_SETTLE_S, force=GRIP_FORCE, speed=GRIP_SPEED)
    for name, p in legs[:2]:
        progress(name, 30.0 if name == "pre-grasp" else 50.0)
        if not arm.go_to_ee_pose(p, quat, name):
            raise Abort(f"{name}: {arm.last_error} -- HOLDING HERE")
    progress("grasp", 70.0, "closing")
    arm.gripper(GRIP_CLOSE, timeout_s=GRIP_SETTLE_S, force=GRIP_FORCE, speed=GRIP_SPEED)

    # One gentle raise, then HOLD. No re-detect check and no auto-reopen:
    # the wrist camera moves with the arm, so a re-detect after lifting isn't
    # reliable -- we just lift a little and keep the grip, confirm by eye.
    progress("raise", 85.0)
    result["raised"] = arm.go_to_ee_pose(p_raise, quat, "raise")
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
