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

GRASP TARGET: the grabber is a protruding RING. The grasp point is the near rim (nearest-depth
slice of the ring mask), NOT the ring center (that's the empty hole over the box).
"""

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
GRASP_DEPTH_M = -0.03

STANDOFF_M = 0.10
RAISE_M = 0.06

N_FRAMES = 5
AGREE_M = 0.01
NEAR_FRAC = 0.20
CONF = 0.5
DEPTH_MIN = 0.2
DEPTH_MAX = 2.0
MIN_PTS = 30

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


def near_rim_point(pts):
    """Point ON the near rim: keep the nearest-depth slice of the ring mask and median it."""
    pts = np.asarray(pts)
    keep = pts[:, 2] <= np.quantile(pts[:, 2], NEAR_FRAC)
    return np.median(pts[keep], axis=0)


def detect_point_cam(camera, model):
    """One agreed near-rim grab point in the CAMERA frame (N frames must agree)."""
    frames = []
    for _ in range(N_FRAMES * 3):
        if len(frames) >= N_FRAMES:
            break
        frame = camera.frame(2.0)
        if frame is None:
            continue
        bgr, dm = frame
        fx, fy, cx, cy = camera.intrinsics()
        r = model.predict(bgr, conf=CONF, retina_masks=True, verbose=False)[0]
        if r.masks is None or len(r.boxes) == 0:
            continue
        idx = [i for i, c in enumerate(r.boxes.cls.cpu().numpy()) if r.names[int(c)] == GRAB_CLASS]
        if not idx:
            continue
        best = max(idx, key=lambda i: float(r.boxes.conf[i]))
        m = r.masks.data[best].cpu().numpy()
        H, W = dm.shape[:2]
        if m.shape != (H, W):
            import cv2

            m = cv2.resize(m, (W, H))
        ys, xs = np.where(m > 0.5)
        pts = []
        for u, v in zip(xs, ys, strict=True):
            z = float(dm[v, u])
            if DEPTH_MIN < z < DEPTH_MAX:
                pts.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
        if len(pts) >= MIN_PTS:
            frames.append(near_rim_point(pts))
    if len(frames) < N_FRAMES:
        raise Abort(f"ring not seen stably ({len(frames)}/{N_FRAMES} frames)")
    arr = np.array(frames)
    if np.max(np.linalg.norm(arr - np.median(arr, axis=0), axis=1)) > AGREE_M:
        raise Abort(f"frames disagree > {AGREE_M * 100:.0f} cm -- not grasping")
    return np.median(arr, axis=0)


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
    near-rim point in the base frame."""
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
    p_cam = detect_point_cam(camera, model)
    pre, p_grasp, p_raise, R = grasp_targets(camera.to_base(p_cam))
    quat = Rotation.from_matrix(R).as_quat()
    log(f"grab point base xyz {np.round(p_grasp, 3)}  approach {np.round(R[:, 2], 2)}")
    result = {
        "executed": execute,
        "grasp_point": p_grasp.tolist(),
        "approach": R[:, 2].tolist(),
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
