"""Planning for putting the held container down at a spot on the table. No rclpy: tests offline.

Ported from RAMMP-table's place_bowl.py (the spot path that placed the bowl on a marked sticky note
on the flatbot, 2026-10-26). Everything is in arm_mount, the driver's gravity-aligned frame: z true
up, x forward, y left, origin on the arm base axis. The driver applies the mount (flatbot: 30 deg
pitch), so there is no tilt correction here.

Where the container is, along the hand's pointing direction (tool z):
  flange --TOOL_Z (12 cm)--> tool frame --tip_ahead--> fingertips --BOWL_RADIUS--> container centre
The fingers pinch the lip, so the lip is at the tool frame's height and the bottom BOWL_DEPTH below.

The hand holds the container level with the camera on its right (tool x up, tool z = the heading),
pointed `pitch` deg down so the far side touches the table before the fingers.
"""

import numpy as np
from capability_engine import sim
from rammp_arm import CONTINUOUS, TOOL_Z
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

# --- the container and how it is held ---
BOWL_RADIUS = 0.055  # m, centre to the wall the fingertips are pushed in to
LIP_WIDTH = 0.03175  # m, 1.25 in: the flat lip past BOWL_RADIUS
BOWL_DEPTH = 0.08  # m, lip (= tool frame height) to the bottom
FLANGE_TO_TIP = 0.17955  # m, flange face to the closed fingertips (Gen3 + 2F-85; not ruler-measured)
PLACE_PITCH_DEG = 5.0  # the hand points this many deg down while placing
# --- the motion ---
SAFE_HEIGHT = 0.30  # m, tool frame above the table while carrying the container over to the spot
PRESS_BELOW = 0.10  # m, the line down is aimed this far below where the bottom meets the table: in
# impedance the hand lags its command by a few cm, and the table, not the command, stops it
MIN_TOOL_HEIGHT = 0.02  # m, the tool frame is never targeted lower than this above the table
MIN_BELOW_MEET = 0.03  # m, the line down must reach at least this far below where the bottom meets the table
MAX_DROP = 0.40  # m, the longest straight line down
PATH_STEP = 0.002  # m, between the line's joint waypoints
RETREAT = 2.5 * LIP_WIDTH  # m, straight back after letting go: the fingers slide off the lip
RETREAT_MIN = 1.5 * LIP_WIDTH  # m, a shorter line is still taken if it gets this far (past the lip)
RETREAT_STEP = 0.004  # m, between the retreat's joint waypoints
RETREAT_RELEASE = 0.25  # the first quarter of the retreat also lifts the setpoint by the hold's press
MAX_ADJUST_DEG = 15.0  # the squaring turn may only correct this much (a bigger one tips the container)
MAX_BOWL_TILT_DEG = 10.0  # container tilt allowed anywhere along a joint move
MAX_LOWER_TILT_DEG = 1.0  # ... and on the straight way down
# --- checks ---
IK_TOL = 0.005  # m
IK_ROT_TOL_DEG = 1.0
MAX_STEP_JUMP_DEG = 120.0  # one joint move
UP = np.array([0.0, 0.0, 1.0])


class Refused(Exception):
    """A plan or check that does not hold: nothing more is commanded."""


def wrap(rad):
    return (np.asarray(rad) + np.pi) % (2 * np.pi) - np.pi


def wrap_deg(rad):
    """Angle difference in radians -> degrees in [-180, 180)."""
    return np.degrees(wrap(rad))


def hold_reach(flange_to_tip=FLANGE_TO_TIP):
    """Container centre in front of the tool frame (m)."""
    return BOWL_RADIUS + flange_to_tip - TOOL_Z


def hand_orientation(heading, camera_left=False, pitch_deg=PLACE_PITCH_DEG):
    """The hand pointing along `heading` (horizontal part), container level, camera on its right
    (camera_left: the same hold turned 180 deg about the hand's axis), pointed pitch_deg down.
    Level, camera right: tool z = heading, tool y = heading x up (right of the hand seen from
    behind), tool x = up."""
    z = np.array([heading[0], heading[1], 0.0], dtype=float)
    z /= np.linalg.norm(z)
    y = np.cross(z, UP)
    x = np.cross(y, z)
    if camera_left:
        x, y = -x, -y
    R = Rotation.from_matrix(np.column_stack([x, y, z]))
    # about z x up (horizontal, across the hand): a negative turn brings the tip down
    return (Rotation.from_rotvec(-np.radians(pitch_deg) * np.cross(z, UP)) * R).as_quat()


def camera_on_left(R):
    """The hand's camera is on its left (tool x points down rather than up)."""
    return bool(R[:, 0] @ UP < 0)


def bowl_tilt_deg(R, pitch_deg=PLACE_PITCH_DEG):
    """How far the container is from how it should be held: tool x straight up (camera right) or
    down (camera left), leaned forward by pitch_deg when the hand points down."""
    h = np.array([R[0, 2], R[1, 2], 0.0])
    h = h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-6 else np.zeros(3)
    p = np.radians(pitch_deg)
    want = (UP * np.cos(p) + h * np.sin(p)) * (-1.0 if camera_on_left(R) else 1.0)
    return float(np.degrees(np.arccos(np.clip(R[:, 0] @ want, -1.0, 1.0))))


def approach_direction(spot):
    """Horizontal unit vector from the arm base axis toward the spot."""
    d = np.array([spot[0], spot[1], 0.0], dtype=float)
    return d / np.linalg.norm(d)


def tool_above_spot(spot, table_z, height, reach):
    """Tool position that puts the container centre over `spot`, tool `height` above the table."""
    pos = np.array([spot[0], spot[1], table_z]) - reach * approach_direction(spot)
    pos[2] = table_z + height
    return pos


class Model:
    """FK and IK of the arm on the process's shared sim copy (capability_engine.sim: the driver's
    model, rooted at arm_mount). fk() gives 4x4 poses in arm_mount of the tool frame (the sim's ee
    link, arm_tool_frame), the flange or the fingertips."""

    def __init__(self, flange_to_tip=FLANGE_TO_TIP, pitch_deg=PLACE_PITCH_DEG):
        import pybullet as p

        self.scene, self.rb = sim.make_sim()
        self.flange_to_tip = float(flange_to_tip)
        self.pitch = float(pitch_deg)
        self.reach = hold_reach(self.flange_to_tip)
        info = [p.getJointInfo(self.rb.robot_id, j, physicsClientId=self.rb.physics_client_id) for j in sim.ARM]
        # (lower, upper) in rad; continuous joints come back as lower > upper
        self.limits = [(float(i[8]), float(i[9])) for i in info]

    def fk(self, q, link="tool"):
        pos, quat = sim.tool_pose(self.scene, self.rb, q)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T[:3, 3] = pos
        offset = {"tool": 0.0, "flange": -TOOL_Z, "tip": self.flange_to_tip - TOOL_Z}[link]
        T[:3, 3] += offset * T[:3, 2]
        return T

    def tilt(self, q):
        return bowl_tilt_deg(self.fk(q)[:3, :3], self.pitch)

    def limit_violations(self, q):
        bad = []
        for i, (v, (lo, hi)) in enumerate(zip(q, self.limits, strict=True)):
            if lo < hi and not (lo <= v <= hi):
                bad.append(f"J{i + 1} {np.degrees(v):.0f} deg (limit {np.degrees(lo):.0f}..{np.degrees(hi):.0f})")
        return bad

    def ik(self, pos, quat, seed, restarts=10, rot_weight=0.2, stay=0.0):
        """Joints putting the tool frame at (pos, quat), or None. stay > 0 pulls the answer toward
        the seed (m per rad): 7 joints for a 6-number pose, so without it consecutive small steps
        slide along the spare freedom. Of the solutions within IK_TOL / IK_ROT_TOL_DEG and the
        joint limits, the one that moves the joints least from the seed; continuous joints in
        [-pi, pi)."""
        seed = np.asarray(seed, dtype=float)
        target = Rotation.from_quat(quat)
        pos = np.asarray(pos, dtype=float)

        def residual(x):
            T = self.fk(x)
            rot = (Rotation.from_matrix(T[:3, :3]) * target.inv()).as_rotvec()
            return np.concatenate([T[:3, 3] - pos, rot_weight * rot, stay * wrap(x - seed)])

        lo = np.array([a if a < b else -2 * np.pi for a, b in self.limits])
        hi = np.array([b if a < b else 2 * np.pi for a, b in self.limits])
        rng = np.random.default_rng(0)
        best = None
        for k in range(restarts + 1):
            x0 = (
                np.clip(seed, lo + 1e-6, hi - 1e-6)
                if k == 0
                else rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi))
            )
            # PyBullet FK is single precision: the default finite-difference step is too small.
            r = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-4, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            q = r.x.copy()
            q[CONTINUOUS] = wrap(q[CONTINUOUS])
            T = self.fk(q)
            err = np.linalg.norm(T[:3, 3] - pos)
            rot = np.degrees((Rotation.from_matrix(T[:3, :3]) * target.inv()).magnitude())
            if err > IK_TOL or rot > IK_ROT_TOL_DEG or self.limit_violations(q):
                continue
            move = float(np.max(np.abs(wrap_deg(q - seed))))
            if best is None or move < best[0]:
                best = (move, q)
            if k == 0 and move < 30.0:
                break  # the seed's own solution is close: good enough
        return None if best is None else best[1]


def check_joint_move(model, q_a, q_b, table_z, samples=20):
    """Problems along a straight joint-space move: the container tilts, or the hand dips."""
    allowed = max(MAX_BOWL_TILT_DEG, model.tilt(q_a) + 1.0, model.tilt(q_b) + 1.0)
    delta = wrap(np.asarray(q_b) - np.asarray(q_a))
    for s in np.linspace(0.0, 1.0, samples):
        T = model.fk(np.asarray(q_a) + s * delta)
        tilt = bowl_tilt_deg(T[:3, :3], model.pitch)
        if tilt > allowed:
            return [f"the container tilts {tilt:.0f} deg"]
        if table_z is not None and T[2, 3] < table_z + MIN_TOOL_HEIGHT:
            return [f"the hand dips to {(T[2, 3] - table_z) * 100:.1f} cm above the table"]
    return []


def plan_segment(model, q_from, pos_to, quat_to, table_z, name, depth=0):
    """Joint waypoints from q_from to the pose, split in Cartesian halves until each joint move
    keeps the container level and above the table. [(name, q), ...]."""
    q_to = model.ik(pos_to, quat_to, q_from)
    if q_to is None:
        raise Refused(f"{name}: no IK solution for the tool at {np.round(pos_to, 3).tolist()}")
    jump = float(np.max(np.abs(wrap_deg(q_to - q_from))))
    problems = check_joint_move(model, q_from, q_to, table_z)
    if jump <= MAX_STEP_JUMP_DEG and not problems:
        return [(name, q_to)]
    if depth >= 4:
        raise Refused(f"{name}: " + ("; ".join(problems) or f"a joint moves {jump:.0f} deg"))
    T_from = model.fk(q_from)
    mid_pos = (T_from[:3, 3] + np.asarray(pos_to)) / 2
    rots = Rotation.from_matrix(np.stack([T_from[:3, :3], Rotation.from_quat(quat_to).as_matrix()]))
    mid_quat = Slerp([0, 1], rots)([0.5])[0].as_quat()
    first = plan_segment(model, q_from, mid_pos, mid_quat, table_z, name + "'", depth + 1)
    return first + plan_segment(model, first[-1][1], pos_to, quat_to, table_z, name, depth + 1)


def plan_descent(model, q_start, floor_z, step=PATH_STEP, log=None):
    """Joint targets every `step` straight down from q_start's tool pose to floor_z, orientation
    kept. Returns (tool z values, joint arrays); the first entry is q_start itself. Ends early (with
    a note to `log`) where the arm can't go further straight down, if it got at least 1 cm."""
    T = model.fk(q_start)
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    zs = np.arange(T[2, 3], floor_z - 1e-9, -step)
    qs, seed = [np.asarray(q_start, dtype=float)], np.asarray(q_start, dtype=float)
    for z in zs[1:]:
        q = model.ik([T[0, 3], T[1, 3], z], quat, seed, restarts=0, stay=0.01)
        if q is None or np.max(np.abs(wrap_deg(q - seed))) > 3.0 * step / PATH_STEP:
            got = T[2, 3] - zs[len(qs) - 1]
            if got >= 0.01:
                if log is not None:
                    log(f"note: the arm can only go {got * 100:.0f} cm straight down from here (reach / joint limits)")
                zs = zs[: len(qs)]
                break
            raise Refused(f"lower: can't go straight down from z={T[2, 3]:.3f}")
        qs.append(q)
        seed = q
    return np.asarray(zs), np.array(qs)


def check_straight(model, q_a, q_b, samples=20, max_dev=0.005):
    """Problems if the joint move q_a -> q_b strays from the straight line between its end poses by
    more than max_dev, or tilts the container more than MAX_LOWER_TILT_DEG."""
    a, b = model.fk(q_a)[:3, 3], model.fk(q_b)[:3, 3]
    ab = b - a
    if ab @ ab < 1e-8:
        return []
    delta = wrap(np.asarray(q_b) - np.asarray(q_a))
    for s in np.linspace(0.0, 1.0, samples):
        T = model.fk(np.asarray(q_a) + s * delta)
        p = T[:3, 3] - a
        dev = np.linalg.norm(p - (p @ ab) / (ab @ ab) * ab)
        if dev > max_dev:
            return [f"leaves the straight line by {dev * 100:.1f} cm"]
        if bowl_tilt_deg(T[:3, :3], model.pitch) > MAX_LOWER_TILT_DEG:
            return [f"tilts the container {bowl_tilt_deg(T[:3, :3], model.pitch):.0f} deg"]
    return []


def plan_place(model, q_now, spot, table_z, max_adjust=MAX_ADJUST_DEG, log=None):
    """The whole place from joints q_now: (moves [(name, q)], descent (zs, qs), info).

    adjust (square the hand in place to the NEAREST hold) -> lift to SAFE_HEIGHT if lower ->
    transit to above the spot -> the straight line down (descent; aimed PRESS_BELOW under where the
    bottom meets the table, at most MAX_DROP). Raises Refused with the reason."""
    T_now = model.fk(q_now)
    heading = T_now[:3, 2].copy()
    heading[2] = 0.0
    if np.linalg.norm(heading) < 0.3:
        raise Refused("the hand points too steeply up or down to tell which way it faces")
    left = camera_on_left(T_now[:3, :3])
    adjust_quat = hand_orientation(heading, left, model.pitch)
    adjust_deg = float(
        np.degrees((Rotation.from_matrix(T_now[:3, :3]) * Rotation.from_quat(adjust_quat).inv()).magnitude())
    )
    if adjust_deg > max_adjust:
        raise Refused(
            f"the hand is {adjust_deg:.0f} deg from holding the container {model.pitch:.0f} deg down "
            f"(max {max_adjust:.0f} to correct with it in hand) -- re-grip it closer"
        )
    spot = np.asarray(spot, dtype=float)
    moves = plan_segment(model, q_now, T_now[:3, 3], adjust_quat, table_z, "adjust")
    approach = approach_direction(spot)
    quat = hand_orientation(approach, left, model.pitch)
    safe_z = table_z + SAFE_HEIGHT
    if T_now[2, 3] < safe_z - 0.01:
        lift = T_now[:3, 3].copy()
        lift[2] = safe_z
        moves += plan_segment(model, moves[-1][1], lift, adjust_quat, table_z, "lift")
    above = tool_above_spot(spot, table_z, SAFE_HEIGHT, model.reach)
    moves += plan_segment(model, moves[-1][1], above, quat, table_z, "transit")
    floor_z = max(table_z + max(BOWL_DEPTH - PRESS_BELOW, MIN_TOOL_HEIGHT), above[2] - MAX_DROP)
    descent = plan_descent(model, moves[-1][1], floor_z, log=log)
    # The line may end early (reach / joint limits); it must still get below where the bottom meets the
    # table, or the lowering stops in mid-air (the flatbot at 0.54 m, 2026-10-29: 4 cm of the 22).
    meet = table_z + BOWL_DEPTH
    if descent[0][-1] > meet - MIN_BELOW_MEET:
        raise Refused(
            f"lower: the arm can only go down to tool z={descent[0][-1]:.3f} here, "
            f"{(descent[0][-1] - meet) * 100:.0f} cm above where the bottom meets the table -- "
            f"the spot is {np.hypot(*spot[:2]):.2f} m from the base axis; move it within reach"
        )
    info = dict(adjust_deg=adjust_deg, approach=approach, table_z=table_z, floor_z=floor_z, camera_left=left)
    return moves, descent, info


def plan_retreat(model, q_start, release=0.0, step=RETREAT_STEP, log=None):
    """Joint waypoints every `step` along the straight horizontal line RETREAT back from q_start's
    tool pose (back = opposite the hand's heading), orientation kept. release: also lift that much
    over the first RETREAT_RELEASE of it (the press lets go gently). The first entry is q_start. If
    the line runs out of reach after RETREAT_MIN it ends there (flatbot at 0.8 m, 2026-10-26: 6-7 cm
    of the 8)."""
    T = model.fk(q_start)
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    back = -np.array([T[0, 2], T[1, 2], 0.0])
    if np.linalg.norm(back) < 0.3:
        raise Refused("retreat: the hand points too steeply up or down to tell which way is back")
    back /= np.linalg.norm(back)
    n = max(1, int(np.ceil(RETREAT / step)))
    qs, seed = [np.asarray(q_start, dtype=float)], np.asarray(q_start, dtype=float)
    for k in range(1, n + 1):
        f = k / n
        pos = T[:3, 3] + f * RETREAT * back + min(1.0, f / RETREAT_RELEASE) * release * UP
        q = model.ik(pos, quat, seed, restarts=0, stay=0.01)
        if q is None or np.max(np.abs(wrap_deg(q - seed))) > 3.0 * step / PATH_STEP:
            done = (k - 1) / n * RETREAT
            if done >= RETREAT_MIN:
                if log is not None:
                    log(f"retreat: only {done * 100:.1f} cm straight back fits from here (of {RETREAT * 100:.0f})")
                break
            raise Refused(f"retreat: can't go straight back from here ({f * RETREAT * 100:.0f} cm in)")
        qs.append(q)
        seed = q
    return qs


def reset_moves(model, q_now, q_start):
    """Back to a saved start: straight UP to the start's height first (the open fingers rise clear
    of the container instead of sweeping across it), then one joint move to the start's joints."""
    T_now, T_start = model.fk(q_now), model.fk(q_start)
    moves, q = [], np.asarray(q_now, dtype=float)
    if T_start[2, 3] - T_now[2, 3] > 0.005:
        up = T_now[:3, 3].copy()
        up[2] = T_start[2, 3]
        moves = plan_segment(model, q, up, Rotation.from_matrix(T_now[:3, :3]).as_quat(), None, "up")
        q = moves[-1][1]
    problems = check_joint_move(model, q, q_start, None)
    jump = float(np.max(np.abs(wrap_deg(np.asarray(q_start) - q))))
    if problems or jump > MAX_STEP_JUMP_DEG:
        raise Refused("back to the start: " + ("; ".join(problems) or f"a joint moves {jump:.0f} deg"))
    return moves + [("start", np.asarray(q_start, dtype=float))]


def hold_point(line_q, line_z, z_cmd):
    """The joints on the planned line down (joint waypoints line_q at flange heights line_z,
    falling) where the flange is at z_cmd: interpolated between neighbouring waypoints (a few mm
    apart). Above the line's start: None (hold where the arm is)."""
    if z_cmd >= line_z[0]:
        return None
    for i in range(len(line_z) - 1):
        if line_z[i] >= z_cmd >= line_z[i + 1]:
            s = (line_z[i] - z_cmd) / max(line_z[i] - line_z[i + 1], 1e-9)
            return np.asarray(line_q[i]) + s * (np.asarray(line_q[i + 1]) - np.asarray(line_q[i]))
    return np.asarray(line_q[-1])


class VzContact:
    """Contact on the way down from the hand's z velocity (/ee_state, + = up): once the hand has
    moved down faster than `moving`, vz at or above `stop` for `hits` new messages in a row = the
    table stopped it. On the flatbot (2026-10-26) the default 0.0 came 4.5 s late: the wrist folded
    about the fingertips and vz sat at ~-0.5 cm/s; -1 cm/s stops at the touch."""

    def __init__(self, stop=-0.01, moving=0.01, hits=2):
        self.stop, self.moving, self.need = stop, moving, hits
        self.moved, self.count = False, 0

    def update(self, vz):
        self.moved = self.moved or vz < -self.moving
        self.count = self.count + 1 if self.moved and vz >= self.stop else 0
        return self.count >= self.need


# self.arm.move_joints times a path minimum-jerk over its whole length (rammp_arm.joint_trajectory):
# at time fraction u the hand is minjerk_distance(u) of the way along, at minjerk_speed(u) x the
# average speed. The descent is one such move, so it is slow at both ends.
CONTACT_GATE = 0.75  # contact counts only while the commanded speed is at least this x lower_speed
MAX_PEAK = 1.5  # the descent's peak speed is at most this x lower_speed


def minjerk_distance(u):
    u = min(max(u, 0.0), 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def minjerk_speed(u):
    u = min(max(u, 0.0), 1.0)
    return 30 * u**2 * (1 - u) ** 2


def minjerk_time(d):
    """The time fraction at which a minimum-jerk move is fraction d of the way along."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if minjerk_distance(mid) < d else (lo, mid)
    return (lo + hi) / 2


def descent_speed_scale(meet, lower_speed, cart_speed):
    """move_joints' speed_scale for the line down so the hand passes `meet` (0..1 of the way, where
    the table should be) at lower_speed, the speed the contact test was tuned at (RAMMP-table: a
    steady 2 cm/s). Its peak speed is cart_speed x the scale (joint_trajectory), at most
    MAX_PEAK x lower_speed."""
    u = minjerk_time(min(max(meet, 0.05), 0.95))
    peak = min(lower_speed / (minjerk_speed(u) / minjerk_speed(0.5)), MAX_PEAK * lower_speed)
    return peak / cart_speed


def contact_gate(u, length, duration, lower_speed):
    """Whether contact may be called at time fraction u of the descent: the commanded speed is up
    (at the start and end of a minimum-jerk move it is slow, and a hand that slows with its command
    must not read as stopped by the table)."""
    return 0.0 < u < 1.0 and minjerk_speed(u) * length / duration >= CONTACT_GATE * lower_speed
