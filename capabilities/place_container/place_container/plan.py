"""Planning for putting the held container down at a spot on the table. No rclpy: tests offline.

Ported from RAMMP-table's place_bowl.py (the spot path that placed the bowl on a marked sticky note
on the flatbot, 2026-10-26; the arm has since moved 0.2032 m forward on the chair, 2026-11-07, so
positions from those runs are 0.2032 m off in x). Everything is in arm_mount, the driver's gravity-aligned frame: z true
up, x forward, y left, origin on the arm base axis. The driver applies the mount (flatbot: 30 deg
pitch), so there is no tilt correction here.

Where the container is, along the hand's pointing direction (tool z):
  flange --TOOL_Z (12 cm)--> tool frame --tip_ahead--> fingertips --BOWL_RADIUS--> container centre
The fingers pinch the lip, so the lip is at the tool frame's height and the bottom BOWL_DEPTH below.

The hand holds the container level with the camera on its right (tool x up, tool z = the heading),
pointed `pitch` deg down so the far side touches the table before the fingers.
"""

import re

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
# Where the straight line back runs out of reach (the elbow at its limit, close spots): turn the hand
# instead, seen from above, about the vertical through the flange. The fingertips, ~18 cm ahead of
# it, swing out along an arc, still straddling the lip (the turn keeps every height), until they
# are past its outer edge; the wrist barely moves, so the elbow needs no room.
SWING_DEG = 60.0  # deg, the full turn, left or right
SWING_CLEAR = 0.01  # m, the fingertips end at least this far outside the lip's outer edge
# m: the swing starts after as much of the straight line back as fits, so the fingertips first leave the
# container wall they are pushed against -- a swing from the wall drags the container round with it by
# friction (flatbot, 2026-11-10). At least this much of it is wanted; less, and the swing still runs.
SWING_BACKOFF = 0.01
MAX_ADJUST_DEG = 15.0  # the squaring turn may only correct this much (a bigger one tips the container)
MAX_BOWL_TILT_DEG = 10.0  # container tilt allowed anywhere along a joint move
MAX_LOWER_TILT_DEG = 1.0  # ... and on the straight way down
# --- checks ---
IK_TOL = 0.005  # m
IK_ROT_TOL_DEG = 1.0
# rad: IK answers stay this far inside each joint limit -- the driver's joint_limit_margin_rad, which
# refuses a goal any closer (GOAL_IN_LIMIT_MARGIN). Without it the solver pins a joint ON its limit
# and a retreat that ran out of elbow room looked complete (flatbot, 2026-11-10: J4 at 2.570).
LIMIT_MARGIN = 0.10
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

        lo = np.array([a + LIMIT_MARGIN if a < b else -2 * np.pi for a, b in self.limits])
        hi = np.array([b - LIMIT_MARGIN if a < b else 2 * np.pi for a, b in self.limits])
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


def check_joint_move(model, q_a, q_b, table_z, samples=20, tilt_check=True):
    """Problems along a straight joint-space move: the container tilts (tilt_check: only with something
    in the hand), or the hand dips."""
    allowed = max(MAX_BOWL_TILT_DEG, model.tilt(q_a) + 1.0, model.tilt(q_b) + 1.0)
    delta = wrap(np.asarray(q_b) - np.asarray(q_a))
    for s in np.linspace(0.0, 1.0, samples):
        T = model.fk(np.asarray(q_a) + s * delta)
        tilt = bowl_tilt_deg(T[:3, :3], model.pitch)
        if tilt_check and tilt > allowed:
            return [f"the container tilts {tilt:.0f} deg"]
        if table_z is not None and T[2, 3] < table_z + MIN_TOOL_HEIGHT:
            return [f"the hand dips to {(T[2, 3] - table_z) * 100:.1f} cm above the table"]
    return []


def plan_segment(model, q_from, pos_to, quat_to, table_z, name, depth=0):
    """Joint waypoints from q_from to the pose, split in Cartesian halves until each joint move
    keeps the container level and above the table. [(name, q), ...]."""
    q_to = model.ik(pos_to, quat_to, q_from)
    if q_to is None:
        raise Refused(
            f"{name}: no arm pose puts the tool at {np.round(pos_to, 3).tolist()} m "
            f"({np.hypot(*pos_to[:2]):.2f} m from the base axis) -- out of reach or joint limits; "
            "move the spot or the start position"
        )
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
            raise Refused(
                f"lower: the arm can't go straight down from tool z={T[2, 3]:.3f} m at "
                f"{np.hypot(T[0, 3], T[1, 3]):.2f} m from the base axis (reach / joint limits) -- move the spot"
            )
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


def reaches(descent, table_z):
    """The line down gets at least MIN_BELOW_MEET below where the bottom meets the table."""
    return descent[0][-1] <= table_z + BOWL_DEPTH - MIN_BELOW_MEET


def reach_refusal(descent, table_z, spot):
    meet = table_z + BOWL_DEPTH
    return (
        f"lower: the arm can only go down to tool z={descent[0][-1]:.3f} here, "
        f"{(descent[0][-1] - meet) * 100:.0f} cm above where the bottom meets the table -- "
        f"the spot is {np.hypot(*spot[:2]):.2f} m from the base axis"
    )


def find_posture(model, pos, quat, q_from, floor_z, table_z, seeds=40, tries=5, log=None):
    """Another arm posture (7 joints: elbow / wrist arranged differently) with the tool at (pos, quat)
    whose straight line down reaches the table, or None: (q, descent), the one moving the joints least
    from q_from. IK from random seeds; a quick check at the bottom of the line first, then the full
    line for at most `tries` of them."""
    rng = np.random.default_rng(1)
    lo = np.array([a if a < b else -np.pi for a, b in model.limits])
    hi = np.array([b if a < b else np.pi for a, b in model.limits])
    found = []
    for _ in range(seeds):
        q = model.ik(pos, quat, rng.uniform(lo, hi), restarts=0)
        if q is not None and not any(np.max(np.abs(wrap_deg(q - f))) < 10.0 for f in found):
            found.append(q)
    found.sort(key=lambda q: float(np.max(np.abs(wrap_deg(q - q_from)))))
    bottom = np.array([pos[0], pos[1], table_z + BOWL_DEPTH - MIN_BELOW_MEET])
    for q in found:
        if tries == 0:
            break
        if model.ik(bottom, quat, q, restarts=0, stay=0.01) is None:
            continue
        tries -= 1
        try:
            descent = plan_descent(model, q, floor_z)
        except Refused:
            continue
        if reaches(descent, table_z):
            if log is not None:
                log(
                    f"another arm posture reaches the table ({float(np.max(np.abs(wrap_deg(q - q_from)))):.0f} deg "
                    "away at the most-moved joint)"
                )
            return q, descent
    return None


def plan_place(
    model,
    q_now,
    spot,
    table_z,
    max_adjust=MAX_ADJUST_DEG,
    allow_reconfigure=False,
    reach=(0.70, 1.10),
    lower_from=0.0,
    log=None,
):
    """The whole place from joints q_now: (moves [(name, q)], descent (zs, qs), info).

    adjust (square the hand in place to the NEAREST hold) -> lift to SAFE_HEIGHT if lower ->
    transit to above the spot -> the straight line down (descent; aimed PRESS_BELOW under where the
    bottom meets the table, at most MAX_DROP). If the line can't reach the table in the arm's posture,
    find_posture looks for another one; it is used only with allow_reconfigure (the move is then
    "transit-reconfigure", to those exact joints). `reach` (m from the base axis): only the hint in a
    refusal. Raises Refused with the reason."""
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
    # The targets of the carrying motion as one go_to_ee_poses would take them. The adjust changes the
    # hand's tilt, so it can't be held; lift keeps adjust's quat exactly (a held target must keep the
    # previous target's orientation); transit keeps the level hold with a new heading.
    waypoints = [dict(name="adjust", pos=T_now[:3, 3].copy(), quat=adjust_quat, hold="none", q=moves[-1][1])]
    approach = approach_direction(spot)
    quat = hand_orientation(approach, left, model.pitch)
    safe_z = table_z + SAFE_HEIGHT
    if T_now[2, 3] < safe_z - 0.01:
        lift = T_now[:3, 3].copy()
        lift[2] = safe_z
        moves += plan_segment(model, moves[-1][1], lift, adjust_quat, table_z, "lift")
        waypoints.append(dict(name="lift", pos=lift, quat=adjust_quat, hold="level", q=moves[-1][1]))
    above = tool_above_spot(spot, table_z, SAFE_HEIGHT, model.reach)
    q_from = moves[-1][1]
    transit = plan_segment(model, q_from, above, quat, table_z, "transit")
    floor_z = max(table_z + max(BOWL_DEPTH - PRESS_BELOW, MIN_TOOL_HEIGHT), above[2] - MAX_DROP)
    descent = plan_descent(model, transit[-1][1], floor_z, log=log)
    # The line may end early (reach / joint limits); it must still get below where the bottom meets the
    # table, or the lowering stops in mid-air (the flatbot at 0.54 m, 2026-10-29: 4 cm of the 22).
    reconfigure = False
    if not reaches(descent, table_z):
        why = reach_refusal(descent, table_z, spot)
        other = find_posture(model, above, quat, q_from, floor_z, table_z, log=log)
        if other is None:
            raise Refused(
                why + f"; no other arm posture reaches it either -- move the spot to {reach[0]:.2f}-{reach[1]:.2f} m "
                "from the base axis (the reach param)"
            )
        if not allow_reconfigure:
            raise Refused(
                why + "; ANOTHER arm posture reaches it: allow_reconfigure:=true to use it "
                "(a bigger move, planned by cuRobo in joint space)"
            )
        q_t, descent = other
        problems = check_joint_move(model, q_from, q_t, table_z)
        if problems:
            raise Refused("transit to the other posture: " + "; ".join(problems))
        transit, reconfigure = [("transit-reconfigure", q_t)], True
    moves += transit
    # joints: True = a joint target (go_to_joint_config to these exact joints), not a pose
    waypoints.append(
        dict(name=transit[-1][0], pos=above, quat=quat, hold="level", q=transit[-1][1], joints=reconfigure)
    )
    down, descent = split_approach(descent, table_z, lower_from)
    moves += down
    if down:
        T = model.fk(down[-1][1])
        waypoints.append(dict(name="approach", pos=T[:3, 3], quat=quat, hold="level", q=down[-1][1]))
    info = dict(
        adjust_deg=adjust_deg,
        approach=approach,
        table_z=table_z,
        floor_z=floor_z,
        camera_left=left,
        reconfigure=reconfigure,
        waypoints=waypoints,
    )
    return moves, descent, info


APPROACH_EVERY = 0.04  # m, a joint waypoint this often on the approach, so it stays near the straight line


def split_approach(descent, table_z, lower_from):
    """The line down (zs, qs) split where the tool is `lower_from` above where the container's bottom
    meets the table: the part above becomes "approach" joint moves for the fast carrying motion
    (position mode, cuRobo, a waypoint every APPROACH_EVERY so it stays near the line), the rest is
    the slow impedance line, which then has only those last cm to cover before the touch. lower_from
    0 (or a line that starts lower already): no approach, the whole line slow as before.
    Returns (approach [(name, q)], descent (zs, qs))."""
    zs, qs = descent
    end = table_z + BOWL_DEPTH + lower_from
    if lower_from <= 0 or zs[0] - end < 0.01:
        return [], descent
    k = int(np.searchsorted(-np.asarray(zs), -end, side="right")) - 1  # the last point at or above `end`
    every = max(1, int(round(APPROACH_EVERY / PATH_STEP)))
    picks = list(range(every, k, every)) + [k]
    return [("approach'" if i != k else "approach", qs[i]) for i in picks], (zs[k:], qs[k:])


def blend_radii(waypoints, table_z, near, far):
    """blend_radius (m) for each waypoint of one blended motion: `far` where the tool passes at the
    carrying height (SAFE_HEIGHT above the table, clear of it), `near` anywhere lower (the table and
    the container are not in cuRobo's world, so a cut corner there is checked only against the
    chair), 0 at the last one, which stops."""
    radii = [far if w["pos"][2] >= table_z + SAFE_HEIGHT - 0.01 else near for w in waypoints]
    return radii[:-1] + [0.0] if radii else []


def plan_retreat(model, q_start, release=0.0, step=RETREAT_STEP, log=None):
    """Joint waypoints every `step` along the straight horizontal line RETREAT back from q_start's
    tool pose (back = opposite the hand's heading), orientation kept. release: also lift that much
    over the first RETREAT_RELEASE of it (the press lets go gently). The first entry is q_start. If
    the line runs out of reach after RETREAT_MIN it ends there (flatbot at 0.8 m, 2026-10-26: 6-7 cm
    of the 8). If it runs out before, the hand swings out instead (plan_swing)."""
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
            straight = f"only {done * 100:.1f} cm straight back fits (needs {RETREAT_MIN * 100:.1f})"
            # swing from the end of the straight part (fingertips off the wall), the lift already done
            lifted = min(1.0, (k - 1) / n / RETREAT_RELEASE) * release
            swing = plan_swing(model, qs[-1], release - lifted, step, q_wall=q_start)
            if isinstance(swing, str):
                raise Refused(f"retreat: can't back out from here: {straight}; {swing}")
            if log is not None:
                off = "" if done >= SWING_BACKOFF else " (WARNING: the fingertips may still touch the wall)"
                log(f"retreat: {straight}; it backs {done * 100:.1f} cm, then the hand swings out: {swing[1]}{off}")
            return qs + swing[0][1:]
        qs.append(q)
        seed = q
    return qs


def plan_swing(model, q_start, release=0.0, step=RETREAT_STEP, q_wall=None):
    """The retreat as a turn of the hand about the vertical through the flange, SWING_DEG left or
    right (seen from above), joint waypoints every ~`step` of fingertip travel; release lifts as
    plan_retreat does. Each side runs until it is done or out of reach; one that ends with the
    fingertips at least SWING_CLEAR outside the lip's outer edge counts, the longer turn wins.
    q_wall: the joints with the fingertips at the container wall (default q_start), which locate the
    container. Returns (waypoints, a note for the log), or why neither side works (a str)."""
    T, T_fl, T_tip = model.fk(q_start), model.fk(q_start, "flange"), model.fk(q_start, "tip")
    q_wall = q_start if q_wall is None else q_wall
    W = model.fk(q_wall)
    heading = np.array([W[0, 2], W[1, 2], 0.0])
    heading /= np.linalg.norm(heading)  # plan_retreat refused a steep hand already
    centre = model.fk(q_wall, "tip")[:3, 3] + BOWL_RADIUS * heading  # the container's axis: tips at its wall
    clear = BOWL_RADIUS + LIP_WIDTH + SWING_CLEAR
    pivot = T_fl[:3, 3]
    lever = np.hypot(*(T_tip[:2, 3] - pivot[:2]))  # flange to fingertips, seen from above
    n = max(1, int(np.ceil(np.radians(SWING_DEG) * lever / step)))
    rot0 = Rotation.from_matrix(T[:3, :3])
    best, reasons = None, []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        qs, seed, out = [np.asarray(q_start, dtype=float)], np.asarray(q_start, dtype=float), False
        for k in range(1, n + 1):
            f = k / n
            turn = Rotation.from_euler("z", sign * f * SWING_DEG, degrees=True)
            pos = pivot + turn.apply(T[:3, 3] - pivot) + min(1.0, f / RETREAT_RELEASE) * release * UP
            q = model.ik(pos, (turn * rot0).as_quat(), seed, restarts=0, stay=0.01)
            if q is None or np.max(np.abs(wrap_deg(q - seed))) > 3.0 * step / PATH_STEP:
                break
            qs.append(q)
            seed = q
            out = np.hypot(*(model.fk(q, "tip")[:2, 3] - centre[:2])) >= clear
        deg = (len(qs) - 1) / n * SWING_DEG
        if not out:
            reasons.append(f"a swing {side} gets only {deg:.0f} deg, the fingertips still over the lip")
        elif best is None or deg > best[1]:
            best = (qs, deg, side)
        if out and len(qs) - 1 == n:
            break  # the full turn, clear of the lip: the other side can't do better (and costs ~50 IK solves)
    if best is None:
        return "; ".join(reasons)
    qs, deg, side = best
    return qs, f"{deg:.0f} deg {side}, fingertips past the lip"


def reset_moves(model, q_now, q_start, empty=False, big_turn=False):
    """Back to a saved start: straight UP to the start's height first (the open fingers rise clear
    of the container instead of sweeping across it), then one joint move to the start's joints.
    empty: nothing in the hand, so the joint move may turn the hand any way (no tilt check)."""
    T_now, T_start = model.fk(q_now), model.fk(q_start)
    moves, q = [], np.asarray(q_now, dtype=float)
    if T_start[2, 3] - T_now[2, 3] > 0.005:
        up = T_now[:3, 3].copy()
        up[2] = T_start[2, 3]
        moves = plan_segment(model, q, up, Rotation.from_matrix(T_now[:3, :3]).as_quat(), None, "up")
        q = moves[-1][1]
    problems = check_joint_move(model, q, q_start, None, tilt_check=not empty)
    jump = float(np.max(np.abs(wrap_deg(np.asarray(q_start) - q))))
    if problems or (jump > MAX_STEP_JUMP_DEG and not (empty and big_turn)):
        raise Refused(
            "back to the start: "
            + ("; ".join(problems) or f"a joint moves {jump:.0f} deg (empty hand: big_turn allows it)")
        )
    return moves + [("start", np.asarray(q_start, dtype=float))]


RETURN_EVERY = 0.015  # m of fingertip travel between the return's waypoints by the container
RETURN_BLEND = 0.005  # m, blend_radius there: corners cut by at most this close to the lip


def return_moves(model, q_hold, q_start, drop=0.0, log=None):
    """After letting go, the way back to the start as ONE blended motion's joint targets
    [(name, q, near)]: the drop straight down (`drop` m, if the arm can), the retreat from there
    (plan_retreat: straight back, or back and the swing; no lift) thinned to every RETURN_EVERY of
    fingertip travel, then reset_moves' way up and over to the start. `near`: by the container (a
    tight blend and the slow speed); False from the way up on."""
    q = np.asarray(q_hold, dtype=float)
    moves = []
    if drop > 0:
        z = float(model.fk(q)[2, 3])
        try:
            q = np.asarray(plan_descent(model, q, z - drop)[1][-1], dtype=float)
            moves.append(("drop", q, True))
        except Refused as e:
            if log is not None:
                log(f"no drop after letting go ({e}): the retreat starts from the hold")
    line = plan_retreat(model, q, log=log)
    tips = [model.fk(x, "tip")[:3, 3] for x in line]
    since = 0.0
    for k in range(1, len(line)):
        since += float(np.linalg.norm(tips[k] - tips[k - 1]))
        if since >= RETURN_EVERY or k == len(line) - 1:
            moves.append(("retreat" if k == len(line) - 1 else "retreat'", np.asarray(line[k], dtype=float), True))
            since = 0.0
    back = reset_moves(model, line[-1], q_start, empty=True)
    return moves + [(name, q_b, False) for name, q_b in back]


def return_radii(model, moves, near):
    """blend_radius (m) of each of return_moves' targets: RETURN_BLEND by the container; on the way
    back up and over, reset_radii's rule (`near` once risen clear of the container); 0 at the start,
    which stops."""
    k0 = next((i for i, (_, _, by) in enumerate(moves) if not by), len(moves))
    radii = [RETURN_BLEND] * k0
    back = [(n, q) for n, q, _ in moves[k0:]]
    if back:
        q_from = moves[k0 - 1][1] if k0 else back[0][1]
        radii += reset_radii(model, q_from, back, near)
    elif radii:
        radii[-1] = 0.0
    return radii


def reset_radii(model, q_now, moves, near):
    """blend_radius (m) of each of reset_moves' joint targets for go_to_joint_configs. Waypoints on
    the straight way up are on one line, so passing them is free. The turn toward the start (the
    last waypoint up) is cut only when the hand has risen far enough by then that the cut can't drag
    the open fingers across the container: at least BOWL_DEPTH + near. The start itself stops."""
    if not moves:
        return []
    radii = [near] * (len(moves) - 1) + [0.0]
    ups = [i for i, (name, _) in enumerate(moves) if name.startswith("up")]
    if ups:
        rise = model.fk(moves[ups[-1]][1])[2, 3] - model.fk(q_now)[2, 3]
        radii[ups[-1]] = near if rise >= BOWL_DEPTH + near else 0.0
    return radii


def contact_zone_checks(model, descent, table_z):
    """The joint moves of a place that don't go through cuRobo (self.arm.move_joints: the line down
    and the retreat), as (label, q_from, q_to) for a cuRobo plan_to_joints check of their ends
    against the chair's collision volumes: the top of the line to its bottom, and where the
    container should meet the table to the end of the retreat from there. A retreat that can't be
    planned from there is left out (the real one starts at the contact and is planned again)."""
    zs, qs = descent
    checks = [("lower", qs[0], qs[-1])]  # to the bottom of the line down
    q_meet = hold_point(qs, zs, table_z + BOWL_DEPTH)
    q_meet = np.asarray(qs[-1] if q_meet is None else q_meet, dtype=float)
    try:
        retreat = plan_retreat(model, q_meet)
    except Refused:
        return checks
    checks.append(("retreat", q_meet, retreat[-1]))  # to its end
    return checks


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
    about the fingertips and vz sat at ~-0.5 cm/s; -1 cm/s stops at the touch.
    update(vz, armed=False) forgets everything seen so far: the sag when the arm goes compliant at
    the start of the descent (a drop, then still) must not read as moving down and stopping."""

    def __init__(self, stop=-0.01, moving=0.01, hits=2):
        self.stop, self.moving, self.need = stop, moving, hits
        self.moved, self.count = False, 0

    def update(self, vz, armed=True):
        if not armed:
            self.moved, self.count = False, 0
            return False
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


CONTACT_ARM_BEFORE = 0.01  # m, contact is armed at least this far above where the table should be


def contact_armed(u, length, ignore_m, meet=None):
    """Whether contact may be called yet at time fraction u of a descent `length` long: the command
    has gone at least `ignore_m` down (at most half the line, and with `meet` -- where the table
    should be, 0..1 of the line -- at least CONTACT_ARM_BEFORE above it, so a short line that starts
    just above the table still feels it). Without a settle, the arm goes compliant as the descent
    starts and sags a little; that is over within the first few cm."""
    return minjerk_distance(u) * length >= arm_distance(length, ignore_m, meet)


def arm_distance(length, ignore_m, meet=None):
    """How far down (m) the descent's command goes before contact is armed (contact_armed)."""
    d = min(ignore_m, 0.5 * length)
    if meet is not None:
        d = min(d, max(meet * length - CONTACT_ARM_BEFORE, 0.0))
    return d


def commanded_z(line_z, u):
    """Where the descent's command is at time fraction u: line_z (heights along the straight line
    down, falling) moved minimum-jerk from its first to its last."""
    return float(line_z[0] - minjerk_distance(u) * (line_z[0] - line_z[-1]))


def check_lowering(lower_speed, vz_stop, cart_speed):
    """Refused unless the contact test works at lower_speed: the line down must not peak above
    cart_speed (move_joints' Cartesian speed), and vz_stop must sit between a hand moving down at
    the gate's speed (CONTACT_GATE x lower_speed) and one stopped, or free motion reads as contact."""
    if lower_speed <= 0:
        raise Refused(f"lower_speed is {lower_speed} m/s -- it must be above 0 (0.02 is the tuned value)")
    top = cart_speed / MAX_PEAK
    if lower_speed > top:
        raise Refused(
            f"lower_speed {lower_speed * 100:.1f} cm/s would peak the line down above arm.cart_speed "
            f"({cart_speed * 100:.0f} cm/s): at most {top * 100:.1f} cm/s"
        )
    if not -CONTACT_GATE * lower_speed < vz_stop <= 0.0:
        raise Refused(
            f"vz_stop {vz_stop * 100:+.1f} cm/s doesn't fit lower_speed {lower_speed * 100:.1f} cm/s: it must be above "
            f"{-CONTACT_GATE * lower_speed * 100:+.1f} cm/s (free motion would read as contact) and at most 0 -- "
            f"about -0.5 x lower_speed ({-0.5 * lower_speed * 100:+.1f} cm/s)"
        )


def contact_gate(u, length, duration, lower_speed):
    """Whether contact may be called at time fraction u of the descent: the commanded speed is up
    (at the start and end of a minimum-jerk move it is slow, and a hand that slows with its command
    must not read as stopped by the table)."""
    return 0.0 < u < 1.0 and minjerk_speed(u) * length / duration >= CONTACT_GATE * lower_speed


# -- failures, explained ------------------------------------------------------------------------

RESET_LEGS = ("up", "start", "reset")  # a reset's targets: a refusal there is about the start position


def _where(name, targets):
    pos = (targets or {}).get(name)
    if pos is None:
        return ""
    r = np.hypot(pos[0], pos[1])
    return f" (tool to x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f} m, {r:.2f} m from the base axis)"


def explain_failure(error, legs, targets=None):
    """A motion's failure as: what happened, where (the target, m in arm_mount), and what to do next,
    followed by the driver's own words. `error` is self.arm.last_error or a cuRobo plan's message;
    the driver prefixes a multi-target goal's planning failure with `leg k/N:` (k from 1), which is
    read against `legs`, our names for the targets in order. `targets`: name -> tool position."""
    raw = re.sub(r"^\s*RammpArm:\s*", "", str(error or "no message")).strip()
    m = re.search(r"\bleg (\d+)/(\d+):\s*(.*)", raw)
    if m and 1 <= int(m.group(1)) <= len(legs):
        leg = legs[int(m.group(1)) - 1]
    else:
        leg = legs[0] if len(legs) == 1 else None
    up = raw.upper()
    on = f"the {leg} leg" if leg else "the motion"
    at = _where(leg, targets) if leg else ""
    nothing = " Nothing moved." if "PLANNING_FAILED" in up or (m is None and "planner" in raw.lower()) else ""
    reset = leg is not None and leg.split("'")[0] in RESET_LEGS
    if "STATETIMEOUT" in up or "NO /JOINT_STATES" in up:
        what = "the arm driver's state (/joint_states, /ee_state, /gripper_state) isn't arriving"
        nxt = "check the arm node is up (sheppy) and ROS_DOMAIN_ID matches, then run again"
    elif "ACTION SERVER" in up:
        what, nxt = (
            "the cuRobo planner is not running",
            "start it with the flatbot profile (sheppy up flatbot), then run again",
        )
    elif "PLANNER BUSY" in up:
        what, nxt = "the planner was busy with another goal", "wait for that goal to end, then run again"
    elif "INVALID_START" in up or "START COLLIDES" in up:
        what = (
            "cuRobo won't plan from where the arm is: it is inside (or within the 2 cm padding of) "
            "the chair's collision volumes"
        )
        nxt = (
            "move the arm clear of the chair and the person (they are physical cardboard: don't push into "
            "them), then run "
            "again. If it is visibly clear, the volumes in rammp-deployments flatbot/curobo may be off"
        )
    elif "COLLI" in up:
        what = (
            f"the {leg or 'motion'}'s target is inside the chair's collision volumes"
            if leg
            else "the path collides with the chair's collision volumes"
        )
        nxt = (
            "the saved start position is inside them: put the arm somewhere clear and run reset with save"
            if reset
            else "move the spot further from the chair (forward, or away from the person's side), mark it again"
        )
    elif re.search(r"\bIK[_ ]?FAIL|\bNO IK\b|\bIK SOLUTION", up):
        what = f"no arm pose reaches {on}'s target"
        nxt = "it is out of the arm's reach or joint limits: move the spot within the reach param (mark it again)"
    elif "CONSTRAINT_VIOLATED" in up or re.search(r"\bHOLD\b", up):
        what = f"cuRobo could not keep the container level on {on}"
        nxt = "run again with adjust_mode:=joints, or blend:=false"
    elif re.search(r"\bBLEND:", up):
        what, at = "every leg planned, but they could not be joined into one motion", ""
        nxt = "lower blend_radius_near / blend_radius_far, or blend:=false"
    elif any(k in up for k in ("TRAJOPT", "DT_EXCEPTION", "FINETUNE", "PLANNING_FAILED", "NO PLAN")):
        what = f"cuRobo found no smooth path for {on}"
        nxt = "run again (cuRobo's search is random); if it repeats, start further from the chair, or blend:=false"
    elif "CM FROM THE GOAL" in up or "JOINTS WITHIN" in up:
        what = f"the arm stopped short of {on}'s target"
        nxt = "look for something it touched that cuRobo doesn't know (the table, a cardboard volume), then reset"
    elif re.search(r"HALTED|E-?STOP|CONTROLTAKEN|TOOK CONTROL", up):
        what = "the driver halted the motion: e-stop, or another sender took control of the arm"
        nxt = "release the e-stop / check who holds control, then reset"
    elif "PATH_TOLERANCE" in up:
        what = "the arm left its planned path (pushed, or blocked by something)"
        nxt = "check what it touched, then reset"
    elif "INVALID_GOAL" in up or "NOT_AUTHORIZED" in up or "SUPERVISOR" in up:
        what = "the driver refused the motion: this node does not hold control of the arm"
        nxt = "check that the orchestrator granted the arm (or nothing else holds it standalone), then run again"
    else:
        what, nxt = f"{on} did not complete", "see the driver's message below; reset, then run again"
    return f"{what}{at}.{nothing} Next: {nxt}. [driver: {raw}]"
