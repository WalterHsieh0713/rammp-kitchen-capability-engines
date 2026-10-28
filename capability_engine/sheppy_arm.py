"""SheppyArm's constants and pure path helpers: Cartesian paths to joint knots, and their timing.

Copied from feeding-deployment microwave-wip@7bc328ae microwave/sheppy_arm.py. Free of rclpy; the
`SheppyArm` class that drives the arm lives below the helpers and imports ROS lazily.
"""

import logging
import threading

import numpy as np
from scipy.spatial.transform import Rotation as R

from capability_engine import sim as sim_mod

JOINTS = [f"arm_joint_{i}" for i in range(1, 8)]
CONTINUOUS = [0, 2, 4, 6]  # J1/J3/J5/J7 spin freely
TOOL_Z = 0.120  # end_effector_link -> tool_frame (Kinova's 2F-85 tool, robot_controller/urdfs)
DT = 0.02  # trajectory sample spacing (cuRobo's interpolation_dt)
# rad/s peak for joint moves (~49 deg/s); was 0.35, 0.5 (10-07), 0.65 then 0.85 (10-24,
# "everything felt slow")
JOINT_SPEED = 0.85
CART_SPEED = 0.16  # m/s peak for Cartesian paths; was 0.06, 0.09 (10-07), 0.12 then 0.16 (10-24)
# Contact moves (the push-open into the door, the push-close push) CAN'T HANDLE MORE SPEED than
# the 10-07 values below -- keep them there however fast the rest goes: at 0.65 rad/s / 0.12 m/s
# the push-open's last leg (door pushing back + a 68 deg hand turn) aborted
# PATH_TOLERANCE_VIOLATED (10-24); it ran clean three times at these.
CONTACT_JOINT_SPEED = 0.5
CONTACT_CART_SPEED = 0.09


def contact_speed_scale():
    """speed_scale for move_joints holding a contact move at CONTACT_*_SPEED."""
    return min(1.0, CONTACT_CART_SPEED / CART_SPEED, CONTACT_JOINT_SPEED / JOINT_SPEED)


MIN_DURATION_S = 1.0
CART_STEP_M = 0.02  # Cartesian segments are cut into steps this long ...
CART_STEP_DEG = 5.0  # ... and turning at most this much (= plan_cartesian)
ROT_EQUIV_M = 0.15  # a turn of 1 rad counts as 15 cm of travel (6 cm/s -> ~23 deg/s)
MAX_KNOT_JUMP_DEG = 20.0  # refuse a step that moves any joint more than this (= MAX_STEP_JUMP_DEG)
PATH_TOL_RAD = 0.15  # ~8.6 deg tracking guard per joint (driver-side abort)
GRIPPER_FORCE = 0.5
GRIPPER_SPEED = 0.5
# Joint impedance for compliant moves (the door swing): the driver's documented teleop starting
# point (kinova-gen3-driver docs/guide/control-modes.md). In impedance mode lag ~ force
# (kq x lag), so the path tolerance doubles as a force stop. NOT USED for the door swing
# (scrapped 10-21): 0.08 rad aborted just after the latch; 0.30 rad let the arm swing wildly once
# the handle slipped out of the fingers (e-stopped).
IMPEDANCE = {
    "kq": (80.0, 80.0, 80.0, 80.0, 30.0, 30.0, 30.0),
    "zeta": 0.5,
    "torque_limit": (39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0),
    "path_tol": 0.08,
}
ERRORS = {
    0: "SUCCESSFUL",
    -1: "INVALID_GOAL",
    -4: "PATH_TOLERANCE_VIOLATED",
    -5: "GOAL_TOLERANCE_VIOLATED",
    -6: "PREEMPTED",
    -7: "PLANNING_FAILED",
    -8: "NOT_AUTHORIZED",
    -9: "HALTED",
}
# go_to_ee_pose / go_to_joint_config (cuRobo in the driver). FRAME: what /ee_state reports and
# the goal must be in (arm_mount on the nightly driver). The goal pose is Kinova's tool_frame,
# = ee_pos here.
FRAME = "arm_mount"
SENDER_ID = "microwave"
# speed_scale of every go_to goal (1.0 = cuRobo's plan at the arm's full speed; the driver
# refuses, not clamps, values outside [0.01, 1]). button_press 10-14: at 1.0 a 51 cm move took
# 1.16 s and hit the microwave cuRobo can't see -- start slow here; the close script's --speed
# scales it too.
GO_TO_SPEED_SCALE = 0.3
GO_TO_TIMEOUT_S = 120.0  # planning + execution
GO_TO_TRACK_M = 0.01  # go_to_ee_pose: tool this far from the goal at the end -> failed
GO_TO_TRACK_DEG = 2.0  # go_to_joint_config: any joint this far from the goal at the end -> failed


def _wrap(q):
    q = np.asarray(q, dtype=float).copy()
    q[CONTINUOUS] = (q[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return q


def _unwrap_from(q_start, knots):
    """Knots made continuous from `q_start`: each free joint steps the short way from the last."""
    out, prev = [], np.asarray(q_start, dtype=float)
    for k in knots:
        d = np.asarray(k, dtype=float) - prev
        d[CONTINUOUS] = (d[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
        prev = prev + d
        out.append(prev.copy())
    return out


def _min_jerk_samples(path, s_knots, duration):
    """Positions + velocities every DT along a piecewise-linear joint path `path` (knots at path
    parameter `s_knots`, 0..L), timed by a minimum-jerk s(t) over `duration`."""
    path, L = np.asarray(path), float(s_knots[-1])
    n = max(2, int(np.ceil(duration / DT)) + 1)
    t = np.linspace(0.0, duration, n)
    u = t / duration
    s = L * (10 * u**3 - 15 * u**4 + 6 * u**5)
    q = np.stack([np.interp(s, s_knots, path[:, j]) for j in range(7)], axis=1)
    qd = np.gradient(q, t, axis=0)
    qd[0] = qd[-1] = 0.0
    return t, q, qd


def cartesian_knots(sim, ee_start, q_start, traj, *, log=None):
    """Joint knots for `traj` from tool pose `ee_start` / joints `q_start`: straight lines with
    slerped orientation between consecutive poses, split into <= CART_STEP_M / CART_STEP_DEG steps,
    each IK'd from the previous (same re-solve as `plan_cartesian`). Returns (knots, segment
    lengths in m), or (None, None) if a step misses the IK tolerance or jumps a joint more than
    MAX_KNOT_JUMP_DEG. Pure sim -- used by `move_cartesian`, and testable offline. `log`: called
    with the refusal message instead of logging it as a warning."""
    from scipy.spatial.transform import Slerp

    scene, rb = sim
    q, prev_p, prev_q = (
        np.asarray(q_start, float),
        np.asarray(ee_start[:3], float),
        np.asarray(ee_start[3:7], float),
    )
    knots, lengths = [], []
    for k, (pos, quat) in enumerate(traj):
        pos, quat = np.asarray(pos, float), np.asarray(quat, float)
        rots = R.from_quat([prev_q, quat])
        ang = float(np.degrees((rots[0].inv() * rots[1]).magnitude()))
        dist = float(np.linalg.norm(pos - prev_p))
        n = max(1, int(np.ceil(max(dist / CART_STEP_M, ang / CART_STEP_DEG))))
        slerp = Slerp([0, 1], rots)
        for i in range(1, n + 1):
            f = i / n
            tp, tq = prev_p + (pos - prev_p) * f, slerp([f]).as_quat()[0]
            nq, err = sim_mod.solve_ik(scene, rb, tp, tq, q)
            for _ in range(3):
                if err <= sim_mod.MAX_IK_ERR:
                    break
                nq, err = sim_mod.solve_ik(scene, rb, tp, tq, nq)
            d = nq - q
            d[CONTINUOUS] = (d[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
            jump = float(np.degrees(np.max(np.abs(d))))
            j4, j6 = abs(np.degrees(nq[3])), abs(np.degrees(nq[5]))
            if (
                err > sim_mod.MAX_IK_ERR
                or jump > MAX_KNOT_JUMP_DEG
                or j4 > sim_mod.J4_GUARD_DEG
                or j6 > sim_mod.J6_GUARD_DEG
            ):
                (log or logging.getLogger("sheppy_arm").warning)(
                    f"  SheppyArm: waypoint {k + 1}/{len(traj)} step {i}/{n}: "
                    f"IK err {err * 100:.1f} cm, "
                    f"jump {jump:.1f} deg, J4 {j4:.1f}, J6 {j6:.1f} -- nothing sent."
                )
                return None, None
            knots.append(nq)
            lengths.append(max(1e-4, dist / n, np.radians(ang / n) * ROT_EQUIV_M))
            q = nq
        prev_p, prev_q = pos, quat
    return knots, lengths


def joint_trajectory(q0, knots, cart_lengths=None, speed_scale=1.0):
    """(t, q, qd) samples for a move from joints `q0` through joint `knots` (wrapped or not), or
    None when the path is too short to move. `cart_lengths`: the Cartesian distance of each
    segment -- given, the path is timed in metres at CART_SPEED, otherwise in joint-space radians
    at JOINT_SPEED."""
    path = [np.asarray(q0, dtype=float)] + _unwrap_from(q0, knots)
    seg = (
        np.asarray(cart_lengths, dtype=float)
        if cart_lengths is not None
        else np.array([np.max(np.abs(b - a)) for a, b in zip(path[:-1], path[1:], strict=False)])
    )
    s_knots = np.concatenate([[0.0], np.cumsum(seg)])
    if s_knots[-1] < 1e-6:
        return None
    # min-jerk peak speed is 1.875 x the average
    speed = (CART_SPEED if cart_lengths is not None else JOINT_SPEED) * speed_scale
    # never faster than JOINT_SPEED on any joint either, whatever the Cartesian length says
    joint_travel = float(
        np.sum([np.max(np.abs(b - a)) for a, b in zip(path[:-1], path[1:], strict=False)])
    )
    duration = max(
        MIN_DURATION_S,
        1.875 * s_knots[-1] / speed,
        1.875 * joint_travel / (JOINT_SPEED * speed_scale),
    )
    return _min_jerk_samples(path, s_knots, duration)


class _StandaloneCtx:
    """The action context of a bench script: nothing pauses or cancels it (Ctrl-C still cancels
    a running goal and re-raises)."""

    def checkpoint(self):
        pass

    def cancelled(self):
        return False


class SheppyArm:
    """SheppyArm for bench scripts: self.arm's ArmInterface on its own node, spun by its own
    MultiThreadedExecutor thread, always in an action until close(). No control token: under
    enforced arbitration the driver refuses its commands. Every attribute is the ArmInterface's
    (get_state(), move_joints(), last_path, monitor_stopped, ...)."""

    def __init__(self, node_name="microwave_sheppy_arm"):
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node

        from capability_engine import arm  # here: arm imports this module

        if not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name)
        self._exec = MultiThreadedExecutor()
        self._exec.add_node(self.node)
        threading.Thread(target=self._exec.spin, daemon=True).start()
        self._arm = arm.ArmInterface(self.node, ReentrantCallbackGroup(), standalone=True)
        ctx = _StandaloneCtx()
        self._arm.begin_action(ctx, token=None)
        self._owns_model = not sim_mod.has_model_provider()
        try:
            if self._owns_model:  # refused when the sim was built from a URDF path
                sim_mod.set_model_provider(self._arm.robot_description)
            # As SheppyArm's constructor: wait for the arm's state, then the trajectory server
            # (polled in POLL_S slices by the arm's own wait).
            self._arm.get_state()
            client = self._arm._action("execute_joint_trajectory")
            if not self._arm._wait_for_server(client, ctx, arm.STATE_WAIT_S):
                raise RuntimeError("no /execute_joint_trajectory action server")
        except BaseException:
            self.close()
            raise

    def __getattr__(self, name):
        arm = self.__dict__.get("_arm")
        if arm is None:
            raise AttributeError(name)
        return getattr(arm, name)

    def close(self):
        """End the action (settling any goal or stream), then stop the executor and destroy the
        node."""
        self._arm.end_action()
        if self._owns_model:
            sim_mod.set_model_provider(None)
        self._exec.shutdown()
        self.node.destroy_node()
