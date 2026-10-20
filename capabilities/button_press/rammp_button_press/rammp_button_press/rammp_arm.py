"""RammpArm: the interface to the real arm (RAMMP's kinova_gen3_node driver).

press_button reads the arm's state and sends every motion through this class.

Reads:
  joints(), state()   /joint_states (J1/J3/J5/J7 wrapped to +-pi)
  ee_pose()           /ee_state (the flange) + TOOL_Z along its z = Kinova's tool_frame
  gripper             /gripper_state (0 open .. 1 closed)
  status()            /control_status (e-stop, arbitration, owner), /diagnostics, tool-frame check

Moves:
  smooth_poses()      short straight legs (approach, press): sends exactly the joint path that
                      kinematics.precheck_poses checked, as one execute_joint_trajectory goal,
                      timed by a minimum-jerk profile capped at CART_SPEED / JOINT_SPEED
  go_to_ee_pose()     long moves (staging, back to start, bail): the driver plans with cuRobo and
                      executes at GO_TO_SPEED_SCALE of its speed. The path cannot be previewed
                      and cuRobo's world has no microwave in it, so only the goal pose is gated
  press_contact()     the press: in along the panel normal until the wrist torques feel the button, on
                      the driver's ee_twist stream
  close_gripper()     once, at preflight, on execute runs only; the gripper is never opened

Safety: a dry run sends nothing; a leg is refused if the arm moved since it was planned
(START_DRIFT_DEG); the driver aborts if a joint lags its reference by PATH_TOL_RAD; a leg that
ends too far from its target raises Abort and the arm holds there. Ctrl-C cancels the running
goal.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTolerance
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from rammp_arm_interfaces.action import ExecuteJointTrajectory, GoToEEPose
from rammp_arm_interfaces.msg import EeState, GripperSetpoint, GripperState, TwistSetpoint
from rammp_arm_interfaces.srv import CloseStream, OpenStream
from rammp_common_interfaces.msg import ControlStatus
from rammp_common_interfaces.srv import AcquireControl, ReleaseControl
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from rammp_button_press import Abort, say
from rammp_button_press.geometry import max_joint_delta_deg
from rammp_button_press.kinematics import (
    JOINTS,
    MAX_REACH_M,
    Z_RANGE_M,
    Kinematics,
)

# /ee_state is the flange (end_effector_link): checked 2026-10-07, tf2 base_link -> end_effector_link
# == /ee_state, and sim tool_frame FK at the real joints == /ee_state + 0.120 m along the flange z
# to 0.1 mm. The button press's PREPRESS_EE_OFFSET_M was measured in Kinova's tool_frame (= the
# Kortex tool pose = flange + 0.120 m), so ee_pose() reports that. preflight re-checks it live
# (TOOL_FRAME_CHECK_M): if the driver's /ee_state ever changes frame, the run refuses.
TOOL_Z = 0.120
TOOL_FRAME_CHECK_M = 0.005
FRESH_STATE_S = 0.5  # /joint_states and /ee_state run at ~100 Hz; older than this = driver down
CONTINUOUS = [0, 2, 4, 6]  # J1/J3/J5/J7 spin freely
DT = 0.02  # trajectory sample spacing (cuRobo's interpolation_dt)
# Speed: the old stack ran with Kortex's low/medium preset. These are peaks of the min-jerk profile,
# the values microwave/sheppy_arm.py ran attended on this arm before "a little faster" (2026-10-07).
CART_SPEED = 0.06  # m/s
JOINT_SPEED = 0.35  # rad/s, on any joint, whatever the Cartesian length says
ROT_EQUIV_M = 0.15  # a 1 rad wrist turn counts as 15 cm of travel when timing a leg
MIN_DURATION_S = 1.0
PATH_TOL_RAD = 0.15  # ~8.6 deg tracking guard per joint (driver-side abort), = sheppy_arm.py
TRACK_ABORT_M = 0.01  # final tool position this far from the target -> hold (= old arm.py)
START_DRIFT_DEG = 0.5  # the arm must still be where a leg was planned from when it is sent
SENDER_ID = "rammp_button_press"
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
DIAG_LEVELS = {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}
# ---- long moves through the driver's go_to_ee_pose (staging, back to start, bail): the driver
# asks RAMMP's cuRobo for a plan from the measured joints and executes it. Its target is the
# Kinova tool_frame in base_link (kinova-gen3-ros2 guide-goto-ee-pose), the same frame as
# ee_pose(). What this side can no longer do: see or gate the planned path before it runs or
# dry-run it (no plan-only mode on this action). Only the goal pose is gated here. cuRobo's world
# (gen3_real.yaml: pedestal + table) does NOT contain the microwave, counter, wheelchair or user.
GO_TO_ACTION = "/go_to_ee_pose"
GO_TO_TIMEOUT_S = 120.0  # planning + execution
# The speed every go_to_ee_pose goal runs at (the goal's speed_scale: 1.0 = cuRobo's plan as it
# is, the arm's full speed; the driver refuses, not clamps, values outside [0.01, 1]). 2026-10-14: at
# 1.0 a 51 cm staging move from HOME took 1.16 s and hit the microwave cuRobo cannot see, faster than
# anyone could reach the e-stop; 0.15 made it ~8 s. Back to 1.0 at the user's request, once the
# return and the bail back straight out from the panel before cuRobo (press_button.RETREAT_OUT_M).
# The move TO staging is still cuRobo's blind path at full speed.
GO_TO_SPEED_SCALE = 1.0
# Preflight's close (close_gripper). /gripper_state.position: 0 open .. 1 closed; the press needs
# the closed fingertips. Speed and force go on every message (the driver keeps neither).
GRIPPER_CLOSED_ABOVE = 0.7
GRIPPER_CLOSE_SPEED = 0.5
GRIPPER_CLOSE_FORCE = 0.5
GRIPPER_CLOSE_TIMEOUT_S = 6.0
GRIPPER_RESEND_S = 0.2  # the setpoint topic is best-effort: re-sent until the fingers are in
# ---- streaming sessions (/open_stream ... /close_stream), used by the contact press. Closing a session
# latches a hold at the measured q (no slew back to the last setpoint).
STREAM_TIMEOUT_S = 0.1  # driver stops the session if setpoints stop for this long
STREAM_RATE_HZ = 100.0  # setpoint rate (10x the timeout)
STREAM_START_CHECK_M = 0.002  # the arm must be at the pre-press spot before a session opens
# ---- contact press (press_contact): box-opening's two-stage press
# (RAMMP-box-opening primitives/core.py press_stroke + press_push, runtime/guards.py TorqueGuard),
# along the panel normal on the driver's ee_twist controller. TOUCH: push until a wrist joint's
# torque moves CONTACT_TOUCH_NM off its baseline -- the fingertip met the button, located by the
# arm, not the camera. PUSH: CONTACT_PUSH_M further from there, position-bounded, the button's
# stop (CONTACT_STOP_NM) ending it early. One stroke until the torque builds up is what shoved
# the microwave and the table under it back 2026-10-15 (as box-opening's single hard stroke
# compressed the whole container, 2026-09-03): a force only the button's bottoming out can give
# must not be what ends the stroke. ee_twist integrates the twist into a STIFF held reference, so
# both bounds apply to the COMMANDED travel, and every stop CLOSES the session (the driver then
# holds at the measured q, which drops the push; a streamed zero would keep pushing).
TWIST_CONTROLLER = "ee_twist"
CONTACT_SPEED_MPS = 0.01  # along the normal, both stages
CONTACT_BASELINE_S = 0.2  # hold still first; the baseline is the wrist torques' mean over it (GUARD_MEAN_S)
CONTACT_WRIST = slice(3, 7)  # joints 4-7: the torques box-opening's guard watches (client.wrist_efforts)
CONTACT_TOUCH_NM = 3.0  # touch = any wrist torque this far off its baseline (box-opening contact_nm)
CONTACT_MIN_TOUCH_M = 0.004  # a touch before this struck something in front of the panel (box-opening
# EARLY_TRIP_FRAC). Was 0.008: 2026-10-15 real contact starts 7-10 mm in and a
# good press that clicked registered at 7.6 mm, refused as "struck something else"
CONTACT_MAX_TOUCH_M = 0.03  # no touch by this: no surface -- the panel is not where the camera put it
CONTACT_PUSH_M = 0.001  # the press: at most this far past the touch. Was 0.008 (box-opening's
# button_travel_m), then 0.004. 2026-10-15: the wrist torques only register the
# touch 2-4 mm after real contact (1-2 N.m of free-air noise keeps the threshold
# at 3), so the finger is already in; pushing on bent it (held 15.5 deg off the
# push line) off the dome and it sprang back over it on the retract: a double
# press. 1 mm / 4 N.m clicked once.
CONTACT_STOP_NM = 4.0  # the button's stop felt during the push: also the press (the backstop). Was 7.0
# (peaks 6.9-7.2 N.m), then 5.0 (2026-10-15): less force, less bend, less spring-back
CONTACT_START_S = 0.2  # launch transient: deviations observed, never judged, before this
CONTACT_LATERAL_ABORT_M = 0.01  # sideways off the press line -> close and hold
CONTACT_TIMEOUT_S = (CONTACT_MAX_TOUCH_M + CONTACT_PUSH_M) / CONTACT_SPEED_MPS + 1.0


def reads_channel(channels, name) -> bool:
    """Does an /open_stream answer's channel list include `name`? The driver gives the bare channel
    ("twist", "pose"; its topic is /setpoint/<channel>), so accept either spelling."""
    return any(c.strip("/") in (name, f"setpoint/{name}") for c in channels)


class ContactPress:
    """The two stages of the contact press, fed one sample per tick (pure: unit-tested).

    update(t, commanded, along, lateral, dev_nm) -> None (keep pushing) or (kind, message).
    t: s since the push started; commanded / along: m into the panel from the pre-press spot,
    commanded and measured; lateral: m off the press line; dev_nm: the largest wrist torque
    deviation from the baseline. Kinds "pressed" (the push bound) and "stop" (the button's stop
    felt during the push) are presses; "early", "no_surface", "lateral", "timeout" are not.
    touch_m / touch_cmd: where the touch tripped (None before)."""

    def __init__(
        self,
        touch_nm=CONTACT_TOUCH_NM,
        min_touch=CONTACT_MIN_TOUCH_M,
        max_touch=CONTACT_MAX_TOUCH_M,
        push=CONTACT_PUSH_M,
        stop_nm=CONTACT_STOP_NM,
        start_s=CONTACT_START_S,
        lateral_abort=CONTACT_LATERAL_ABORT_M,
        timeout_s=CONTACT_TIMEOUT_S,
    ):
        self.touch_nm, self.min_touch, self.max_touch, self.push = touch_nm, min_touch, max_touch, push
        self.stop_nm, self.start_s, self.lateral_abort, self.timeout_s = stop_nm, start_s, lateral_abort, timeout_s
        self.touch_m = self.touch_cmd = None

    @property
    def phase(self):
        return "touch" if self.touch_m is None else "push"

    def update(self, t, commanded, along, lateral, dev_nm):
        if lateral > self.lateral_abort:
            return "lateral", (
                f"arm moved {lateral*1000:.1f} mm sideways off the press line " f"(limit {self.lateral_abort*1000:.0f})"
            )
        if t > self.timeout_s:
            return "timeout", f"still pushing after {t:.1f} s ({along*1000:.1f} mm in)"
        judged = t >= self.start_s
        if self.touch_m is None:
            if judged and dev_nm > self.touch_nm:
                if along < self.min_touch:
                    return "early", (
                        f"touched at {along*1000:.1f} mm in ({dev_nm:.1f} N.m), before the panel "
                        f"(>= {self.min_touch*1000:.0f} mm) -- struck something else"
                    )
                self.touch_m, self.touch_cmd = along, commanded
                return None
            if commanded >= self.max_touch:
                return "no_surface", (
                    f"no touch within {self.max_touch*1000:.0f} mm -- the panel is not where "
                    "the camera put it, or the microwave is sliding"
                )
            return None
        if dev_nm > self.stop_nm:
            return (
                "stop",
                f"the button's stop felt {(along - self.touch_m)*1000:.1f} mm past the touch ({dev_nm:.1f} N.m)",
            )
        if commanded - self.touch_cmd >= self.push:
            return "pressed", f"{self.push*1000:.0f} mm past the touch (at {self.touch_m*1000:.1f} mm in)"
        return None


def wrap(q):
    """Joint vector with the continuous joints (J1/J3/J5/J7) wrapped to [-pi, pi)."""
    q = np.asarray(q, dtype=float).copy()
    q[CONTINUOUS] = (q[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return q


def unwrap_from(q_start, knots):
    """Knots made continuous from `q_start`: each free joint steps the short way from the previous knot."""
    out, prev = [], np.asarray(q_start, dtype=float)
    for k in knots:
        d = np.asarray(k, dtype=float) - prev
        d[CONTINUOUS] = (d[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
        prev = prev + d
        out.append(prev.copy())
    return out


def min_jerk_samples(path, s_knots, duration):
    """Positions + velocities every DT along a joint path (knots at path parameter `s_knots`, 0..L),
    timed by a minimum-jerk s(t) over `duration`, piecewise-linear between knots."""
    path, length = np.asarray(path), float(s_knots[-1])
    n = max(2, int(np.ceil(duration / DT)) + 1)
    t = np.linspace(0.0, duration, n)
    u = t / duration
    s = length * (10 * u**3 - 15 * u**4 + 6 * u**5)
    q = np.stack([np.interp(s, s_knots, path[:, j]) for j in range(path.shape[1])], axis=1)
    qd = np.gradient(q, t, axis=0)
    qd[0] = qd[-1] = 0.0
    return t, q, qd


def leg_timing(q0, knots, poses, p0, R0, cart_speed=CART_SPEED, joint_speed=JOINT_SPEED):
    """Time a leg of tool `poses` that starts at joints q0 / tool pose (p0, R0).

    Returns (joint path from q0, path length at each knot, duration). The duration keeps the
    min-jerk peak speed (1.875 x the average) under `cart_speed` (CART_SPEED unless a caller asks
    for more) in tool travel (a wrist turn counts via ROT_EQUIV_M) and under `joint_speed` (JOINT_SPEED)
    on every joint."""
    path = [np.asarray(q0, dtype=float)] + unwrap_from(q0, knots)
    seg, prev_p, prev_R = [], np.asarray(p0, dtype=float), np.asarray(R0, dtype=float)
    for pt, Rt in poses:
        turn = Rotation.from_matrix(np.asarray(Rt) @ prev_R.T).magnitude()
        seg.append(max(1e-4, float(np.linalg.norm(np.asarray(pt) - prev_p)), turn * ROT_EQUIV_M))
        prev_p, prev_R = np.asarray(pt, dtype=float), np.asarray(Rt, dtype=float)
    s_knots = np.concatenate([[0.0], np.cumsum(seg)])
    joint_travel = float(np.sum([np.max(np.abs(b - a)) for a, b in zip(path[:-1], path[1:], strict=False)]))
    duration = max(MIN_DURATION_S, 1.875 * s_knots[-1] / cart_speed, 1.875 * joint_travel / joint_speed)
    return np.array(path), s_knots, duration


class RammpArm:
    """The live arm: state from the driver's topics, motion through its actions and streams.

    With execute=False every motion is gated and logged but nothing is sent. The constructor
    waits up to 10 s for joint, tool and gripper state, and raises Abort if any is missing.
    """

    def __init__(
        self,
        execute: bool,
        acquire_control: bool = False,
        frame: str = "base_link",
        joint_prefix: str = "",
        mount_rpy_deg=(0.0, 0.0, 0.0),
    ):
        """frame: the frame /ee_state reports and go_to_ee_pose takes (anything else is INVALID_GOAL):
        base_link on the 1.1.0 driver, arm_mount on the nightly. joint_prefix: the driver's joint names
        are joint_prefix + joint_1..7. mount_rpy_deg: the arm base's orientation in `frame`, so the sim
        plans in it too."""
        self.execute = execute
        self.acquire = acquire_control
        self.node, self._exec, self._spin = spin_node("rammp_button_press_arm")
        node = self.node
        self.frame = frame
        self.joint_names = [joint_prefix + n for n in JOINTS]
        self.kin = Kinematics(base_rpy_deg=mount_rpy_deg)
        self._lock = threading.Lock()
        self._js = self._ee = self._grip = self._status = None
        self._js_names, self._js_idx = None, None
        self._diag: dict[str, tuple[int, str]] = {}
        self._token = bytes(16)
        self._active_gh = None
        self.cancel_requested = False  # set by cancel_motion(); a caller clears it per task
        self.sent_goals = 0  # trajectories the driver accepted (press_server: "did it move")
        node.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)
        node.create_subscription(EeState, "/ee_state", self._on_ee, qos_profile_sensor_data)
        node.create_subscription(GripperState, "/gripper_state", self._on_grip, qos_profile_sensor_data)
        node.create_subscription(
            ControlStatus,
            "/control_status",
            self._on_status,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        node.create_subscription(DiagnosticArray, "/diagnostics", self._on_diag, 10)
        self._traj = ActionClient(node, ExecuteJointTrajectory, "/execute_joint_trajectory")
        self._acq = node.create_client(AcquireControl, "/acquire_control")
        self._rel = node.create_client(ReleaseControl, "/release_control")
        # Contact press: the publisher exists from the start, so DDS discovery has settled
        # long before a session opens (the driver doc's "create your publisher BEFORE you open").
        self._tw_pub = node.create_publisher(TwistSetpoint, "/setpoint/twist", qos_profile_sensor_data)
        self._grip_pub = node.create_publisher(GripperSetpoint, "/setpoint/gripper", qos_profile_sensor_data)
        self._goto = ActionClient(node, GoToEEPose, GO_TO_ACTION)
        self._open_stream = node.create_client(OpenStream, "/open_stream")
        self._close_stream = node.create_client(CloseStream, "/close_stream")
        t0 = time.time()
        while (self._js is None or self._ee is None or self._grip is None) and time.time() - t0 < 10.0:
            if not rclpy.ok():  # shutting down (SIGTERM): stop waiting
                break
            time.sleep(0.05)
        if self._js is None or self._ee is None or self._grip is None:
            self.close()  # the caller may try again: leave no node or thread behind
            missing = [
                n
                for n, v in (("/joint_states", self._js), ("/ee_state", self._ee), ("/gripper_state", self._grip))
                if v is None
            ]
            raise Abort(
                f"no {', '.join(missing)} in 10 s -- is the sheppy `arm` node up (and the arm powered)? "
                f"Joint names expected: {self.joint_names[0]}..{self.joint_names[-1]}"
            )

    # -- subscriptions ------------------------------------------------------------------------
    # Callbacks run at ~100 Hz each and share the GIL with dome detection (2026-10-12: with the
    # old per-message SciPy rotation on a MultiThreadedExecutor, a 0.5 s re-detect took 1.5 s).
    # So they only store raw values; state() does the maths when someone asks.
    def _on_js(self, m):
        names = tuple(m.name)
        if names != self._js_names:
            want = self.joint_names
            self._js_idx = [names.index(n) for n in want] if all(n in names for n in want) else None
            self._js_names = names
        if self._js_idx is not None:
            pos = m.position
            q = [pos[i] for i in self._js_idx]
            eff = m.effort
            tau = [eff[i] for i in self._js_idx] if len(eff) == len(pos) else None
            with self._lock:
                self._js = (time.monotonic(), q, tau)

    def _on_ee(self, m):
        p, o = m.pose.position, m.pose.orientation
        with self._lock:
            self._ee = (time.monotonic(), (p.x, p.y, p.z, o.x, o.y, o.z, o.w))

    def _on_grip(self, m):
        with self._lock:
            self._grip = (time.monotonic(), float(m.position), bool(m.present))

    def _on_status(self, m):
        with self._lock:
            self._status = m

    def _on_diag(self, m):
        with self._lock:
            for s in m.status:
                self._diag[s.name] = (
                    int.from_bytes(s.level, "little") if isinstance(s.level, bytes) else int(s.level),
                    s.message,
                )

    def close(self):
        self.release_control()
        self._exec.shutdown(timeout_sec=1.0)
        self._spin.join(timeout=2.0)  # no callback may still be running when the node goes
        self.node.destroy_node()
        self.kin.close()

    # -- state --------------------------------------------------------------------------------
    def _fresh(self, item, what):
        if item is None or time.monotonic() - item[0] > FRESH_STATE_S:
            raise Abort(f"{what} is stale (> {FRESH_STATE_S} s) -- arm driver down?")
        return item

    def state(self):
        """Latest arm state: {position (joints, wrapped), ee_pos (tool xyz + quat xyzw), gripper_pos}.

        Same shape as the old ArmInterfaceClient.get_state(). Raises Abort if the driver's
        topics are stale (driver down)."""
        with self._lock:
            js, ee, grip = self._js, self._ee, self._grip
        js, ee = self._fresh(js, "/joint_states"), self._fresh(ee, "/ee_state")
        flange, quat = np.array(ee[1][:3]), np.array(ee[1][3:])
        tool = flange + Rotation.from_quat(quat).apply([0.0, 0.0, TOOL_Z])
        return {
            "position": wrap(np.array(js[1], dtype=float)),
            "ee_pos": np.concatenate([tool, quat]),
            "gripper_pos": grip[1],
        }

    def joints(self):
        return np.asarray(self.state()["position"], dtype=float)

    def joint_torques(self):
        """Measured arm joint torques (N.m, joints 1-7) from /joint_states.effort; Abort if there are none."""
        with self._lock:
            js = self._js
        tau = self._fresh(js, "/joint_states")[2]
        if tau is None:
            raise Abort("/joint_states carries no arm efforts -- the contact press needs joint torques")
        return np.array(tau, dtype=float)

    def raw_joints(self):
        """Measured joints as the driver reports them (not wrapped): the trajectory start."""
        with self._lock:
            js = self._js
        return np.array(self._fresh(js, "/joint_states")[1], dtype=float)

    def ee_pos(self):
        return np.asarray(self.state()["ee_pos"][:3], dtype=float)

    def ee_pose(self):
        """(xyz, 3x3 rotation) of the Kinova tool frame in the arm base frame."""
        ee = self.state()["ee_pos"]
        return np.asarray(ee[:3], dtype=float), Rotation.from_quat(ee[3:7]).as_matrix()

    def gripper_present(self):
        with self._lock:
            return self._grip is not None and self._grip[2]

    def close_gripper(self):
        """Close the gripper (the press uses the closed fingertips) and wait until it reads closed.

        Execute runs only; keep everything clear of the fingers. Returns the position reached;
        raises Abort if it does not close within GRIPPER_CLOSE_TIMEOUT_S."""
        if not self.execute:
            raise Abort("close_gripper called in a dry run")
        self.take_control()  # under enforced arbitration the setpoint needs the token
        msg = GripperSetpoint(
            position=1.0, speed=GRIPPER_CLOSE_SPEED, force=GRIPPER_CLOSE_FORCE, token=list(self._token)
        )
        t0 = last = time.time()
        self._grip_pub.publish(msg)
        while time.time() - t0 < GRIPPER_CLOSE_TIMEOUT_S:
            pos = float(self.state()["gripper_pos"])
            if pos > GRIPPER_CLOSED_ABOVE:
                return pos
            if time.time() - last >= GRIPPER_RESEND_S:
                self._grip_pub.publish(msg)
                last = time.time()
            time.sleep(0.02)
        raise Abort(
            f"gripper did not close in {GRIPPER_CLOSE_TIMEOUT_S:.0f} s (position "
            f"{float(self.state()['gripper_pos']):.2f}) -- something between the fingers?"
        )

    def status(self):
        """Pre-run health check. Returns (problems, info lines); no problems = ready.

        Checks /control_status (e-stop, arbitration, who owns the arm), /diagnostics, and that the
        sim FK at the live joints matches /ee_state + TOOL_Z. If that last check fails, the stored
        pre-press offset would be in the wrong frame, so the run must refuse."""
        with self._lock:
            st, diag = self._status, dict(self._diag)
        problems, info = [], []
        if st is None:
            problems.append("no /control_status (latched) -- arm driver not up?")
        else:
            info.append(
                f"control_status : estopped={st.estopped} "
                f"arbitration={'enforced' if st.arbitration_enabled else 'disabled'}"
                f" owned={st.owned}{' by ' + repr(st.owner_id) if st.owned else ''} rejected={st.rejected_count}"
            )
            if st.estopped:
                problems.append("arm is E-STOPPED (latched) -- clear it first")
            if st.arbitration_enabled and not self.acquire:
                problems.append(
                    "arbitration is ENFORCED: goals need a token. Rerun with --acquire-control "
                    "(takes control only if nobody holds the arm)"
                )
            if st.arbitration_enabled and st.owned and st.owner_id != SENDER_ID:
                problems.append(
                    f"the arm is owned by {st.owner_id!r} -- acquiring would SEIZE it and halt their "
                    "motion; refusing"
                )
        if not diag:
            info.append("diagnostics    : none yet (1 Hz)")
        for name, (level, msg) in sorted(diag.items()):
            info.append(f"diagnostics    : {name}: {DIAG_LEVELS.get(level, level)} {msg}")
            if level >= 2 and "Arm" in name:
                problems.append(f"diagnostics {name} {DIAG_LEVELS.get(level, level)}: {msg}")
        # Live tool-frame check: our sim FK at the measured joints vs the driver's pose + TOOL_Z.
        try:
            s = self.state()
            p_fk, q_fk = self.kin.fk(s["position"])
            dp = float(np.linalg.norm(p_fk - s["ee_pos"][:3]))
            dr = float(np.degrees((Rotation.from_quat(q_fk) * Rotation.from_quat(s["ee_pos"][3:7]).inv()).magnitude()))
            info.append(f"tool frame     : sim FK vs /ee_state+{TOOL_Z} m: {dp*1000:.1f} mm, {dr:.2f} deg")
            if dp > TOOL_FRAME_CHECK_M or dr > 1.0:
                problems.append(
                    f"tool frame mismatch: sim tool_frame FK is {dp*1000:.1f} mm / {dr:.1f} deg from "
                    f"/ee_state + {TOOL_Z} m -- /ee_state is not the flange any more? The pre-press "
                    "offset would be wrong; refusing"
                )
        except Abort as e:
            problems.append(str(e))
        return problems, info

    # -- arbitration --------------------------------------------------------------------------
    def take_control(self):
        """Under enforced arbitration, acquire a token -- only if nobody holds the arm (acquire SEIZES)."""
        with self._lock:
            st = self._status
        if st is None or not st.arbitration_enabled:
            return
        if st.owned and st.owner_id != SENDER_ID:
            raise Abort(f"arm owned by {st.owner_id!r}; not seizing it")
        if not self._acq.wait_for_service(timeout_sec=3.0):
            raise Abort("/acquire_control unavailable")
        res = self._call(self._acq, AcquireControl.Request(owner_id=SENDER_ID))
        if res is None or not res.accepted:
            raise Abort(f"/acquire_control refused: {getattr(res, 'message', 'no answer')}")
        self._token = bytes(res.token)
        say(f"  control acquired (generation {res.generation})")

    def release_control(self):
        if self._token == bytes(16):
            return
        res = self._call(self._rel, ReleaseControl.Request(token=list(self._token)))
        say(f"  control released: {getattr(res, 'released', False)} {getattr(res, 'message', '')}")
        self._token = bytes(16)

    def _call(self, client, req, timeout_s=5.0):
        return self._wait(client.call_async(req), timeout_s)

    @staticmethod
    def _wait(fut, timeout_s):
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout_s:
                return None
            time.sleep(0.01)
        return fut.result()

    # -- planning (pure sim) ------------------------------------------------------------------
    def precheck_poses(self, poses, name, log, posture=None, start=None):
        """Gate a path of (xyz, R) tool poses in the sim (kinematics.precheck_poses), chained from
        `start` = (q, xyz, R) -- the arm's current joints and tool pose by default. Returns (worst IK
        err m, worst joint step deg, end state, joint solution of every point)."""
        if start is None:
            p, R = self.ee_pose()
            start = (self.joints(), p, R)
        return self.kin.precheck_poses(poses, name, log, start, posture=posture)

    # -- execution ----------------------------------------------------------------------------
    def smooth_poses(
        self,
        poses,
        knots,
        start,
        name,
        log,
        track_abort_m=TRACK_ABORT_M,
        cart_speed=CART_SPEED,
    ):
        """Execute a pre-checked straight leg and verify where it ended.

        Sends `knots` (the joint solutions precheck_poses found, planned from `start` = (joints,
        tool xyz, R)) as one timed joint trajectory, then waits for the arm to stop. Refuses if
        the arm has moved more than START_DRIFT_DEG since planning. Raises Abort (arm holds) if
        the driver reports an error or the tool ends more than `track_abort_m` from the target.
        Returns the final tool xyz (dry run: logs only and returns the current xyz)."""
        target = np.asarray(poses[-1][0], dtype=float)
        rec = {"step": name, "smooth": True, "target": target.tolist(), "n_points": len(poses), "executed": False}
        if not self.execute:
            log(rec)
            return self.ee_pos()
        q_now = self.raw_joints()
        drift = max_joint_delta_deg(q_now, start[0])
        if drift > START_DRIFT_DEG:
            raise Abort(f"{name}: the arm moved {drift:.1f} deg since this leg was planned -- nothing sent")
        path, s_knots, duration = leg_timing(q_now, knots, poses, start[1], start[2], cart_speed)
        t, q, qd = min_jerk_samples(path, s_knots, duration)
        code, msg = self._send(t, q, qd)
        fin = self._wait_still(target)
        err = float(np.linalg.norm(fin - target))
        rec.update(executed=True, returned=code, error=msg, duration_s=duration, ee_after=fin.tolist(), final_err_m=err)
        log(rec)
        say(f"  {name:22s} reached {np.round(fin, 3)}  ({err*1000:.1f} mm from target, {duration:.1f} s)")
        if code != 0:
            raise Abort(f"{name}: trajectory ended {ERRORS.get(code, code)} {msg} -- HOLDING HERE")
        if err > track_abort_m:
            raise Abort(f"{name}: ended {err*100:.1f} cm from the target -- HOLDING HERE")
        return fin

    def _wait_still(self, target, timeout_s=5.0, settle_reads=4, tol_m=5e-4):
        """The goal's result arrives when the reference ends; wait for the tool to stop."""
        deadline, prev, steady = time.time() + timeout_s, None, 0
        cur = self.ee_pos()
        while time.time() < deadline and steady < settle_reads:
            time.sleep(0.05)
            cur = self.ee_pos()
            steady = steady + 1 if prev is not None and np.linalg.norm(cur - prev) < tol_m else 0
            prev = cur
        return cur

    # -- long moves through the driver's go_to_ee_pose -----------------------------------------
    def check_goal_pose(self, p_goal, name):
        """Gate a goal whose path the driver plans: reach and height band only. Raises Abort."""
        tb = np.asarray(p_goal, dtype=float) - self.kin.base_pos
        bad = []
        if np.linalg.norm(tb) > MAX_REACH_M:
            bad.append(f"reach {np.linalg.norm(tb):.3f} > {MAX_REACH_M}")
        if not Z_RANGE_M[0] <= tb[2] <= Z_RANGE_M[1]:
            bad.append(f"z {tb[2]:.3f} outside {Z_RANGE_M}")
        if bad:
            raise Abort(f"{name}: goal fails the gates: {'; '.join(bad)} -- arm not commanded")

    def go_to_ee_pose(self, p_goal, R_goal, name, log, track_abort_m=TRACK_ABORT_M):
        """Long move: send the tool to (p_goal, R_goal) via the driver's go_to_ee_pose.

        cuRobo plans from the measured joints and the driver executes at GO_TO_SPEED_SCALE of
        cuRobo's speed; the path is not straight and is never previewed. Only the goal is gated (dry run: that is all it
        does). Raises Abort (arm holds) on a driver error or if the tool ends more than
        `track_abort_m` from the goal. Returns the tool xyz reached (dry run: the current one)."""
        self.check_goal_pose(p_goal, name)
        target = np.asarray(p_goal, dtype=float)
        travel = float(np.linalg.norm(target - self.ee_pos()))
        say(
            f"  {name:16s} go_to_ee_pose  goal ok ({travel*100:.0f} cm away; path planned by cuRobo "
            f"in the driver, not previewed; run at {GO_TO_SPEED_SCALE:g} x its speed)"
        )
        rec = {
            "step": name,
            "planner": "go_to_ee_pose",
            "target": target.tolist(),
            "executed": False,
            "speed_scale": GO_TO_SPEED_SCALE,
        }
        if not self.execute:
            log(rec)
            return self.ee_pos()
        if self.cancel_requested:
            raise Abort("cancelled -- nothing sent, HOLDING HERE")
        if not self._goto.wait_for_server(timeout_sec=5.0):
            raise Abort(f"{name}: no {GO_TO_ACTION} action server -- arm driver up? Arm not commanded")
        g = GoToEEPose.Goal()
        g.target = PoseStamped()
        g.target.header.frame_id = self.frame
        g.target.header.stamp = self.node.get_clock().now().to_msg()
        g.target.pose.position.x, g.target.pose.position.y, g.target.pose.position.z = map(float, target)
        o = g.target.pose.orientation
        o.x, o.y, o.z, o.w = map(float, Rotation.from_matrix(R_goal).as_quat())
        g.sender_id = SENDER_ID
        g.token = list(self._token)
        g.speed_scale = GO_TO_SPEED_SCALE
        phase = []

        def on_feedback(fb):
            ph = fb.feedback.phase
            if not phase or phase[-1] != ph:
                phase.append(ph)
                say(f"  {name:16s} go_to_ee_pose: {ph}")

        gh = self._wait(self._goto.send_goal_async(g, feedback_callback=on_feedback), 5.0)
        if gh is None or not gh.accepted:
            raise Abort(f"{name}: go_to_ee_pose goal REJECTED -- nothing moved")
        self._active_gh = gh
        self.sent_goals += 1
        try:
            if self.cancel_requested:  # cancelled while this goal was being sent
                gh.cancel_goal_async()
            res = self._wait(gh.get_result_async(), GO_TO_TIMEOUT_S)
        except KeyboardInterrupt:
            gh.cancel_goal_async()
            time.sleep(0.2)  # let the cancel go out before the exception unwinds the process
            say("  RammpArm: Ctrl-C -- goal cancelled")
            raise
        finally:
            self._active_gh = None
        if res is None:
            gh.cancel_goal_async()
            code, msg = -100, f"no result in {GO_TO_TIMEOUT_S:.0f} s -- goal cancelled"
        else:
            code, msg = int(res.result.error_code), res.result.error_string
        fin = self._wait_still(target)
        e = float(np.linalg.norm(fin - target))
        rec.update(executed=True, returned=code, error=msg, phases=phase, ee_after=fin.tolist(), final_err_m=e)
        log(rec)
        say(f"  {name:22s} reached {np.round(fin, 3)}  ({e*1000:.1f} mm from target)")
        if code != 0:
            raise Abort(f"{name}: go_to_ee_pose ended {ERRORS.get(code, code)} {msg} -- HOLDING HERE")
        if e > track_abort_m:
            raise Abort(f"{name}: ended {e*100:.1f} cm from the target -- HOLDING HERE")
        return fin

    def press_contact(self, p_spot, n_out, name, log, push=CONTACT_PUSH_M, stop_nm=CONTACT_STOP_NM):
        """Contact press (ContactPress): on ee_twist, hold still CONTACT_BASELINE_S (the wrist torque
        baseline), then push along -n_out at CONTACT_SPEED_MPS until the touch, and `push` past it
        (or until a wrist torque is `stop_nm` off its baseline); then close the session at once (the
        driver holds at the measured q). Does not come back out: the caller moves back to the spot.
        Every tick goes into the log (`trace`).

        Returns {stop, touch_m, depth_m, peak_nm, t_s}; raises Abort (session closed, holding)
        unless it pressed. Dry run: logs only."""
        rec = {
            "step": name,
            "contact": True,
            "speed": CONTACT_SPEED_MPS,
            "touch_nm": CONTACT_TOUCH_NM,
            "push_m": push,
            "stop_nm": stop_nm,
            "executed": False,
        }
        if self.cancel_requested:
            raise Abort(f"{name}: cancelled -- no session opened, HOLDING HERE")
        if not self.execute:
            log(rec)
            return None
        n_out = np.asarray(n_out, dtype=float) / np.linalg.norm(n_out)
        p0 = self.ee_pos()
        if np.linalg.norm(p0 - np.asarray(p_spot)) > STREAM_START_CHECK_M:
            raise Abort(
                f"{name}: arm is {np.linalg.norm(p0 - p_spot)*1000:.1f} mm from the pre-press spot "
                "-- no session opened"
            )
        self.joint_torques()  # Abort before anything moves if there are none

        def setpoint(v):
            m = TwistSetpoint()
            m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = map(float, v)
            m.token = list(self._token)
            return m

        zero = np.zeros(3)
        for _ in range(5):  # prime: the session must find a (zero) target waiting
            self._tw_pub.publish(setpoint(zero))
        if not self._open_stream.wait_for_service(timeout_sec=3.0):
            raise Abort("/open_stream unavailable -- nothing moved")
        res = self._call(
            self._open_stream,
            OpenStream.Request(controller=TWIST_CONTROLLER, timeout_s=STREAM_TIMEOUT_S, token=list(self._token)),
        )
        if res is not None and res.accepted:
            self.sent_goals += 1
        if res is None or not res.accepted:
            raise Abort(f"{name}: /open_stream refused: {getattr(res, 'message', 'no answer')} -- nothing moved")
        if not reads_channel(res.channels, "twist"):
            self._close()
            raise Abort(f"{name}: {TWIST_CONTROLLER} reads {list(res.channels)}, not /setpoint/twist -- closed")
        press = ContactPress(push=push, stop_nm=stop_nm)
        v_cmd = -CONTACT_SPEED_MPS * n_out
        peak = t = 0.0
        stop = None
        trace = []
        try:
            taus, t_b = [], time.monotonic()
            while time.monotonic() - t_b < CONTACT_BASELINE_S:  # hold still: the baseline
                self._tw_pub.publish(setpoint(zero))
                taus.append(self.joint_torques()[CONTACT_WRIST])
                time.sleep(1.0 / STREAM_RATE_HZ)
            tau0 = np.mean(taus, axis=0)
            t_start = time.monotonic()
            while stop is None:
                t = time.monotonic() - t_start
                p = self.ee_pos()  # raises Abort if the state goes stale
                dev = np.abs(self.joint_torques()[CONTACT_WRIST] - tau0)
                rel = p - p0
                along = -float(rel @ n_out)  # + = into the panel
                lat = float(np.linalg.norm(rel + along * n_out))
                peak = max(peak, float(dev.max()))
                trace.append([round(t, 3), round(along * 1000, 2), *np.round(dev, 2).tolist()])
                stop = press.update(t, CONTACT_SPEED_MPS * t, along, lat, float(dev.max()))
                if stop is None:
                    self._tw_pub.publish(setpoint(v_cmd))
                    time.sleep(1.0 / STREAM_RATE_HZ)
        except KeyboardInterrupt:
            self._close()
            say("  RammpArm: Ctrl-C -- twist session closed, holding")
            raise
        except Abort as e:
            stop = ("error", str(e))
        self._close()  # hold at the measured q: the push stops here
        fin = self.ee_pos()
        depth = -float((fin - p0) @ n_out)
        rec.update(
            executed=True,
            stop=stop[0],
            why=stop[1],
            touch_m=press.touch_m,
            depth_m=depth,
            peak_nm=peak,
            t_s=t,
            ee_after=fin.tolist(),
            trace_cols=["t_s", "in_mm", "dev_j4", "dev_j5", "dev_j6", "dev_j7"],
            trace=trace,
        )
        log(rec)
        touch = "no touch" if press.touch_m is None else f"touch at {press.touch_m*1000:.1f} mm"
        say(
            f"  {name:22s} contact: {stop[0]} -- {stop[1]}; {touch}, peak {peak:.1f} N.m; "
            f"holding {depth*1000:.1f} mm in after {t:.2f} s"
        )
        if stop[0] not in ("pressed", "stop"):
            raise Abort(f"{name}: {stop[1]} -- session closed, HOLDING HERE")
        return {"stop": stop[0], "touch_m": press.touch_m, "depth_m": depth, "peak_nm": peak, "t_s": t}

    def cancel_motion(self):
        """Stop the trajectory in flight, if any (the driver then holds where the arm is), and
        make every later leg of this task refuse to start. Safe from another thread."""
        self.cancel_requested = True
        gh = self._active_gh
        if gh is not None:
            gh.cancel_goal_async()

    def _close(self):
        """Close the stream session (the driver then holds at the measured q)."""
        res = self._call(self._close_stream, CloseStream.Request(token=list(self._token)), timeout_s=2.0)
        if res is None or not res.closed:
            say(
                f"  WARNING: /close_stream: {getattr(res, 'message', 'no answer')} "
                f"(the session also ends {STREAM_TIMEOUT_S:.1f} s after the last setpoint)"
            )

    def _send(self, t, q, qd):
        """Send one timed joint trajectory (with PATH_TOL_RAD tracking tolerance) and wait for it.
        Returns (error_code, message); 0 = success."""
        if self.cancel_requested:
            raise Abort("cancelled -- nothing sent, HOLDING HERE")
        if not self._traj.wait_for_server(timeout_sec=5.0):
            raise Abort("no /execute_joint_trajectory action server -- nothing sent")
        g = ExecuteJointTrajectory.Goal()
        g.trajectory.joint_names = self.joint_names
        for ti, qi, qdi in zip(t, q, qd, strict=False):
            pt = JointTrajectoryPoint()
            pt.positions, pt.velocities = qi.tolist(), qdi.tolist()
            sec = int(ti)
            pt.time_from_start = Duration(sec=sec, nanosec=int(round((ti - sec) * 1e9)))
            g.trajectory.points.append(pt)
        tol = []
        for n in self.joint_names:
            jt = JointTolerance()
            jt.name, jt.position = n, PATH_TOL_RAD
            tol.append(jt)
        g.path_tolerance = tol
        g.control_mode = 0  # POSITION
        g.preemption = 0  # QUEUE
        g.sender_id = SENDER_ID
        g.token = list(self._token)
        gh = self._wait(self._traj.send_goal_async(g), 5.0)
        if gh is None or not gh.accepted:
            raise Abort("trajectory goal REJECTED -- nothing moved")
        self._active_gh = gh
        self.sent_goals += 1
        try:
            if self.cancel_requested:  # cancelled while this leg was being sent
                gh.cancel_goal_async()
            res = self._wait(gh.get_result_async(), float(t[-1]) + 30.0)
        except KeyboardInterrupt:
            gh.cancel_goal_async()
            time.sleep(0.2)  # let the cancel go out before the exception unwinds the process
            say("  RammpArm: Ctrl-C -- goal cancelled")
            raise
        finally:
            self._active_gh = None
        if res is None:
            gh.cancel_goal_async()
            return -100, "no result in time -- goal cancelled"
        return int(res.result.error_code), res.result.error_string


def spin_node(name: str) -> tuple[Node, SingleThreadedExecutor, threading.Thread]:
    """A node on its own executor, spun on a daemon thread (Perception spins the global one)."""
    if not rclpy.ok():
        rclpy.init()
    node = Node(name)
    ex = SingleThreadedExecutor()  # main thread only polls futures; one thread is cheaper
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    return node, ex, th
