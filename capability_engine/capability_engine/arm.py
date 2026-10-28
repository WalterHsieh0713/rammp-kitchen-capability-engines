"""`self.arm`: the arm driver (kinova-gen3-ros2) and its cuRobo planner, for a capability.

`ArmInterface` lives on the engine's node and callback group and owns no node, executor or thread.
Its ROS entities exist only inside an action: the first arm call after `begin_action()` creates
them all (`_open`), and `end_action()` settles any goal or stream, destroys them and forgets the
state they received. An action that never touches the arm costs nothing, and no state read in one
action is returned in the next.

In mock mode (`mock=True`) it is a dry run that makes no ROS entity at all: motions and goals log
"mock: not sent" and succeed, and the state is the last commanded joints (from MOCK_JOINTS), with
the tool pose from the sim's FK when a sim exists or can be built from MOCK_URDF. That FK sets the
arm joints of the process's one shared sim copy, as solve_ik does.
"""

import copy
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import rammp_arm
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTolerance
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from rammp_arm import CONTINUOUS, ERRORS, JOINTS, _wrap
from rammp_arm_interfaces.action import (
    ExecuteJointTrajectory,
    GoToEEPose,
    GoToJointConfig,
    GoToPreset,
)
from rammp_arm_interfaces.msg import EeState as EeStateMsg
from rammp_arm_interfaces.msg import (
    GripperSetpoint,
    ImpedanceGains,
    JointImpedanceGainValues,
    JointSetpoint,
    PoseSetpoint,
    TwistSetpoint,
)
from rammp_arm_interfaces.msg import GripperState as GripperStateMsg
from rammp_arm_interfaces.msg import StreamStatus as StreamStatusMsg
from rammp_arm_interfaces.srv import CloseStream, ListControllers, OpenStream, SetGains
from rammp_common_interfaces.msg import ControlStatus as ControlStatusMsg
from rammp_common_interfaces.msg import EStop
from rammp_common_interfaces.srv import AcquireControl, ReleaseControl, RevokeControl
from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
from rammp_curobo_interfaces.srv import SetWorld
from rclpy.action import ActionClient
from rclpy.callback_groups import CallbackGroup
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from capability_engine import sim
from capability_engine.cornell_commands import (
    CartesianTrajectoryCommand,
    CloseGripperCommand,
    JointCommand,
    OpenGripperCommand,
)
from capability_engine.driver_state import (
    ESTOPPED_REASON,
    ArmError,
    DriverState,
    EStopped,
    ModeConflict,
    NotInAction,
    StateTimeout,
)

STATE_WAIT_S = 10.0  # how long the first arm call in an action waits for the driver's state
STATE_STALE_S = 0.5  # /joint_states, /ee_state or /gripper_state older than this is an error
DIAGNOSTICS_STALE_S = 3.0  # /diagnostics (1 Hz) older than this is an error
GOAL_ACCEPT_S = 5.0  # how long a motion waits for the driver to accept or reject its goal
RESULT_MARGIN_S = 30.0  # a trajectory's result may come this long after its planned duration
POLL_S = 0.01  # how often a running motion polls its result, its monitor and the action's cancel
CANCEL_SETTLE_S = 0.4  # after cancelling a goal, how long to wait for the driver to settle it
SUCCESSFUL = 0  # error_code of a goal that succeeded
PREEMPTED = -6  # error_code of a goal settled by a cancel
HALTED = -9  # error_code of a goal the driver halted: control taken, or an e-stop
HALT_STATUS_WAIT_S = 0.2  # after a HALTED goal, how long to wait for /control_status to show
# an e-stop (the result can arrive before the status), before taking it as control taken
ACTION_ENDED_MESSAGE = "action ended"  # last_error of a motion the end of its action stopped
PLANNER_WAIT_S = 5.0  # how long a go_to_* call waits for its action server (the cuRobo planner)
GO_TO_SETTLE_S = 0.3  # after a go_to goal, let /ee_state and /joint_states catch up with its end
GRIPPER_PERIOD_S = 0.02  # gripper() streams /setpoint/gripper at 50 Hz
GRIPPER_MOVED = 0.002  # gripper(): a position change (0..1) above this means the fingers move
GRIPPER_JUDGE_AFTER_S = 0.5  # gripper(): the fingers are judged still only after this long
GRIPPER_STILL_S = 0.3  # gripper(): still this long -> settled, the stream stops
PLAN_TIMEOUT_S = 30.0  # how long a plan_to_* call waits for cuRobo's plan
SERVICE_WAIT_S = 5.0  # how long a service call waits for its server to come up
SERVICE_CALL_S = 10.0  # how long a service call waits for its answer (set_world reloads cuRobo)
STREAM_TIMEOUT_S = 0.1  # stream(): the driver closes the session after this long with no setpoint
PUBLISHER_MATCH_S = 1.0  # how long a new setpoint or /estop publisher waits for the driver to
# subscribe before its first message (DDS discovery), so that message is not lost
CLOSE_STREAM_WAIT_S = 1.0  # how long closing a stream waits for /close_stream's answer
MOCK_JOINTS = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]  # mock mode: the joints it starts at
MOCK_URDF = ""  # mock mode: the URDF file make_sim() loads ("": none, make_sim() refuses)
MOCK_NOT_SENT = "mock: not sent"
MOCK_NO_MODEL_MESSAGE = "mock mode has no driver model: set arm.mock_urdf to a URDF file"
CONTROL_FROM_ORCHESTRATOR_MESSAGE = "control comes from the orchestrator"
STREAM_ENDED_MESSAGE = "the stream is closed: send() works only inside its with block"

# The driver's goal fields, for the keyword arguments of the goal methods (the driver's values).
HOLD_NONE = 0  # go_to_ee_pose orientation_hold: the planner reorients as it likes
HOLD_LEVEL = 1  # keep the tool's tilt; spin about arm_mount z is free
HOLD_FIXED = 2  # keep the orientation entirely
POSITION = 0  # control_mode: stiff position control
IMPEDANCE = 1  # control_mode: compliant joint impedance
QUEUE = 0  # execute_joint_trajectory preemption: wait for the running goal to finish
LATEST_WINS = 1  # execute_joint_trajectory preemption: replace the running goal
# `gains` by name: the driver's impedance profiles ("default": the session's default)
GAIN_PROFILES = {
    "default": ImpedanceGains.PROFILE_SESSION_DEFAULT,
    "soft": ImpedanceGains.PROFILE_SOFT,
    "medium": ImpedanceGains.PROFILE_MEDIUM,
    "stiff": ImpedanceGains.PROFILE_STIFF,
}
GAIN_KEYS = ("kq", "zeta", "torque_limit")  # `gains` as a dict: custom joint impedance

# QoS as in RammpArm (rammp_arm.py@7bc328ae:190-201) and the driver's docs/interface.md.
LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
SETPOINT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
)
ESTOP_QOS = 10  # /estop is volatile; a reliable publisher reaches any subscriber
# What stream() sends on each /setpoint/<channel> a controller reads, and what send() takes.
SETPOINTS = {
    "joint_position": (JointSetpoint, "7 joint positions (rad)"),
    "joint_velocity": (JointSetpoint, "7 joint velocities (rad/s)"),
    "joint_torque": (JointSetpoint, "7 joint torques (N m)"),
    "pose": (PoseSetpoint, "(position xyz, quaternion xyzw) of /ee_state's frame in arm_mount"),
    "twist": (TwistSetpoint, "6 floats: linear (m/s) then angular (rad/s), in arm_mount axes"),
}

ACTIONS = {
    "execute_joint_trajectory": (ExecuteJointTrajectory, "/execute_joint_trajectory"),
    "go_to_ee_pose": (GoToEEPose, "/go_to_ee_pose"),
    "go_to_joint_config": (GoToJointConfig, "/go_to_joint_config"),
    "go_to_preset": (GoToPreset, "/go_to_preset"),
    "plan_to_pose": (PlanToPose, "/rammp_curobo/plan_to_pose"),
    "plan_to_joints": (PlanToJoints, "/rammp_curobo/plan_to_joints"),
}
SERVICES = {
    "open_stream": (OpenStream, "/open_stream"),
    "close_stream": (CloseStream, "/close_stream"),
    "list_controllers": (ListControllers, "/list_controllers"),
    "set_gains": (SetGains, "/set_gains"),
    "acquire_control": (AcquireControl, "/acquire_control"),
    "release_control": (ReleaseControl, "/release_control"),
    "revoke_control": (RevokeControl, "/revoke_control"),
    "set_world": (SetWorld, "/rammp_curobo/set_world"),
}
# The first arm call in an action waits for these (RammpArm's constructor did).
MOTION_STATE = ("/joint_states", "/ee_state", "/gripper_state")
NOT_IN_ACTION_MESSAGE = "the arm is only available inside an action"
NO_STATE_MESSAGE = (
    "no /joint_states, /ee_state or /gripper_state -- is the sheppy arm node up, "
    "and ROS_DOMAIN_ID / RMW_IMPLEMENTATION exported?"
)


@dataclass
class JointStates:
    """/joint_states for arm_joint_1..7, as the driver reports them (not wrapped)."""

    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    age_s: float = 0.0


@dataclass
class EeState:
    """/ee_state: the flange (not the tool frame) in `frame_id`, its twist and measured wrench."""

    frame_id: str
    position: np.ndarray  # x, y, z
    orientation: np.ndarray  # x, y, z, w
    linear: np.ndarray
    angular: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    age_s: float = 0.0


@dataclass
class GripperState:
    position: float  # 0 (open) .. 1 (closed)
    effort: float
    current: float
    present: bool
    age_s: float = 0.0


@dataclass
class ControlStatus:
    arbitration_enabled: bool
    estopped: bool
    owned: bool
    owner_id: str
    generation: int
    rejected_count: int
    age_s: float = 0.0


@dataclass
class StreamStatus:
    open: bool
    controller: str
    channels: list[str]
    timeout_s: float
    rejected_count: int
    age_s: float = 0.0


@dataclass
class DiagnosticStatus:
    level: int
    name: str
    message: str
    hardware_id: str
    values: dict[str, str] = field(default_factory=dict)


@dataclass
class Diagnostics:
    status: list[DiagnosticStatus]
    age_s: float = 0.0


@dataclass
class TrajectoryResult:
    """execute_joint_trajectory's result, as the driver settled the goal."""

    error_code: int  # 0 SUCCESSFUL, or one of rammp_arm.ERRORS
    error_string: str
    final_error: np.ndarray  # joint position error (reference - actual) at the end, rad


@dataclass
class GoalFeedback:
    """A running goal's feedback."""

    fraction_complete: float  # execution progress, 0 .. 1 (0 while planning)
    actual: np.ndarray  # measured joint positions, rad
    phase: str = ""  # go_to goals: "planning" or "executing"
    planner_state: str = ""  # go_to goals: cuRobo's state while planning


@dataclass
class Plan:
    """A cuRobo plan (nothing moved): sample times (s), joints and joint velocities, (N, 7) in
    JOINTS order. A failed plan has success False, the planner's message and no samples."""

    success: bool
    message: str
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, len(JOINTS))))
    qd: np.ndarray = field(default_factory=lambda: np.zeros((0, len(JOINTS))))
    planning_time: float = 0.0  # s


@dataclass
class Controller:
    """A streaming controller of the driver (/list_controllers)."""

    name: str
    channels: list[str]  # the /setpoint/<channel> topics it reads
    available: bool  # whether the running driver can open it


def _goal_feedback(m):
    return GoalFeedback(
        float(m.fraction_complete),
        np.array(m.actual.positions),
        getattr(m, "phase", ""),
        getattr(m, "planner_state", ""),
    )


def _impedance_gains(gains) -> ImpedanceGains:
    """`gains` as the driver's ImpedanceGains: None (the message default), a profile name in
    GAIN_PROFILES, or a dict {"kq": 7 floats, "zeta": float, "torque_limit": 7 floats} (custom
    joint impedance, RammpArm's IMPEDANCE without path_tol). ArmError for anything else."""
    if gains is None:
        return ImpedanceGains()
    if isinstance(gains, str) and gains in GAIN_PROFILES:
        return ImpedanceGains(profile=GAIN_PROFILES[gains])
    if isinstance(gains, dict) and set(gains) == set(GAIN_KEYS):
        try:
            kq = np.asarray(gains["kq"], float)
            zeta = float(gains["zeta"])
            torque_limit = np.asarray(gains["torque_limit"], float)
        except (TypeError, ValueError):
            kq = None
        if kq is not None and kq.shape == torque_limit.shape == (len(JOINTS),):
            return ImpedanceGains(
                profile=ImpedanceGains.PROFILE_CUSTOM,
                custom=JointImpedanceGainValues(
                    kq=kq.tolist(), zeta=zeta, torque_limit=torque_limit.tolist()
                ),
            )
    raise ArmError(
        f"gains must be one of {', '.join(GAIN_PROFILES)} or a dict "
        f'{{"kq": 7 floats, "zeta": float, "torque_limit": 7 floats}}, not {gains!r}'
    )


def _refusal(why):
    """The driver API's exception for not_ready() reason `why`: EStopped for an e-stop (as
    Stream.send raises), ArmError otherwise."""
    return EStopped if why == ESTOPPED_REASON else ArmError


def _duration(t):
    sec = int(t)
    return Duration(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def _seconds(d):
    return d.sec + d.nanosec * 1e-9


def _joint_index(names):
    """Where arm_joint_1..7 are in `names`, in JOINTS order, when `names` are exactly those or
    joint_1..7 (the sim's rule: cuRobo's config names them joint_<i>); None otherwise."""
    names = list(names)
    for fmt in ("arm_joint_{}", "joint_{}"):
        wanted = [fmt.format(i) for i in range(1, len(JOINTS) + 1)]
        if sorted(names) == sorted(wanted):
            return [names.index(n) for n in wanted]
    return None


def _plan(res):
    """A planner's result as a Plan, its joints reordered to JOINTS; None when its joints are
    not arm_joint_1..7 or joint_1..7 (_joint_index) or a point lacks positions or velocities
    for them."""
    names = list(res.trajectory.joint_names)
    idx = _joint_index(names)
    if idx is None:
        return None
    points = res.trajectory.points
    if any(len(p.positions) != len(names) or len(p.velocities) != len(names) for p in points):
        return None
    return Plan(
        True,
        res.message,
        np.array([_seconds(p.time_from_start) for p in points]),
        np.array([[p.positions[i] for i in idx] for p in points]).reshape(-1, len(JOINTS)),
        np.array([[p.velocities[i] for i in idx] for p in points]).reshape(-1, len(JOINTS)),
        float(res.planning_time),
    )


def _setpoint(channel, value):
    """`value` as the message on /setpoint/<channel> (see SETPOINTS), or None when it is not
    of that channel's shape."""
    msg_type = SETPOINTS[channel][0]
    try:
        if channel == "pose":
            pos, quat = (np.asarray(v, float) for v in value)
            if pos.shape != (3,) or quat.shape != (4,):
                return None
            m = msg_type()
            p, o = m.pose.position, m.pose.orientation
            p.x, p.y, p.z = pos.tolist()
            o.x, o.y, o.z, o.w = quat.tolist()
            return m
        v = np.asarray(value, float)
    except (TypeError, ValueError):
        return None
    if channel == "twist":
        if v.shape != (6,):
            return None
        m = msg_type()
        lin, ang = m.twist.linear, m.twist.angular
        lin.x, lin.y, lin.z, ang.x, ang.y, ang.z = v.tolist()
        return m
    if v.shape != (len(JOINTS),):
        return None
    return msg_type(values=v.tolist())


def _tolerances(names, position):
    """A position tolerance (rad) per joint: `position` is one for all or one per joint; None
    gives none (the driver's default)."""
    if position is None:
        return []
    per_joint = np.broadcast_to(np.asarray(position, float), (len(names),))
    return [
        JointTolerance(name=n, position=float(p)) for n, p in zip(names, per_joint, strict=True)
    ]


def _vec(v):
    return np.array([v.x, v.y, v.z])


def _joint_states(m):
    if not set(JOINTS) <= set(m.name):
        return None
    idx = [list(m.name).index(n) for n in JOINTS]
    return JointStates(
        np.array([m.position[i] for i in idx]),
        np.array([m.velocity[i] for i in idx]),
        np.array([m.effort[i] for i in idx]),
    )


def _ee_state(m):
    o = m.pose.orientation
    return EeState(
        m.header.frame_id,
        _vec(m.pose.position),
        np.array([o.x, o.y, o.z, o.w]),
        _vec(m.twist.linear),
        _vec(m.twist.angular),
        _vec(m.wrench.force),
        _vec(m.wrench.torque),
    )


def _tool_position(ee):
    """The tool frame's position: the flange + TOOL_Z along the flange z."""
    return ee.position + R.from_quat(ee.orientation).apply([0.0, 0.0, rammp_arm.TOOL_Z])


def _gripper_state(m):
    return GripperState(float(m.position), float(m.effort), float(m.current), bool(m.present))


def _control_status(m):
    return ControlStatus(
        m.arbitration_enabled, m.estopped, m.owned, m.owner_id, m.generation, m.rejected_count
    )


def _stream_status(m):
    return StreamStatus(m.open, m.controller, list(m.channels), m.timeout_s, m.rejected_count)


def _diagnostics(m):
    return Diagnostics(
        [
            DiagnosticStatus(
                ord(s.level),  # a ROS byte field: one-byte bytes
                s.name,
                s.message,
                s.hardware_id,
                {kv.key: kv.value for kv in s.values},
            )
            for s in m.status
        ]
    )


class _Gate(CallbackGroup):
    """The engine's callback group, closable. rclpy (Humble) raises InvalidHandle out of the
    executor's spin() when an entity is destroyed after the executor found it ready but before it
    took the message, so close() lets no new callback start and waits for the running ones; the
    entities in the gate can then be destroyed while the engine's executor spins. close() waits
    for running callbacks with no timeout, so a callback on a gate must never block or call
    end_action()."""

    def __init__(self, group):
        super().__init__()
        self._group = group
        self._cond = threading.Condition()
        self._running = 0
        self._closed = False

    def add_entity(self, entity):
        super().add_entity(entity)
        self._group.add_entity(entity)

    def can_execute(self, entity):
        return not self._closed and self._group.can_execute(entity)

    def beginning_execution(self, entity):
        with self._cond:
            if self._closed or not self._group.beginning_execution(entity):
                return False
            self._running += 1
            return True

    def ending_execution(self, entity):
        self._group.ending_execution(entity)
        with self._cond:
            self._running -= 1
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.wait_for(lambda: self._running == 0)


class Stream:
    """A streaming session open on the driver, from ArmInterface.stream(): send() a setpoint
    at least every timeout_s, or the driver closes the session."""

    def __init__(self, arm, controller, channel, ctx, driver):
        self.controller = controller
        self.channel = channel  # the /setpoint/<channel> it publishes on
        self._arm = arm
        self._ctx, self._driver, self._token = ctx, driver, driver.token
        self._pub = None  # created before open_stream, destroyed when the session closes
        self._opened = False  # the driver accepted open_stream

    def send(self, setpoint):
        """Send one setpoint: 7 joints for joint controllers, (pos, quat xyzw) of /ee_state's frame
        (not get_state()'s tool frame) for pose controllers, 6 floats (linear, then angular, in
        arm_mount axes) for twist controllers. Never blocks. Raises Cancelled once the action is
        cancelled, StreamClosed, ControlTaken or EStopped when the driver will not take it, and
        NotInAction outside the with block; leaving the block then zeroes and closes the session."""
        self._arm._send_setpoint(self, setpoint)

    def _message(self, value):
        """`value` as this session's setpoint message, with the token; None for a wrong shape."""
        m = _setpoint(self.channel, value)
        if m is not None and self._token is not None:
            m.token = list(self._token)
        return m


class ArmInterface:
    """`self.arm`: RammpArm's API and the driver's, side by side, on the engine's node."""

    def __init__(self, node, callback_group, *, standalone: bool = False, mock: bool = False):
        self._node = node
        self._group = callback_group
        self._standalone = standalone
        self._mock = mock
        self._driver = DriverState(standalone=standalone)
        self._ctx = None
        self._in_action = False
        self._open_lock = threading.Lock()
        self._opened = False
        self._gate = None  # the callback group of the entities _open() made
        self._subs, self._actions, self._clients, self._pubs = {}, {}, {}, {}
        self._cond = threading.Condition()
        self._latest = {}  # topic -> (dataclass, time.monotonic() when received)
        self.last_error = ""  # the most recent refusal or failure, also logged
        self.last_path = []  # the last trajectory's measured tool path: [(t, x, y, z)]
        self.monitor_stopped = False  # the last trajectory was stopped by its monitor
        self._goal = None  # the send future of the running goal: end_action() cancels it
        self._path_t0 = None  # time.monotonic() its goal started; set: /ee_state adds to last_path
        self._stream = None  # the open Stream: end_action() zeroes and closes it
        self._mock_q = np.array(MOCK_JOINTS, float)  # mock mode: the last commanded joints
        self._mock_gripper = 0.0  # mock mode: the last commanded gripper position (0 = open)

    # -- lifecycle ----------------------------------------------------------------------------

    def begin_action(self, ctx, token: bytes | None):
        """Start an action: arm calls are allowed until end_action(). Creates no ROS entity."""
        if self._in_action or self._opened:  # the last action never ended: carry nothing over
            self.end_action()
        self._ctx = ctx
        # A fresh baseline: this action's /control_status generation is its own, not the last's.
        self._driver = DriverState(standalone=self._standalone)
        self._driver.set_token(token)
        self._in_action = True

    def end_action(self):
        """End the action: settle any goal or stream, destroy every ROS entity _open() made and
        forget the state they received, so the next action starts from the driver afresh.
        Worst case, it waits GOAL_ACCEPT_S + CANCEL_SETTLE_S for a goal the driver has not yet
        answered, to accept it and then to settle its cancel, or CLOSE_STREAM_WAIT_S for a stream
        to close."""
        with self._cond:
            self._in_action = False
            self._cond.notify_all()  # a call waiting for state wakes with NotInAction
        try:
            self._settle()
        finally:
            with self._open_lock:
                self._destroy()
            with self._cond:
                self._latest.clear()
                self._cond.notify_all()
            self._ctx = None

    def close(self):
        """A no-op under the engine, which owns the node."""

    def _settle(self):
        """Cancel any running goal and close any open stream, waiting for each to settle.
        Called after the action has ended: uses the stored handles only, never _ensure_open()."""
        with self._cond:
            send, self._goal = self._goal, None
        if send is not None:  # sent, perhaps not yet accepted: wait for the handle to cancel it
            gh = self._wait(send, GOAL_ACCEPT_S)
            if gh is not None and gh.accepted:
                self._cancel_goal(gh, gh.get_result_async())
        stream = self._stream
        if stream is not None:
            self._close_stream(stream)

    def _check_in_action(self):
        if not self._in_action:
            self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)

    def _ensure_open(self, wait_state=True):
        """Create the arm's entities on the first call in an action, and (`wait_state`) wait for
        the arm's state; the e-stop, the planner and the services need none. The wait is outside
        _open_lock, so a call that needs no state is never held up by one that waits."""
        self._check_in_action()
        with self._open_lock:
            # Again under the lock: end_action() may have destroyed everything since the check.
            if not self._in_action:
                self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
            opened_now = not self._opened
            if opened_now:
                self._open()
        if opened_now and wait_state and not all(self._wait_for(t) for t in MOTION_STATE):
            self._fail(StateTimeout, NO_STATE_MESSAGE)

    def _open(self):
        """Create the subscriptions, clients and publishers."""
        with self._cond:
            self._latest.clear()
        node = self._node
        group = self._gate = _Gate(self._group)
        topics = (
            (JointState, "/joint_states", qos_profile_sensor_data, _joint_states),
            (EeStateMsg, "/ee_state", qos_profile_sensor_data, _ee_state),
            (GripperStateMsg, "/gripper_state", qos_profile_sensor_data, _gripper_state),
            (ControlStatusMsg, "/control_status", LATCHED, self._on_control_status),
            (StreamStatusMsg, "/stream_status", LATCHED, self._on_stream_status),
            (DiagnosticArray, "/diagnostics", qos_profile_sensor_data, _diagnostics),
        )
        for msg_type, topic, qos, convert in topics:
            self._subs[topic] = node.create_subscription(
                msg_type,
                topic,
                lambda m, t=topic, c=convert: self._store(t, c(m)),
                qos,
                callback_group=group,
            )
        for name, (action_type, path) in ACTIONS.items():
            self._actions[name] = ActionClient(node, action_type, path, callback_group=group)
        for name, (srv_type, path) in SERVICES.items():
            self._clients[name] = node.create_client(srv_type, path, callback_group=group)
        self._pubs["/setpoint/gripper"] = node.create_publisher(
            GripperSetpoint, "/setpoint/gripper", SETPOINT_QOS
        )
        self._pubs["/estop"] = node.create_publisher(EStop, "/estop", ESTOP_QOS)
        self._opened = True

    def _destroy(self):
        if self._gate is not None:
            self._gate.close()
        node = self._node
        for sub in self._subs.values():
            node.destroy_subscription(sub)
        for action_client in self._actions.values():
            action_client.destroy()
        for client in self._clients.values():
            node.destroy_client(client)
        for pub in self._pubs.values():
            node.destroy_publisher(pub)
        self._subs, self._actions, self._clients, self._pubs = {}, {}, {}, {}
        self._gate = None
        self._opened = False

    def _fail(self, error, message):
        self.last_error = message
        self._node.get_logger().error(message)
        raise error(message)

    def _refuse(self, message):
        """RammpArm's failure path: log `message` (its print), keep it in last_error, False."""
        self.last_error = message
        self._node.get_logger().error(message)
        return False

    def _check_command(self):
        """Raise (and log) ControlTaken or StreamClosed. An e-stop is not raised here: RammpArm's
        motions return False for it, through not_ready(), with RammpArm's message."""
        try:
            self._driver.check_command()
        except EStopped:
            pass
        except ArmError as e:
            self._fail(type(e), str(e))

    def _action(self, name):
        """Action client `name`, opening the arm's entities if needed; NotInAction, never a
        KeyError, once end_action() has destroyed them."""
        self._ensure_open()
        with self._open_lock:  # not while _destroy() empties the dict
            client = self._actions.get(name)
        if client is None:
            self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
        return client

    def _wait_for_server(self, client, ctx, timeout_s):
        """Whether the action server came up within `timeout_s`, polled in POLL_S slices so a
        cancel of the action (Cancelled) or its end (NotInAction) is not held up."""
        deadline = time.monotonic() + timeout_s
        while True:
            # Under _cond: end_action() clears _in_action under it before it destroys the client.
            with self._cond:
                if not self._in_action:
                    self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
                if client.server_is_ready():
                    return True
            if ctx.cancelled():
                ctx.checkpoint()  # raises the engine's Cancelled
            if time.monotonic() > deadline:
                return False
            time.sleep(POLL_S)

    def _estopped_within(self, timeout_s):
        """Whether /control_status shows the arm e-stopped, waiting up to `timeout_s` for it."""

        def estopped():
            latest = self._latest.get("/control_status")
            return latest is not None and latest[0].estopped

        with self._cond:
            return self._cond.wait_for(estopped, timeout_s)

    # -- mock mode ------------------------------------------------------------------------------

    def _mock_sent(self, label, q=None):
        """Mock mode: nothing goes to the driver. Log it; a joint target `q` becomes the state.
        True, as for a motion that succeeded."""
        self._check_in_action()
        if q is not None:
            self._mock_q = np.asarray(q, float).copy()
        self._node.get_logger().info(f"  {label}: {MOCK_NOT_SENT}")
        return True

    def _mock_sim(self):
        """Mock mode: the process's sim copy (built from MOCK_URDF, through the model provider,
        if none exists yet), or None when there is none and no MOCK_URDF to build it from."""
        try:
            return sim.make_sim()
        except sim.SimModelError:
            return None

    def _mock_read(self, topic):
        """Mock mode's state: the last commanded joints, at rest; the tool from the sim's FK when
        there is a sim (which sets the shared sim's arm joints), else at the origin; no e-stop,
        no stream, no diagnostics."""
        zeros = np.zeros(len(JOINTS))
        if topic == "/joint_states":
            return JointStates(self._mock_q.copy(), zeros, zeros.copy())
        if topic == "/ee_state":
            model = self._mock_sim()
            if model is None:
                pos, quat = np.zeros(3), R.identity().as_quat()
            else:
                pos, quat = sim.tool_pose(*model, self._mock_q)
            flange = pos - R.from_quat(quat).apply([0.0, 0.0, rammp_arm.TOOL_Z])
            return EeState(rammp_arm.FRAME, flange, quat, *(np.zeros(3) for _ in range(4)))
        if topic == "/gripper_state":
            return GripperState(self._mock_gripper, 0.0, 0.0, True)
        if topic == "/control_status":
            return ControlStatus(False, False, False, "", 0, 0)
        if topic == "/stream_status":
            return StreamStatus(False, "", [], 0.0, 0)
        return Diagnostics([])

    # -- state --------------------------------------------------------------------------------

    def _store(self, topic, value):
        if value is None:
            return
        now = time.monotonic()
        with self._cond:
            self._latest[topic] = (value, now)
            self._cond.notify_all()
        t0 = self._path_t0
        if topic == "/ee_state" and t0 is not None:  # a trajectory runs: record the tool path
            x, y, z = _tool_position(value)
            self.last_path.append((now - t0, float(x), float(y), float(z)))

    def _on_control_status(self, m):
        self._driver.on_control_status(
            m.estopped, m.arbitration_enabled, m.owned, m.owner_id, m.generation
        )
        return _control_status(m)

    def _on_stream_status(self, m):
        self._driver.on_stream_status(m.open)
        return _stream_status(m)

    def _wait_for(self, topic):
        """Wait up to STATE_WAIT_S for the first message on `topic`: its (value, receipt time),
        or None if none came. NotInAction if the action ends meanwhile."""
        with self._cond:
            self._cond.wait_for(lambda: topic in self._latest or not self._in_action, STATE_WAIT_S)
            if not self._in_action:
                self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
            return self._latest.get(topic)

    def _read(self, topic, stale_s=None):
        """The latest value on `topic` with its age; an error when there is none, or when it is
        older than `stale_s` (None: never stale)."""
        if self._mock:
            self._check_in_action()
            return self._mock_read(topic)
        self._ensure_open()
        latest = self._wait_for(topic)
        if latest is None:
            if topic in MOTION_STATE:
                self._fail(StateTimeout, NO_STATE_MESSAGE)
            self._fail(StateTimeout, f"no {topic} in {STATE_WAIT_S:g} s -- is the arm driver up?")
        value, received = latest
        age = time.monotonic() - received
        if stale_s is not None and age > stale_s:
            self._fail(
                ArmError, f"{topic} is stale: last message {age:.2f} s ago (limit {stale_s:g} s)"
            )
        return replace(copy.deepcopy(value), age_s=age)

    def joint_states(self) -> JointStates:
        return self._read("/joint_states", STATE_STALE_S)

    def ee_state(self) -> EeState:
        return self._read("/ee_state", STATE_STALE_S)

    def gripper_state(self) -> GripperState:
        return self._read("/gripper_state", STATE_STALE_S)

    def control_status(self) -> ControlStatus:
        return self._read("/control_status")  # latched, on change: never stale

    def stream_status(self) -> StreamStatus:
        return self._read("/stream_status")  # latched, on change: never stale

    def diagnostics(self) -> Diagnostics:
        return self._read("/diagnostics", DIAGNOSTICS_STALE_S)

    def get_state(self) -> dict:
        """RammpArm's state dict: joints wrapped to +-pi, `ee_pos` the tool frame (flange +
        TOOL_Z along the flange z) as [x, y, z, qx, qy, qz, qw]. Like RammpArm, the latest
        values however old: only the driver-API reads (joint_states(), ...) refuse stale state."""
        js, ee, grip = (self._read(topic) for topic in MOTION_STATE)
        tool = _tool_position(ee)
        return {
            "position": _wrap(js.position),
            "velocity": js.velocity,
            "effort": js.effort,
            "ee_pos": np.concatenate([tool, ee.orientation]),
            "gripper_pos": grip.position,
        }

    def raw_joints(self) -> np.ndarray:
        """Measured joints as the driver reports them (not wrapped): where a cuRobo plan must
        start. The latest value however old, like get_state()."""
        return self._read("/joint_states").position

    def not_ready(self) -> str | None:
        """Why the arm must not be commanded now (from /control_status), or None when it may be."""
        if self._mock:
            self._check_in_action()
            return None
        self._ensure_open()
        self._wait_for("/control_status")
        return self._driver.not_ready(rammp_arm.SENDER_ID)

    def robot_description(self) -> str:
        """The driver's URDF, from /arm/robot_description (latched), through a one-shot
        subscription destroyed after the first message. Opens none of the arm's other entities.
        In mock mode, MOCK_URDF's text; SimModelError when it is unset (this is make_sim()'s
        model provider, and a missing model is a sim error)."""
        self._check_in_action()
        if self._mock:
            if not MOCK_URDF:
                raise sim.SimModelError(MOCK_NO_MODEL_MESSAGE)
            return Path(MOCK_URDF).read_text()
        received = {}
        done = threading.Event()

        def on_description(m):
            received["urdf"] = m.data
            done.set()

        gate = _Gate(self._group)
        sub = self._node.create_subscription(
            String, "/arm/robot_description", on_description, LATCHED, callback_group=gate
        )
        try:
            if not done.wait(STATE_WAIT_S):
                self._fail(
                    StateTimeout,
                    f"no /arm/robot_description in {STATE_WAIT_S:g} s -- is the arm driver up?",
                )
        finally:
            gate.close()
            self._node.destroy_subscription(sub)
        return received["urdf"]

    # -- motions (RammpArm's, rammp_arm.py@7bc328ae:260-458) -------------------------------

    def execute_command(self, cmd):
        if isinstance(cmd, JointCommand):
            return self.move_joints([np.asarray(cmd.pos, dtype=float)])
        if isinstance(cmd, CartesianTrajectoryCommand):
            return self.move_cartesian(cmd.traj)
        if isinstance(cmd, OpenGripperCommand):
            return self.gripper(0.0)
        if isinstance(cmd, CloseGripperCommand):
            return self.gripper(1.0)
        raise TypeError(f"RammpArm: unsupported command {type(cmd).__name__}")

    def move_joints(self, knots, cart_lengths=None, speed_scale=1.0, impedance=None, monitor=None):
        """Through joint `knots` (wrapped or not) from the measured joints. `cart_lengths`: the
        Cartesian distance of each segment -- given, the path is timed in metres at CART_SPEED,
        otherwise in joint-space radians at JOINT_SPEED. `impedance`: a dict like IMPEDANCE -> run
        compliant (joint impedance) instead of stiff position control. `monitor`: polled at ~100 Hz
        while the trajectory runs; when it returns True the goal is cancelled (the driver stops the
        arm and holds), `monitor_stopped` is set and the call returns True."""
        self._checkpoint()
        samples = rammp_arm.joint_trajectory(self.raw_joints(), knots, cart_lengths, speed_scale)
        if samples is None:
            return True
        t, q, qd = samples
        # fridge runs only: the wheelchair keep-out, every 2nd sample + end
        if sim.keep_out_enabled():
            rb = sim.make_sim()[1]
            for qi in list(q[::2]) + [q[-1]]:
                msg = sim.keep_out_message(rb, qi)
                if msg is not None:
                    return self._refuse(
                        f"  RammpArm: the trajectory enters the keep-out zone ({msg}) -- "
                        "nothing sent."
                    )
        return self._send(t, q, qd, impedance, monitor)

    def move_cartesian(self, traj, impedance=None, speed_scale=1.0):
        """`traj`: [(pos, quat xyzw), ...] in the base frame, like CartesianTrajectoryCommand. Each
        segment (from the current pose) is a straight line with slerped orientation -- what Kortex
        did -- cut into <= CART_STEP_M / CART_STEP_DEG steps before IK, the same density as
        `plan_cartesian`, so the joint path is the one the sim gates checked. In mock mode with a
        sim the same gates run, so a path the real arm would refuse returns False here too."""
        self._checkpoint()
        if self._mock and self._mock_sim() is None:  # no model to solve the path on: joints stay
            return self._mock_sent("move_cartesian")
        refusals = []
        knots, lengths = rammp_arm.cartesian_knots(
            sim.make_sim(), self.get_state()["ee_pos"], self.raw_joints(), traj, log=refusals.append
        )
        if knots is None:
            return self._refuse(refusals[-1])
        return self.move_joints(
            knots, cart_lengths=lengths, impedance=impedance, speed_scale=speed_scale
        )

    # -- planned moves (RammpArm's go_to_*, rammp_arm.py@7bc328ae:317-378) -------------------

    def go_to_ee_pose(
        self,
        pos,
        quat,
        label="go_to_ee_pose",
        speed_scale=None,
        *,
        orientation_hold=HOLD_NONE,
        control_mode=POSITION,
        gains=None,
        feedback=None,
    ):
        """Tool (tool_frame = ee_pos) to (pos, quat xyzw) in FRAME: cuRobo plans from the measured
        joints and the driver executes. Blocking; True if the driver succeeded and the tool ended
        within GO_TO_TRACK_M. `orientation_hold` (HOLD_*), `control_mode` (POSITION or IMPEDANCE)
        and `gains` (see _impedance_gains()) are the driver's goal fields. `feedback`: called with
        each GoalFeedback, on the executor (it must not block). In mock mode with a sim, the
        joints become the sim's IK solution for the pose when it is within MAX_IK_ERR (seeded
        from the mock joints); otherwise they stay."""
        if self._mock:
            self._checkpoint()
            model, q = self._mock_sim(), None
            if model is not None:
                solution, err = sim.solve_ik(*model, pos[:3], quat, self._mock_q)
                q = solution if err <= sim.MAX_IK_ERR else None
            return self._mock_sent(label, q)
        g = GoToEEPose.Goal()
        g.target = PoseStamped()
        g.target.header.frame_id = rammp_arm.FRAME
        g.target.header.stamp = self._node.get_clock().now().to_msg()
        p = g.target.pose.position
        p.x, p.y, p.z = map(float, pos[:3])
        o = g.target.pose.orientation
        o.x, o.y, o.z, o.w = map(float, quat)
        g.orientation_hold = orientation_hold
        if not self._go_to("go_to_ee_pose", g, label, speed_scale, control_mode, gains, feedback):
            return False
        tool = _tool_position(self._read("/ee_state"))
        e = float(np.linalg.norm(tool - np.asarray(pos[:3], float)))
        message = f"  {label}: tool {e * 100:.1f} cm from the goal"
        if e > rammp_arm.GO_TO_TRACK_M:
            return self._refuse(message)
        self._node.get_logger().info(message)
        return True

    def go_to_joint_config(
        self,
        q,
        label="go_to_joint_config",
        speed_scale=None,
        *,
        control_mode=POSITION,
        gains=None,
        feedback=None,
    ):
        """Joints to `q` (rad, J1..J7): cuRobo plans in joint space and the driver executes.
        Blocking; True if the driver succeeded and every joint ended within GO_TO_TRACK_DEG (free
        joints compared the short way). Keyword arguments as in go_to_ee_pose."""
        if self._mock:
            self._checkpoint()
            return self._mock_sent(label, q)
        g = GoToJointConfig.Goal()
        g.target_joints = [float(x) for x in q]
        if not self._go_to(
            "go_to_joint_config", g, label, speed_scale, control_mode, gains, feedback
        ):
            return False
        d = np.asarray(q, float) - self.raw_joints()
        d[CONTINUOUS] = (d[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
        e = float(np.degrees(np.max(np.abs(d))))
        message = f"  {label}: joints within {e:.1f} deg of the goal"
        if e > rammp_arm.GO_TO_TRACK_DEG:
            return self._refuse(message)
        self._node.get_logger().info(message)
        return True

    def go_to_preset(
        self, name: str, speed_scale=None, *, control_mode=POSITION, gains=None, feedback=None
    ):
        """The driver's named preset pose (its registry), planned by cuRobo and executed by the
        driver, like go_to_joint_config. Blocking; True if the driver succeeded. Keyword
        arguments as in go_to_ee_pose."""
        label = f"go_to_preset {name}"
        if self._mock:  # the presets are the driver's, unknown here: the joints stay
            self._checkpoint()
            return self._mock_sent(label)
        g = GoToPreset.Goal()
        g.preset_name = name
        return self._go_to("go_to_preset", g, label, speed_scale, control_mode, gains, feedback)

    def _go_to(self, name, g, label, speed_scale, control_mode, gains, feedback):
        g.gains = _impedance_gains(gains)  # a bad `gains` raises before anything is sent
        self._checkpoint()
        ctx = self._ctx
        self._check_command()
        why = self.not_ready()
        if why is not None:
            return self._refuse(f"  RammpArm: {label}: {why} -- nothing sent.")
        client = self._action(name)
        if not self._wait_for_server(client, ctx, PLANNER_WAIT_S):
            return self._refuse(
                f"  RammpArm: {label}: no action server (the go_to_* actions need the cuRobo "
                "planner: sheppy profile real-arm-camera-planner) -- nothing sent."
            )
        g.speed_scale = float(rammp_arm.GO_TO_SPEED_SCALE if speed_scale is None else speed_scale)
        g.control_mode = control_mode
        self._node.get_logger().info(
            f"  {label}: cuRobo plans the path (not previewed), run at {g.speed_scale:g} x its "
            "speed ..."
        )
        timeout_s = rammp_arm.GO_TO_TIMEOUT_S
        res = self._run_goal(
            name,
            client,
            g,
            timeout_s,
            ctx,
            f"  RammpArm: {label}: goal REJECTED -- nothing moved",
            f"  RammpArm: {label}: no result in {timeout_s:.0f} s -- goal cancelled",
            feedback=feedback,
        )
        if res is None:
            return False
        code = res.error_code
        if code != 0:
            return self._refuse(
                f"  RammpArm: {label} ended {ERRORS.get(code, code)} {res.error_string}"
            )
        time.sleep(GO_TO_SETTLE_S)  # let /ee_state and /joint_states catch up with the move's end
        return True

    def _checkpoint(self):
        """Before every motion: NotInAction outside an action, wait while the action is paused,
        Cancelled once it is cancelled."""
        self._check_in_action()
        self._ctx.checkpoint()

    def _send(self, t, q, qd, impedance=None, monitor=None):
        if self._mock:
            self.last_path, self.monitor_stopped = [], False
            return self._mock_sent("trajectory", q[-1])
        ctx = self._ctx  # end_action() clears self._ctx; an ended action fails in not_ready()
        self._check_command()
        why = self.not_ready()
        if why is not None:
            return self._refuse(f"  RammpArm: {why} -- nothing sent.")
        g = ExecuteJointTrajectory.Goal()
        g.trajectory.joint_names = JOINTS
        for ti, qi, qdi in zip(t, q, qd, strict=False):
            pt = JointTrajectoryPoint()
            pt.positions, pt.velocities = qi.tolist(), qdi.tolist()
            pt.time_from_start = _duration(ti)
            g.trajectory.points.append(pt)
        path_tol = impedance["path_tol"] if impedance else rammp_arm.PATH_TOL_RAD
        g.path_tolerance = _tolerances(JOINTS, path_tol)
        g.control_mode = POSITION
        if impedance:
            g.control_mode = IMPEDANCE
            g.gains = _impedance_gains({k: impedance[k] for k in GAIN_KEYS})
        g.preemption = QUEUE
        client = self._action("execute_joint_trajectory")
        if not self._wait_for_server(client, ctx, STATE_WAIT_S):
            self._fail(StateTimeout, "no /execute_joint_trajectory action server")
        res = self._run_goal(
            "execute_joint_trajectory",
            client,
            g,
            t[-1] + RESULT_MARGIN_S,
            ctx,
            "  RammpArm: trajectory goal REJECTED",
            "  RammpArm: no result in time -- goal cancelled",
            monitor=monitor,
            trajectory=True,
        )
        if res is None:
            return False
        code = res.error_code
        if self.monitor_stopped and code == PREEMPTED:
            return True
        if code != 0:
            self._refuse(
                f"  RammpArm: trajectory ended {ERRORS.get(code, code)} {res.error_string}"
            )
            fe = list(res.final_error.positions)
            if fe:
                self._node.get_logger().error(
                    "  RammpArm: final joint error (ref - actual) deg J1-J7 "
                    f"{np.round(np.degrees(fe), 1)}"
                )
        return code == 0

    def _run_goal(
        self,
        name,
        client,
        g,
        timeout_s,
        ctx,
        rejected,
        timed_out,
        *,
        monitor=None,
        feedback=None,
        trajectory=False,
    ):
        """Send goal `g` on action `name` with SENDER_ID and the token, and wait up to `timeout_s`
        for its result: the action's Result. A cancel of the action cancels the goal and raises
        Cancelled; a goal HALTED under the token raises ControlTaken (unless the arm e-stopped).
        None, with last_error set, when the goal was refused (`rejected`), had no result in time
        (`timed_out`; the goal is cancelled) or its action ended. `monitor`: polled while the goal
        runs; True cancels the goal and sets monitor_stopped. `feedback`: called with each
        GoalFeedback, on the executor (it must not block). `trajectory`: record last_path."""
        g.sender_id = rammp_arm.SENDER_ID
        token = self._driver.token
        if token is not None:
            g.token = list(token)

        def on_feedback(m):
            # On the executor: an exception here would escape its spin(), so it is only logged.
            try:
                feedback(_goal_feedback(m.feedback))
            except Exception as e:  # noqa: BLE001 -- the caller's callback, any error
                self._node.get_logger().error(f"{name}: the feedback callback raised {e!r}")

        try:
            self._driver.begin_goal(name)
        except ModeConflict as e:
            self._fail(ModeConflict, str(e))
        try:
            # The client is used under _cond, after _in_action is checked: end_action() clears
            # _in_action under _cond before it settles self._goal and destroys the client.
            with self._cond:
                send = None
                if self._in_action:
                    send = self._goal = client.send_goal_async(
                        g, feedback_callback=None if feedback is None else on_feedback
                    )
            if send is None:
                self._refuse(ACTION_ENDED_MESSAGE)
                return None
            try:
                return self._goal_result(
                    send, timeout_s, ctx, token, rejected, timed_out, monitor, trajectory
                )
            finally:
                with self._cond:  # once the action has ended, end_action() settles the goal
                    if self._in_action and self._goal is send:
                        self._goal = None
        finally:
            self._driver.end_goal()

    def _goal_result(self, send, timeout_s, ctx, token, rejected, timed_out, monitor, trajectory):
        gh, rf = self._wait(send, GOAL_ACCEPT_S), None
        with self._cond:
            ended = not self._in_action
            if not ended and gh is not None and gh.accepted:
                rf = gh.get_result_async()
        if ended:
            self._refuse(ACTION_ENDED_MESSAGE)
            return None
        if gh is None or not gh.accepted:
            self._refuse(rejected)
            return None
        if trajectory:
            # record the measured tool path while the goal runs (`last_path`: [(t, x, y, z)], from
            # the /ee_state callback) -- in impedance mode this is where the door actually let the
            # hand go
            self.last_path = []
            self._path_t0 = time.monotonic()
            self.monitor_stopped = False
        res = None
        try:
            t_end = time.monotonic() + timeout_s
            while not rf.done() and time.monotonic() < t_end and self._in_action:
                if ctx.cancelled():
                    ctx.checkpoint()  # raises the engine's Cancelled: cancelled below
                if monitor is not None and not self.monitor_stopped and monitor():
                    self.monitor_stopped = True
                    self._cancel(gh)  # the driver stops the arm, then settles PREEMPTED
                time.sleep(POLL_S)
            if rf.done():
                res = rf.result()
        except BaseException as e:
            # Anything raised while the goal runs (a cancel, Ctrl-C, the monitor, a state read)
            # must not leave it running: cancel it and let the driver settle it, then re-raise.
            self._cancel(gh)
            self._wait(rf, CANCEL_SETTLE_S)
            if isinstance(e, KeyboardInterrupt):
                self._node.get_logger().warning("  RammpArm: Ctrl-C -- goal cancelled")
            raise
        finally:
            self._path_t0 = None
        if not self._in_action:
            self._refuse(ACTION_ENDED_MESSAGE)
            return None
        if res is None:
            self._cancel(gh)
            self._refuse(timed_out)
            return None
        # HALTED under our token: control taken, unless the driver e-stopped (an e-stop drops
        # grants without a new generation), which is the caller's plain failure. The planner's
        # goals carry no token (and no error_code).
        if (
            token is not None
            and res.result.error_code == HALTED
            and not self._estopped_within(HALT_STATUS_WAIT_S)
        ):
            self._driver.on_goal_halted()
            self._check_command()  # raises ControlTaken: another sender took the arm from us
        return res.result

    def _cancel(self, gh):
        """Cancel goal `gh`, unless its action has ended: end_action() then cancels it (_settle)
        before it destroys the client, which must not be used after that."""
        with self._cond:  # end_action() clears _in_action under _cond before it destroys
            if self._in_action:
                gh.cancel_goal_async()

    def _cancel_goal(self, gh, rf):
        """Cancel a goal and wait up to CANCEL_SETTLE_S for the driver to settle it (_settle,
        before the client is destroyed)."""
        gh.cancel_goal_async()
        deadline = time.monotonic() + CANCEL_SETTLE_S
        while not rf.done() and time.monotonic() < deadline:
            time.sleep(POLL_S)

    @staticmethod
    def _wait(fut, timeout_s):
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout_s:
                return None
            time.sleep(POLL_S)
        return fut.result()

    def gripper(self, position, timeout_s=4.0, force=None, *, speed=None):
        """Stream /setpoint/gripper at 50 Hz until the fingers stop moving (or `timeout_s`). The
        2F-85 latches the last command, so it holds after the stream stops. `speed`: a fraction
        of the closing speed (default GRIPPER_SPEED). A cancel of the action stops the stream and
        raises Cancelled."""
        self._checkpoint()
        if self._mock:
            self._mock_gripper = float(position)
            return self._mock_sent("gripper")
        ctx = self._ctx
        self._check_command()
        m = GripperSetpoint()
        m.position = float(position)
        m.speed = float(rammp_arm.GRIPPER_SPEED if speed is None else speed)
        # a current ceiling: lower = looser hold
        m.force = float(rammp_arm.GRIPPER_FORCE if force is None else force)
        token = self._driver.token
        if token is not None:
            m.token = list(token)
        t0, last, still_since = time.monotonic(), self._read("/gripper_state").position, None
        with self._open_lock:  # after _read() has opened the arm's entities
            pub = self._pubs.get("/setpoint/gripper")
        while time.monotonic() - t0 < timeout_s:
            if ctx.cancelled():
                ctx.checkpoint()  # raises the engine's Cancelled
            # Under _cond: end_action() clears _in_action under it before it destroys `pub`.
            with self._cond:
                if not self._in_action:
                    self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
                pub.publish(m)
            time.sleep(GRIPPER_PERIOD_S)
            g = self._read("/gripper_state").position
            now = time.monotonic()
            if abs(g - last) > GRIPPER_MOVED:
                last, still_since = g, None
            elif now - t0 > GRIPPER_JUDGE_AFTER_S:
                still_since = still_since or now
                if now - still_since > GRIPPER_STILL_S:
                    break
        return True

    # -- the driver's goals ----------------------------------------------------------------------

    def execute_joint_trajectory(
        self,
        names,
        points,
        *,
        path_tolerance=None,
        goal_tolerance=None,
        goal_time_tolerance=0.0,
        control_mode=POSITION,
        preemption=QUEUE,
        gains=None,
        speed_scale=1.0,
        feedback=None,
    ) -> TrajectoryResult:
        """The driver's /execute_joint_trajectory, every field: joints `names` through `points`,
        [(t from start s, positions, velocities), ...] sent as given (no timing or interpolation
        added). Tolerances: position (rad), one for all joints or one per joint; None leaves the
        driver's default. `gains`: see _impedance_gains(). `feedback`: called with each
        GoalFeedback on the executor (it must not block). Blocking; a cancel of the action
        cancels the goal and raises Cancelled, its end raises NotInAction. Raises ArmError when
        the arm must not be commanded (EStopped when e-stopped), the goal is refused or no result
        comes in time."""
        if speed_scale <= 0:
            self._fail(
                ArmError, f"execute_joint_trajectory: speed_scale must be > 0, not {speed_scale}"
            )
        gains = _impedance_gains(gains)
        self._checkpoint()
        if self._mock:  # the last point's joints become the state, when they are the arm's
            points, idx, q = list(points), _joint_index(names), None
            if points:
                last = np.asarray(points[-1][1], float)
                if last.shape != (len(names),):
                    self._fail(
                        ArmError,
                        f"execute_joint_trajectory: a point has {last.size} positions for "
                        f"{len(names)} joints",
                    )
                q = None if idx is None else last[idx]
            self._mock_sent("execute_joint_trajectory", q)
            return TrajectoryResult(SUCCESSFUL, "", np.zeros(len(names)))
        ctx = self._ctx
        self._check_command()
        why = self.not_ready()
        if why is not None:
            self._fail(_refusal(why), f"execute_joint_trajectory: {why} -- nothing sent.")
        g = ExecuteJointTrajectory.Goal()
        g.trajectory.joint_names = list(names)
        duration = 0.0
        for t, q, qd in points:
            duration = t / speed_scale
            g.trajectory.points.append(
                JointTrajectoryPoint(
                    positions=[float(x) for x in q],
                    velocities=[float(x) for x in qd],
                    time_from_start=_duration(t),
                )
            )
        g.path_tolerance = _tolerances(names, path_tolerance)
        g.goal_tolerance = _tolerances(names, goal_tolerance)
        g.goal_time_tolerance = _duration(goal_time_tolerance)
        g.control_mode, g.preemption, g.gains = control_mode, preemption, gains
        g.speed_scale = float(speed_scale)
        client = self._action("execute_joint_trajectory")
        if not self._wait_for_server(client, ctx, STATE_WAIT_S):
            self._fail(StateTimeout, "no /execute_joint_trajectory action server")
        timeout_s = duration + goal_time_tolerance + RESULT_MARGIN_S
        res = self._run_goal(
            "execute_joint_trajectory",
            client,
            g,
            timeout_s,
            ctx,
            "execute_joint_trajectory: goal REJECTED -- nothing moved",
            f"execute_joint_trajectory: no result in {timeout_s:.0f} s -- goal cancelled",
            feedback=feedback,
        )
        if res is None:
            raise (ArmError if self._in_action else NotInAction)(self.last_error)
        return TrajectoryResult(
            res.error_code, res.error_string, np.array(res.final_error.positions)
        )

    # -- the planner (cuRobo: plans only, nothing moves) -----------------------------------------

    def plan_to_pose(self, pos, quat, start_joints, hold=HOLD_NONE) -> Plan:
        """cuRobo's plan for the tool frame to (pos, quat xyzw) in arm_mount, from `start_joints`
        (rad, arm_joint_1..7, as the driver reports them: raw_joints()). `hold` (HOLD_*): how much
        of the tool's orientation the path keeps. A failed plan is Plan(success=False, message)."""
        g = PlanToPose.Goal()
        p, o = g.target.position, g.target.orientation
        p.x, p.y, p.z = map(float, pos[:3])
        o.x, o.y, o.z, o.w = map(float, quat)
        g.start_joints = [float(x) for x in start_joints]
        g.hold = hold
        return self._plan_goal("plan_to_pose", g)

    def plan_to_joints(self, q_goal, start_joints) -> Plan:
        """cuRobo's plan to joints `q_goal` from `start_joints` (rad, arm_joint_1..7), as
        plan_to_pose."""
        g = PlanToJoints.Goal()
        g.target_joints = [float(x) for x in q_goal]
        g.start_joints = [float(x) for x in start_joints]
        return self._plan_goal("plan_to_joints", g)

    def set_world(self, world: str) -> tuple[bool, str]:
        """Swap cuRobo's collision world for the packaged world `world`: (success, message)."""
        if self._mock:
            return self._mock_sent("set_world"), MOCK_NOT_SENT
        res = self._call("set_world", SetWorld.Request(world=world))
        return res.success, res.message

    def _plan_goal(self, name, g):
        if self._mock:  # a successful plan with no samples
            self._mock_sent(name)
            return Plan(True, MOCK_NOT_SENT)
        self._ensure_open(wait_state=False)  # the planner needs no arm state
        ctx = self._ctx
        client = self._action(name)
        path = ACTIONS[name][1]
        if not self._wait_for_server(client, ctx, PLANNER_WAIT_S):
            self._refuse(f"{name}: no {path} action server (the cuRobo planner)")
            return Plan(False, self.last_error)
        with self._cond:  # as in _run_goal: end_action() settles self._goal before it destroys
            send = None
            if self._in_action:
                send = self._goal = client.send_goal_async(g)
        if send is None:
            self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
        try:
            res = self._goal_result(
                send,
                PLAN_TIMEOUT_S,
                ctx,
                None,
                f"{name}: goal REJECTED",
                f"{name}: no plan in {PLAN_TIMEOUT_S:g} s -- goal cancelled",
                None,
                False,
            )
        finally:
            with self._cond:
                if self._in_action and self._goal is send:
                    self._goal = None
        if res is None:
            if not self._in_action:
                self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
            return Plan(False, self.last_error)
        if not res.success:
            self._refuse(f"{name}: {res.message}")
            return Plan(False, res.message, planning_time=float(res.planning_time))
        plan = _plan(res)
        if plan is None:
            names = list(res.trajectory.joint_names)
            self._refuse(
                f"{name}: the plan is not positions and velocities of arm_joint_1..7 or "
                f"joint_1..7 at every point (its joints: {names})"
            )
            return Plan(False, self.last_error)
        return plan

    # -- the driver's services -----------------------------------------------------------------

    def list_controllers(self) -> list[Controller]:
        """The driver's streaming controllers, and whether each can be opened now (none in mock
        mode)."""
        if self._mock:
            self._mock_sent("list_controllers")
            return []
        res = self._call("list_controllers", ListControllers.Request())
        return [Controller(c.name, list(c.channels), bool(c.available)) for c in res.controllers]

    def set_gains(self, gains) -> tuple[bool, str]:
        """The session's default impedance gains (see _impedance_gains()): (accepted, message)."""
        if self._mock:
            return self._mock_sent("set_gains"), MOCK_NOT_SENT
        req = SetGains.Request(spec=_impedance_gains(gains))
        token = self._driver.token
        if token is not None:
            req.token = list(token)
        res = self._call("set_gains", req)
        return res.accepted, res.message

    def acquire_control(self, owner_id, token=None) -> tuple[bool, bytes, int, str]:
        """Standalone only (control comes from the orchestrator under the engine). Seize control
        as `owner_id`, with the pre-generated key `token` or a fresh one: (accepted, token,
        generation, message)."""
        self._check_standalone()
        req = AcquireControl.Request(owner_id=owner_id)
        if token is not None:
            req.token = list(token)
        res = self._call("acquire_control", req)
        return res.accepted, bytes(res.token), int(res.generation), res.message

    def release_control(self, token=None) -> tuple[bool, str]:
        """Standalone only. Release `token` (default: this action's): (released, message)."""
        self._check_standalone()
        token = self._driver.token if token is None else token
        req = ReleaseControl.Request()
        if token is not None:
            req.token = list(token)
        res = self._call("release_control", req)
        return res.released, res.message

    def revoke_control(self, reason) -> tuple[bool, str]:
        """Standalone only. Take control from whoever holds it: (revoked, message)."""
        self._check_standalone()
        res = self._call("revoke_control", RevokeControl.Request(reason=reason))
        return res.revoked, res.message

    def _check_standalone(self):
        if not self._standalone:
            self._fail(ArmError, CONTROL_FROM_ORCHESTRATOR_MESSAGE)

    def _call(self, name, request):
        """Call service `name` and return its response: ArmError when the server is not up within
        SERVICE_WAIT_S or does not answer within SERVICE_CALL_S, NotInAction once the action
        ends, Cancelled once it is cancelled (both waits poll every POLL_S). Waits for no arm
        state."""
        self._ensure_open(wait_state=False)
        ctx = self._ctx
        with self._open_lock:  # not while _destroy() empties the dict
            client = self._clients.get(name)
        if client is None:
            self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
        path = SERVICES[name][1]
        deadline = time.monotonic() + SERVICE_WAIT_S
        future = None
        while future is None:
            # Under _cond: end_action() clears _in_action under it before it destroys the client.
            with self._cond:
                if not self._in_action:
                    self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
                if client.service_is_ready():
                    future = client.call_async(request)
            if future is None:
                if ctx.cancelled():
                    ctx.checkpoint()  # raises the engine's Cancelled
                if time.monotonic() > deadline:
                    self._fail(
                        ArmError, f"no {path} service in {SERVICE_WAIT_S:g} s -- is the driver up?"
                    )
                time.sleep(POLL_S)
        deadline = time.monotonic() + SERVICE_CALL_S
        while not future.done():
            if not self._in_action:
                self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
            if ctx.cancelled():
                ctx.checkpoint()  # raises the engine's Cancelled
            if time.monotonic() > deadline:
                self._fail(ArmError, f"{path}: no answer in {SERVICE_CALL_S:g} s")
            time.sleep(POLL_S)
        return future.result()

    # -- streaming -------------------------------------------------------------------------------

    @contextmanager
    def stream(self, controller: str, gains=None, timeout_s=None):
        """A streaming session on `controller` (see list_controllers()): yields a Stream to send()
        setpoints on, at least every `timeout_s` (default STREAM_TIMEOUT_S). `gains` (see
        _impedance_gains()) apply to impedance controllers, at open. One session is one motion: a
        pause is honoured at open only. Leaving the block, however (an exception, a cancel, the
        driver closing it), sends the hold setpoint (zeros for twist and velocity controllers, the
        measured pose or joints for pose and position ones; nothing for torque), closes the
        session and destroys its publisher. ModeConflict while a goal or another stream runs;
        ArmError when the arm must not be commanded (EStopped when e-stopped) or the driver has no
        such controller available. In mock mode the session opens and send() does nothing."""
        gains = _impedance_gains(gains)
        timeout_s = STREAM_TIMEOUT_S if timeout_s is None else timeout_s
        self._checkpoint()
        ctx, driver = self._ctx, self._driver
        if self._mock:
            self._mock_sent(f"stream {controller!r}")
            yield Stream(self, controller, None, ctx, driver)
            return
        self._check_command()
        why = self.not_ready()
        if why is not None:
            self._fail(_refusal(why), f"stream {controller!r}: {why} -- nothing sent.")
        try:
            driver.begin_stream(controller)
        except ModeConflict as e:
            self._fail(ModeConflict, str(e))
        stream = None
        try:
            stream = self._new_stream(controller, ctx, driver)
            req = OpenStream.Request(controller=controller, timeout_s=float(timeout_s), gains=gains)
            if stream._token is not None:
                req.token = list(stream._token)
            res = self._call("open_stream", req)
            if not res.accepted:
                self._fail(
                    ArmError,
                    f"stream {controller!r}: refused ({res.error_code}) {res.message}",
                )
            stream._opened = True
            if list(res.channels) != [stream.channel]:
                self._fail(
                    ArmError,
                    f"stream {controller!r}: opened on {list(res.channels)}, not "
                    f"[{stream.channel!r}] as /list_controllers said",
                )
            yield stream
        finally:
            try:
                if stream is not None:
                    self._close_stream(stream)
            finally:
                driver.end_stream()

    def _new_stream(self, controller, ctx, driver):
        """A Stream on `controller`'s channel, its publisher made and matched (the driver's
        OpenStream: create the publisher and let discovery settle BEFORE opening, or the first
        setpoints go nowhere and the session expires). ArmError for a controller the driver
        does not list as available."""
        listed = {c.name: c for c in self.list_controllers()}
        row = listed.get(controller)
        if row is None:
            self._fail(
                ArmError, f"stream: no controller {controller!r} (the driver has {sorted(listed)})"
            )
        if not row.available:
            self._fail(ArmError, f"stream: controller {controller!r} is not available")
        if len(row.channels) != 1 or row.channels[0] not in SETPOINTS:
            self._fail(
                ArmError,
                f"stream: controller {controller!r} reads {row.channels}, not one of "
                f"{sorted(SETPOINTS)}",
            )
        stream = Stream(self, controller, row.channels[0], ctx, driver)
        with self._open_lock:  # registered: end_action() closes it from here on
            if not self._in_action:
                self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
            msg_type = SETPOINTS[stream.channel][0]
            topic = f"/setpoint/{stream.channel}"
            stream._pub = self._node.create_publisher(msg_type, topic, SETPOINT_QOS)
            self._stream = stream
        self._wait_matched(stream._pub)
        return stream

    def _send_setpoint(self, stream, setpoint):
        if self._mock:
            return
        if self._stream is not stream:  # closed already: the block outlived it, or its action
            self._fail(NotInAction, STREAM_ENDED_MESSAGE)
        if stream._ctx.cancelled():
            stream._ctx.checkpoint()  # raises the engine's Cancelled; the block's exit closes
        try:
            stream._driver.check_command()
        except ArmError as e:
            self._fail(type(e), str(e))
        m = stream._message(setpoint)
        if m is None:
            self._fail(
                ArmError,
                f"stream {stream.controller!r}: a {stream.channel} setpoint is "
                f"{SETPOINTS[stream.channel][1]}, not {setpoint!r}",
            )
        with self._open_lock:  # not while the session closes and destroys its publisher
            if self._stream is not stream:
                self._fail(NotInAction, STREAM_ENDED_MESSAGE)
            stream._pub.publish(m)

    def _close_stream(self, stream):
        """Send the hold setpoint, close the session at the driver and destroy its publisher,
        once: on leaving the stream's block, or in end_action() (_settle), whichever comes first.
        Stored handles only; waits at most CLOSE_STREAM_WAIT_S."""
        with self._open_lock:  # not while _destroy() destroys the close_stream client
            if self._stream is not stream:
                return
            self._stream = None
            try:
                if stream._opened:
                    hold = self._hold_setpoint(stream.channel)
                    if hold is not None:
                        stream._pub.publish(stream._message(hold))
                    req = CloseStream.Request()
                    if stream._token is not None:
                        req.token = list(stream._token)
                    res = self._wait(
                        self._clients["close_stream"].call_async(req), CLOSE_STREAM_WAIT_S
                    )
                    log = self._node.get_logger()
                    if res is None:
                        log.warning(f"stream {stream.controller!r}: /close_stream: no answer")
                    elif not res.closed:  # the driver had closed it already
                        log.debug(f"stream {stream.controller!r}: /close_stream: {res.message}")
            finally:
                self._node.destroy_publisher(stream._pub)

    def _hold_setpoint(self, channel):
        """What holds the arm as a session closes: zeros for twist and velocity, the latest
        measured pose or joints for pose and position; None for torque (nothing is sent)."""
        if channel == "twist":
            return np.zeros(6)
        if channel == "joint_velocity":
            return np.zeros(len(JOINTS))
        with self._cond:
            if channel == "pose" and "/ee_state" in self._latest:
                ee = self._latest["/ee_state"][0]
                return ee.position, ee.orientation
            if channel == "joint_position" and "/joint_states" in self._latest:
                return self._latest["/joint_states"][0].position
        return None

    def _wait_matched(self, pub):
        """Wait up to PUBLISHER_MATCH_S for `pub`, one of this action's publishers, to have a
        subscriber. NotInAction once the action ends (end_action() then destroys `pub`)."""
        deadline = time.monotonic() + PUBLISHER_MATCH_S
        while True:
            # Under _cond: end_action() clears _in_action under it before it destroys `pub`.
            with self._cond:
                if not self._in_action:
                    self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
                if pub.get_subscription_count() > 0:
                    return
            if time.monotonic() > deadline:
                return
            time.sleep(POLL_S)

    # -- e-stop ----------------------------------------------------------------------------------

    def estop(self, engaged: bool, reason: str):
        """Engage (True) or clear (False) the e-stop on /estop, as this capability's node. Sent at
        once: it waits for no arm state, only (up to PUBLISHER_MATCH_S) for a subscriber."""
        if self._mock:
            self._mock_sent("estop")
            return
        self._ensure_open(wait_state=False)
        m = EStop(engaged=bool(engaged), source=self._node.get_name(), reason=reason)
        m.header.stamp = self._node.get_clock().now().to_msg()
        with self._open_lock:  # after _ensure_open() has made it
            pub = self._pubs.get("/estop")
        if pub is None:
            self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
        self._wait_matched(pub)
        # Under _cond: end_action() clears _in_action under it before it destroys `pub`.
        with self._cond:
            if not self._in_action:
                self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
            pub.publish(m)
