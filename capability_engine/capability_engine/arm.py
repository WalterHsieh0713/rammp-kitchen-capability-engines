"""`self.arm`: the arm driver (kinova-gen3-ros2) and its cuRobo planner, for a capability.

`ArmInterface` lives on the engine's node and callback group and owns no node, executor or thread.
Its ROS entities exist only inside an action: the first arm call after `begin_action()` creates
them all (`_open`), and `end_action()` settles any goal or stream, destroys them and forgets the
state they received. An action that never touches the arm costs nothing, and no state read in one
action is returned in the next.
"""

import copy
import threading
import time
from dataclasses import dataclass, field, replace

import numpy as np
import sheppy_arm
from diagnostic_msgs.msg import DiagnosticArray
from rammp_arm_interfaces.action import (
    ExecuteJointTrajectory,
    GoToEEPose,
    GoToJointConfig,
    GoToPreset,
)
from rammp_arm_interfaces.msg import EeState as EeStateMsg
from rammp_arm_interfaces.msg import GripperSetpoint
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
from sheppy_arm import JOINTS, TOOL_Z, _wrap
from std_msgs.msg import String

from capability_engine.driver_state import ArmError, DriverState, NotInAction, StateTimeout

STATE_WAIT_S = 10.0  # how long the first arm call in an action waits for the driver's state
STATE_STALE_S = 0.5  # /joint_states, /ee_state or /gripper_state older than this is an error
DIAGNOSTICS_STALE_S = 3.0  # /diagnostics (1 Hz) older than this is an error

# QoS as in SheppyArm (sheppy_arm.py@7bc328ae:190-201) and the driver's docs/interface.md.
LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
SETPOINT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
)
ESTOP_QOS = 10  # /estop is volatile; a reliable publisher reaches any subscriber

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
# The first arm call in an action waits for these (SheppyArm's constructor did).
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


class ArmInterface:
    """`self.arm`: SheppyArm's API and the driver's, side by side, on the engine's node."""

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
        forget the state they received, so the next action starts from the driver afresh."""
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
        """Cancel any running goal and close any open stream, waiting for each to settle."""

    def _check_in_action(self):
        if not self._in_action:
            self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)

    def _ensure_open(self):
        self._check_in_action()
        with self._open_lock:
            # Again under the lock: end_action() may have destroyed everything since the check.
            if not self._in_action:
                self._fail(NotInAction, NOT_IN_ACTION_MESSAGE)
            if not self._opened:
                self._open()

    def _open(self):
        """Create the subscriptions, clients and publishers, then wait for the arm's state."""
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
        if not all(self._wait_for(t) for t in MOTION_STATE):
            self._fail(StateTimeout, NO_STATE_MESSAGE)

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

    # -- state --------------------------------------------------------------------------------

    def _store(self, topic, value):
        if value is None:
            return
        with self._cond:
            self._latest[topic] = (value, time.monotonic())
            self._cond.notify_all()

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
        """SheppyArm's state dict: joints wrapped to +-pi, `ee_pos` the tool frame (flange +
        TOOL_Z along the flange z) as [x, y, z, qx, qy, qz, qw]. Like SheppyArm, the latest
        values however old: only the driver-API reads (joint_states(), ...) refuse stale state."""
        js, ee, grip = (self._read(topic) for topic in MOTION_STATE)
        tool = ee.position + R.from_quat(ee.orientation).apply([0.0, 0.0, TOOL_Z])
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
        self._ensure_open()
        self._wait_for("/control_status")
        return self._driver.not_ready(sheppy_arm.SENDER_ID)

    def robot_description(self) -> str:
        """The driver's URDF, from /arm/robot_description (latched), through a one-shot
        subscription destroyed after the first message. Opens none of the arm's other entities."""
        self._check_in_action()
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
