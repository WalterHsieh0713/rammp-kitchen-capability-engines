"""`self.arm`: the arm driver (kinova-gen3-ros2) and its cuRobo planner, for a capability.

`ArmInterface` lives on the engine's node and callback group and owns no node, executor or thread.
Its ROS entities exist only inside an action: each is made the first time a call in the action
needs it (`_get`), and `end_action()` settles any goal or stream, destroys them and forgets the
state they received. An action pays only for what it uses (one that only plans subscribes to none
of the 50 Hz state topics), and no state read in one action is returned in the next.

Threading: any thread may call it during an action, and end_action() may run in another thread
while a call is in progress; executor callbacks (state, goal answers, feedback) must never block.
Every use of the action's entities is inside `_alive()`, which holds _cond: end_action() marks the
action ended under _cond before it settles or destroys anything, so inside `_alive()` every entity
exists and every request sent is one end_action() will settle. After the end, a cancel is left to
end_action()'s settle (_cancel, _cancel_late_goal) and a state message is dropped (_store). The
uses outside `_alive()` are a Stream's send() and close(), under the stream's lock instead:
end_action() closes the stream (setting it closed) before it closes the scope, and send() does
not publish on a closed stream.

In mock mode (`mock=True`) it is a dry run that makes no ROS entity at all: motions and goals log
"mock: not sent" and succeed, and the state is the last commanded joints (from MOCK_JOINTS), with
the tool pose from the sim's FK when a sim exists or can be built from MOCK_URDF. That FK sets the
arm joints of the process's one shared sim copy, as solve_ik does.
"""

import copy
import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import rammp_arm
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped
from rammp_arm import CONTINUOUS, ERRORS, JOINTS, _wrap
from rammp_arm_interfaces.action import (
    ExecuteJointTrajectory,
    GoToEEPose,
    GoToEEPoses,
    GoToJointConfig,
    GoToJointConfigs,
    GoToPreset,
)
from rammp_arm_interfaces.msg import EEPoseTarget, GripperSetpoint, JointConfigTarget
from rammp_arm_interfaces.msg import EeState as EeStateMsg
from rammp_arm_interfaces.msg import GripperState as GripperStateMsg
from rammp_arm_interfaces.msg import StreamStatus as StreamStatusMsg
from rammp_arm_interfaces.srv import CloseStream, ListControllers, OpenStream, SetGains
from rammp_common_interfaces.msg import ControlStatus as ControlStatusMsg
from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
from rammp_curobo_interfaces.srv import SetWorld
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
from capability_engine.action_scope import ActionScope
from capability_engine.arm_messages import (  # re-exports: the arm's dataclasses and constants
    GAIN_KEYS,
    GAIN_PROFILES,  # noqa: F401
    HOLD_FIXED,  # noqa: F401
    HOLD_LEVEL,  # noqa: F401
    HOLD_NONE,
    SETPOINTS,
    Controller,
    ControlStatus,
    Diagnostics,
    DiagnosticStatus,  # noqa: F401
    EeState,
    GoalFeedback,  # noqa: F401
    GripperState,
    JointStates,
    JointTarget,  # noqa: F401
    Plan,
    PoseTarget,  # noqa: F401
    StreamStatus,
    TrajectoryResult,
    _control_status,
    _diagnostics,
    _duration,
    _ee_state,
    _goal_feedback,
    _gripper_state,
    _impedance_gains,
    _joint_index,
    _joint_states,
    _plan,
    _setpoint,
    _speed_scale,
    _stream_status,
    _tolerances,
    _tool_position,
)
from capability_engine.cornell_commands import (
    CartesianTrajectoryCommand,
    CloseGripperCommand,
    JointCommand,
    OpenGripperCommand,
)
from capability_engine.driver_state import (
    ArmError,
    DriverState,
    EStopped,
    NotInAction,
    Refused,
    StateTimeout,
)

STATE_WAIT_S = 10.0  # how long a call waits for the driver's state to arrive
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
PUBLISHER_MATCH_S = 1.0  # how long a new setpoint publisher waits for the driver to
# subscribe before its first message (DDS discovery), so that message is not lost
CLOSE_STREAM_WAIT_S = 1.0  # how long closing a stream waits for /close_stream's answer
MOCK_JOINTS = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]  # mock mode: the joints it starts at
MOCK_URDF = ""  # mock mode: the URDF file make_sim() loads ("": none, make_sim() refuses)
MOCK_NOT_SENT = "mock: not sent"
MOCK_NO_MODEL_MESSAGE = "mock mode has no driver model: set arm.mock_urdf to a URDF file"
STREAM_ENDED_MESSAGE = "the stream is closed: send() works only inside its with block"

# The driver's goal fields, for the keyword arguments of the goal methods (the driver's values).
POSITION = 0  # control_mode: stiff position control
IMPEDANCE = 1  # control_mode: compliant joint impedance
QUEUE = 0  # execute_joint_trajectory preemption: wait for the running goal to finish
LATEST_WINS = 1  # execute_joint_trajectory preemption: replace the running goal

# QoS as in RammpArm (rammp_arm.py@7bc328ae:190-201) and the driver's docs/interface.md.
LATCHED = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
SETPOINT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
)
ACTIONS = {
    "execute_joint_trajectory": (ExecuteJointTrajectory, "/execute_joint_trajectory"),
    "go_to_ee_pose": (GoToEEPose, "/go_to_ee_pose"),
    "go_to_joint_config": (GoToJointConfig, "/go_to_joint_config"),
    "go_to_ee_poses": (GoToEEPoses, "/go_to_ee_poses"),
    "go_to_joint_configs": (GoToJointConfigs, "/go_to_joint_configs"),
    "go_to_preset": (GoToPreset, "/go_to_preset"),
    "plan_to_pose": (PlanToPose, "/rammp_curobo/plan_to_pose"),
    "plan_to_joints": (PlanToJoints, "/rammp_curobo/plan_to_joints"),
}
SERVICES = {
    "open_stream": (OpenStream, "/open_stream"),
    "close_stream": (CloseStream, "/close_stream"),
    "list_controllers": (ListControllers, "/list_controllers"),
    "set_gains": (SetGains, "/set_gains"),
    "set_world": (SetWorld, "/rammp_curobo/set_world"),
}
# Every call that reads them or commands the arm waits for these first (RammpArm's constructor
# did).
MOTION_STATE = ("/joint_states", "/ee_state", "/gripper_state")
NOT_IN_ACTION_MESSAGE = "the arm is only available inside an action"
NO_STATE_MESSAGE = (
    "no /joint_states, /ee_state or /gripper_state -- is the sheppy arm node up, "
    "and ROS_DOMAIN_ID / RMW_IMPLEMENTATION exported?"
)


def _rammp(method):
    """RammpArm's failure path over a method of the core, which raises: a Refused (already
    logged and kept in last_error) returns False, and so does a call its action's end cut short
    (NotInAction from a call begun inside the action, which has since ended); every other error
    propagates."""

    @functools.wraps(method)
    def call(self, *args, **kwargs):
        began = self._in_action
        try:
            return method(self, *args, **kwargs)
        except Refused:
            return False
        except NotInAction:
            if not began or self._in_action:  # called outside an action, or not its end
                raise
            self.last_error = ACTION_ENDED_MESSAGE
            return False

    return call


def wait_until(ready, timeout_s, ctx=None):
    """Poll ready() every POLL_S: True once it is, False after `timeout_s`. Raises the engine's
    Cancelled once `ctx` (an action's) is cancelled, and whatever ready() raises (NotInAction
    once the action has ended). Every wait that observes a cancel goes through here; the few that
    do not (settling a cancelled goal or a stream, GO_TO_SETTLE_S, HALT_STATUS_WAIT_S) are bounded
    and counted in ArmInterface.worst_settle_s()."""
    deadline = time.monotonic() + timeout_s
    while True:
        if ready():
            return True
        if ctx is not None and ctx.cancelled():
            ctx.checkpoint()  # raises the engine's Cancelled
        if time.monotonic() > deadline:
            return False
        time.sleep(POLL_S)


def _result_within(fut, timeout_s):
    """`fut`'s result, or None if it is not done within `timeout_s` (no cancel: settling)."""
    return fut.result() if wait_until(fut.done, timeout_s) else None


class Stream:
    """A streaming session on the driver, from ArmInterface.stream(): send() a setpoint at
    least every timeout_s, or the driver closes the session. close() ends it, once: on leaving
    the stream's block, or in end_action(), whichever comes first. Its publisher and the
    close_stream client are the action's (None in mock mode, where nothing is sent): they outlive
    the session, and end_action() destroys them only after closing it."""

    def __init__(self, arm, controller, channel, pub, close_client):
        self.controller = controller
        self.channel = channel  # the /setpoint/<channel> it publishes on
        self._arm = arm
        self._ctx, self._driver, self._token = arm._ctx, arm._driver, arm._driver.token
        self._pub, self._close_client = pub, close_client
        self._lock = threading.Lock()  # send()'s publish and close(), one at a time
        self._closed = False
        self._opened = False  # the driver accepted open_stream
        self._open_sent = None  # open_stream's future once sent: settled by close()

    def open(self, timeout_s, gains):
        """Open the session at the driver: Refused when it says no."""
        req = OpenStream.Request(controller=self.controller, timeout_s=float(timeout_s))
        req.gains = gains
        if self._token is not None:
            req.token = list(self._token)
        res = self._arm._call("open_stream", req, on_sent=lambda f: setattr(self, "_open_sent", f))
        if not res.accepted:
            self._arm._fail(
                Refused, f"stream {self.controller!r}: refused ({res.error_code}) {res.message}"
            )
        self._opened = True
        if list(res.channels) != [self.channel]:
            self._arm._fail(
                ArmError,
                f"stream {self.controller!r}: opened on {list(res.channels)}, not "
                f"[{self.channel!r}] as /list_controllers said",
            )

    def send(self, setpoint):
        """Send one setpoint: 7 joints for joint controllers, (pos, quat xyzw) of /ee_state's frame
        (not get_state()'s tool frame) for pose controllers, 6 floats (linear, then angular, in
        arm_mount axes) for twist controllers. Never blocks. Raises Cancelled once the action is
        cancelled, StreamClosed, ControlTaken or EStopped when the driver will not take it, and
        NotInAction outside the with block; leaving the block then zeroes and closes the session."""
        fail = self._arm._fail
        if self._closed:  # the block outlived it, or its action
            fail(NotInAction, STREAM_ENDED_MESSAGE)
        if self._ctx.cancelled():
            self._ctx.checkpoint()  # raises the engine's Cancelled; the block's exit closes
        if self._pub is None:  # mock mode: a dry run
            return
        self._arm._recorded(self._driver.check_command)
        m = self._message(setpoint)
        if m is None:
            fail(
                ArmError,
                f"stream {self.controller!r}: a {self.channel} setpoint is "
                f"{SETPOINTS[self.channel][1]}, not {setpoint!r}",
            )
        with self._lock:
            if self._closed:
                fail(NotInAction, STREAM_ENDED_MESSAGE)
            self._pub.publish(m)

    def close(self):
        """Send the hold setpoint and close the session at the driver, once. Waits at most
        twice CLOSE_STREAM_WAIT_S."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._opened and self._open_sent is not None:
                # Sent, unanswered (cancelled, timed out, the action ended): close it if the
                # driver opened it after all. Never otherwise: CloseStream closes whatever
                # session is open, perhaps another sender's.
                res = _result_within(self._open_sent, CLOSE_STREAM_WAIT_S)
                self._opened = res is not None and res.accepted
            # Closed by the driver already: no hold, no CloseStream, which closes whatever
            # session is open now, perhaps another sender's.
            if not self._opened or self._driver.stream_closed:
                return
            hold = self._arm._hold_setpoint(self.channel)
            if hold is not None:
                self._pub.publish(self._message(hold))
            req = CloseStream.Request()
            if self._token is not None:
                req.token = list(self._token)
            res = _result_within(self._close_client.call_async(req), CLOSE_STREAM_WAIT_S)
        log = self._arm._node.get_logger()
        if res is None:
            log.warning(f"stream {self.controller!r}: /close_stream: no answer")
        elif not res.closed:  # the driver had closed it already
            log.debug(f"stream {self.controller!r}: /close_stream: {res.message}")

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
        # One end_action() at a time: one in another thread returns only once the first is done.
        # begin_action() holds it from ending the last action to installing the next, so an
        # end_action() in another thread ends a whole action, never a half-installed one.
        self._end_lock = threading.Lock()
        self._scope = None  # the action's ActionScope: the entities _get() made live in it
        # This action's entities, by action or service name, or topic (they never collide).
        self._entities = {}
        self._cond = threading.Condition()  # _in_action, _latest, _goals and _entities: _alive()
        self._latest = {}  # topic -> (dataclass, time.monotonic() when received)
        self.last_error = ""  # the most recent refusal or failure, also logged
        self.last_path = []  # the last trajectory's measured tool path: [(t, x, y, z)]
        self.monitor_stopped = False  # the last trajectory was stopped by its monitor
        # Send futures of this action's goals whose end the driver has not reported: each leaves
        # once refused or finished; a goal its call gave up on is cancelled if accepted later
        # (_cancel_late_goal), and end_action() cancels whatever is left.
        self._goals = []
        self._path_t0 = None  # time.monotonic() its goal started; set: /ee_state adds to last_path
        self._stream = None  # the open Stream: end_action() zeroes and closes it
        self._mock_q = np.array(MOCK_JOINTS, float)  # mock mode: the last commanded joints
        self._mock_gripper = 0.0  # mock mode: the last commanded gripper position (0 = open)

    # -- lifecycle ----------------------------------------------------------------------------

    def begin_action(self, ctx, token: bytes | None):
        """Start an action: arm calls are allowed until end_action(). Creates no ROS entity: the
        first call that needs them makes them in the arm's own scope for this action, which
        end_action() closes."""
        with self._end_lock:
            self._end_locked()  # the last action never ended: carry nothing
            scope = ActionScope(self._node, self._group)
            driver = DriverState(standalone=self._standalone)  # a fresh baseline: this action's
            driver.set_token(token)  # /control_status generation is its own
            with self._cond:
                self._ctx, self._scope, self._driver = ctx, scope, driver
                self._in_action = True

    def end_action(self):
        """End the action: settle any goal or stream, close its scope (destroying the entities
        _get() made) and forget the state they received, so the next action starts from the
        driver afresh. Worst case, it waits GOAL_ACCEPT_S + CANCEL_SETTLE_S for the goals the
        driver has not yet answered (all at once), to accept them and then to settle their
        cancels, and twice CLOSE_STREAM_WAIT_S for a stream to open late and then close."""
        with self._end_lock:
            self._end_locked()

    def _end_locked(self):
        """end_action()'s work, under _end_lock."""
        with self._cond:
            self._in_action = False
            self._cond.notify_all()  # a call waiting for state wakes with NotInAction
        try:
            self._settle()
        finally:
            with self._cond:  # nothing of the ended action is kept
                scope, self._scope, self._entities, self._ctx = self._scope, None, {}, None
            if scope is not None:
                scope.close()
            with self._cond:
                self._latest.clear()
                self._cond.notify_all()

    def close(self):
        """A no-op under the engine, which owns the node."""

    @staticmethod
    def worst_settle_s() -> float:
        """How long a cancelled action's arm calls may still take to end, at most: the running
        call's own settle, then end_action()'s. The call either cancels its goal and waits
        CANCEL_SETTLE_S, or is in a go_to goal's GO_TO_SETTLE_S, or after a HALTED goal waits
        HALT_STATUS_WAIT_S for an e-stop status, or closes its stream; end_action() waits
        GOAL_ACCEPT_S + CANCEL_SETTLE_S for the goals left and closes a stream still open. A stream
        closes once (twice CLOSE_STREAM_WAIT_S), in the call or in end_action(), so it counts
        once. From the arm.* parameters; 7.8 s by default."""
        call = max(CANCEL_SETTLE_S, GO_TO_SETTLE_S, HALT_STATUS_WAIT_S)
        return call + GOAL_ACCEPT_S + CANCEL_SETTLE_S + 2 * CLOSE_STREAM_WAIT_S

    def _settle(self):
        """Cancel any running goal and close any open stream, waiting for each to settle.
        Called after the action has ended: uses the stored handles only, never _get()."""
        with self._cond:
            goals, self._goals = self._goals, []
            stream, self._stream = self._stream, None
        # Sent, perhaps not yet accepted: wait for all their answers at once, cancel each one
        # accepted, then wait for all to settle, so however many are left the settle takes at
        # most GOAL_ACCEPT_S + CANCEL_SETTLE_S (inside the engine's stop_wait_s()).
        wait_until(lambda: all(send.done() for send in goals), GOAL_ACCEPT_S)
        results = []
        for send in goals:
            gh = send.result() if send.done() else None
            if gh is not None and gh.accepted:
                gh.cancel_goal_async()
                results.append(gh.get_result_async())
        wait_until(lambda: all(rf.done() for rf in results), CANCEL_SETTLE_S)
        if stream is not None:
            stream.close()  # waits for a close the stream's block has begun

    def _check_in_action(self, message=NOT_IN_ACTION_MESSAGE):
        """NotInAction(`message`) unless an action is running."""
        if not self._in_action:
            self._fail(NotInAction, message)

    @contextmanager
    def _alive(self, message=NOT_IN_ACTION_MESSAGE):
        """The one guard of the action's entities: hold it to make or use one. It holds _cond,
        under which end_action() marks the action ended before it settles or destroys anything,
        and raises as _check_in_action() does."""
        with self._cond:
            self._check_in_action(message)
            yield

    def _live(self, use):
        """use() inside _alive(), for a wait_until() condition."""
        with self._alive():
            return use()

    def _get(self, key, make):
        """This action's entity `key`, made by make(scope) the first time a call needs it: an
        action pays only for what it uses (rclpy's executor spends ~2 ms per message on the Orin,
        Demo-Software #263). NotInAction, never a new entity, once end_action() has run."""
        with self._alive():
            if "/control_status" not in self._entities:
                # With the action's first entity: the generation it first reports is the takeover
                # baseline, so another sender taking the arm before the first command still
                # raises ControlTaken. Latched, it only changes on arbitration events.
                self._entities["/control_status"] = self._subscribe("/control_status")(self._scope)
            if key not in self._entities:
                self._entities[key] = make(self._scope)
            return self._entities[key]

    def _subscription(self, topic):
        """Subscribe to the driver's state `topic`, its messages kept in _latest."""
        return self._get(topic, self._subscribe(topic))

    def _subscribe(self, topic):
        """make(scope) for the subscription to `topic`, for _get."""
        msg_type, qos, convert = {
            "/joint_states": (JointState, qos_profile_sensor_data, _joint_states),
            "/ee_state": (EeStateMsg, qos_profile_sensor_data, _ee_state),
            "/gripper_state": (GripperStateMsg, qos_profile_sensor_data, _gripper_state),
            "/control_status": (ControlStatusMsg, LATCHED, self._on_control_status),
            "/stream_status": (StreamStatusMsg, LATCHED, self._on_stream_status),
            "/diagnostics": (DiagnosticArray, qos_profile_sensor_data, _diagnostics),
        }[topic]
        return lambda scope: scope.create_subscription(
            msg_type, topic, lambda m: self._store(topic, convert(m)), qos
        )

    def _action(self, name):
        """Action client `name` (ACTIONS)."""
        action_type, path = ACTIONS[name]
        return self._get(name, lambda scope: scope.create_action_client(action_type, path))

    def _client(self, name):
        """Service client `name` (SERVICES)."""
        srv_type, path = SERVICES[name]
        return self._get(name, lambda scope: scope.create_client(srv_type, path))

    def _motion_state(self):
        """Subscribe to MOTION_STATE and wait for each, as RammpArm's constructor did: before a
        read of them or a command. The wait is outside _alive(), so a call that needs no state
        is never held up by one that waits. StateTimeout when one does not come."""
        for topic in MOTION_STATE:
            self._subscription(topic)
        if not all(self._wait_for(t) for t in MOTION_STATE):
            self._fail(StateTimeout, NO_STATE_MESSAGE)

    def _record(self, message):
        """Log `message` (RammpArm's print) and keep it in last_error."""
        self.last_error = message
        self._node.get_logger().error(message)

    def _fail(self, error, message):
        """Raise `error`(`message`), recorded: every ArmError the arm raises goes through here."""
        self._record(message)
        raise error(message)

    def _recorded(self, check):
        """Run check(), a DriverState check: an ArmError from it is recorded, as by _fail()."""
        try:
            check()
        except ArmError as e:
            self._fail(type(e), str(e))

    def _check_command(self):
        """Raise ControlTaken or StreamClosed. An e-stop is not raised here: _check_ready()
        reports it with not_ready()'s reason."""
        try:
            self._driver.check_command()
        except EStopped:
            pass
        except ArmError as e:
            self._fail(type(e), str(e))

    def _check_ready(self, prefix):
        """Before a command: ControlTaken or StreamClosed, else EStopped or Refused when
        not_ready() gives a reason (f"{prefix}{why} -- nothing sent.")."""
        self._check_command()
        why, estopped = self._refusal()
        if why is not None:
            self._fail(EStopped if estopped else Refused, f"{prefix}{why} -- nothing sent.")

    def _wait_for_server(self, client, ctx, timeout_s):
        """Whether the action server came up within `timeout_s`; Cancelled once the action is
        cancelled, NotInAction once it ends."""
        return wait_until(lambda: self._live(client.server_is_ready), timeout_s, ctx)

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
            # Belt and braces: messages still arrive while end_action() settles, and it clears
            # _latest once the scope is closed, so none would reach the next action anyway.
            if not self._in_action:
                return
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
        or None if none came. NotInAction if the action ends meanwhile, Cancelled once it is
        cancelled."""
        wait_until(lambda: self._live(lambda: topic in self._latest), STATE_WAIT_S, self._ctx)
        with self._alive():
            return self._latest.get(topic)

    def _read(self, topic, stale_s=None):
        """The latest value on `topic` with its age; an error when there is none, or when it is
        older than `stale_s` (None: never stale)."""
        if self._mock:
            self._check_in_action()
            return self._mock_read(topic)
        if topic in MOTION_STATE:
            self._motion_state()
        else:
            self._subscription(topic)
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
        return self._refusal()[0]

    def _refusal(self):
        """(not_ready()'s reason, whether the arm is e-stopped), from one /control_status."""
        if self._mock:
            self._check_in_action()
            return None, False
        # /control_status before any command: the e-stop, and takeover by its generation.
        self._subscription("/control_status")
        self._motion_state()
        self._wait_for("/control_status")
        return self._driver.refusal(rammp_arm.SENDER_ID)

    def robot_description(self) -> str:
        """The driver's URDF, from /arm/robot_description (latched), through a one-shot
        subscription of its own, destroyed as the call returns (NotInAction once the action
        ends). Opens none of the arm's other entities.
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

        scope = ActionScope(self._node, self._group)  # its own: destroyed when the call returns
        try:
            with self._alive():  # never made once the action has ended
                scope.create_subscription(String, "/arm/robot_description", on_description, LATCHED)
            if not wait_until(lambda: self._live(done.is_set), STATE_WAIT_S, self._ctx):
                self._fail(
                    StateTimeout,
                    f"no /arm/robot_description in {STATE_WAIT_S:g} s -- is the arm driver up?",
                )
        finally:
            scope.close()
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

    @_rammp
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
                    self._fail(
                        Refused,
                        f"  RammpArm: the trajectory enters the keep-out zone ({msg}) -- "
                        "nothing sent.",
                    )
        return self._send(t, q, qd, impedance, monitor)

    @_rammp
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
            self._fail(Refused, refusals[-1])
        return self.move_joints(
            knots, cart_lengths=lengths, impedance=impedance, speed_scale=speed_scale
        )

    # -- planned moves (RammpArm's go_to_*, rammp_arm.py@7bc328ae:317-378) -------------------

    @_rammp
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
        g.speed_scale = _speed_scale(speed_scale)
        self._go_to("go_to_ee_pose", g, label, control_mode, gains, feedback)
        return self._tool_reached(label, pos)

    @_rammp
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
        g.speed_scale = _speed_scale(speed_scale)
        self._go_to("go_to_joint_config", g, label, control_mode, gains, feedback)
        return self._joints_reached(label, q)

    @_rammp
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
        g.speed_scale = _speed_scale(speed_scale)
        self._go_to("go_to_preset", g, label, control_mode, gains, feedback)
        return True

    @_rammp
    def go_to_ee_poses(
        self, targets, label="go_to_ee_poses", *, control_mode=POSITION, gains=None, feedback=None
    ):
        """go_to_ee_pose through `targets` (PoseTarget) in ONE motion: the driver plans every leg
        (cuRobo), blends them so the arm passes the poses between still moving where each
        target's blend_radius allows, and executes once (/go_to_ee_poses). Blocking; True if the
        driver succeeded and the tool ended within GO_TO_TRACK_M of the last target. Keyword
        arguments as in go_to_ee_pose, one for the whole motion. In mock mode it is go_to_ee_pose
        to the last target."""
        self._checkpoint()  # NotInAction / Cancelled first, as for every other call
        targets = list(targets)
        if not targets:
            self._fail(ArmError, f"{label}: no targets")
        # The driver refuses a target whose quaternion is not unit (go_to_ee_pose's does not):
        # send each normalized, and refuse one that is no rotation at all.
        quats = []
        for k, t in enumerate(targets):
            q = np.asarray(t.quat, float)
            n = float(np.linalg.norm(q))
            if not (np.isfinite(n) and n > 0.0):
                self._fail(ArmError, f"{label}: targets[{k}]: quaternion {t.quat} is no rotation")
            quats.append(q / n)
        last = targets[-1]
        if self._mock:
            return self.go_to_ee_pose(last.pos, quats[-1], label)
        g = GoToEEPoses.Goal()
        stamp = self._node.get_clock().now().to_msg()
        for t, q in zip(targets, quats, strict=True):
            m = EEPoseTarget(orientation_hold=t.hold, blend_radius=float(t.blend_radius))
            m.speed_scale = _speed_scale(t.speed_scale)
            m.pose.header.frame_id, m.pose.header.stamp = rammp_arm.FRAME, stamp
            p, o = m.pose.pose.position, m.pose.pose.orientation
            p.x, p.y, p.z = map(float, t.pos[:3])
            o.x, o.y, o.z, o.w = map(float, q)
            g.targets.append(m)
        self._go_to("go_to_ee_poses", g, label, control_mode, gains, feedback, legs=len(targets))
        return self._tool_reached(label, last.pos)

    @_rammp
    def go_to_joint_configs(
        self,
        targets,
        label="go_to_joint_configs",
        *,
        control_mode=POSITION,
        gains=None,
        feedback=None,
    ):
        """go_to_joint_config through `targets` (JointTarget) in ONE motion (/go_to_joint_configs),
        as go_to_ee_poses. Blocking; True if the driver succeeded and every joint ended within
        GO_TO_TRACK_DEG of the last target. In mock mode it is go_to_joint_config to the last
        target."""
        self._checkpoint()  # NotInAction / Cancelled first, as for every other call
        targets = list(targets)
        if not targets:
            self._fail(ArmError, f"{label}: no targets")
        last = targets[-1]
        if self._mock:
            return self.go_to_joint_config(last.joints, label)
        g = GoToJointConfigs.Goal()
        for t in targets:
            g.targets.append(
                JointConfigTarget(
                    target_joints=[float(x) for x in t.joints],
                    speed_scale=_speed_scale(t.speed_scale),
                    blend_radius=float(t.blend_radius),
                )
            )
        self._go_to(
            "go_to_joint_configs", g, label, control_mode, gains, feedback, legs=len(targets)
        )
        return self._joints_reached(label, last.joints)

    def _tool_reached(self, label, pos):
        """True when the tool ended within GO_TO_TRACK_M of `pos` (logged); Refused if not."""
        tool = _tool_position(self._read("/ee_state"))
        e = float(np.linalg.norm(tool - np.asarray(pos[:3], float)))
        message = f"  {label}: tool {e * 100:.1f} cm from the goal"
        if e > rammp_arm.GO_TO_TRACK_M:
            self._fail(Refused, message)
        self._node.get_logger().info(message)
        return True

    def _joints_reached(self, label, q):
        """True when every joint ended within GO_TO_TRACK_DEG of `q` (free joints compared the
        short way; logged); Refused if not."""
        d = np.asarray(q, float) - self.raw_joints()
        d[CONTINUOUS] = (d[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
        e = float(np.degrees(np.max(np.abs(d))))
        message = f"  {label}: joints within {e:.1f} deg of the goal"
        if e > rammp_arm.GO_TO_TRACK_DEG:
            self._fail(Refused, message)
        self._node.get_logger().info(message)
        return True

    def _go_to(self, name, g, label, control_mode, gains, feedback, legs=None):
        """Run go_to goal `g` on action `name`. `legs`: the number of targets of a
        GoToEEPoses / GoToJointConfigs goal, each with its own speed; None for a one-target goal,
        whose speed_scale the caller has set."""
        g.gains = _impedance_gains(gains)  # a bad `gains` raises before anything is sent
        self._checkpoint()
        self._check_ready(f"  RammpArm: {label}: ")
        if not self._wait_for_server(self._action(name), self._ctx, PLANNER_WAIT_S):
            self._fail(
                Refused,
                f"  RammpArm: {label}: no action server (the go_to_* actions need the cuRobo "
                "planner: sheppy profile real-arm-camera-planner) -- nothing sent.",
            )
        if legs is None:
            how, legs = f"run at {g.speed_scale:g} x its speed", 1
        else:
            how = f"{legs} targets in one motion, each leg at its own speed"
        g.control_mode = control_mode
        self._node.get_logger().info(f"  {label}: cuRobo plans the path (not previewed), {how} ...")
        timeout_s = rammp_arm.GO_TO_TIMEOUT_S * legs  # planning + execution, per leg
        res = self._run_goal(name, g, timeout_s, f"  RammpArm: {label}", feedback=feedback)
        code = res.error_code
        if code != 0:
            self._fail(
                Refused, f"  RammpArm: {label} ended {ERRORS.get(code, code)} {res.error_string}"
            )
        time.sleep(GO_TO_SETTLE_S)  # let /ee_state and /joint_states catch up with the move's end

    def _checkpoint(self):
        """Before every motion: NotInAction outside an action, wait while the action is paused,
        Cancelled once it is cancelled."""
        ctx = self._ctx  # first: end_action() clears it only after the action has ended
        self._check_in_action()
        ctx.checkpoint()

    def _send(self, t, q, qd, impedance=None, monitor=None):
        if self._mock:
            self.last_path, self.monitor_stopped = [], False
            return self._mock_sent("trajectory", q[-1])
        self._check_ready("  RammpArm: ")
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
        name = "execute_joint_trajectory"
        if not self._wait_for_server(self._action(name), self._ctx, STATE_WAIT_S):
            self._fail(StateTimeout, "no /execute_joint_trajectory action server")
        timeout_s = t[-1] + RESULT_MARGIN_S
        res = self._run_goal(
            name, g, timeout_s, "  RammpArm: trajectory", monitor=monitor, path=True
        )
        code = res.error_code
        if code == 0 or (self.monitor_stopped and code == PREEMPTED):
            return True
        fe = list(res.final_error.positions)
        if fe:
            self._node.get_logger().error(
                "  RammpArm: final joint error (ref - actual) deg J1-J7 "
                f"{np.round(np.degrees(fe), 1)}"
            )
        self._fail(
            Refused, f"  RammpArm: trajectory ended {ERRORS.get(code, code)} {res.error_string}"
        )

    def _run_goal(self, name, g, timeout_s, label, *, monitor=None, feedback=None, path=False):
        """Send the driver's goal `g` on action `name` with SENDER_ID and the token, and wait for
        its result as _goal_result() does; a goal HALTED under the token then raises ControlTaken
        (unless the arm e-stopped). `feedback`: called with each GoalFeedback, on the executor (it
        must not block)."""
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

        client = self._action(name)
        self._recorded(lambda: self._driver.begin_goal(name))
        try:
            send = self._send_goal(client, g, None if feedback is None else on_feedback)
            res = self._goal_result(send, timeout_s, label, monitor, path)
        finally:
            self._driver.end_goal()
        # HALTED under our token: control taken, unless the driver e-stopped (an e-stop drops
        # grants without a new generation), which is the caller's plain failure.
        if (
            token is not None
            and res.error_code == HALTED
            and not self._estopped_within(HALT_STATUS_WAIT_S)
        ):
            self._driver.on_goal_halted()
            self._check_command()  # raises ControlTaken: another sender took the arm from us
        return res

    def _send_goal(self, client, g, feedback_callback=None):
        """Send goal `g`, kept in _goals until the driver reports its end. NotInAction ("action
        ended") once the action has ended: the call that began in it was cut short."""
        with self._alive(ACTION_ENDED_MESSAGE):
            send = client.send_goal_async(g, feedback_callback=feedback_callback)
            self._goals.append(send)
        return send

    def _goal_result(self, send, timeout_s, label, monitor=None, path=False, moves=True):
        """Wait for goal `send`'s acceptance, then up to `timeout_s` for its result: the action's
        Result. A cancel of the action cancels the goal and raises Cancelled. Refused (`label`
        first) when the goal is refused or unanswered, or has no result in time (it is then
        cancelled); NotInAction ("action ended") when its action ends. `monitor`: polled while
        the goal runs; True cancels the goal and sets monitor_stopped. `path`: record last_path.
        `moves`: a goal that moves the arm (not a plan), so a refusal says nothing moved."""
        ctx = self._ctx
        wait_until(lambda: send.done() or not self._in_action, GOAL_ACCEPT_S, ctx)
        gh = send.result() if send.done() else None
        with self._alive(ACTION_ENDED_MESSAGE):  # end_action() settles it then
            rf = gh.get_result_async() if gh is not None and gh.accepted else None
        if gh is None:  # not answered: the driver may still accept it, so cancel it then
            send.add_done_callback(self._cancel_late_goal)
            self._fail(
                Refused,
                f"{label}: no answer from the driver in {GOAL_ACCEPT_S:g} s -- "
                "cancelled if it starts",
            )
        if not gh.accepted:
            self._forget(send)
            self._fail(Refused, f"{label}: goal REJECTED" + (" -- nothing moved" if moves else ""))
        rf.add_done_callback(lambda _: self._forget(send))
        if path:
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
            _result_within(rf, CANCEL_SETTLE_S)
            if isinstance(e, KeyboardInterrupt):
                self._node.get_logger().warning("  RammpArm: Ctrl-C -- goal cancelled")
            raise
        finally:
            self._path_t0 = None
        self._check_in_action(ACTION_ENDED_MESSAGE)
        if res is None:
            self._cancel(gh)
            self._fail(Refused, f"{label}: no result in {timeout_s:.0f} s -- goal cancelled")
        return res.result

    def _cancel(self, gh):
        """Cancel goal `gh`, unless its action has ended: end_action() then cancels it (_settle)
        before it destroys the client, which must not be used after that."""
        with self._cond:  # end_action() clears _in_action under _cond before it destroys
            if self._in_action:
                gh.cancel_goal_async()

    def _forget(self, send):
        """The driver has reported goal `send`'s end (refused, or its result): nothing to settle."""
        with self._cond:
            if send in self._goals:
                self._goals.remove(send)

    def _cancel_late_goal(self, send):
        """On the executor, once the driver answers a goal its call had given up on: cancel it if
        accepted. After the action has ended it is left to _settle, which still holds the client."""
        gh = send.result()
        if not gh.accepted:
            self._forget(send)
            return
        with self._cond:
            if not self._in_action or send not in self._goals:
                return
            gh.cancel_goal_async()
            rf = gh.get_result_async()
        rf.add_done_callback(lambda _: self._forget(send))

    @_rammp
    def gripper(self, position, timeout_s=4.0, force=None, *, speed=None):
        """Stream /setpoint/gripper at 50 Hz until the fingers stop moving (or `timeout_s`). The
        2F-85 latches the last command, so it holds after the stream stops. `speed`: a fraction
        of the closing speed (default GRIPPER_SPEED). A cancel of the action stops the stream and
        raises Cancelled. False, as for every motion, when not_ready() refuses (e.g. an e-stop),
        before or while it streams."""
        self._checkpoint()
        if self._mock:
            self._mock_gripper = float(position)
            return self._mock_sent("gripper")
        ctx = self._ctx
        # Made before the state wait, so discovery has that long (_wait_matched below).
        pub = self._get(
            "/setpoint/gripper",
            lambda scope: scope.create_publisher(
                GripperSetpoint, "/setpoint/gripper", SETPOINT_QOS
            ),
        )
        self._check_ready("  RammpArm: gripper: ")
        m = GripperSetpoint()
        m.position = float(position)
        m.speed = float(rammp_arm.GRIPPER_SPEED if speed is None else speed)
        # a current ceiling: lower = looser hold
        m.force = float(rammp_arm.GRIPPER_FORCE if force is None else force)
        token = self._driver.token
        if token is not None:
            m.token = list(token)
        self._wait_matched(pub)  # its first setpoints are not lost while the driver discovers it
        t0, last, still_since = time.monotonic(), self._read("/gripper_state").position, None
        while time.monotonic() - t0 < timeout_s:
            if ctx.cancelled():
                ctx.checkpoint()  # raises the engine's Cancelled
            why = self._driver.not_ready(rammp_arm.SENDER_ID)  # e.g. e-stopped mid-stream
            if why is not None:
                self._fail(Refused, f"  RammpArm: gripper: {why}")
            with self._alive():
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
        cancels the goal and raises Cancelled, its end raises NotInAction. Raises Refused when
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
        self._check_ready("execute_joint_trajectory: ")
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
        name = "execute_joint_trajectory"
        if not self._wait_for_server(self._action(name), self._ctx, STATE_WAIT_S):
            self._fail(StateTimeout, "no /execute_joint_trajectory action server")
        timeout_s = duration + goal_time_tolerance + RESULT_MARGIN_S
        res = self._run_goal(name, g, timeout_s, name, feedback=feedback)
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
        client = self._action(name)  # the planner needs no arm state
        path = ACTIONS[name][1]
        try:
            if not self._wait_for_server(client, self._ctx, PLANNER_WAIT_S):
                self._fail(Refused, f"{name}: no {path} action server (the cuRobo planner)")
            send = self._send_goal(client, g)
            res = self._goal_result(send, PLAN_TIMEOUT_S, name, moves=False)
        except Refused as e:  # a failed plan
            return Plan(False, str(e))
        if not res.success:
            self._record(f"{name}: {res.message}")
            return Plan(False, res.message, planning_time=float(res.planning_time))
        plan = _plan(res)
        if plan is None:
            names = list(res.trajectory.joint_names)
            self._record(
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
        self._subscription("/control_status")  # before any command, as in not_ready()
        token = self._driver.token
        if token is not None:
            req.token = list(token)
        res = self._call("set_gains", req)
        return res.accepted, res.message

    def _call(self, name, request, on_sent=None):
        """Call service `name` and return its response: Refused when the server is not up within
        SERVICE_WAIT_S or does not answer within SERVICE_CALL_S, NotInAction once the action
        ends, Cancelled once it is cancelled (both waits poll every POLL_S). Waits for no arm
        state. `on_sent`: called with the request's future once it is sent, so a caller can
        settle a request the driver answers after this call has given up on it."""
        client = self._client(name)
        ctx = self._ctx
        path = SERVICES[name][1]
        if not wait_until(lambda: self._live(client.service_is_ready), SERVICE_WAIT_S, ctx):
            self._fail(Refused, f"no {path} service in {SERVICE_WAIT_S:g} s -- is the driver up?")
        with self._alive():  # end_action() settles a request sent here (`on_sent`)
            future = client.call_async(request)
            if on_sent is not None:
                on_sent(future)
        if not wait_until(lambda: self._live(future.done), SERVICE_CALL_S, ctx):
            self._fail(Refused, f"{path}: no answer in {SERVICE_CALL_S:g} s")
        return future.result()

    # -- streaming -------------------------------------------------------------------------------

    @contextmanager
    def stream(self, controller: str, gains=None, timeout_s=None):
        """A streaming session on `controller` (see list_controllers()): yields a Stream to send()
        setpoints on, at least every `timeout_s` (default STREAM_TIMEOUT_S). `gains` (see
        _impedance_gains()) apply to impedance controllers, at open. One session is one motion: a
        pause is honoured at open only. Leaving the block, however (an exception, a cancel, the
        driver closing it), sends the hold setpoint (zeros for twist and velocity controllers, the
        measured pose or joints for pose and position ones; nothing for torque) and closes the
        session; its publisher is the action's. ModeConflict while a goal or another stream runs;
        Refused when the arm must not be commanded (EStopped when e-stopped), ArmError when the
        driver has no such controller available. In mock mode the session opens and send() does
        nothing."""
        gains = _impedance_gains(gains)
        timeout_s = STREAM_TIMEOUT_S if timeout_s is None else timeout_s
        self._checkpoint()
        if self._mock:
            self._mock_sent(f"stream {controller!r}")
            stream = Stream(self, controller, None, None, None)
            with self._alive():  # registered, as a real one: send() honours a cancel and the end
                self._stream = stream
            try:
                yield stream
            finally:
                stream.close()
            return
        self._check_ready(f"stream {controller!r}: ")
        # Before open_stream: /stream_status shows the driver closing the session (DriverState),
        # and the stream closes with the close_stream client after the action has ended.
        self._subscription("/stream_status")
        close_client = self._client("close_stream")
        driver = self._driver
        self._recorded(lambda: driver.begin_stream(controller))
        stream = None
        try:
            channel = self._channel(controller)
            topic = f"/setpoint/{channel}"
            pub = self._get(
                topic,
                lambda scope: scope.create_publisher(SETPOINTS[channel][0], topic, SETPOINT_QOS),
            )
            stream = Stream(self, controller, channel, pub, close_client)
            with self._alive():  # registered: end_action() closes it from here on
                self._stream = stream
            # The driver's OpenStream: create the publisher and let discovery settle BEFORE
            # opening, or the first setpoints go nowhere and the session expires.
            self._wait_matched(pub)
            stream.open(timeout_s, gains)
            yield stream
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                driver.end_stream()

    def _channel(self, controller):
        """The /setpoint/<channel> `controller` reads; ArmError for a controller the driver does
        not list as available."""
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
        return row.channels[0]

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
        subscriber. NotInAction once the action ends (end_action() then destroys `pub`),
        Cancelled once it is cancelled."""
        wait_until(
            lambda: self._live(lambda: pub.get_subscription_count() > 0),
            PUBLISHER_MATCH_S,
            self._ctx,
        )
