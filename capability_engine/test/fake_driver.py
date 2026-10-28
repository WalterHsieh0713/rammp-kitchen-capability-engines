"""FakeDriver: the kinova-gen3-ros2 arm driver and the cuRobo planner, faked on one node for tests.

It publishes the driver's state (/joint_states, /ee_state, /gripper_state at ~50 Hz, /diagnostics
at 1 Hz, /control_status and /stream_status latched, /arm/robot_description latched with the
flatbot URDF), serves every driver and planner action and service, and records what it receives:

    .goals[name]       every goal, accepted or not, by short name ("execute_joint_trajectory", ...)
    .requests[name]    every service request, by short name ("open_stream", ...)
    .setpoints[topic]  every message on /setpoint/* and /estop, by full topic name

The names and types below are written out independently of arm.py on purpose: a wrong topic or
action name in arm.py then fails the tests instead of being mirrored here. Test controls:

    state:        set_joints(q7), set_flange(pos, quat), set_gripper(pos), move_gripper_to(pos, s),
                  pause_publishing(), resume_publishing(), set_publish_rate(hz)
    arbitration:  enforce_arbitration(), seize(owner_id)
    goals:        accept(name, bool), set_result(name, code), set_result_after_cancel(name, code),
                  hold_goals(seconds), finish_at(name, joints=None, flange=None), cancelled(name)
    streams:      set_controllers([(name, channels, available)]), publish_stream_status(open)
    planner:      plan_result(points, joint_names=JOINTS)

Also here: FakeCtx, which stands in for the engine's Context, and counts(node).
"""

import secrets
import threading
import time
from collections import defaultdict
from pathlib import Path

from builtin_interfaces.msg import Duration
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rammp_arm_interfaces.action import (
    ExecuteJointTrajectory,
    GoToEEPose,
    GoToJointConfig,
    GoToPreset,
)
from rammp_arm_interfaces.msg import (
    ControllerCapability,
    EeState,
    GripperSetpoint,
    GripperState,
    JointSetpoint,
    PoseSetpoint,
    StreamStatus,
    TwistSetpoint,
)
from rammp_arm_interfaces.srv import CloseStream, ListControllers, OpenStream, SetGains
from rammp_common_interfaces.msg import ControlStatus, EStop
from rammp_common_interfaces.srv import AcquireControl, ReleaseControl, RevokeControl
from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose
from rammp_curobo_interfaces.srv import SetWorld
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import JointState
from sheppy_arm import ERRORS, JOINTS
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from capability_engine.engine import Cancelled

GRIPPER_JOINT = "arm_robotiq_85_left_knuckle_joint"
STATE_HZ = 50.0  # /joint_states, /ee_state and /gripper_state
DIAGNOSTICS_HZ = 1.0
GOAL_POLL_S = 0.01  # how often a held goal checks for a cancel or a seize
PLAN_DT_S = 0.1  # time between the points of a plan_result trajectory
PAUSE_POLL_S = 0.01  # how often a paused FakeCtx.checkpoint() looks for resume or cancel
FLATBOT_URDF = Path("/fixtures/urdf/flatbot.urdf")
FRAME = "arm_mount"
SUCCESSFUL, PREEMPTED, HALTED, STREAM_REJECTED = 0, -6, -9, -10

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
    "go_to_preset": (GoToPreset, "/go_to_preset"),
    "plan_to_pose": (PlanToPose, "/rammp_curobo/plan_to_pose"),
    "plan_to_joints": (PlanToJoints, "/rammp_curobo/plan_to_joints"),
}
PLANNER_ACTIONS = {"plan_to_pose", "plan_to_joints"}
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
SUBSCRIBED = {
    "/setpoint/joint_position": JointSetpoint,
    "/setpoint/joint_velocity": JointSetpoint,
    "/setpoint/joint_torque": JointSetpoint,
    "/setpoint/pose": PoseSetpoint,
    "/setpoint/twist": TwistSetpoint,
    "/setpoint/gripper": GripperSetpoint,
}
# The driver's streaming controllers on nightly (stream_server.cpp), all available by default.
CONTROLLERS = [
    ("joint_position", ["joint_position"], True),
    ("joint_impedance", ["joint_position"], True),
    ("ee_pose_position", ["pose"], True),
    ("ee_pose_impedance", ["pose"], True),
    ("joint_torque", ["joint_torque"], True),
    ("joint_velocity", ["joint_velocity"], True),
    ("ee_twist", ["twist"], True),
    ("ee_twist_impedance", ["twist"], True),
]


class FakeCtx:
    """Stands in for engine.Context: checkpoint() blocks while paused and raises Cancelled once
    cancelled."""

    def __init__(self):
        self._paused = threading.Event()
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def checkpoint(self):
        while self._paused.is_set() and not self._cancelled.is_set():
            self._cancelled.wait(PAUSE_POLL_S)
        if self._cancelled.is_set():
            raise Cancelled()


def counts(node):
    """(subscriptions, service clients, action clients and other waitables, publishers)."""
    return tuple(
        len(list(entities))
        for entities in (node.subscriptions, node.clients, node.waitables, node.publishers)
    )


class FakeDriver(Node):
    def __init__(self, node_name="fake_arm_driver"):
        super().__init__(node_name)
        self._group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self.goals = defaultdict(list)
        self.requests = defaultdict(list)
        self.setpoints = defaultdict(list)

        # Arm state, as the next publish reports it.
        self._joints = [0.0] * 7
        self._flange = ([0.5, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0])
        self._gripper_ramp = (0.0, 0.0, 0.0, 0.0)  # (from, to, start time, seconds)
        self._publishing = True
        # /control_status
        self._arbitration = False
        self._estopped = False
        self._owner_id = ""
        self._generation = 0
        # Goal outcomes, by action name.
        self._accept = {}
        self._results = {}
        self._results_after_cancel = {}
        self._finish = {}
        self._cancelled = set()
        self._hold_s = 0.0
        self._plan = None
        self._controllers = list(CONTROLLERS)

        self._js_pub = self.create_publisher(JointState, "/joint_states", qos_profile_sensor_data)
        self._ee_pub = self.create_publisher(EeState, "/ee_state", qos_profile_sensor_data)
        self._grip_pub = self.create_publisher(
            GripperState, "/gripper_state", qos_profile_sensor_data
        )
        self._diag_pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", qos_profile_sensor_data
        )
        self._control_pub = self.create_publisher(ControlStatus, "/control_status", LATCHED)
        self._stream_pub = self.create_publisher(StreamStatus, "/stream_status", LATCHED)
        self._description_pub = self.create_publisher(String, "/arm/robot_description", LATCHED)
        self._description_pub.publish(String(data=FLATBOT_URDF.read_text()))
        self._publish_control_status()
        self.publish_stream_status(False)

        for name, (action_type, path) in ACTIONS.items():
            ActionServer(
                self,
                action_type,
                path,
                execute_callback=lambda handle, n=name: self._execute(n, handle),
                goal_callback=lambda goal, n=name: self._on_goal(n, goal),
                cancel_callback=lambda _, n=name: self._on_cancel(n),
                callback_group=self._group,
            )
        for name, (srv_type, path) in SERVICES.items():
            handler = getattr(self, f"_srv_{name}")
            self.create_service(
                srv_type,
                path,
                lambda req, res, n=name, h=handler: self._serve(n, h, req, res),
                callback_group=self._group,
            )
        for topic, msg_type in SUBSCRIBED.items():
            self.create_subscription(
                msg_type,
                topic,
                lambda m, t=topic: self.setpoints[t].append(m),
                SETPOINT_QOS,
                callback_group=self._group,
            )
        self.create_subscription(EStop, "/estop", self._on_estop, 10, callback_group=self._group)
        self._state_timer = self.create_timer(
            1.0 / STATE_HZ, self._publish_state, callback_group=self._group
        )
        self.create_timer(
            1.0 / DIAGNOSTICS_HZ, self._publish_diagnostics, callback_group=self._group
        )

    # -- test controls: state -------------------------------------------------------------------

    def set_joints(self, q7):
        with self._lock:
            self._joints = [float(x) for x in q7]

    def set_flange(self, pos, quat):
        """The flange (/ee_state) pose in arm_mount; quat is x, y, z, w."""
        with self._lock:
            self._flange = ([float(x) for x in pos], [float(x) for x in quat])

    def set_gripper(self, pos):
        with self._lock:
            self._gripper_ramp = (float(pos), float(pos), 0.0, 0.0)

    def move_gripper_to(self, pos, seconds):
        """The gripper moves linearly from where it is to `pos` over `seconds`, starting now."""
        now = time.monotonic()
        with self._lock:
            start = self._gripper_at(now)
            self._gripper_ramp = (start, float(pos), now, float(seconds))

    def set_publish_rate(self, hz):
        """Publish /joint_states, /ee_state and /gripper_state at `hz` instead of STATE_HZ."""
        self.destroy_timer(self._state_timer)
        self._state_timer = self.create_timer(
            1.0 / hz, self._publish_state, callback_group=self._group
        )

    def pause_publishing(self):
        self._publishing = False

    def resume_publishing(self):
        self._publishing = True

    # -- test controls: arbitration -------------------------------------------------------------

    def enforce_arbitration(self):
        """Arbitration on: goals, streams and gains without a token are refused (NOT_AUTHORIZED)."""
        with self._lock:
            self._arbitration = True
        self._publish_control_status()

    def seize(self, owner_id):
        """Another sender acquires control: generation bumps, running goals settle HALTED.
        Returns the new token."""
        with self._lock:
            self._owner_id = owner_id
            self._generation += 1
        self._publish_control_status()
        return secrets.token_bytes(16)

    # -- test controls: goals -------------------------------------------------------------------

    def accept(self, name, accepted):
        self._accept[name] = bool(accepted)

    def set_result(self, name, code):
        """The error_code a goal of `name` finishes with when not cancelled (default 0)."""
        self._results[name] = code

    def set_result_after_cancel(self, name, code):
        """The error_code a cancelled goal of `name` settles with (default -6 PREEMPTED)."""
        self._results_after_cancel[name] = code

    def hold_goals(self, seconds):
        """Goals run this long before finishing unless cancelled (default: they end at once)."""
        self._hold_s = float(seconds)

    def finish_at(self, name, joints=None, flange=None):
        """Where a goal of `name` leaves the arm when it ends: joints (q7), flange ((pos, quat))."""
        self._finish[name] = (joints, flange)

    def cancelled(self, name):
        """Whether a cancel was requested for any goal of `name`."""
        return name in self._cancelled

    # -- test controls: streams and planner -----------------------------------------------------

    def set_controllers(self, controllers):
        """What /list_controllers reports: [(name, channels, available), ...]."""
        self._controllers = [(n, list(c), bool(a)) for n, c, a in controllers]

    def publish_stream_status(self, open, controller="", channels=(), timeout_s=0.0):  # noqa: A002
        self._stream_pub.publish(
            StreamStatus(
                open=open, controller=controller, channels=list(channels), timeout_s=timeout_s
            )
        )

    def plan_result(self, trajectory_points, joint_names=JOINTS):
        """What plan_to_pose / plan_to_joints return: points (position lists) PLAN_DT_S apart."""
        traj = JointTrajectory(joint_names=list(joint_names))
        for i, positions in enumerate(trajectory_points):
            t = i * PLAN_DT_S
            sec = int(t)
            traj.points.append(
                JointTrajectoryPoint(
                    positions=[float(x) for x in positions],
                    velocities=[0.0] * len(positions),
                    time_from_start=Duration(sec=sec, nanosec=round((t - sec) * 1e9)),
                )
            )
        self._plan = traj

    # -- actions --------------------------------------------------------------------------------

    def _on_goal(self, name, goal):
        self.goals[name].append(goal)
        if not self._accept.get(name, True) or not self._authorized(goal):
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, name):
        self._cancelled.add(name)
        return CancelResponse.ACCEPT

    def _execute(self, name, handle):
        generation = self._generation
        deadline = time.monotonic() + self._hold_s
        while (
            time.monotonic() < deadline
            and not handle.is_cancel_requested
            and self._generation == generation
        ):
            time.sleep(GOAL_POLL_S)
        joints, flange = self._finish.get(name, (None, None))
        if joints is not None:
            self.set_joints(joints)
        if flange is not None:
            self.set_flange(*flange)
        if handle.is_cancel_requested:
            code = self._results_after_cancel.get(name, PREEMPTED)
            handle.canceled()
        else:
            code = HALTED if self._generation != generation else self._results.get(name, SUCCESSFUL)
            if code == SUCCESSFUL:
                handle.succeed()
            else:
                handle.abort()
        result = ACTIONS[name][0].Result()
        if name in PLANNER_ACTIONS:
            result.success = code == SUCCESSFUL and self._plan is not None
            result.message = "" if result.success else "FakeDriver: no plan_result set, or failed"
            result.trajectory = self._plan or JointTrajectory()
        else:
            result.error_code = code
            result.error_string = ERRORS.get(code, "")
        return result

    def _authorized(self, request):
        """False when arbitration is enforced and the request carries no token."""
        return not (self._arbitration and hasattr(request, "token") and not any(request.token))

    # -- services -------------------------------------------------------------------------------

    def _serve(self, name, handler, request, response):
        self.requests[name].append(request)
        return handler(request, response)

    def _srv_open_stream(self, request, response):
        row = next((c for c in self._controllers if c[0] == request.controller), None)
        if row is None or not row[2] or not self._authorized(request):
            response.error_code = STREAM_REJECTED
            response.message = f"FakeDriver: refused controller {request.controller!r}"
            return response
        response.accepted, response.channels = True, row[1]
        self.publish_stream_status(True, row[0], row[1], request.timeout_s)
        return response

    def _srv_close_stream(self, request, response):
        response.closed = True
        self.publish_stream_status(False)
        return response

    def _srv_list_controllers(self, request, response):
        response.controllers = [
            ControllerCapability(name=n, channels=c, available=a) for n, c, a in self._controllers
        ]
        return response

    def _srv_set_gains(self, request, response):
        response.accepted = self._authorized(request)
        return response

    def _srv_acquire_control(self, request, response):
        response.token = list(self.seize(request.owner_id))
        response.accepted, response.generation = True, self._generation
        return response

    def _srv_release_control(self, request, response):
        self._release()
        response.released = True
        return response

    def _srv_revoke_control(self, request, response):
        self._release()
        response.revoked = True
        return response

    def _srv_set_world(self, request, response):
        response.success = True
        return response

    def _release(self):
        with self._lock:
            self._owner_id = ""
        self._publish_control_status()

    # -- publishing -----------------------------------------------------------------------------

    def _on_estop(self, msg):
        self.setpoints["/estop"].append(msg)
        with self._lock:
            self._estopped = msg.engaged
        self._publish_control_status()

    def _publish_control_status(self):
        with self._lock:
            msg = ControlStatus(
                arbitration_enabled=self._arbitration,
                estopped=self._estopped,
                owned=bool(self._owner_id),
                owner_id=self._owner_id,
                generation=self._generation,
            )
        msg.header.stamp = self.get_clock().now().to_msg()
        self._control_pub.publish(msg)

    def _gripper_at(self, now):
        start, end, t0, seconds = self._gripper_ramp
        if seconds <= 0.0 or now >= t0 + seconds:
            return end
        return start + (end - start) * (now - t0) / seconds

    def _publish_state(self):
        if not self._publishing:
            return
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            joints, (pos, quat) = list(self._joints), self._flange
            gripper = self._gripper_at(time.monotonic())
        # The gripper joint goes first, so a reader that slices instead of looking names up fails.
        js = JointState(
            name=[GRIPPER_JOINT, *JOINTS],
            position=[gripper, *joints],
            velocity=[float("nan")] + [0.0] * 7,
            effort=[float("nan")] + [0.0] * 7,
        )
        js.header.stamp = stamp
        self._js_pub.publish(js)
        ee = EeState()
        ee.header.stamp, ee.header.frame_id = stamp, FRAME
        p, o = ee.pose.position, ee.pose.orientation
        p.x, p.y, p.z = pos
        o.x, o.y, o.z, o.w = quat
        self._ee_pub.publish(ee)
        grip = GripperState(position=gripper, present=True)
        grip.header.stamp = stamp
        self._grip_pub.publish(grip)

    def _publish_diagnostics(self):
        if not self._publishing:
            return
        msg = DiagnosticArray(
            status=[
                DiagnosticStatus(level=DiagnosticStatus.OK, name=n, message="OK")
                for n in ("Arbitration", "Arm", "Gripper")
            ]
        )
        msg.header.stamp = self.get_clock().now().to_msg()
        self._diag_pub.publish(msg)
