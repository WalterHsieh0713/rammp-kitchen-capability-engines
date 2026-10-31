"""Base class every kitchen capability builds on.

A capability subclasses `CapabilityEngine`, declares its actions, and writes
`execute()`. This class implements the rest of the RAMMP orchestrator's engine
contract (rammp_orchestrator_interfaces) once, for every capability:

- serves /<id>/activate, /<id>/execute_action, /<id>/control, /<id>/hibernate
- registers with /rammp_orchestrator/register_engine, retrying until the
  orchestrator appears, and never exiting because it is absent
- publishes /<id>/status (JSON in std_msgs/String) at 1 Hz and on every change
- accepts cancel from Control.CANCEL and from an action cancel alike
- gives each action ROS entities that live only while it runs (self.arm's, and the
  capability's own through create_action_subscription() and friends)
- on SIGTERM or SIGINT, cancels the running action, lets it end, then exits

The same node runs standalone with `-p use_orchestrator:=false`: it skips
registration, and the orchestrator calls on `ctx` become logged no-ops. Drive
it with `ros2 action send_goal /<id>/execute_action ...` (see `make action`).
"""

import json
import os
import signal
import threading
import time
from dataclasses import dataclass, field

import rammp_arm
import rclpy
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import (
    ActionAvailability,
    ActionSpec,
    CapabilityManifest,
    Progress,
    Result,
)
from rammp_orchestrator_interfaces.srv import (
    Activate,
    Control,
    Hibernate,
    RegisterEngine,
    RequestIntent,
    UpdateObjectState,
    WriteWorldState,
)
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from capability_engine import sim
from capability_engine.action_scope import ENDED_MESSAGE, ActionScope, ScopeClosed
from capability_engine.arm import ArmInterface, wait_until
from capability_engine.arm_params import PARAMS

ORCHESTRATOR = "/rammp_orchestrator"
ARM_PARAMS_FIXED_MESSAGE = "arm.* parameters are read at startup; restart to change"
REGISTER_RETRY_S = 5.0
CALL_TIMEOUT_S = 10.0
INTENT_TIMEOUT_S = 120.0
STOP_MARGIN_S = 1.0  # on a stop, past self.arm's worst settle: for the action's own code
START_WAIT_S = 1.0  # an accepted goal not handed to execute within this was lost by rclpy
SPIN_STEP_S = 0.1  # run() spins in steps this long, to notice a stop between them


@dataclass
class Action:
    """One action this capability offers. `params_schema` is JSON Schema."""

    action_id: str
    label: str
    params_schema: dict = field(default_factory=dict)
    sticky_followup: bool = False
    perception_classes: list[str] = field(default_factory=list)


def stop_wait_s() -> float:
    """On SIGTERM/SIGINT, how long the cancelled action may take to end: self.arm's worst settle
    (ArmInterface.worst_settle_s(), 7.8 s by default) plus STOP_MARGIN_S. docker stop's grace
    (10 s) must cover it."""
    return ArmInterface.worst_settle_s() + STOP_MARGIN_S


class Cancelled(Exception):
    """Raised by `ctx.checkpoint()` once the action has been cancelled."""


class EngineError(Exception):
    """Raise from `execute()` to fail the action with an error code."""

    def __init__(self, error_code: str, message: str = "", recoverable: bool = False):
        super().__init__(message or error_code)
        self.error_code = error_code
        self.recoverable = recoverable


class _Run:
    """One accepted goal, from its acceptance until its result is reported: its cancel and pause,
    and (once it executes) the scope its per-action ROS entities live in."""

    def __init__(self, action_id: str, request):
        self.action_id = action_id
        self.request = request  # the goal request, the same object rclpy gives its goal handle
        self.accepted_at = time.monotonic()
        self.goal = None  # its goal handle, once rclpy has sent the acceptance (_on_accepted)
        self.cancelled = threading.Event()
        self.paused = threading.Event()
        self.scope = None


class Context:
    """What `execute()` uses to report progress, honour pause/cancel, and talk
    to the orchestrator."""

    def __init__(self, engine: "CapabilityEngine", goal_handle, run: _Run):
        self._engine = engine
        self._goal = goal_handle
        self._run = run

    def progress(self, phase: str, pct: float, message: str = "", pausable: bool = True):
        feedback = ExecuteAction.Feedback()
        feedback.progress = Progress(
            phase=phase, pct=float(pct), message=message, pausable=pausable
        )
        self._goal.publish_feedback(feedback)

    def checkpoint(self):
        """Call between steps: blocks while paused, raises Cancelled if cancelled."""
        while self._run.paused.is_set() and not self.cancelled():
            self._run.cancelled.wait(0.1)
        if self.cancelled():
            raise Cancelled()

    def cancelled(self) -> bool:
        """Whether the action has been cancelled. Never blocks, even while paused: self.arm
        polls it while a motion runs."""
        return self._run.cancelled.is_set() or self._goal.is_cancel_requested

    def request_intent(self, intent_type: str, params: dict | None = None) -> Result:
        """Ask the orchestrator to delegate an intent (e.g. MOVE_BASE) and wait (Cancelled
        once the action is cancelled)."""
        request = RequestIntent.Request(
            requesting_engine_id=self._engine.get_name(),
            intent_type=intent_type,
            params_json=json.dumps(params or {}),
        )
        response = self._engine._call("request_intent", request, self, INTENT_TIMEOUT_S)
        if response is None:
            return Result(status=Result.STATUS_FAILED, error_code="orchestrator_unavailable")
        return response.result

    def write_world_state(self, **updates) -> bool:
        """Merge into the orchestrator's world state. Its schema is fixed
        (gripper, doors, objects, arm, base_pose); unknown keys are rejected."""
        request = WriteWorldState.Request(updates_json=json.dumps(updates))
        return self._engine._write("write_world_state", request, self)

    def update_object(self, instance_id: str, **fields) -> bool:
        request = UpdateObjectState.Request(instance_id=instance_id, fields_json=json.dumps(fields))
        return self._engine._write("update_object_state", request, self)


class CapabilityEngine(Node):
    """Subclass, set the class attributes, and implement `execute()`."""

    display_name: str = ""
    perception_classes: list[str] = []
    required_resources: list[str] = []
    state_reads: list[str] = []
    state_writes: list[str] = []
    intents_handled: list[str] = []
    actions: list[Action] = []

    def __init__(self, engine_id: str, **node_kwargs):
        # The orchestrator addresses an engine as /<engine_id>/..., so the
        # node name is the engine id. `node_kwargs` go to rclpy's Node.
        super().__init__(engine_id, **node_kwargs)
        self.declare_parameter("use_orchestrator", True)
        self.declare_parameter("mock", False)
        self.orchestrated = self.get_parameter("use_orchestrator").value
        self.mock = self.get_parameter("mock").value

        self._group = ReentrantCallbackGroup()
        self._declare_arm_params(engine_id)
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.arm = ArmInterface(self, self._group, mock=self.mock)
        # make_sim() loads the driver's model (in mock mode, arm.mock_urdf's)
        sim.set_model_provider(self.arm.robot_description)
        # The lifecycle: idle (no _current), running one goal from its acceptance until its
        # result is reported (_current), and stopping (_stopping set: no goal is accepted again).
        self._lifecycle = threading.Condition()
        self._current = None
        self._stopping = None  # time.monotonic() stop() was first called
        self._registered = False
        self._registering = False
        self._register_sent = 0.0  # time.monotonic() the pending RegisterEngine call was sent
        self.instance_id = ""

        self.create_service(Activate, "~/activate", self._on_activate, callback_group=self._group)
        self.create_service(Control, "~/control", self._on_control, callback_group=self._group)
        self.create_service(
            Hibernate, "~/hibernate", self._on_hibernate, callback_group=self._group
        )
        ActionServer(
            self,
            ExecuteAction,
            "~/execute_action",
            execute_callback=self._on_execute,
            goal_callback=self._on_goal,
            handle_accepted_callback=self._on_accepted,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self._group,
        )
        self._orchestrator_clients = {
            name: self.create_client(srv, f"{ORCHESTRATOR}/{name}", callback_group=self._group)
            for name, srv in (
                ("register_engine", RegisterEngine),
                ("request_intent", RequestIntent),
                ("write_world_state", WriteWorldState),
                ("update_object_state", UpdateObjectState),
            )
        }

        self._status = self.create_publisher(String, "~/status", 10)
        self.create_timer(1.0, self._publish_status, callback_group=self._group)
        if self.orchestrated:
            self.create_timer(REGISTER_RETRY_S, self._try_register, callback_group=self._group)
            self._try_register()

        mode = "orchestrated" if self.orchestrated else "standalone"
        self.get_logger().info(f"{engine_id} up ({mode}{', mock' if self.mock else ''})")

    # -- what a capability implements ---------------------------------------

    def availability(self, instance_id: str, world_state: dict) -> list[ActionAvailability]:
        """Which actions can run on this object now. Default: all of them."""
        return [ActionAvailability(action_id=a.action_id, available=True) for a in self.actions]

    def execute(self, action_id: str, params: dict, ctx: Context) -> dict:
        """Run one action. Return the result payload; raise EngineError to fail.
        Call ctx.checkpoint() between steps so pause and cancel work."""
        raise NotImplementedError

    def hibernate(self):
        """Drop any per-selection state. Default: nothing to drop."""

    # -- ROS entities for the running action only ---------------------------

    def create_action_subscription(self, msg_type, topic, callback, qos_profile):
        """Node.create_subscription, for the running action only: destroyed, safely, when it
        ends (an idle subscription still costs CPU for every message). Call it inside execute()."""
        return self._action_scope().create_subscription(msg_type, topic, callback, qos_profile)

    def create_action_publisher(self, msg_type, topic, qos_profile):
        """Node.create_publisher, for the running action only, as create_action_subscription()."""
        return self._action_scope().create_publisher(msg_type, topic, qos_profile)

    def create_action_tf_buffer(self):
        """A tf2_ros.Buffer kept current for the running action only. Use it instead of a
        TransformListener, whose unregister() is not safe while the executor spins."""
        return self._action_scope().create_tf_buffer()

    def _action_scope(self) -> ActionScope:
        with self._lifecycle:
            run = self._current
        if run is None or run.scope is None:
            raise ScopeClosed(ENDED_MESSAGE)
        return run.scope

    # -- orchestrator plumbing ----------------------------------------------

    def _declare_arm_params(self, engine_id):
        """Declare every arm.<name> parameter and write its value to the constant it sets. Those
        are module constants, so the values are process-wide: the last engine made sets them."""
        for prm in PARAMS:
            # The design: under the engine, the sender id on goals is the capability's.
            default = engine_id if prm.target == (rammp_arm, "SENDER_ID") else prm.default
            value = self.declare_parameter(
                f"arm.{prm.name}", default, ParameterDescriptor(description=prm.description)
            ).value
            setattr(*prm.target, list(value) if isinstance(prm.default, list) else value)

    def _on_set_parameters(self, params):
        """Refuse arm.* changes after startup: their values were written once, in __init__."""
        if any(prm.name.startswith("arm.") for prm in params):
            return SetParametersResult(successful=False, reason=ARM_PARAMS_FIXED_MESSAGE)
        return SetParametersResult(successful=True)

    def _manifest(self) -> CapabilityManifest:
        return CapabilityManifest(
            engine_id=self.get_name(),
            display_name=self.display_name or self.get_name(),
            perception_classes=list(self.perception_classes),
            actions=[
                ActionSpec(
                    action_id=a.action_id,
                    label=a.label,
                    params_schema_json=json.dumps(a.params_schema),
                    sticky_followup=a.sticky_followup,
                    perception_classes=list(a.perception_classes),
                )
                for a in self.actions
            ],
            required_resources=list(self.required_resources),
            state_reads=list(self.state_reads),
            state_writes=list(self.state_writes),
            intents_handled=list(self.intents_handled),
        )

    def _try_register(self):
        client = self._orchestrator_clients["register_engine"]
        if not client.service_is_ready():
            # Gone: a restarted orchestrator has forgotten every registration, and a call it
            # never answered never will be. Register afresh once it is back.
            if self._registered or self._registering:
                self._registered = self._registering = False
                self._publish_status()
            self.get_logger().info(
                "waiting for the orchestrator to register "
                "(run with -p use_orchestrator:=false to run standalone)",
                throttle_duration_sec=10.0,
            )
            return
        if self._registering and time.monotonic() - self._register_sent > CALL_TIMEOUT_S:
            self._registering = False  # never answered: ask again
        if self._registered or self._registering:
            return
        self._registering = True
        self._register_sent = time.monotonic()
        future = client.call_async(RegisterEngine.Request(manifest=self._manifest()))
        future.add_done_callback(self._on_registered)

    def _on_registered(self, future):
        self._registering = False
        response = future.result()
        # The orchestrator keeps registrations for its lifetime and rejects a
        # duplicate id, so after this node restarts it is still registered.
        if response.accepted or "already registered" in response.message:
            self._registered = True
            self.get_logger().info(f"registered with the orchestrator: {response.message}")
            self._publish_status()
        else:
            self.get_logger().warning(f"registration rejected, will retry: {response.message}")

    def _call(self, name: str, request, ctx: Context, timeout: float = CALL_TIMEOUT_S):
        """Blocking call to an orchestrator service, from inside execute(): None when it is
        unavailable or does not answer, Cancelled once the action is cancelled."""
        if not self.orchestrated:
            self.get_logger().warning(f"standalone: skipping orchestrator call '{name}'")
            return None
        client = self._orchestrator_clients[name]
        if not wait_until(client.service_is_ready, timeout, ctx):
            self.get_logger().error(f"orchestrator service '{name}' unavailable")
            return None
        future = client.call_async(request)
        if not wait_until(future.done, timeout, ctx):
            self.get_logger().error(f"orchestrator service '{name}' timed out")
            return None
        return future.result()

    def _write(self, name: str, request, ctx: Context) -> bool:
        response = self._call(name, request, ctx)
        if response is not None and not response.success:
            self.get_logger().warning(f"orchestrator rejected {name}: {response.message}")
        return bool(response and response.success)

    # -- servers ---------------------------------------------------------------

    def _on_activate(self, request, response):
        self.instance_id = request.instance_id
        try:
            world = json.loads(request.world_state_snapshot_json or "{}")
            response.availability = self.availability(request.instance_id, world)
        except Exception as error:  # bad input or a bug in availability() must not kill the node
            self.get_logger().error(f"activate {request.instance_id!r} raised: {error!r}")
        return response

    def _on_hibernate(self, request, response):
        self.instance_id = ""
        try:
            self.hibernate()
            response.success = True
        except Exception as error:  # a bug in hibernate() must not kill the node
            self.get_logger().error(f"hibernate raised: {error!r}")
        return response

    def _on_control(self, request, response):
        with self._lifecycle:
            run = self._current  # pause and cancel belong to the goal running, if any
        if request.command not in (
            Control.Request.PAUSE,
            Control.Request.RESUME,
            Control.Request.CANCEL,
        ):
            response.message = f"unknown command {request.command}"
            return response
        if run is not None:
            if request.command == Control.Request.PAUSE:
                run.paused.set()
            elif request.command == Control.Request.RESUME:
                run.paused.clear()
            else:
                run.cancelled.set()
        response.accepted = True
        response.message = "ok" if run is not None else "ok (no action running)"
        self._publish_status()
        return response

    def _on_goal(self, goal):
        if goal.action_id not in {a.action_id for a in self.actions}:
            return GoalResponse.REJECT
        # Accepted only when idle, and its run is made here: a Control.CANCEL from now on, even
        # before it executes, applies to this goal.
        with self._lifecycle:
            run = self._current
            if (
                run is not None
                and run.goal is None
                and time.monotonic() - run.accepted_at > START_WAIT_S
            ):
                # rclpy accepted it but failed before handing it on (e.g. its client went away
                # before the acceptance was sent): it will never execute or end.
                self.get_logger().warning(f"{run.action_id}: accepted but never started; dropped")
                run = self._current = None
            if run is not None or self._stopping is not None:
                return GoalResponse.REJECT
            self._current = _Run(goal.action_id, goal)
        return GoalResponse.ACCEPT

    def _on_accepted(self, goal_handle):
        """rclpy has sent the acceptance: bind the run _on_goal made for this goal (by its
        request, never a newer goal's run), then execute."""
        with self._lifecycle:
            run = self._current
            if run is not None and run.goal is None and run.request is goal_handle.request:
                run.goal = goal_handle
        goal_handle.execute()

    def _on_execute(self, goal_handle):
        goal = goal_handle.request
        with self._lifecycle:
            run = self._current
        if run is None or run.goal is not goal_handle:  # its run was dropped as never started
            goal_handle.abort()
            return ExecuteAction.Result(
                result=Result(status=Result.STATUS_FAILED, error_code="not_started")
            )
        result = Result()
        try:
            run.scope = ActionScope(self, self._group)
            self._publish_status()
            ctx = Context(self, goal_handle, run)
            try:
                # Follow-up: ExecuteAction carries no control token yet, so self.arm sends none.
                self.arm.begin_action(ctx, token=None)
                params = json.loads(goal.params_json or "{}")
                payload = self.execute(goal.action_id, params, ctx)
                result.status = Result.STATUS_SUCCESS
                result.payload_json = json.dumps(payload or {})
            except Cancelled:
                result.status = Result.STATUS_CANCELLED
            except EngineError as error:
                result.status = Result.STATUS_FAILED
                result.error_code = error.error_code
                result.recoverable = error.recoverable
                self.get_logger().error(f"{goal.action_id} failed: {error}")
            except Exception as error:  # a bug in execute() must not kill the node
                result.status = Result.STATUS_FAILED
                result.error_code = "exception"
                self.get_logger().error(f"{goal.action_id} raised: {error!r}")
            finally:
                try:
                    try:
                        self.arm.end_action()  # settles its goals and stream, closes its scope
                    finally:
                        run.scope.close()  # destroys the capability's entities of this action
                except Exception as error:  # the run must still end and report its result
                    result.status = Result.STATUS_FAILED
                    result.error_code = "exception"
                    self.get_logger().error(f"{goal.action_id}: ending it raised: {error!r}")

            # canceled() is only a legal transition after an action-level cancel
            # request; a Control.CANCEL is reported through result.status instead.
            if result.status == Result.STATUS_SUCCESS:
                goal_handle.succeed()
            elif result.status == Result.STATUS_CANCELLED and goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            self._publish_status()
        finally:
            with self._lifecycle:  # idle again only once the result is reported: stop() waits
                self._current = None
                self._lifecycle.notify_all()
            self._publish_status()
        return ExecuteAction.Result(result=result)

    def stop(self):
        """Begin stopping: no goal is accepted from now on, and the running one is cancelled."""
        with self._lifecycle:
            if self._stopping is None:
                self._stopping = time.monotonic()
            if self._current is not None:
                self._current.cancelled.set()

    def _stopped(self) -> bool:
        """Whether stop() was called and the running action has since ended (or stop_wait_s()
        has passed)."""
        with self._lifecycle:
            if self._stopping is None:
                return False
            return self._current is None or time.monotonic() - self._stopping > stop_wait_s()

    def _publish_status(self):
        with self._lifecycle:
            run = self._current
        if run is None:
            state = "idle"
        else:
            state = "paused" if run.paused.is_set() else "executing"
        status = {
            "state": state,
            "action": "" if run is None else run.action_id,
            "instance_id": self.instance_id,
            "mode": "orchestrated" if self.orchestrated else "standalone",
            "registered": self._registered,
            "mock": self.mock,
        }
        self._status.publish(String(data=json.dumps(status)))


def run(engine_class: type[CapabilityEngine], args=None):
    """Entry point for a capability: spin until SIGTERM or Ctrl-C."""
    # rclpy's own handlers would shut ROS down at once, under a running action.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = engine_class()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # A container's PID 1 receives no default action for SIGTERM, so without
    # this handler `docker stop` — and every `sheppy woof` — would wait out the
    # full grace period and then SIGKILL. SIGINT is the image's STOPSIGNAL and
    # Ctrl-C. Either cancels the running action and stops accepting goals; the
    # executor spins on until the action has ended and reported its result (at
    # most stop_wait_s()), so self.arm sees the driver settle. ROS is shut down
    # here, in the spinning thread: rclpy.try_shutdown() from another thread
    # can land mid-spin and raise RCLError out of it.
    signal.signal(signal.SIGTERM, lambda *_: node.stop())
    signal.signal(signal.SIGINT, lambda *_: node.stop())

    try:
        while rclpy.ok() and not node._stopped():
            executor.spin_once(timeout_sec=SPIN_STEP_S)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop()  # cancels a running execute(), which shutdown() waits for
        if node._current is not None:
            # It ignored the cancel for stop_wait_s(): shutdown() would wait for it forever.
            node.get_logger().error(
                f"{node._current.action_id} did not end within {stop_wait_s():g} s of the stop; "
                "exiting without it"
            )
            os._exit(1)
        executor.shutdown()
        rclpy.try_shutdown()
