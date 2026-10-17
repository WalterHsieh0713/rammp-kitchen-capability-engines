"""Base class every kitchen capability builds on.

A capability subclasses `CapabilityEngine`, declares its actions, and writes
`execute()`. This class implements the rest of the RAMMP orchestrator's engine
contract (rammp_orchestrator_interfaces) once, for every capability:

- serves /<id>/activate, /<id>/execute_action, /<id>/control, /<id>/hibernate
- registers with /rammp_orchestrator/register_engine, retrying until the
  orchestrator appears, and never exiting because it is absent
- publishes /<id>/status (JSON in std_msgs/String) at 1 Hz and on every change
- accepts cancel from Control.CANCEL and from an action cancel alike
- exits on SIGTERM

The same node runs standalone with `-p use_orchestrator:=false`: it skips
registration, and the orchestrator calls on `ctx` become logged no-ops. Drive
it with `ros2 action send_goal /<id>/execute_action ...` (see `make action`).
"""

import json
import signal
import threading
from dataclasses import dataclass, field

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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

ORCHESTRATOR = "/rammp_orchestrator"
REGISTER_RETRY_S = 5.0
CALL_TIMEOUT_S = 10.0
INTENT_TIMEOUT_S = 120.0


@dataclass
class Action:
    """One action this capability offers. `params_schema` is JSON Schema."""

    action_id: str
    label: str
    params_schema: dict = field(default_factory=dict)
    sticky_followup: bool = False
    perception_classes: list[str] = field(default_factory=list)


class Cancelled(Exception):
    """Raised by `ctx.checkpoint()` once the action has been cancelled."""


class EngineError(Exception):
    """Raise from `execute()` to fail the action with an error code."""

    def __init__(self, error_code: str, message: str = "", recoverable: bool = False):
        super().__init__(message or error_code)
        self.error_code = error_code
        self.recoverable = recoverable


class Context:
    """What `execute()` uses to report progress, honour pause/cancel, and talk
    to the orchestrator."""

    def __init__(self, engine: "CapabilityEngine", goal_handle):
        self._engine = engine
        self._goal = goal_handle

    def progress(self, phase: str, pct: float, message: str = "", pausable: bool = True):
        feedback = ExecuteAction.Feedback()
        feedback.progress = Progress(
            phase=phase, pct=float(pct), message=message, pausable=pausable
        )
        self._goal.publish_feedback(feedback)

    def checkpoint(self):
        """Call between steps: blocks while paused, raises Cancelled if cancelled."""
        engine = self._engine
        while engine._paused.is_set() and not engine._cancel.is_set():
            engine._cancel.wait(0.1)
        if engine._cancel.is_set() or self._goal.is_cancel_requested:
            raise Cancelled()

    def request_intent(self, intent_type: str, params: dict | None = None) -> Result:
        """Ask the orchestrator to delegate an intent (e.g. MOVE_BASE) and wait."""
        request = RequestIntent.Request(
            requesting_engine_id=self._engine.get_name(),
            intent_type=intent_type,
            params_json=json.dumps(params or {}),
        )
        response = self._engine._call("request_intent", request, INTENT_TIMEOUT_S)
        if response is None:
            return Result(status=Result.STATUS_FAILED, error_code="orchestrator_unavailable")
        return response.result

    def write_world_state(self, **updates) -> bool:
        """Merge into the orchestrator's world state. Its schema is fixed
        (gripper, doors, objects, arm, base_pose); unknown keys are rejected."""
        request = WriteWorldState.Request(updates_json=json.dumps(updates))
        return self._engine._write("write_world_state", request)

    def update_object(self, instance_id: str, **fields) -> bool:
        request = UpdateObjectState.Request(instance_id=instance_id, fields_json=json.dumps(fields))
        return self._engine._write("update_object_state", request)


class CapabilityEngine(Node):
    """Subclass, set the class attributes, and implement `execute()`."""

    display_name: str = ""
    perception_classes: list[str] = []
    required_resources: list[str] = []
    state_reads: list[str] = []
    state_writes: list[str] = []
    intents_handled: list[str] = []
    actions: list[Action] = []

    def __init__(self, engine_id: str):
        # The orchestrator addresses an engine as /<engine_id>/..., so the
        # node name is the engine id.
        super().__init__(engine_id)
        self.declare_parameter("use_orchestrator", True)
        self.declare_parameter("mock", False)
        self.orchestrated = self.get_parameter("use_orchestrator").value
        self.mock = self.get_parameter("mock").value

        self._group = ReentrantCallbackGroup()
        self._busy = threading.Lock()
        self._paused = threading.Event()
        self._cancel = threading.Event()
        self._registered = False
        self._registering = False
        self._active_action = ""
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

    # -- orchestrator plumbing ----------------------------------------------

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
        if self._registered or self._registering:
            return
        client = self._orchestrator_clients["register_engine"]
        if not client.service_is_ready():
            self.get_logger().info(
                "waiting for the orchestrator to register "
                "(run with -p use_orchestrator:=false to run standalone)",
                throttle_duration_sec=10.0,
            )
            return
        self._registering = True
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

    def _call(self, name: str, request, timeout: float = CALL_TIMEOUT_S):
        """Blocking call to an orchestrator service, from inside execute()."""
        if not self.orchestrated:
            self.get_logger().warning(f"standalone: skipping orchestrator call '{name}'")
            return None
        client = self._orchestrator_clients[name]
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"orchestrator service '{name}' unavailable")
            return None
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            self.get_logger().error(f"orchestrator service '{name}' timed out")
            return None
        return future.result()

    def _write(self, name: str, request) -> bool:
        response = self._call(name, request)
        if response is not None and not response.success:
            self.get_logger().warning(f"orchestrator rejected {name}: {response.message}")
        return bool(response and response.success)

    # -- servers ---------------------------------------------------------------

    def _on_activate(self, request, response):
        self.instance_id = request.instance_id
        world = json.loads(request.world_state_snapshot_json or "{}")
        response.availability = self.availability(request.instance_id, world)
        return response

    def _on_hibernate(self, request, response):
        self.instance_id = ""
        self.hibernate()
        response.success = True
        return response

    def _on_control(self, request, response):
        if request.command == Control.Request.PAUSE:
            self._paused.set()
        elif request.command == Control.Request.RESUME:
            self._paused.clear()
        elif request.command == Control.Request.CANCEL:
            self._cancel.set()
        else:
            response.message = f"unknown command {request.command}"
            return response
        response.accepted = True
        response.message = "ok" if self._active_action else "ok (no action running)"
        self._publish_status()
        return response

    def _on_goal(self, goal):
        known = {a.action_id for a in self.actions}
        if goal.action_id not in known or self._busy.locked():
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_execute(self, goal_handle):
        goal = goal_handle.request
        result = Result()
        with self._busy:
            self._cancel.clear()
            self._paused.clear()
            self._active_action = goal.action_id
            self._publish_status()
            try:
                params = json.loads(goal.params_json or "{}")
                payload = self.execute(goal.action_id, params, Context(self, goal_handle))
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
                self._active_action = ""

        # canceled() is only a legal transition after an action-level cancel
        # request; a Control.CANCEL is reported through result.status instead.
        if result.status == Result.STATUS_SUCCESS:
            goal_handle.succeed()
        elif result.status == Result.STATUS_CANCELLED and goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        self._publish_status()
        return ExecuteAction.Result(result=result)

    def _publish_status(self):
        if self._active_action:
            state = "paused" if self._paused.is_set() else "executing"
        else:
            state = "idle"
        status = {
            "state": state,
            "action": self._active_action,
            "instance_id": self.instance_id,
            "mode": "orchestrated" if self.orchestrated else "standalone",
            "registered": self._registered,
            "mock": self.mock,
        }
        self._status.publish(String(data=json.dumps(status)))


def run(engine_class: type[CapabilityEngine], args=None):
    """Entry point for a capability: spin until SIGTERM or Ctrl-C."""
    rclpy.init(args=args)
    node = engine_class()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # A container's PID 1 receives no default action for SIGTERM, so without
    # this handler `docker stop` — and every `sheppy woof` — would wait out the
    # full grace period and then SIGKILL.
    signal.signal(signal.SIGTERM, lambda *_: rclpy.try_shutdown())

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        rclpy.try_shutdown()
