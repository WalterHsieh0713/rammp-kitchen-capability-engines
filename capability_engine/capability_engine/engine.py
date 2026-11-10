"""Base class every kitchen capability builds on.

A capability subclasses `CapabilityEngine`, declares its actions, and writes
`execute()`. This class implements the rest of the RAMMP orchestrator's engine
contract (rammp_orchestrator_interfaces) once, for every capability:

- serves /<id>/activate, /<id>/execute_action, /<id>/control, /<id>/hibernate,
  /<id>/acquire_control, /<id>/release_control
- registers with /rammp_orchestrator/register_engine, retrying until the
  orchestrator appears, and never exiting because it is absent
- publishes /<id>/status (JSON in std_msgs/String) at 1 Hz and on every change
- accepts cancel from Control.CANCEL and from an action cancel alike
- holds the per-execution control token the orchestrator pushes via
  /<id>/acquire_control right before dispatching execute_action, and passes
  it on to self.arm so its commands carry the same token as the arm driver's
  own grant for that execution
- gives each action ROS entities that live only while it runs (self.arm's, and the
  capability's own through create_action_subscription() and friends)
- on SIGTERM or SIGINT, cancels the running action, lets it end, then exits
- lets execute() talk to the UI bridge directly (rammp_ui_bridge_interfaces),
  bypassing the orchestrator, via ctx.update_user_panel()/ctx.poll_ui_event()
  (a live, engine-owned panel -- progress bar, buttons, tappable objects --
  independent of ctx.progress()'s orchestrator-only feedback) and
  ctx.publish_overlay_image() (this capability's own camera view), all using
  the same per-execution token as self.arm, never acquiring one of its own

The same node runs standalone with `-p use_orchestrator:=false`: it skips
registration, and the orchestrator calls on `ctx` become logged no-ops. Drive
it with `ros2 action send_goal /<id>/execute_action ...` (see `make action`).
"""

import json
import os
import secrets
import signal
import threading
import time
from dataclasses import dataclass, field

import rammp_arm
import rclpy
from rammp_common_interfaces.srv import AcquireControl, ReleaseControl
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
from rammp_ui_bridge_interfaces.action import RequestUserInput
from rammp_ui_bridge_interfaces.msg import EngineOverlayImage
from rammp_ui_bridge_interfaces.srv import UpdateUserInputPanel
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
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
        # UI-bridge panel state for this run only (see Context.update_user_panel/
        # poll_ui_event below): the currently-open RequestUserInput goal handle (or
        # None if none is open, e.g. before the first update_user_panel() call, or
        # after the last one ended on a button press), and the most recent
        # not-yet-read button-press/object-tap event. Entirely independent of
        # ctx.progress()'s ExecuteAction feedback.
        self.ui_lock = threading.Lock()
        self.ui_goal_handle = None
        self.ui_pending_event: dict | None = None
        # Set by _ui_close() once execute() has ended. A RequestUserInput goal
        # accepted after that point (e.g. its send was in flight when the run
        # ended) must be cancelled immediately on arrival instead of being
        # left dangling against a token this engine no longer holds.
        self.ui_closed = False


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

    def update_user_panel(
        self,
        panel_json: dict | None = None,
        objects: list[dict] | None = None,
        selected_instance_id: str | None = None,
    ) -> bool:
        """Push panel content directly to the UI bridge (rammp_ui_bridge_interfaces'
        RequestUserInput/UpdateUserInputPanel), fully independent of `progress()`
        above (which only reports to the orchestrator and is never shown to the
        user) -- call either, both, or neither, as this capability needs. Each
        argument left as None is omitted from the update, i.e. "leave unchanged";
        pass one to replace just that aspect (`panel_json` entirely, `objects`
        entirely, or just the highlight). At least one of `panel_json`/`objects`
        must be given the first time this is called in a run, to have something
        for the bridge to open a goal with.

        Non-blocking: opens a RequestUserInput goal (lazily, on first call, or
        again if the previous one just ended -- any button press ends it, see
        poll_ui_event() below) or refreshes the open one via
        UpdateUserInputPanel, then returns immediately without waiting for the
        user to do anything, or for the bridge's service/action server to be
        ready or respond -- readiness is checked immediately (never waited
        for) and the request/response itself is tracked through callbacks, not
        a blocking wait in this thread. Use poll_ui_event() to read back
        button presses and object taps as they happen.

        Returns False (no-op, logged) if this run holds no session token (e.g.
        standalone with no orchestrator) or the bridge's client/server isn't
        immediately ready. A True return means the request was dispatched, not
        that the bridge has necessarily accepted it yet -- a refusal is
        resolved transparently in the background (see _ui_update_panel)."""
        return self._engine._ui_update_panel(
            self, panel_json=panel_json, objects=objects, selected_instance_id=selected_instance_id
        )

    def poll_ui_event(self) -> dict | None:
        """Non-blocking: returns (and consumes) the most recent event the bridge
        has reported since the last call -- {"type": "button", "button_id": ...}
        or {"type": "object_tapped", "instance_id": ...} -- or None if nothing
        new. A button event means the panel's goal has ended; the next
        update_user_panel() call transparently opens a fresh one."""
        return self._engine._ui_poll_event(self)

    def publish_overlay_image(self, image) -> bool:
        """Publish one frame of this capability's own camera (e.g. a wrist cam)
        to the UI bridge's shared overlay topic, stamped with this run's session
        token -- `image` is a sensor_msgs/Image. Returns False (no-op, logged)
        if this run holds no session token."""
        return self._engine._ui_publish_overlay(image)

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
        # Per-execution control token: the orchestrator pushes one shared
        # session token here (and to the arm driver's own ~/acquire_control)
        # right before dispatching this engine's execute_action goal, rather
        # than this engine having to pull one reactively (see
        # RemoteEngineAdapter.push_execution_token() in orchestrator_node.py
        # and _on_acquire_control below). bytes(16) (all-zero) means "nobody
        # holds it".
        self._assigned_token: bytes = bytes(16)
        self._control_generation = 0

        self.create_service(Activate, "~/activate", self._on_activate, callback_group=self._group)
        self.create_service(Control, "~/control", self._on_control, callback_group=self._group)
        self.create_service(
            Hibernate, "~/hibernate", self._on_hibernate, callback_group=self._group
        )
        self.create_service(
            AcquireControl,
            "~/acquire_control",
            self._on_acquire_control,
            callback_group=self._group,
        )
        self.create_service(
            ReleaseControl,
            "~/release_control",
            self._on_release_control,
            callback_group=self._group,
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

        # Direct engine<->UI-bridge wiring (rammp-UI-bridge's
        # docs/ENGINE_UI_INTEGRATION_GUIDE.md section 2): bypasses the
        # orchestrator entirely, and reuses the per-execution token pushed to
        # self._assigned_token above instead of this engine acquiring its own
        # ui_control grant -- see Context.update_user_panel/poll_ui_event/
        # publish_overlay_image and the _ui_* helpers below.
        self.declare_parameter("ui_bridge_node_name", "ui_bridge_node")
        ui_ns = f"/{self.get_parameter('ui_bridge_node_name').value}"
        self._ui_request_client = ActionClient(
            self, RequestUserInput, f"{ui_ns}/request_user_input", callback_group=self._group
        )
        self._ui_update_client = self.create_client(
            UpdateUserInputPanel, f"{ui_ns}/request_user_input/update", callback_group=self._group
        )
        # One shared topic, not per-engine -- see EngineOverlayImage.msg's own
        # doc comment; the bridge tells engines apart by the token on each frame.
        self._ui_overlay_pub = self.create_publisher(
            EngineOverlayImage, "/engine_overlay/image_raw", 10
        )

        self._status = self.create_publisher(String, "~/status", 10)
        self.create_timer(1.0, self._publish_status, callback_group=self._group)
        if self.orchestrated:
            self.create_timer(REGISTER_RETRY_S, self._try_register, callback_group=self._group)
            self._try_register()

        mode = "orchestrated" if self.orchestrated else "standalone"
        self.get_logger().info(f"{engine_id} up ({mode}{', mock' if self.mock else ''})")

    # -- what a capability implements ---------------------------------------

    def availability(self, instance_id: str, world_state: dict) -> list[ActionAvailability]:
        """Which actions can run on this object now. Default: every action
        whose own `perception_classes` applies to this instance's class (or
        declares none, i.e. applies to every class), all reported
        available. Mirrors `MockEngine.activate()`'s reference semantics
        (`rammp-orchestrator`'s `mocks/mock_engine.py`): an action whose
        `perception_classes` doesn't include this instance's class is
        omitted entirely (not just marked unavailable), since it doesn't
        apply to this object at all, not merely blocked right now -- e.g. a
        fridge-only action on a selected microwave instance. The instance's
        class is looked up from `world_state["objects"][instance_id]
        ["perception_class"]`; unknown (not yet reported by Perception, or
        no such key) falls back to offering every action, same as
        `MockEngine`. Object-state-dependent gating (e.g. "open" unavailable
        once already open) is not implemented here -- override this method
        to add it, using `world_state` the same way."""
        instance = world_state.get("objects", {}).get(instance_id, {})
        instance_class = instance.get("perception_class")
        availability = []
        for a in self.actions:
            if (
                a.perception_classes
                and instance_class is not None
                and (instance_class not in a.perception_classes)
            ):
                continue
            availability.append(ActionAvailability(action_id=a.action_id, available=True))
        return availability

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

    # -- UI-bridge plumbing (Context.update_user_panel/poll_ui_event/publish_overlay_image) --
    #
    # Everything here is genuinely non-blocking, matching ctx.update_user_panel()'s
    # docstring: readiness is checked immediately (never waited for), and every
    # send's outcome (accept/refusal, result) is handled by a done-callback
    # rather than a blocking wait in the caller's (execute()'s) thread.

    def _ui_update_panel(
        self,
        ctx: Context,
        *,
        panel_json: dict | None,
        objects: list[dict] | None,
        selected_instance_id: str | None,
    ) -> bool:
        run = ctx._run
        token = self._assigned_token
        if token == bytes(16):
            self.get_logger().warning("standalone/no session token: skipping update_user_panel")
            return False
        request_payload: dict = {}
        if panel_json is not None:
            request_payload["panel_json"] = json.dumps(panel_json)
        if objects is not None:
            request_payload["objects"] = objects
        if selected_instance_id is not None:
            request_payload["selected_instance_id"] = selected_instance_id

        with run.ui_lock:
            goal_handle = run.ui_goal_handle
        if goal_handle is not None:
            if not self._ui_update_client.service_is_ready():
                self.get_logger().warning(
                    "ui_bridge_node's ~/request_user_input/update unavailable"
                )
                return False
            future = self._ui_update_client.call_async(
                UpdateUserInputPanel.Request(
                    token=list(token), request_json=json.dumps(request_payload)
                )
            )
            future.add_done_callback(
                lambda f: self._ui_on_update_response(run, token, request_payload, f)
            )
            return True

        if "panel_json" not in request_payload and "objects" not in request_payload:
            self.get_logger().warning(
                "update_user_panel: need panel_json or objects to open a new goal"
            )
            return False
        return self._ui_open_goal(run, request_payload, token)

    def _ui_on_update_response(
        self, run: "_Run", token: bytes, request_payload: dict, future
    ) -> None:
        """Done-callback for the UpdateUserInputPanel call `_ui_update_panel`
        fires without waiting for. Reopens a fresh goal in the background if
        the update was refused -- most likely the previous goal just ended (a
        button press, or the bridge force-aborted it) since we last checked --
        same as if this were the first call."""
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(
                f"ui_bridge_node's ~/request_user_input/update failed: {error!r}"
            )
            return
        if response.accepted:
            return
        self.get_logger().info(f"update_user_panel update refused, reopening: {response.message}")
        with run.ui_lock:
            run.ui_goal_handle = None
        self._ui_open_goal(run, request_payload, token)

    def _ui_open_goal(self, run: "_Run", request_payload: dict, token: bytes) -> bool:
        if not self._ui_request_client.server_is_ready():
            self.get_logger().warning("ui_bridge_node's ~/request_user_input server unavailable")
            return False
        goal = RequestUserInput.Goal(
            engine_id=self.get_name(), token=list(token), request_json=json.dumps(request_payload)
        )
        send_future = self._ui_request_client.send_goal_async(
            goal, feedback_callback=lambda feedback_msg: self._ui_on_feedback(run, feedback_msg)
        )
        # Always tracked via this callback, not a blocking wait on
        # send_future -- so a goal accepted after a slow send is never
        # abandoned/untracked, however long it actually takes.
        send_future.add_done_callback(lambda f: self._ui_on_goal_response(run, f))
        return True

    def _ui_on_goal_response(self, run: "_Run", send_future) -> None:
        try:
            goal_handle = send_future.result()
        except Exception as error:
            self.get_logger().warning(f"RequestUserInput goal send failed: {error!r}")
            return
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning("RequestUserInput goal rejected")
            return
        with run.ui_lock:
            if run.ui_closed:
                # The run ended (_ui_close() already ran) while this goal's
                # acceptance was still in flight -- don't leave it dangling
                # against a token this engine no longer holds.
                already_closed = True
            else:
                run.ui_goal_handle = goal_handle
                already_closed = False
        if already_closed:
            goal_handle.cancel_goal_async()
            return
        goal_handle.get_result_async().add_done_callback(
            lambda result_future: self._ui_on_result(run, goal_handle, result_future)
        )

    def _ui_on_feedback(self, run: "_Run", feedback_msg) -> None:
        try:
            payload = json.loads(feedback_msg.feedback.feedback_json)
        except (json.JSONDecodeError, AttributeError):
            return
        if payload.get("event_type") == "object_tapped":
            with run.ui_lock:
                run.ui_pending_event = {
                    "type": "object_tapped",
                    "instance_id": payload.get("instance_id", ""),
                }

    def _ui_on_result(self, run: "_Run", goal_handle, result_future) -> None:
        """A RequestUserInput goal always ends on a button press, an engine
        cancel, or a bridge-forced abort (see the .action file's own doc
        comment) -- there's no "button that keeps the goal open" concept.
        Record the outcome, and clear ui_goal_handle only if it is still this
        goal's handle: update_user_panel() may already have opened a newer
        goal (e.g. after this one was refused) by the time this result
        arrives, and must not have that replacement's handle wiped out from
        under it."""
        try:
            payload = json.loads(result_future.result().result.result_json)
        except Exception as error:
            payload = {"status": "aborted", "reason": repr(error)}
        with run.ui_lock:
            if run.ui_goal_handle is goal_handle:
                run.ui_goal_handle = None
            if payload.get("status") == "button":
                run.ui_pending_event = {"type": "button", "button_id": payload.get("button_id", "")}
            elif payload.get("status") == "aborted":
                self.get_logger().warning(
                    f"RequestUserInput goal aborted: {payload.get('reason', '')}"
                )

    def _ui_poll_event(self, ctx: Context) -> dict | None:
        run = ctx._run
        with run.ui_lock:
            event = run.ui_pending_event
            run.ui_pending_event = None
            return event

    def _ui_publish_overlay(self, image) -> bool:
        token = self._assigned_token
        if token == bytes(16):
            self.get_logger().warning("standalone/no session token: skipping publish_overlay_image")
            return False
        self._ui_overlay_pub.publish(EngineOverlayImage(image=image, token=list(token)))
        return True

    def _ui_close(self, run: "_Run") -> None:
        """Called once execute() ends: cancels any still-open panel goal rather
        than leaving it dangling against a token this engine no longer holds.
        Also marks the run closed, so a goal whose acceptance is still in
        flight (see _ui_on_goal_response) is cancelled immediately on arrival
        instead of being left dangling too."""
        with run.ui_lock:
            goal_handle = run.ui_goal_handle
            run.ui_goal_handle = None
            run.ui_pending_event = None
            run.ui_closed = True
        if goal_handle is not None:
            goal_handle.cancel_goal_async()

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

    def _on_acquire_control(self, request, response):
        """Grants this engine's own per-execution control token -- see the
        ``__init__`` comment above ``self._assigned_token``. Always SEIZES
        (same contract as every other ``AcquireControl`` server in this
        project), since by design only the orchestrator ever calls this,
        right before dispatching an ``ExecuteAction`` goal to this same
        engine -- there is no legitimate concurrent holder to protect
        against. A supplied ``request.token`` (non-zero) is used verbatim as
        the shared session key instead of minting a fresh one, so this
        engine's commands carry the same token as the arm driver's own
        grant for the same execution."""
        if bytes(request.token) == bytes(16):
            token = secrets.token_bytes(16)
        else:
            token = bytes(request.token)
        self._assigned_token = token
        self._control_generation += 1
        response.accepted = True
        response.token = list(token)
        response.generation = self._control_generation
        response.message = f"granted to '{request.owner_id}'"
        self.get_logger().info(
            f"acquire_control: '{request.owner_id}' assigned token "
            f"{token.hex()[:8]}... (generation {self._control_generation})"
        )
        return response

    def _on_release_control(self, request, response):
        if bytes(request.token) == self._assigned_token and self._assigned_token != bytes(16):
            self._assigned_token = bytes(16)
            response.released = True
            response.message = ""
            self.get_logger().info("release_control: token released")
        else:
            response.released = False
            response.message = "token mismatch or already released"
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
                # Use whatever token the orchestrator pushed via ~/acquire_control
                # right before this goal was dispatched (bytes(16)/all-zero means
                # "nobody holds it", e.g. running standalone with no orchestrator).
                token = self._assigned_token if self._assigned_token != bytes(16) else None
                self.arm.begin_action(ctx, token=token)
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
                # Cancel any still-open UI-bridge panel goal (Context.update_user_panel)
                # rather than leaving it dangling against a token about to be released.
                self._ui_close(run)
                # Clear the pushed session token now that this execution is over: the
                # orchestrator reclaims/replaces its copy on the arm driver independently,
                # but this engine's own copy (self._assigned_token) has no other owner to
                # reset it, so do it here rather than leaving a stale-but-still-valid token
                # sitting around until the next execution's push overwrites it.
                self._assigned_token = bytes(16)

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
