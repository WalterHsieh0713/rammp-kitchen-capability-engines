"""self.arm in the engine: per-action resources, parameters, the sim model, standalone RammpArm
and mock mode. Each engine runs beside a FakeDriver and is driven through its execute_action."""

import itertools
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import rammp_arm
from fake_driver import FLATBOT_URDF, FakeDriver, counts
from geometry_msgs.msg import TransformStamped
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rammp_orchestrator_interfaces.srv import (
    Activate,
    Control,
    Hibernate,
    RegisterEngine,
    RequestIntent,
)
from rcl_interfaces.msg import Log
from rclpy.action import ActionClient, GoalResponse
from rclpy.duration import Duration as RclpyDuration
from rclpy.exceptions import InvalidHandle
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time as RclpyTime
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from capability_engine import arm as arm_module
from capability_engine import engine as engine_module
from capability_engine import sim
from capability_engine.action_scope import TF_STATIC_QOS, ScopeClosed
from capability_engine.arm_params import PARAMS
from capability_engine.driver_state import ArmError, NotInAction
from capability_engine.engine import ORCHESTRATOR, Action, CapabilityEngine

WAIT_S = 5.0  # how long a test waits for a server, a goal's acceptance or a node to go
ACTION_WAIT_S = 30.0  # how long a test waits for an action's result
POLL_S = 0.01
# One engine id per test: Humble defers a node's teardown, and a lingering node with the same
# name takes the shared /rosout publisher with it, losing the next engine's log lines.
ENGINE_IDS = itertools.count()
TARGET = [0.3] * 7  # the joint target of the tests' moves
STRUCTURE = {"JOINTS", "CONTINUOUS", "ERRORS", "IMPEDANCE"}  # rammp_arm constants, not tunables
CART_STEP = np.array([0.02, 0.0, 0.0])  # the mock Cartesian move: 2 cm along x
PRE_STEP = np.array([0.0, 0.05, -0.05])  # the mock go_to_ee_pose: from the start to a line-up
UNREACHABLE = [5.0, 0.0, 0.0]  # a tool position 5 m out: no IK solution
SHORT_WAIT_S = 0.5  # STATE_WAIT_S in the test of a driver with no trajectory server
REGISTER_RETRY_TEST_S = 0.1  # REGISTER_RETRY_S in the registration test
REGISTER_TIMEOUT_TEST_S = 0.5  # CALL_TIMEOUT_S there: how long an unanswered call is waited on
SHUTDOWN_LIMIT_S = 5.0  # how long a SIGTERMed engine may take to exit
REPLY_WAIT_S = 1.0  # how long the SIGTERM test waits for one Control reply
INTERRUPT_AFTER_S = 0.3  # when the lifecycle tests interrupt a running action
START_WAIT_TEST_S = 0.2  # START_WAIT_S in the never-started test
# The stuck-action test's stop budget, set through the arm.* parameters it is computed from.
FAST_SETTLE_ARGS = ["-p", "arm.goal_accept_s:=0.1", "-p", "arm.cancel_settle_s:=0.1"]
FAST_SETTLE_ARGS += ["-p", "arm.close_stream_wait_s:=0.1", "-p", "arm.go_to_settle_s:=0.1"]
EXECUTE_DELAY_S = 0.5  # how long the two-goal test holds back each goal's execute


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(POLL_S)


class ArmEngine(CapabilityEngine):
    """One action, "run", whose execute() calls `body(engine, ctx)`."""

    actions = [Action("run", "Run")]

    def __init__(self, engine_id, body, **node_kwargs):
        super().__init__(engine_id, **node_kwargs)
        self.body = body

    def execute(self, action_id, params, ctx):
        return self.body(self, ctx)


@pytest.fixture
def env(ros):
    """make(body, **params) -> a spun ArmEngine; run(engine) -> its "run" action's Result."""
    saved = [(prm.target, getattr(*prm.target)) for prm in PARAMS]
    sim.reset_for_tests()
    fake = FakeDriver()
    client_node = Node("engine_arm_test_client")
    logs = []  # every /rosout line: mock mode's "not sent" is only logged
    client_node.create_subscription(Log, "/rosout", logs.append, 100)
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    executor.add_node(client_node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    engines = []
    engine_id = f"arm_test_engine_{next(ENGINE_IDS)}"

    def make(body, **params):
        overrides = [Parameter("use_orchestrator", value=False)]
        overrides += [Parameter(name, value=value) for name, value in params.items()]
        engine = ArmEngine(engine_id, body, parameter_overrides=overrides)
        engines.append(engine)
        executor.add_node(engine)
        return engine

    def run(engine):
        client = ActionClient(client_node, ExecuteAction, f"/{engine_id}/execute_action")
        try:
            assert client.wait_for_server(timeout_sec=WAIT_S)
            send = client.send_goal_async(ExecuteAction.Goal(action_id="run"))
            wait_until(send.done)
            assert send.result().accepted
            result = send.result().get_result_async()
            wait_until(result.done, ACTION_WAIT_S)
            return result.result().result.result
        finally:
            client.destroy()

    yield SimpleNamespace(
        fake=fake, make=make, run=run, logs=logs, engine_id=engine_id, executor=executor
    )
    for engine in engines:
        engine.stop()  # cancels an action a failed test left running, so shutdown() returns
    executor.shutdown()
    spinner.join()
    for engine in engines:
        engine.destroy_node()
    client_node.destroy_node()
    fake.destroy_node()
    sim.reset_for_tests()
    for (module, attr), value in saved:
        setattr(module, attr, value)


def test_engine_has_arm_and_releases_everything(env):
    seen = {}

    def body(engine, ctx):
        seen["state"] = engine.arm.get_state()
        seen["moved"] = engine.arm.move_joints([TARGET])
        seen["counts"] = counts(engine)

    engine = env.make(body)
    base = counts(engine)  # (subscriptions, clients, action clients and waitables, publishers)
    assert env.run(engine).status == Result.STATUS_SUCCESS
    assert seen["moved"]
    assert seen["counts"] != base
    assert counts(engine) == base
    assert len(env.fake.goals["execute_joint_trajectory"]) == 1


def test_token_none_until_orchestrator_sends_one(env):
    engine = env.make(lambda engine, ctx: {"moved": engine.arm.move_joints([TARGET])})
    assert env.run(engine).status == Result.STATUS_SUCCESS
    goal = env.fake.goals["execute_joint_trajectory"][-1]
    assert list(goal.token) == [0] * 16
    assert goal.sender_id == env.engine_id  # SENDER_ID comes from the capability


def test_arm_outside_execute_raises(env):
    engine = env.make(lambda engine, ctx: {})
    base = counts(engine)
    errors = []

    def tick():
        timer.cancel()
        try:
            engine.arm.get_state()
        except Exception as e:  # noqa: BLE001 -- recorded for the assertion below
            errors.append(e)

    timer = engine.create_timer(POLL_S, tick)
    wait_until(lambda: errors)
    assert isinstance(errors[0], NotInAction)
    assert "only available inside an action" in str(errors[0])
    engine.destroy_timer(timer)
    assert counts(engine) == base


def test_close_is_noop_under_engine(env):
    seen = {}

    def body(engine, ctx):
        engine.arm.close()
        seen["state"] = engine.arm.get_state()

    engine = env.make(body)
    base = counts(engine)
    engine.arm.close()  # outside an action too
    assert env.run(engine).status == Result.STATUS_SUCCESS
    assert set(seen["state"]) == {"position", "velocity", "effort", "ee_pos", "gripper_pos"}
    assert counts(engine) == base


def test_params_declared_with_defaults(env):
    targets = {prm.target for prm in PARAMS}
    for name in dir(rammp_arm):
        if name.isupper() and name not in STRUCTURE and not callable(getattr(rammp_arm, name)):
            assert (rammp_arm, name) in targets, name
    defaults = {prm.name: prm.default for prm in PARAMS}
    assert defaults["state_wait_s"] == 10.0
    assert defaults["state_stale_s"] == 0.5
    assert defaults["stream_timeout_s"] == 0.1
    assert defaults["mock_joints"] == [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
    for prm in PARAMS:
        assert prm.name == prm.target[1].lower()
        assert prm.description
        assert getattr(*prm.target) == prm.default, prm.name

    engine = env.make(lambda engine, ctx: {}, **{"arm.cart_speed": 0.08})
    for prm in PARAMS:
        name = f"arm.{prm.name}"
        assert engine.has_parameter(name), name
        assert engine.describe_parameter(name).description == prm.description
        value = engine.get_parameter(name).value
        if prm.name == "cart_speed":
            assert value == 0.08
        elif prm.name == "sender_id":
            assert value == env.engine_id
        elif isinstance(prm.default, list):
            assert list(value) == prm.default, name
        else:
            assert value == prm.default, name
    assert rammp_arm.CART_SPEED == 0.08
    assert rammp_arm.SENDER_ID == env.engine_id


def test_arm_params_refused_after_startup(env):
    engine = env.make(lambda engine, ctx: {})
    (res,) = engine.set_parameters([Parameter("arm.cart_speed", value=0.2)])
    assert not res.successful
    assert res.reason == "arm.* parameters are read at startup; restart to change"
    assert engine.get_parameter("arm.cart_speed").value == rammp_arm.CART_SPEED
    assert rammp_arm.CART_SPEED != 0.2


def test_sim_provider_uses_robot_description(env):
    seen = {}

    def body(engine, ctx):
        _, rb = sim.make_sim()
        seen["robot"] = p.getBodyInfo(rb.robot_id, physicsClientId=rb.physics_client_id)[1]
        try:
            sim.make_sim("/fixtures/cornell/robot.urdf")
        except sim.SimModelError as e:
            seen["explicit"] = e

    engine = env.make(body)
    assert env.run(engine).status == Result.STATUS_SUCCESS
    assert seen["robot"].decode() == ET.parse(FLATBOT_URDF).getroot().get("name")
    assert "call it with no path" in str(seen["explicit"])


def assert_destroyed(node, other):
    """`node` is destroyed: its handle is gone and the graph (seen from `other`) lost it."""
    with pytest.raises(InvalidHandle):
        node.get_name()
    wait_until(lambda: "rammp_arm_test" not in other.get_node_names())


def test_standalone_rammp_arm(env):
    arm = rammp_arm.RammpArm(node_name="rammp_arm_test")
    node = arm.node
    try:
        assert sim.has_model_provider()
        assert set(arm.get_state()) == {"position", "velocity", "effort", "ee_pos", "gripper_pos"}
        assert arm.move_joints([TARGET])
        assert arm.monitor_stopped is False
        assert isinstance(arm.last_path, list)
        goal = env.fake.goals["execute_joint_trajectory"][-1]
        assert goal.sender_id == "microwave"
    finally:
        arm.close()
    assert not sim.has_model_provider()  # the closed arm's provider is gone with it
    assert_destroyed(node, env.fake)
    wait_until(lambda: env.fake.count_subscribers("/joint_states") == 0)


def test_standalone_rammp_arm_needs_trajectory_server(ros, monkeypatch):
    monkeypatch.setattr(arm_module, "STATE_WAIT_S", SHORT_WAIT_S)
    sim.reset_for_tests()
    fake = FakeDriver(without_actions=("execute_joint_trajectory",))
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        # An earlier test's fake driver may linger in the graph until discovery drops it.
        send_goal = "/execute_joint_trajectory/_action/send_goal"
        wait_until(lambda: send_goal not in dict(fake.get_service_names_and_types()))
        with pytest.raises(RuntimeError) as raised:
            rammp_arm.RammpArm(node_name="rammp_arm_test")
        assert type(raised.value) is RuntimeError
        assert str(raised.value) == "no /execute_joint_trajectory action server"
        assert not sim.has_model_provider()  # the failed constructor closed itself
        wait_until(lambda: "rammp_arm_test" not in fake.get_node_names())
        wait_until(lambda: fake.count_subscribers("/joint_states") == 0)
    finally:
        executor.shutdown()
        spinner.join()
        fake.destroy_node()
        sim.reset_for_tests()


def test_mock_dry_run(env):
    seen = {}

    def body(engine, ctx):
        arm = engine.arm
        seen["start"] = arm.get_state()
        seen["moved"] = arm.move_joints([TARGET])
        seen["after_move"] = arm.get_state()["position"]
        seen["goto"] = arm.go_to_joint_config(np.zeros(7))
        seen["after_goto"] = arm.raw_joints()
        seen["pose"] = arm.go_to_ee_pose([0.4, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])
        seen["after_pose"] = arm.raw_joints()
        seen["gripper"] = arm.gripper(1.0)
        seen["grip"] = arm.get_state()["gripper_pos"]
        seen["not_ready"] = arm.not_ready()
        with arm.stream("ee_twist") as stream:
            stream.send(np.zeros(6))
        try:
            sim.make_sim()
        except sim.SimModelError as e:
            seen["make_sim"] = e
        seen["counts"] = counts(engine)

    engine = env.make(body, mock=True)
    base = counts(engine)
    assert env.run(engine).status == Result.STATUS_SUCCESS
    mock_joints = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
    np.testing.assert_allclose(seen["start"]["position"], mock_joints)
    np.testing.assert_allclose(seen["start"]["ee_pos"], [0, 0, 0, 0, 0, 0, 1])  # no sim yet
    assert seen["moved"] and seen["goto"] and seen["pose"] and seen["gripper"]
    np.testing.assert_allclose(seen["after_move"], TARGET, atol=1e-9)
    np.testing.assert_allclose(seen["after_goto"], np.zeros(7))
    np.testing.assert_allclose(seen["after_pose"], np.zeros(7))  # a pose target leaves them
    assert seen["grip"] == 1.0
    assert seen["not_ready"] is None
    assert str(seen["make_sim"]) == (
        "mock mode has no driver model: set arm.mock_urdf to a URDF file"
    )
    wait_until(
        lambda: any(
            m.name == env.engine_id and m.msg == "  trajectory: mock: not sent" for m in env.logs
        )
    )
    assert seen["counts"] == base  # no ROS entity to the driver at all
    assert counts(engine) == base
    assert not any(env.fake.goals.values())
    assert not any(env.fake.requests.values())
    assert not any(env.fake.setpoints.values())


def test_mock_tool_pose_from_sim(env):
    seen = {}

    def body(engine, ctx):
        arm = engine.arm
        ee = arm.get_state()["ee_pos"]  # FK on the sim built from arm.mock_urdf
        _, rb = sim.make_sim()
        seen["robot"] = p.getBodyInfo(rb.robot_id, physicsClientId=rb.physics_client_id)[1]
        target = ee[:3] + CART_STEP
        seen["moved"] = arm.move_cartesian([(target, ee[3:])])
        seen["start"], seen["target"], seen["ee"] = ee, target, arm.get_state()["ee_pos"]

    engine = env.make(body, **{"mock": True, "arm.mock_urdf": str(FLATBOT_URDF)})
    assert env.run(engine).status == Result.STATUS_SUCCESS
    assert seen["robot"].decode() == ET.parse(FLATBOT_URDF).getroot().get("name")
    assert np.linalg.norm(seen["start"][:3]) > CART_STEP[0]  # not the no-model origin
    assert seen["moved"]
    np.testing.assert_allclose(seen["ee"][:3], seen["target"], atol=sim.MAX_IK_ERR)
    assert not any(env.fake.goals.values())


def test_mock_go_to_ee_pose_then_move_cartesian(env):
    """The spec's flow in mock: go_to_ee_pose moves the mock joints to the pose's IK solution,
    so the move_cartesian after it starts there."""
    seen = {}

    def body(engine, ctx):
        arm = engine.arm
        ee = arm.get_state()["ee_pos"]
        q0 = arm.raw_joints()
        seen["unreachable"] = arm.go_to_ee_pose(UNREACHABLE, ee[3:])
        seen["q_unreachable"], seen["q0"] = arm.raw_joints(), q0
        pre = ee[:3] + PRE_STEP
        seen["goto"] = arm.go_to_ee_pose(pre, ee[3:])
        seen["at_pre"] = arm.get_state()["ee_pos"]
        target = pre + CART_STEP
        seen["moved"] = arm.move_cartesian([(target, ee[3:])])
        seen["pre"], seen["target"], seen["ee"] = pre, target, arm.get_state()["ee_pos"]

    engine = env.make(body, **{"mock": True, "arm.mock_urdf": str(FLATBOT_URDF)})
    assert env.run(engine).status == Result.STATUS_SUCCESS
    assert seen["unreachable"]  # a mock motion succeeds, but the joints stay
    np.testing.assert_allclose(seen["q_unreachable"], seen["q0"])
    assert seen["goto"] and seen["moved"]
    np.testing.assert_allclose(seen["at_pre"][:3], seen["pre"], atol=sim.MAX_IK_ERR)
    np.testing.assert_allclose(seen["ee"][:3], seen["target"], atol=sim.MAX_IK_ERR)
    assert not any(env.fake.goals.values())


def test_mock_raw_trajectory(env):
    seen = {}
    names = [f"joint_{i}" for i in range(7, 0, -1)]  # cuRobo's names, J7 first
    q = np.arange(1, 8) / 10.0  # J1..J7

    def body(engine, ctx):
        arm = engine.arm
        res = arm.execute_joint_trajectory(names, [(1.0, q[::-1], np.zeros(7))])
        seen["code"], seen["q"] = res.error_code, arm.raw_joints()
        try:
            arm.execute_joint_trajectory(names, [(1.0, q[:3], np.zeros(3))])
        except ArmError as e:
            seen["short"] = e

    engine = env.make(body, mock=True)
    assert env.run(engine).status == Result.STATUS_SUCCESS
    assert seen["code"] == 0
    np.testing.assert_allclose(seen["q"], q)
    assert type(seen["short"]) is ArmError
    assert "3 positions for 7 joints" in str(seen["short"])
    assert not any(env.fake.goals.values())


# -- the engine's own lifecycle: goals, registration, shutdown ------------------------------------


def test_two_goals_at_once_accept_only_one(env, monkeypatch):
    execute = CapabilityEngine._on_execute

    def late_execute(self, goal_handle):  # both goals are decided before either one executes
        time.sleep(EXECUTE_DELAY_S)
        return execute(self, goal_handle)

    monkeypatch.setattr(CapabilityEngine, "_on_execute", late_execute)
    release = threading.Event()
    env.make(lambda engine, ctx: release.wait(WAIT_S))
    client_node = Node(f"{env.engine_id}_goals")
    executor = MultiThreadedExecutor()
    executor.add_node(client_node)
    threading.Thread(target=executor.spin, daemon=True).start()
    client = ActionClient(client_node, ExecuteAction, f"/{env.engine_id}/execute_action")
    try:
        assert client.wait_for_server(timeout_sec=WAIT_S)
        sends = [client.send_goal_async(ExecuteAction.Goal(action_id="run")) for _ in range(2)]
        wait_until(lambda: all(s.done() for s in sends))
        accepted = [s.result() for s in sends if s.result().accepted]
        assert len(accepted) == 1
        release.set()
        result = accepted[0].get_result_async()
        wait_until(result.done, ACTION_WAIT_S)
    finally:
        release.set()
        client.destroy()
        executor.shutdown()
        client_node.destroy_node()


def registrar(name, calls):
    """A node serving /rammp_orchestrator/register_engine that records each call. Until it is
    spun, the service is discoverable but never replies."""
    node = Node(name)

    def on_register(request, response):
        calls.append(request.manifest.engine_id)
        response.accepted = True
        response.message = "registered"
        return response

    node.create_service(RegisterEngine, f"{ORCHESTRATOR}/register_engine", on_register)
    return node


def test_reregisters_after_the_orchestrator_restarts(env, monkeypatch):
    monkeypatch.setattr(engine_module, "REGISTER_RETRY_S", REGISTER_RETRY_TEST_S)
    monkeypatch.setattr(engine_module, "CALL_TIMEOUT_S", REGISTER_TIMEOUT_TEST_S)
    calls = []
    silent = registrar(f"{env.engine_id}_silent", [])  # never spun: never answers
    engine = env.make(lambda engine, ctx: None, use_orchestrator=True)
    client = engine._orchestrator_clients["register_engine"]
    sent = []
    call_async = client.call_async
    monkeypatch.setattr(client, "call_async", lambda req: sent.append(req) or call_async(req))
    # An orchestrator that never answers: the engine asks again after CALL_TIMEOUT_S.
    wait_until(lambda: len(sent) >= 2)
    assert not engine._registered
    silent.destroy_node()
    orchestrator = registrar(f"{env.engine_id}_orchestrator", calls)
    env.executor.add_node(orchestrator)
    wait_until(lambda: engine._registered)
    assert calls == [env.engine_id]
    # It restarts and forgets every registration: once it is gone the engine registers again.
    env.executor.remove_node(orchestrator)
    orchestrator.destroy_node()
    wait_until(lambda: not engine._registered)
    restarted = registrar(f"{env.engine_id}_restarted", calls)
    env.executor.add_node(restarted)
    wait_until(lambda: engine._registered)
    assert calls == [env.engine_id, env.engine_id]
    env.executor.remove_node(restarted)
    restarted.destroy_node()


SHUTDOWN_SCRIPT = """
import time
from capability_engine.engine import Action, CapabilityEngine, run


class Paused(CapabilityEngine):
    actions = [Action("run", "Run")]

    def __init__(self):
        super().__init__("{engine_id}")

    def execute(self, action_id, params, ctx):
        while True:
            ctx.checkpoint()
            time.sleep(0.05)


run(Paused)
"""


MOVING_SCRIPT = """
from capability_engine.engine import Action, CapabilityEngine, run


class Moving(CapabilityEngine):
    actions = [Action("run", "Run")]

    def __init__(self):
        super().__init__("{engine_id}")

    def execute(self, action_id, params, ctx):
        return {{"moved": self.arm.move_joints([[0.3] * 7])}}


run(Moving)
"""


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT], ids=["sigterm", "sigint"])
def test_a_signal_mid_motion_cancels_the_goal_at_the_driver(env, tmp_path, sig):
    """docker stop (SIGINT, the image's STOPSIGNAL) or a SIGTERM while a trajectory runs: the
    engine cancels the goal at the driver and reports the action's result before ROS shuts down,
    rather than leaving the arm moving or failing on a dead context."""
    env.fake.hold_goals(ACTION_WAIT_S)
    engine_id = f"arm_test_engine_{next(ENGINE_IDS)}"
    script = tmp_path / "moving.py"
    script.write_text(MOVING_SCRIPT.format(engine_id=engine_id))
    proc = subprocess.Popen(
        [sys.executable, str(script), "--ros-args", "-p", "use_orchestrator:=false"],
        stderr=subprocess.PIPE,
        text=True,
    )
    node = Node(f"{engine_id}_client")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    client = ActionClient(node, ExecuteAction, f"/{engine_id}/execute_action")
    try:
        assert client.wait_for_server(timeout_sec=ACTION_WAIT_S)
        client.send_goal_async(ExecuteAction.Goal(action_id="run"))  # its reply may be lost
        wait_until(lambda: env.fake.goals["execute_joint_trajectory"], ACTION_WAIT_S)
        proc.send_signal(sig)
        _, logs = proc.communicate(timeout=SHUTDOWN_LIMIT_S)
        assert env.fake.cancelled("execute_joint_trajectory")
        assert "Error" not in logs, logs
    finally:
        proc.kill()
        proc.wait()
        client.destroy()
        executor.shutdown()
        node.destroy_node()


def reply(client, request):
    """`client`'s reply message to `request`, or None when none arrives within REPLY_WAIT_S."""
    future = client.call_async(request)
    deadline = time.monotonic() + REPLY_WAIT_S
    while not future.done() and time.monotonic() < deadline:
        time.sleep(POLL_S)
    return future.result().message if future.done() else None


def test_sigterm_exits_while_an_action_is_paused(ros, tmp_path):
    engine_id = f"arm_test_engine_{next(ENGINE_IDS)}"
    script = tmp_path / "paused.py"
    script.write_text(SHUTDOWN_SCRIPT.format(engine_id=engine_id))
    proc = subprocess.Popen(
        [sys.executable, str(script), "--ros-args", "-p", "use_orchestrator:=false"]
    )
    node = Node(f"{engine_id}_client")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    client = ActionClient(node, ExecuteAction, f"/{engine_id}/execute_action")
    control = node.create_client(Control, f"/{engine_id}/control")
    try:
        assert client.wait_for_server(timeout_sec=ACTION_WAIT_S)
        assert control.wait_for_service(timeout_sec=WAIT_S)
        # Humble may drop a reply sent before the server has discovered the client, so neither
        # the goal's acceptance nor one Control reply is waited on: PAUSE is repeated until the
        # engine answers that it paused a running action.
        client.send_goal_async(ExecuteAction.Goal(action_id="run"))
        pause = Control.Request(command=Control.Request.PAUSE)
        wait_until(lambda: reply(control, pause) == "ok", ACTION_WAIT_S)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=SHUTDOWN_LIMIT_S)  # TimeoutExpired: the paused action held the exit
    finally:
        proc.kill()
        proc.wait()
        client.destroy()
        executor.shutdown()
        node.destroy_node()


# -- the lifecycle, interrupted: every way an action can be cut short ------------------------------


class GoalClient:
    """A client node for one engine: sends its "run" goal, calls its control service."""

    def __init__(self, engine_id):
        self.node = Node(f"{engine_id}_lifecycle_client")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        threading.Thread(target=self.executor.spin, daemon=True).start()
        self.action = ActionClient(self.node, ExecuteAction, f"/{engine_id}/execute_action")
        self.control_client = self.node.create_client(Control, f"/{engine_id}/control")
        assert self.action.wait_for_server(timeout_sec=WAIT_S)
        assert self.control_client.wait_for_service(timeout_sec=WAIT_S)

    def send(self):
        """The goal's handle: accepted or not."""
        send = self.action.send_goal_async(ExecuteAction.Goal(action_id="run"))
        wait_until(send.done)
        return send.result()

    def result(self, handle):
        future = handle.get_result_async()
        wait_until(future.done, ACTION_WAIT_S)
        return future.result().result.result

    def control(self, command):
        future = self.control_client.call_async(Control.Request(command=command))
        wait_until(future.done)
        return future.result()

    def close(self):
        self.action.destroy()
        self.executor.shutdown()
        self.node.destroy_node()


def trajectory_body(engine, ctx):
    return {"moved": engine.arm.move_joints([TARGET])}


def subscription_body(engine, ctx):
    engine.create_action_subscription(String, "/lifecycle_test", lambda m: None, 10)
    while True:
        ctx.checkpoint()
        time.sleep(POLL_S)


def control_cancel(client, handle, engine):
    client.control(Control.Request.CANCEL)


def goal_cancel(client, handle, engine):
    handle.cancel_goal_async()


def pause_then_cancel(client, handle, engine):
    client.control(Control.Request.PAUSE)
    time.sleep(INTERRUPT_AFTER_S)
    client.control(Control.Request.CANCEL)


def stop(client, handle, engine):
    engine.stop()


@pytest.mark.parametrize(
    "interrupt", [control_cancel, goal_cancel, pause_then_cancel, stop], ids=lambda f: f.__name__
)
@pytest.mark.parametrize("body", [trajectory_body, subscription_body], ids=lambda f: f.__name__)
def test_an_interrupted_action_ends_cleanly(env, body, interrupt):
    """Cancelled however: the result says so, nothing of the action is left at the driver or on
    the node, the engine is idle, and it takes the next goal (none once stopping)."""
    env.fake.hold_goals(ACTION_WAIT_S)
    engine = env.make(body)
    base = counts(engine)
    client = GoalClient(env.engine_id)
    try:
        handle = client.send()
        assert handle.accepted
        time.sleep(INTERRUPT_AFTER_S)  # running: the trajectory sent, the subscription made
        assert counts(engine) != base
        interrupt(client, handle, engine)
        assert client.result(handle).status == Result.STATUS_CANCELLED
        wait_until(lambda: engine._current is None)
        assert counts(engine) == base
        wait_until(lambda: env.fake.running() == 0)
        following = client.send()
        assert following.accepted == (interrupt is not stop)
        if following.accepted:
            client.control(Control.Request.CANCEL)
            client.result(following)
    finally:
        client.close()


def test_action_entities_only_inside_execute(env):
    engine = env.make(lambda engine, ctx: None)
    with pytest.raises(ScopeClosed):
        engine.create_action_subscription(String, "/lifecycle_test", lambda m: None, 10)


def test_action_tf_buffer_sees_static_transforms(env):
    broadcaster = Node(f"{env.engine_id}_tf")
    pub = broadcaster.create_publisher(TFMessage, "/tf_static", TF_STATIC_QOS)
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = "world", "lifecycle_cam"
    t.transform.translation.x, t.transform.rotation.w = 0.5, 1.0
    pub.publish(TFMessage(transforms=[t]))
    seen = {}

    def body(engine, ctx):
        tf = engine.create_action_tf_buffer()
        assert tf.can_transform(
            "world", "lifecycle_cam", RclpyTime(), RclpyDuration(seconds=WAIT_S)
        )
        seen["x"] = tf.lookup_transform(
            "world", "lifecycle_cam", RclpyTime()
        ).transform.translation.x

    engine = env.make(body)
    try:
        assert env.run(engine).status == Result.STATUS_SUCCESS
        assert seen["x"] == pytest.approx(0.5)
    finally:
        broadcaster.destroy_node()


def test_a_failing_end_still_reports_and_idles(env, monkeypatch):
    engine = env.make(lambda engine, ctx: None)

    def broken():
        raise RuntimeError("end_action broke")

    monkeypatch.setattr(engine.arm, "end_action", broken)
    result = env.run(engine)
    assert (result.status, result.error_code) == (Result.STATUS_FAILED, "exception")
    assert engine._current is None
    assert env.run(engine).status == Result.STATUS_FAILED  # a second goal is taken


def test_request_intent_honours_cancel(env, monkeypatch):
    """An orchestrator that never answers an intent: a cancel ends the action at once, not after
    INTENT_TIMEOUT_S."""
    silent = Node(f"{env.engine_id}_intents")  # never spun: discoverable, never answers
    silent.create_service(RequestIntent, f"{ORCHESTRATOR}/request_intent", lambda q, r: r)
    engine = env.make(lambda engine, ctx: ctx.request_intent("MOVE_BASE"), use_orchestrator=True)
    client = GoalClient(env.engine_id)
    try:
        handle = client.send()
        assert handle.accepted
        time.sleep(INTERRUPT_AFTER_S)
        t0 = time.monotonic()
        client.control(Control.Request.CANCEL)
        assert client.result(handle).status == Result.STATUS_CANCELLED
        assert time.monotonic() - t0 < WAIT_S
    finally:
        client.close()
        silent.destroy_node()
        engine.destroy_node()


def test_a_failing_activate_or_hibernate_keeps_the_engine_alive(env, monkeypatch):
    """Bad input or a bug in availability()/hibernate() is the call's failure, not the node's."""
    engine = env.make(lambda engine, ctx: None)

    def broken(*_):
        raise KeyError("world_state")

    monkeypatch.setattr(engine, "availability", broken)
    monkeypatch.setattr(engine, "hibernate", broken)
    node = Node(f"{env.engine_id}_activate_client")
    env.executor.add_node(node)
    activate = node.create_client(Activate, f"/{env.engine_id}/activate")
    hibernate = node.create_client(Hibernate, f"/{env.engine_id}/hibernate")
    try:
        assert activate.wait_for_service(timeout_sec=WAIT_S)
        for _ in range(2):  # the second call is answered too: the executor still spins
            future = activate.call_async(
                Activate.Request(instance_id="counter_1", world_state_snapshot_json="{not json")
            )
            wait_until(future.done)
            assert list(future.result().availability) == []
            future = hibernate.call_async(Hibernate.Request())
            wait_until(future.done)
            assert future.result().success is False
        assert env.run(engine).status == Result.STATUS_SUCCESS
    finally:
        env.executor.remove_node(node)
        node.destroy_node()


def test_a_goal_rclpy_never_starts_does_not_block_the_next(env, monkeypatch):
    """rclpy can accept a goal and then fail before executing it (its client gone before the
    acceptance went out): that run is dropped, and the next goal is taken."""
    monkeypatch.setattr(engine_module, "START_WAIT_S", START_WAIT_TEST_S)
    engine = env.make(lambda engine, ctx: None)
    assert engine._on_goal(ExecuteAction.Goal(action_id="run")) == GoalResponse.ACCEPT
    time.sleep(2 * START_WAIT_TEST_S)  # never handed on
    assert env.run(engine).status == Result.STATUS_SUCCESS


STUCK_SCRIPT = """
import time

from capability_engine import engine
from capability_engine.engine import Action, CapabilityEngine, run

engine.STOP_MARGIN_S = 0.2


class Stuck(CapabilityEngine):
    actions = [Action("run", "Run")]

    def __init__(self):
        super().__init__("{engine_id}")

    def execute(self, action_id, params, ctx):
        while True:  # never checks for a cancel
            time.sleep(0.05)


run(Stuck)
"""


def test_a_stop_is_bounded_when_execute_ignores_the_cancel(ros, tmp_path):
    """The action does not end: the engine still exits, stop_wait_s() after the signal, saying so,
    instead of waiting for it until docker's SIGKILL."""
    engine_id = f"arm_test_engine_{next(ENGINE_IDS)}"
    script = tmp_path / "stuck.py"
    script.write_text(STUCK_SCRIPT.format(engine_id=engine_id))
    proc = subprocess.Popen(
        [sys.executable, str(script), "--ros-args", "-p", "use_orchestrator:=false"]
        + FAST_SETTLE_ARGS,
        stderr=subprocess.PIPE,
        text=True,
    )
    node = Node(f"{engine_id}_client")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    client = ActionClient(node, ExecuteAction, f"/{engine_id}/execute_action")
    control = node.create_client(Control, f"/{engine_id}/control")
    try:
        assert client.wait_for_server(timeout_sec=ACTION_WAIT_S)
        assert control.wait_for_service(timeout_sec=WAIT_S)
        client.send_goal_async(ExecuteAction.Goal(action_id="run"))  # its reply may be lost
        resume = Control.Request(command=Control.Request.RESUME)
        wait_until(lambda: reply(control, resume) == "ok", ACTION_WAIT_S)  # it runs
        proc.send_signal(signal.SIGTERM)
        _, logs = proc.communicate(timeout=SHUTDOWN_LIMIT_S)
        assert proc.returncode == 1
        assert "did not end within" in logs, logs
    finally:
        proc.kill()
        proc.wait()
        client.destroy()
        executor.shutdown()
        node.destroy_node()


def test_a_late_acceptance_does_not_take_over_the_next_run(env, monkeypatch):
    """A goal whose acceptance comes so late that its run was dropped (and the next goal's made)
    must not be bound to that next run: runs are bound by their goal request, not by timing."""
    monkeypatch.setattr(engine_module, "START_WAIT_S", START_WAIT_TEST_S)
    engine = env.make(lambda engine, ctx: None)
    goal_a, goal_b = ExecuteAction.Goal(action_id="run"), ExecuteAction.Goal(action_id="run")
    assert engine._on_goal(goal_a) == GoalResponse.ACCEPT
    time.sleep(2 * START_WAIT_TEST_S)  # A's run is dropped when B arrives
    assert engine._on_goal(goal_b) == GoalResponse.ACCEPT
    late_a = SimpleNamespace(request=goal_a, execute=lambda: None)
    engine._on_accepted(late_a)
    assert engine._current.goal is None  # B's run is still B's
    handle_b = SimpleNamespace(request=goal_b, execute=lambda: None)
    engine._on_accepted(handle_b)
    assert engine._current.goal is handle_b


def test_the_stop_budget_follows_the_arm_parameters(env):
    """A profile that lets self.arm wait longer gets a longer stop, not an exit mid-settle."""
    params = {"arm.goal_accept_s": 10.0, "arm.halt_status_wait_s": 2.0}
    env.make(lambda engine, ctx: None, **params)
    assert engine_module.stop_wait_s() == pytest.approx(
        2.0  # the running call's settle: the longest of its waits, here the HALTED one
        + 10.0
        + arm_module.CANCEL_SETTLE_S  # end_action()'s
        + 2 * arm_module.CLOSE_STREAM_WAIT_S
        + engine_module.STOP_MARGIN_S
    )
