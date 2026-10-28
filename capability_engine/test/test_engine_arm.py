"""self.arm in the engine: per-action resources, parameters, the sim model, standalone SheppyArm
and mock mode. Each engine runs beside a FakeDriver and is driven through its execute_action."""

import importlib.util
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import sheppy_arm
from fake_driver import FLATBOT_URDF, FakeDriver, counts
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_orchestrator_interfaces.msg import Result
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from capability_engine import sim
from capability_engine.arm_params import PARAMS
from capability_engine.driver_state import NotInAction
from capability_engine.engine import Action, CapabilityEngine

# tests/arm/conftest.py holds the Cornell sim. Loaded by path under its own name, as in
# test_arm_motion.py: `from conftest import ...` reaches whichever conftest pytest loaded last.
_spec = importlib.util.spec_from_file_location(
    "arm_goldens_engine", Path(__file__).parents[2] / "tests" / "arm" / "conftest.py"
)
_goldens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_goldens)
cornell_sim = _goldens.cornell_sim

WAIT_S = 5.0  # how long a test waits for a server, a goal's acceptance or a node to go
ACTION_WAIT_S = 30.0  # how long a test waits for an action's result
POLL_S = 0.01
ENGINE_ID = "arm_test_engine"
TARGET = [0.3] * 7  # the joint target of the tests' moves
STRUCTURE = {"JOINTS", "CONTINUOUS", "ERRORS", "IMPEDANCE"}  # sheppy_arm constants, not tunables
CART_STEP = np.array([0.02, 0.0, 0.0])  # the mock Cartesian move: 2 cm along x


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(POLL_S)


class ArmEngine(CapabilityEngine):
    """One action, "run", whose execute() calls `body(engine, ctx)`."""

    actions = [Action("run", "Run")]

    def __init__(self, body, **node_kwargs):
        super().__init__(ENGINE_ID, **node_kwargs)
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
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    executor.add_node(client_node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    engines = []

    def make(body, **params):
        overrides = [Parameter("use_orchestrator", value=False)]
        overrides += [Parameter(name, value=value) for name, value in params.items()]
        engine = ArmEngine(body, parameter_overrides=overrides)
        engines.append(engine)
        executor.add_node(engine)
        return engine

    def run(engine):
        client = ActionClient(client_node, ExecuteAction, f"/{ENGINE_ID}/execute_action")
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

    yield SimpleNamespace(fake=fake, make=make, run=run)
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
    assert goal.sender_id == ENGINE_ID  # SENDER_ID comes from the capability


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
    for name in dir(sheppy_arm):
        if name.isupper() and name not in STRUCTURE and not callable(getattr(sheppy_arm, name)):
            assert (sheppy_arm, name) in targets, name
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
            assert value == ENGINE_ID
        elif isinstance(prm.default, list):
            assert list(value) == prm.default, name
        else:
            assert value == prm.default, name
    assert sheppy_arm.CART_SPEED == 0.08
    assert sheppy_arm.SENDER_ID == ENGINE_ID


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
    assert "driver model is live" in str(seen["explicit"])


def test_standalone_sheppy_arm(env):
    arm = sheppy_arm.SheppyArm(node_name="sheppy_arm_test")
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
    wait_until(lambda: env.fake.count_subscribers("/joint_states") == 0)


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
    assert seen["counts"] == base  # no ROS entity to the driver at all
    assert counts(engine) == base
    assert not any(env.fake.goals.values())
    assert not any(env.fake.requests.values())
    assert not any(env.fake.setpoints.values())


def test_mock_tool_pose_from_sim(env):
    seen = {}

    def body(engine, ctx):
        arm = engine.arm
        _, rb = cornell_sim()  # mock registers no driver model, so a path is allowed
        ee = arm.get_state()["ee_pos"]
        target = ee[:3] + CART_STEP
        seen["moved"] = arm.move_cartesian([(target, ee[3:])])
        seen["target"], seen["ee"] = target, arm.get_state()["ee_pos"]

    engine = env.make(body, mock=True)
    assert env.run(engine).status == Result.STATUS_SUCCESS
    assert seen["moved"]
    np.testing.assert_allclose(seen["ee"][:3], seen["target"], atol=sim.MAX_IK_ERR)
    assert not any(env.fake.goals.values())
