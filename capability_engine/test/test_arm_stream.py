import time

import numpy as np
import pytest
from fake_driver import FakeCtx, counts
from rammp_arm_interfaces.msg import ImpedanceGains
from rammp_curobo_interfaces.action import PlanToPose
from rclpy.callback_groups import ReentrantCallbackGroup
from sheppy_arm import JOINTS

from capability_engine import arm as arm_module
from capability_engine.arm import HOLD_LEVEL, ArmInterface, Controller, Plan
from capability_engine.driver_state import (
    ArmError,
    ModeConflict,
    NotInAction,
    StreamClosed,
)
from capability_engine.engine import Cancelled

TOKEN = b"\x03" * 16
WAIT_S = 2.0  # how long a test waits for a message to reach the fake or the arm
POLL_S = 0.01
TWIST = np.ones(6) * 0.01


def wait_until(condition, timeout_s=WAIT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(POLL_S)


def twist_values(m):
    t = m.twist
    return [t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z]


def last_twist(fake):
    wait_until(lambda: fake.setpoints["/setpoint/twist"])
    return twist_values(fake.setpoints["/setpoint/twist"][-1])


def test_stream_opens_sends_and_closes(arm_env):
    arm, node, fake = arm_env
    base = counts(node)  # (subscriptions, clients, action clients, publishers)
    arm.begin_action(FakeCtx(), TOKEN)
    with arm.stream("ee_twist", gains="default") as s:
        assert len(fake.requests["close_stream"]) == 0
        for _ in range(5):
            s.send(TWIST)
            time.sleep(POLL_S)
        wait_until(lambda: len(fake.setpoints["/setpoint/twist"]) >= 1)
    req = fake.requests["open_stream"][-1]
    assert req.controller == "ee_twist"
    assert req.timeout_s == pytest.approx(0.1)
    assert bytes(req.token) == TOKEN
    assert req.gains.profile == ImpedanceGains.PROFILE_SESSION_DEFAULT
    sent = fake.setpoints["/setpoint/twist"]
    np.testing.assert_allclose(twist_values(sent[0]), TWIST)
    assert bytes(sent[0].token) == TOKEN
    assert len(fake.requests["close_stream"]) == 1
    assert bytes(fake.requests["close_stream"][-1].token) == TOKEN
    assert arm._driver.mode == "idle"
    arm.end_action()
    assert counts(node) == base


def test_stream_refuses_unavailable_controller(arm_env):
    arm, _, fake = arm_env
    fake.set_controllers([("ee_twist", ["twist"], True), ("ee_twist_impedance", ["twist"], False)])
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(ArmError, match="ee_twist_impedance"):
        with arm.stream("ee_twist_impedance"):
            pass
    assert len(fake.requests["list_controllers"]) == 1
    assert not fake.requests["open_stream"]
    assert arm._driver.mode == "idle"


def test_stream_refuses_unknown_controller(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(ArmError, match="'warp_drive'"):
        with arm.stream("warp_drive"):
            pass
    assert not fake.requests["open_stream"]


def test_stream_closes_on_exception(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(RuntimeError, match="boom"):
        with arm.stream("ee_twist") as s:
            s.send(TWIST)
            raise RuntimeError("boom")
    assert last_twist(fake) == [0.0] * 6
    assert len(fake.requests["close_stream"]) == 1
    assert arm._driver.mode == "idle"


@pytest.mark.parametrize(
    ("controller", "topic"),
    [
        ("joint_velocity", "/setpoint/joint_velocity"),
        ("joint_position", "/setpoint/joint_position"),
        ("ee_pose_position", "/setpoint/pose"),
    ],
)
def test_stream_exit_holds_the_arm(arm_env, controller, topic):
    """The exit setpoint: zeros for velocity controllers, the measured state for position ones."""
    arm, _, fake = arm_env
    q, pos, quat = [0.1 * i for i in range(7)], [0.4, 0.1, 0.3], [0.0, 1.0, 0.0, 0.0]
    fake.set_joints(q)
    fake.set_flange(pos, quat)
    arm.begin_action(FakeCtx(), None)
    with arm.stream(controller):
        pass
    wait_until(lambda: fake.setpoints[topic])
    m = fake.setpoints[topic][-1]
    if controller == "joint_velocity":
        np.testing.assert_allclose(m.values, [0.0] * 7)
    elif controller == "joint_position":
        np.testing.assert_allclose(m.values, q)
    else:
        p, o = m.pose.position, m.pose.orientation
        np.testing.assert_allclose([p.x, p.y, p.z], pos)
        np.testing.assert_allclose([o.x, o.y, o.z, o.w], quat)


def test_stream_sends_joints_and_poses(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with arm.stream("joint_impedance", gains="soft") as s:
        s.send(np.arange(7.0))
        wait_until(lambda: fake.setpoints["/setpoint/joint_position"])
        with pytest.raises(ArmError, match="joint_position setpoint"):
            s.send(np.arange(6.0))
    assert fake.requests["open_stream"][-1].gains.profile == ImpedanceGains.PROFILE_SOFT
    np.testing.assert_allclose(fake.setpoints["/setpoint/joint_position"][0].values, range(7))
    with arm.stream("ee_pose_impedance") as s:
        s.send(([0.4, 0.1, 0.3], [0.0, 0.0, 0.0, 1.0]))
        wait_until(lambda: fake.setpoints["/setpoint/pose"])  # before the exit's hold setpoint
    p = fake.setpoints["/setpoint/pose"][0].pose.position
    np.testing.assert_allclose([p.x, p.y, p.z], [0.4, 0.1, 0.3])


def test_nested_stream_refused(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with arm.stream("ee_twist"):
        with pytest.raises(ModeConflict, match="ee_twist"):
            with arm.stream("joint_velocity"):
                pass
    assert len(fake.requests["list_controllers"]) == 1  # refused before any service call
    assert len(fake.requests["open_stream"]) == 1


def test_goal_during_stream_refused(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with arm.stream("ee_twist"):
        with pytest.raises(ModeConflict, match="ee_twist"):
            arm.go_to_joint_config([0.0] * 7)
    assert not fake.goals["go_to_joint_config"]


def test_stream_closed_by_driver_raises(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(StreamClosed, match="stopped sending for more than"):
        with arm.stream("ee_twist") as s:
            s.send(TWIST)
            fake.publish_stream_status(False)
            wait_until(lambda: not arm.stream_status().open)
            s.send(TWIST)
    assert len(fake.requests["close_stream"]) == 1
    assert arm._driver.mode == "idle"


def test_late_close_of_the_last_session_is_ignored(arm_env):
    """The driver closes session A; A's late "closed" lands after B began, before B opens."""
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(StreamClosed):
        with arm.stream("ee_twist") as s:
            fake.publish_stream_status(False)
            wait_until(lambda: not arm.stream_status().open)
            s.send(TWIST)
    fake.late_close_on_next_open()
    with arm.stream("ee_twist") as s:
        wait_until(lambda: arm.stream_status().open)  # the late "closed" has come in before it
        for _ in range(5):
            s.send(TWIST)
            time.sleep(POLL_S)
        wait_until(lambda: twist_values(fake.setpoints["/setpoint/twist"][-1]) != [0.0] * 6)
    assert len(fake.requests["open_stream"]) == 2


def test_stream_cancel_zeroes_and_closes(arm_env):
    arm, _, fake = arm_env
    ctx = FakeCtx()
    arm.begin_action(ctx, None)
    with pytest.raises(Cancelled):
        with arm.stream("ee_twist") as s:
            s.send(TWIST)
            ctx.cancel()
            s.send(TWIST)
    assert last_twist(fake) == [0.0] * 6
    assert len(fake.requests["close_stream"]) == 1


def test_end_action_closes_an_open_stream(arm_env):
    arm, node, fake = arm_env
    base = counts(node)
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(NotInAction):
        with arm.stream("ee_twist") as s:
            s.send(TWIST)
            arm.end_action()
            assert counts(node) == base
            s.send(TWIST)
    assert last_twist(fake) == [0.0] * 6
    assert len(fake.requests["close_stream"]) == 1


def test_control_services_refused_under_engine(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    for call in (
        lambda: arm.acquire_control("bench"),
        lambda: arm.release_control(),
        lambda: arm.revoke_control("test"),
    ):
        with pytest.raises(ArmError, match="control comes from the orchestrator"):
            call()
    for name in ("acquire_control", "release_control", "revoke_control"):
        assert not fake.requests[name]


def test_control_services_standalone(arm_env):
    _, node, fake = arm_env
    arm = ArmInterface(node, ReentrantCallbackGroup(), standalone=True)
    arm.begin_action(FakeCtx(), None)
    try:
        accepted, token, generation, _ = arm.acquire_control("bench")
        assert accepted and len(token) == 16 and generation == 1
        assert fake.requests["acquire_control"][-1].owner_id == "bench"
        assert arm.release_control(token)[0] is True
        assert bytes(fake.requests["release_control"][-1].token) == token
        assert arm.revoke_control("done")[0] is True
        assert fake.requests["revoke_control"][-1].reason == "done"
    finally:
        arm.end_action()


def test_plan_to_pose_round_trip(arm_env):
    arm, _, fake = arm_env
    q = np.arange(21.0).reshape(3, 7) / 10
    order = [6, 0, 1, 2, 3, 4, 5]  # the planner may name the joints in any order
    fake.plan_result(q[:, order], joint_names=[JOINTS[i] for i in order])
    arm.begin_action(FakeCtx(), None)
    start = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
    plan = arm.plan_to_pose([0.4, 0.1, 0.3], [0.0, 1.0, 0.0, 0.0], start, hold=HOLD_LEVEL)
    assert isinstance(plan, Plan) and plan.success
    assert plan.q.shape == (3, 7)
    np.testing.assert_allclose(plan.q, q)
    np.testing.assert_allclose(plan.t, [0.0, 0.1, 0.2])
    assert plan.qd.shape == (3, 7)
    g = fake.goals["plan_to_pose"][-1]
    np.testing.assert_allclose(g.start_joints, start)
    assert g.hold == PlanToPose.Goal.HOLD_LEVEL
    p = g.target.position
    np.testing.assert_allclose([p.x, p.y, p.z], [0.4, 0.1, 0.3])


def test_plan_with_planner_joint_names(arm_env):
    """cuRobo's config names the joints joint_1..7; they map to arm_joint_1..7 by number."""
    arm, _, fake = arm_env
    q = np.arange(14.0).reshape(2, 7) / 10
    order = [3, 4, 5, 6, 0, 1, 2]
    fake.plan_result(q[:, order], joint_names=[f"joint_{i + 1}" for i in order])
    arm.begin_action(FakeCtx(), None)
    plan = arm.plan_to_joints([0.1] * 7, [0.0] * 7)
    assert plan.success
    np.testing.assert_allclose(plan.q, q)


@pytest.mark.parametrize(
    ("names", "velocities"),
    [([f"j{i}" for i in range(1, 8)], True), (JOINTS, False)],
    ids=["unknown_names", "no_velocities"],
)
def test_unusable_plan_fails(arm_env, names, velocities):
    arm, _, fake = arm_env
    fake.plan_result([[0.0] * 7, [0.1] * 7], joint_names=names, velocities=velocities)
    arm.begin_action(FakeCtx(), None)
    plan = arm.plan_to_joints([0.1] * 7, [0.0] * 7)
    assert plan.success is False
    assert names[0] in plan.message
    assert plan.q.shape == (0, 7)


def test_planner_and_estop_need_no_arm_state(arm_env):
    """With no /joint_states, /ee_state or /gripper_state, both go out at once."""
    arm, _, fake = arm_env
    fake.pause_publishing()
    fake.plan_result([[0.0] * 7, [0.1] * 7])
    arm.begin_action(FakeCtx(), None)
    t0 = time.monotonic()
    arm.estop(True, "test")
    assert arm.plan_to_joints([0.1] * 7, [0.0] * 7).success
    assert time.monotonic() - t0 < WAIT_S  # far below STATE_WAIT_S
    wait_until(lambda: fake.setpoints["/estop"])


def test_plan_to_joints_and_failure(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    plan = arm.plan_to_joints([0.1] * 7, [0.0] * 7)  # no plan_result set: the planner fails
    assert plan.success is False
    assert "no plan_result" in plan.message
    assert plan.q.shape == (0, 7)
    fake.plan_result([[0.0] * 7, [0.1] * 7])
    plan = arm.plan_to_joints([0.1] * 7, [0.0] * 7)
    assert plan.success and plan.q.shape == (2, 7)
    np.testing.assert_allclose(fake.goals["plan_to_joints"][-1].target_joints, [0.1] * 7)


def test_hold_constants_match_the_planner():
    goal = PlanToPose.Goal
    assert (arm_module.HOLD_NONE, arm_module.HOLD_LEVEL, arm_module.HOLD_FIXED) == (
        goal.HOLD_NONE,
        goal.HOLD_LEVEL,
        goal.HOLD_FIXED,
    )


def test_services(arm_env):
    arm, _, fake = arm_env
    fake.set_controllers([("ee_twist", ["twist"], True), ("ee_twist_impedance", ["twist"], False)])
    arm.begin_action(FakeCtx(), TOKEN)
    assert arm.list_controllers() == [
        Controller("ee_twist", ["twist"], True),
        Controller("ee_twist_impedance", ["twist"], False),
    ]
    assert arm.set_gains("stiff") == (True, "")
    req = fake.requests["set_gains"][-1]
    assert req.spec.profile == ImpedanceGains.PROFILE_STIFF
    assert bytes(req.token) == TOKEN
    assert arm.set_world("kitchen") == (True, "")
    assert fake.requests["set_world"][-1].world == "kitchen"


def test_estop_publishes(arm_env):
    arm, node, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    arm.estop(True, "test")
    wait_until(lambda: fake.setpoints["/estop"])
    m = fake.setpoints["/estop"][-1]
    assert m.engaged is True
    assert m.reason == "test"
    assert m.source == node.get_name()
