import threading
import time

import numpy as np
import pytest
from fake_driver import FakeCtx, counts
from rammp_arm import JOINTS
from rammp_arm_interfaces.msg import ImpedanceGains
from rammp_curobo_interfaces.action import PlanToPose
from rammp_curobo_interfaces.srv import SetWorld
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from capability_engine import arm as arm_module
from capability_engine.arm import HOLD_LEVEL, ArmInterface, Controller, Plan, Stream
from capability_engine.driver_state import (
    ArmError,
    DriverState,
    EStopped,
    ModeConflict,
    NotInAction,
    StreamClosed,
)
from capability_engine.engine import Cancelled

TOKEN = b"\x03" * 16
WAIT_S = 2.0  # how long a test waits for a message to reach the fake or the arm
POLL_S = 0.01
TWIST = np.ones(6) * 0.01
OPEN_DELAY_S = 0.5  # how long the fake takes to answer open_stream, past CANCEL_AFTER_S
CANCEL_AFTER_S = 0.3  # when a test cancels the action during a service call
SLOW_ANSWER_S = 2.0  # how long the slow test service takes to answer (> CANCEL_AFTER_S)
END_LANDS_S = 0.1  # how long a test holds a call back while end_action() starts


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
        assert "ee_twist" in arm.last_error  # refused through _fail, like stream()
    assert not fake.goals["go_to_joint_config"]


def test_driver_api_raises_estopped(arm_env):
    arm, _, fake = arm_env
    fake.engage_estop()
    arm.begin_action(FakeCtx(), TOKEN)
    with pytest.raises(EStopped, match="arm is E-STOPPED -- clear it first"):
        arm.execute_joint_trajectory(JOINTS, [(1.0, [0.1] * 7, [0.0] * 7)])
    with pytest.raises(EStopped, match="arm is E-STOPPED -- clear it first"):
        with arm.stream("ee_twist"):
            pass
    assert not fake.goals["execute_joint_trajectory"]
    assert not fake.requests["open_stream"]


def test_a_refusal_takes_its_type_and_reason_from_one_status(arm_env, monkeypatch):
    """/control_status can change between taking the reason and choosing the error (a callback
    in between): both come from one snapshot, so an e-stop reason is always EStopped. Here the
    e-stop clears just after the reason is taken, so any later look at the status disagrees."""
    arm, _, fake = arm_env
    fake.engage_estop()
    arm.begin_action(FakeCtx(), TOKEN)
    reason = DriverState._reason

    def reason_then_clear(ds, st, token, sender_id):
        why = reason(ds, st, token, sender_id)
        ds._status = (False, *ds._status[1:])  # a /control_status callback lands here
        return why

    monkeypatch.setattr(DriverState, "_reason", reason_then_clear)
    with pytest.raises(EStopped, match="arm is E-STOPPED -- clear it first"):
        arm.execute_joint_trajectory(JOINTS, [(1.0, [0.1] * 7, [0.0] * 7)])


@pytest.mark.parametrize("phase", ["server", "answer"])
def test_cancel_during_service_call(arm_env, monkeypatch, phase):
    """A cancel interrupts a service call waiting for its server, or for its answer."""
    arm, node, _ = arm_env
    path = f"/test_set_world_{phase}"
    if phase == "answer":

        def slow(request, response):
            time.sleep(SLOW_ANSWER_S)
            return response

        node.create_service(SetWorld, path, slow, callback_group=ReentrantCallbackGroup())
    monkeypatch.setitem(arm_module.SERVICES, "set_world", (SetWorld, path))
    ctx = FakeCtx()
    arm.begin_action(ctx, None)
    threading.Timer(CANCEL_AFTER_S, ctx.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        arm.set_world("kitchen")
    assert time.monotonic() - t0 < SLOW_ANSWER_S


def test_stream_closed_by_driver_raises(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(StreamClosed, match="stopped sending for more than"):
        with arm.stream("ee_twist") as s:
            s.send(TWIST)
            fake.publish_stream_status(False)
            wait_until(lambda: not arm.stream_status().open)
            s.send(TWIST)
    # Closed by the driver: its exit neither holds nor closes (a CloseStream now could close
    # another sender's session).
    assert not fake.requests["close_stream"]
    assert twist_values(fake.setpoints["/setpoint/twist"][-1]) != [0.0] * 6
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


def test_a_send_racing_the_close_publishes_nothing_after_it(arm_env):
    """The stream closes (end_action() in another thread) while send() is between its first
    check and its publish: send() raises NotInAction and publishes nothing after the hold."""
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with arm.stream("ee_twist") as s:
        s.send(TWIST)
        message = s._message

        def close_then_message(value):  # runs between send()'s first check and its publish
            s._message = message
            # In another thread, as end_action() would, so the test fails, not hangs, if
            # _message ever runs under send()'s (non-reentrant) lock.
            closer = threading.Thread(target=s.close, daemon=True)
            closer.start()
            closer.join(WAIT_S)
            assert not closer.is_alive(), "close() is waiting for send()'s lock"
            return message(value)

        s._message = close_then_message
        with pytest.raises(NotInAction):
            s.send(TWIST)
    time.sleep(POLL_S * 10)  # a setpoint published after the hold would have arrived
    assert last_twist(fake) == [0.0] * 6
    assert len(fake.requests["close_stream"]) == 1


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
    assert plan.message == (
        "plan_to_joints: the plan is not positions and velocities of arm_joint_1..7 or "
        f"joint_1..7 at every point (its joints: {names})"
    )
    assert plan.q.shape == (0, 7)
    if not velocities:  # the same plan with velocities is fine: they were the failure
        fake.plan_result([[0.0] * 7, [0.1] * 7], joint_names=names)
        assert arm.plan_to_joints([0.1] * 7, [0.0] * 7).success


def test_planner_needs_no_arm_state(arm_env):
    """With no /joint_states, /ee_state or /gripper_state, a plan goes out at once."""
    arm, _, fake = arm_env
    fake.pause_publishing()
    fake.plan_result([[0.0] * 7, [0.1] * 7])
    arm.begin_action(FakeCtx(), None)
    t0 = time.monotonic()
    assert arm.plan_to_joints([0.1] * 7, [0.0] * 7).success
    assert time.monotonic() - t0 < WAIT_S  # far below STATE_WAIT_S


def test_plan_subscribes_to_no_arm_state(arm_env):
    """An action that only plans pays for none of the 50 Hz state subscriptions (~2 ms a message
    on the Orin): each entity is made when a call first needs it."""
    arm, node, fake = arm_env
    fake.plan_result([[0.0] * 7, [0.1] * 7])
    arm.begin_action(FakeCtx(), None)
    assert arm.plan_to_joints([0.1] * 7, [0.0] * 7).success
    topics = {s.topic_name for s in node.subscriptions}
    assert not topics & {"/joint_states", "/ee_state", "/gripper_state"}
    assert fake.count_subscribers("/joint_states") == 0


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


def test_a_rejected_plan_says_so_without_claiming_nothing_moved(arm_env):
    """A plan never moves the arm: its REJECTED message has no "-- nothing moved"."""
    arm, _, fake = arm_env
    fake.accept("plan_to_joints", False)
    arm.begin_action(FakeCtx(), None)
    plan = arm.plan_to_joints([0.1] * 7, [0.0] * 7)
    assert not plan.success
    assert plan.message == arm.last_error == "plan_to_joints: goal REJECTED"


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


@pytest.mark.parametrize("accept", [True, False], ids=["opened_late", "refused_late"])
def test_cancel_while_opening_closes_only_a_stream_the_driver_opened(arm_env, accept):
    """Cancelled while /open_stream is answering: a session the driver opens after all is closed;
    one it refuses is left alone (CloseStream would close whatever session is open)."""
    arm, _, fake = arm_env
    fake.delay_open_stream(OPEN_DELAY_S, accept=accept)
    ctx = FakeCtx()
    arm.begin_action(ctx, None)
    threading.Timer(CANCEL_AFTER_S, ctx.cancel).start()
    with pytest.raises(Cancelled):
        with arm.stream("ee_twist"):
            pass
    assert len(fake.requests["close_stream"]) == (1 if accept else 0)


def test_end_action_just_as_open_stream_is_sent_closes_the_late_session(arm_env, monkeypatch):
    """end_action() in another thread lands just as /open_stream is sent, before the stream
    has noted the request, and the driver opens the session late: end_action()'s settle still
    sees the request and closes the session."""
    arm, _, fake = arm_env
    fake.delay_open_stream(OPEN_DELAY_S)  # answered within the close's CLOSE_STREAM_WAIT_S
    arm.begin_action(FakeCtx(), None)
    ender = threading.Thread(target=arm.end_action)

    def end_then_note(stream, future):  # the stream notes the sent request (on_sent)
        if future is not None:
            ender.start()
            time.sleep(END_LANDS_S)  # end_action() is under way before the note
        stream.__dict__["_open_sent"] = future

    noted = property(lambda stream: stream.__dict__["_open_sent"], end_then_note)
    monkeypatch.setattr(Stream, "_open_sent", noted, raising=False)
    with pytest.raises(NotInAction):
        with arm.stream("ee_twist"):
            pass
    ender.join(WAIT_S + OPEN_DELAY_S)
    assert not ender.is_alive()
    time.sleep(OPEN_DELAY_S)  # the driver has answered, and opened the session, by now
    assert len(fake.requests["open_stream"]) == 1
    assert not fake.session_open(), "the session the driver opened late is still open"


def test_mock_stream_honours_cancel_and_its_end(ros):
    """mock:=true: send() after the block raises NotInAction, and a send loop ends on a cancel."""
    node = Node("arm_test_mock")
    arm = ArmInterface(node, ReentrantCallbackGroup(), mock=True)
    ctx = FakeCtx()
    arm.begin_action(ctx, None)
    try:
        with arm.stream("ee_twist") as s:
            s.send(TWIST)
        with pytest.raises(NotInAction):
            s.send(TWIST)
        threading.Timer(CANCEL_AFTER_S, ctx.cancel).start()
        deadline = time.monotonic() + WAIT_S  # a bound, so the test fails rather than hangs
        with pytest.raises(Cancelled):
            with arm.stream("ee_twist") as s:
                while time.monotonic() < deadline:
                    s.send(TWIST)
                    time.sleep(POLL_S)
    finally:
        arm.end_action()
        node.destroy_node()
