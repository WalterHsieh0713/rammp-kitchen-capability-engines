import dataclasses
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fake_driver import FakeCtx, counts

from capability_engine import arm as arm_module
from capability_engine.driver_state import ArmError, NotInAction, StateTimeout

STRESS_CYCLES = 200  # actions in test_end_action_while_state_streams_in
STRESS_HZ = 1000.0  # the fake's state rate there: a message is nearly always in flight
WAKE_DELAY_S = 0.3  # how long a call waits for state before end_action() in the wake-up test
WAKE_LIMIT_S = 2.0  # it must return well before STATE_WAIT_S (10 s)
SHEPPY_NO_STATE = (
    "no /joint_states, /ee_state or /gripper_state -- is the sheppy arm node up, "
    "and ROS_DOMAIN_ID / RMW_IMPLEMENTATION exported?"
)

LINGER_S = 0.3  # how long the no-state-after-the-end test waits for late messages
RACE_HOLD_S = 0.3  # the begin/end race test: how long the ending thread is held mid-end


def test_no_resources_until_first_call(arm_env):
    arm, node, fake = arm_env
    base = counts(node)  # (subscriptions, clients, action clients, publishers)
    arm.begin_action(ctx=FakeCtx(), token=None)
    assert counts(node) == base
    arm.get_state()
    assert counts(node) != base
    arm.end_action()
    assert counts(node) == base


def test_get_state_shape_and_tool_offset(arm_env):
    arm, _, fake = arm_env
    fake.set_joints([0.1, 0.2, 3.3, -1.0, 0.0, 0.5, -3.2])
    fake.set_flange(pos=[0.5, 0.0, 0.4], quat=[0, 0, 0, 1])
    arm.begin_action(FakeCtx(), None)
    s = arm.get_state()
    assert set(s) == {"position", "velocity", "effort", "ee_pos", "gripper_pos"}
    np.testing.assert_allclose(s["position"][[2, 6]], [3.3 - 2 * np.pi, -3.2 + 2 * np.pi])
    np.testing.assert_allclose(s["ee_pos"][:3], [0.5, 0.0, 0.4 + 0.120])


def test_state_cleared_between_actions(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    arm.get_state()
    arm.end_action()
    fake.set_joints([0.5] * 7)
    arm.begin_action(FakeCtx(), None)
    np.testing.assert_allclose(arm.get_state()["position"], [0.5] * 7)


def test_call_outside_action_raises(arm_env):
    arm, _, _ = arm_env
    with pytest.raises(NotInAction, match="only available inside an action"):
        arm.get_state()


def test_stale_state_is_an_error(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    arm.get_state()
    fake.pause_publishing()
    time.sleep(arm_module.STATE_STALE_S + 0.1)
    with pytest.raises(ArmError, match="stale"):
        arm.joint_states()


def test_state_reads_are_dataclasses_with_age(arm_env):
    arm, _, fake = arm_env
    fake.set_joints([0.1, 0.2, 3.3, -1.0, 0.0, 0.5, -3.2])
    arm.begin_action(FakeCtx(), None)
    reads = (
        arm.joint_states,
        arm.ee_state,
        arm.gripper_state,
        arm.control_status,
        arm.stream_status,
        arm.diagnostics,
    )
    for read in reads:
        value = read()
        assert dataclasses.is_dataclass(value), read.__name__
        assert value.age_s >= 0.0, read.__name__
    assert arm.ee_state().frame_id == "arm_mount"
    assert arm.stream_status().open is False
    np.testing.assert_allclose(arm.raw_joints(), [0.1, 0.2, 3.3, -1.0, 0.0, 0.5, -3.2])


def test_robot_description_is_one_shot(arm_env):
    arm, node, _ = arm_env
    base = counts(node)
    arm.begin_action(FakeCtx(), None)
    assert arm.robot_description() == Path("/fixtures/urdf/flatbot.urdf").read_text()
    assert counts(node) == base  # its own subscription is gone, and nothing else was opened


def test_new_action_ignores_the_last_actions_generation(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), b"\x01" * 16)
    assert arm.control_status().generation == 0
    arm.end_action()
    fake.seize("orchestrator")  # the orchestrator re-acquires control between actions
    arm.begin_action(FakeCtx(), b"\x02" * 16)
    assert arm.control_status().generation == 1
    arm._driver.check_command()  # a new action's baseline: no "control taken"


def test_refusals_are_kept_in_last_error(arm_env):
    arm, _, _ = arm_env
    with pytest.raises(NotInAction):
        arm.not_ready()
    assert arm.last_error == "the arm is only available inside an action"


def test_end_action_while_state_streams_in(arm_env):
    # rclpy (Humble) raises InvalidHandle out of the executor's spin() when an entity is destroyed
    # while a message for it is being taken; end_action() must leave the executor spinning.
    arm, _, fake = arm_env
    fake.set_publish_rate(STRESS_HZ)
    for _ in range(STRESS_CYCLES):
        arm.begin_action(FakeCtx(), None)
        arm.get_state()
        arm.end_action()


def test_get_state_returns_stale_values_like_rammp_arm(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    arm.get_state()
    fake.pause_publishing()
    time.sleep(arm_module.STATE_STALE_S + 0.1)
    assert set(arm.get_state()) == {"position", "velocity", "effort", "ee_pos", "gripper_pos"}
    assert arm.raw_joints().shape == (7,)


def test_state_timeout_keeps_rammp_arms_message(arm_env, monkeypatch):
    arm, _, fake = arm_env
    monkeypatch.setattr(arm_module, "STATE_WAIT_S", 0.2)
    fake.pause_publishing()
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(StateTimeout, match=re.escape(SHEPPY_NO_STATE)) as raised:
        arm.get_state()
    assert isinstance(raised.value, RuntimeError)  # what RammpArm's callers catch


def test_end_action_while_an_entity_is_made(arm_env, monkeypatch):
    """Another thread's end_action() lands just after a call's in-action check, as it makes the
    action's first entity: the check and the making are one guarded step, so the entity is
    made before the end (and destroyed by it), and the call raises NotInAction."""
    arm, node, fake = arm_env
    base = counts(node)
    fake.pause_publishing()  # no state: the call is still waiting when the end lands
    arm.begin_action(FakeCtx(), None)
    ender = threading.Thread(target=arm.end_action)
    check = arm._check_in_action

    def check_then_end(*args):
        check(*args)
        if ender.ident is None:
            ender.start()
            time.sleep(0.1)  # end_action() is under way before the entity is made

    monkeypatch.setattr(arm, "_check_in_action", check_then_end)
    with pytest.raises(NotInAction):
        arm.get_state()
    ender.join(WAKE_LIMIT_S)
    assert not ender.is_alive()
    assert counts(node) == base


def test_end_action_wakes_a_waiting_call(arm_env):
    arm, node, fake = arm_env
    base = counts(node)
    fake.pause_publishing()
    arm.begin_action(FakeCtx(), None)
    errors = []

    def call():
        try:
            arm.get_state()
        except ArmError as error:
            errors.append(error)

    caller = threading.Thread(target=call)
    caller.start()
    time.sleep(WAKE_DELAY_S)
    t0 = time.monotonic()
    arm.end_action()
    caller.join(WAKE_LIMIT_S)
    assert not caller.is_alive() and time.monotonic() - t0 < WAKE_LIMIT_S
    assert [type(e) for e in errors] == [NotInAction]
    assert counts(node) == base


def test_begin_action_ends_an_unended_action(arm_env):
    arm, node, _ = arm_env
    base = counts(node)
    arm.begin_action(FakeCtx(), None)
    arm.get_state()
    arm.begin_action(FakeCtx(), None)  # the last action never got end_action()
    assert counts(node) == base


def test_end_action_racing_begin_action_leaves_a_whole_action_or_none(arm_env, monkeypatch):
    """end_action() in another thread while begin_action() sets the next action up: the arm is
    left with that action whole, or with none, never in an action with no scope or ctx."""
    arm, _, _ = arm_env
    arm.begin_action(FakeCtx(), None)
    ender, settling = threading.Thread(target=arm.end_action), threading.Event()
    settle, make_driver = arm._settle, arm_module.DriverState

    def slow_settle():
        if threading.current_thread() is ender:
            settling.set()
            time.sleep(RACE_HOLD_S)  # begin_action() sets up meanwhile, unless it holds the lock
        settle()

    def driver(**kwargs):  # in begin_action(), after it has ended the last action
        if ender.ident is None:
            ender.start()
            settling.wait(RACE_HOLD_S)
        return make_driver(**kwargs)

    monkeypatch.setattr(arm, "_settle", slow_settle)
    monkeypatch.setattr(arm_module, "DriverState", driver)
    arm.begin_action(FakeCtx(), None)
    ender.join()
    assert arm._in_action == (arm._scope is not None) == (arm._ctx is not None)


def test_no_state_is_stored_after_the_action_ends(arm_env, monkeypatch):
    """end_action() closes the scope only after it settles: a message arriving in between must
    not be kept, or the next action would start from the last one's state."""
    arm, _, _ = arm_env
    arm.begin_action(FakeCtx(), None)
    arm.get_state()
    settle, received = arm._settle, {}

    def slow_settle():
        received["at_end"] = {topic: t for topic, (_, t) in arm._latest.items()}
        time.sleep(LINGER_S)  # the fake keeps publishing to the still-open subscriptions
        received["after"] = {topic: t for topic, (_, t) in arm._latest.items()}
        settle()

    monkeypatch.setattr(arm, "_settle", slow_settle)
    arm.end_action()
    assert received["after"] == received["at_end"]


def test_end_action_holds_nothing_of_the_action(arm_env):
    arm, _, _ = arm_env
    arm.begin_action(FakeCtx(), None)
    arm.get_state()
    arm.end_action()
    assert arm._ctx is None and arm._scope is None and arm._stream is None
    assert not arm._entities and not arm._latest and not arm._goals
