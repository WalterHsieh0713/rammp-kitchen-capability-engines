import dataclasses
import time
from pathlib import Path

import numpy as np
import pytest
from fake_driver import FakeCtx, counts

from capability_engine import arm as arm_module
from capability_engine.driver_state import ArmError, NotInAction

STRESS_CYCLES = 200  # actions in test_end_action_while_state_streams_in
STRESS_HZ = 1000.0  # the fake's state rate there: a message is nearly always in flight


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
    arm.begin_action(FakeCtx(), None)
    arm.get_state()
    opened = counts(node)
    assert arm.robot_description() == Path("/fixtures/urdf/flatbot.urdf").read_text()
    assert counts(node) == opened


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
