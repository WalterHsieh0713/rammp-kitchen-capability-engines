import threading
import time

import numpy as np
import pytest
from fake_driver import FakeCtx
from rammp_arm_interfaces.msg import ImpedanceGains
from sheppy_arm import SENDER_ID, TOOL_Z

from capability_engine.arm import HOLD_LEVEL, IMPEDANCE, LATEST_WINS
from capability_engine.driver_state import NotInAction
from capability_engine.engine import Cancelled

HOLD_S = 5.0  # how long the fake runs a goal in the tests that need one running
TOKEN = b"\x02" * 16
UP = [0.0, 0.0, 0.0, 1.0]  # identity quaternion, x y z w


def pt_time(p):
    return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9


def flange_for_tool(pos):
    """The flange pose (identity orientation) that puts the tool frame at `pos`."""
    return ([pos[0], pos[1], pos[2] - TOOL_Z], UP)


def test_go_to_ee_pose_goal_fields(arm_env):
    arm, _, fake = arm_env
    pos, quat = [0.4, 0.1, 0.3], [0.0, 1.0, 0.0, 0.0]
    fake.finish_at("go_to_ee_pose", flange=flange_for_tool(pos))
    arm.begin_action(FakeCtx(), TOKEN)
    assert arm.go_to_ee_pose(pos, quat) is True
    g = fake.goals["go_to_ee_pose"][-1]
    assert g.target.header.frame_id == "arm_mount"
    p, o = g.target.pose.position, g.target.pose.orientation
    np.testing.assert_allclose([p.x, p.y, p.z], pos)
    np.testing.assert_allclose([o.x, o.y, o.z, o.w], quat)
    assert g.speed_scale == pytest.approx(0.3)
    assert g.sender_id == SENDER_ID
    assert bytes(g.token) == TOKEN


def test_go_to_ee_pose_end_check(arm_env):
    arm, _, fake = arm_env
    pos = [0.4, 0.1, 0.3]
    fake.finish_at("go_to_ee_pose", flange=flange_for_tool([0.4, 0.1, 0.315]))  # 1.5 cm short
    arm.begin_action(FakeCtx(), None)
    assert arm.go_to_ee_pose(pos, UP) is False
    assert "tool 1.5 cm from the goal" in arm.last_error


def test_go_to_joint_config_end_check_short_way(arm_env):
    arm, _, fake = arm_env
    fake.finish_at("go_to_joint_config", joints=[-np.pi + 0.01] + [0.0] * 6)
    arm.begin_action(FakeCtx(), None)
    assert arm.go_to_joint_config([np.pi - 0.01] + [0.0] * 6) is True  # 1.1 deg the short way
    np.testing.assert_allclose(
        fake.goals["go_to_joint_config"][-1].target_joints, [np.pi - 0.01] + [0.0] * 6
    )


def test_go_to_driver_kwargs(arm_env):
    arm, _, fake = arm_env
    pos = [0.4, 0.1, 0.3]
    fake.finish_at("go_to_ee_pose", flange=flange_for_tool(pos))
    arm.begin_action(FakeCtx(), None)
    soft = ImpedanceGains(profile=ImpedanceGains.PROFILE_SOFT)
    assert arm.go_to_ee_pose(
        pos, UP, orientation_hold=HOLD_LEVEL, control_mode=IMPEDANCE, gains=soft
    )
    g = fake.goals["go_to_ee_pose"][-1]
    assert g.orientation_hold == HOLD_LEVEL
    assert g.control_mode == IMPEDANCE
    assert g.gains.profile == ImpedanceGains.PROFILE_SOFT


def test_go_to_preset_home(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    assert arm.go_to_preset("home") is True
    g = fake.goals["go_to_preset"][-1]
    assert g.preset_name == "home"
    assert g.speed_scale == pytest.approx(0.3)


def test_gripper_streams_until_settled(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), TOKEN)
    fake.move_gripper_to(1.0, 0.4)
    t0 = time.monotonic()
    assert arm.gripper(1.0, force=0.25) is True
    assert time.monotonic() - t0 < 2.0  # settled, well before the 4 s timeout
    sent = fake.setpoints["/setpoint/gripper"]
    assert sent
    for m in sent:
        assert (m.position, m.speed, m.force) == pytest.approx((1.0, 0.5, 0.25))
        assert bytes(m.token) == TOKEN


def test_gripper_cancel_raises(arm_env):
    arm, _, _ = arm_env
    ctx = FakeCtx()
    arm.begin_action(ctx, None)
    threading.Timer(0.3, ctx.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        arm.gripper(1.0)  # the fake's gripper never moves: would stream until settled (~0.8 s)
    assert time.monotonic() - t0 < 0.5


def test_cancel_during_go_to(arm_env):
    arm, _, fake = arm_env
    ctx = FakeCtx()
    fake.hold_goals(HOLD_S)
    arm.begin_action(ctx, None)
    threading.Timer(0.3, ctx.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        arm.go_to_ee_pose([0.4, 0.1, 0.3], UP)
    assert time.monotonic() - t0 < 0.8
    assert fake.cancelled("go_to_ee_pose")


def test_execute_joint_trajectory_raw(arm_env):
    arm, _, fake = arm_env
    fake.set_result("execute_joint_trajectory", -4)
    fake.hold_goals(0.3)  # the goal runs long enough for its feedback to arrive
    arm.begin_action(FakeCtx(), TOKEN)
    names = ["arm_joint_1", "arm_joint_2"]
    points = [(0.5, [0.1, 0.2], [0.0, 0.0]), (1.25, [0.3, 0.4], [0.1, 0.0])]
    soft = ImpedanceGains(profile=ImpedanceGains.PROFILE_SOFT)
    feedback = []
    res = arm.execute_joint_trajectory(
        names,
        points,
        path_tolerance=0.2,
        goal_tolerance=[0.01, 0.02],
        goal_time_tolerance=0.5,
        control_mode=IMPEDANCE,
        preemption=LATEST_WINS,
        gains=soft,
        speed_scale=0.5,
        feedback=feedback.append,
    )
    assert res.error_code == -4
    assert res.error_string == "PATH_TOLERANCE_VIOLATED"
    assert feedback
    g = fake.goals["execute_joint_trajectory"][-1]
    assert list(g.trajectory.joint_names) == names
    assert [pt_time(p) for p in g.trajectory.points] == pytest.approx([0.5, 1.25])
    np.testing.assert_allclose([p.positions for p in g.trajectory.points], [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose([p.velocities for p in g.trajectory.points], [[0, 0], [0.1, 0]])
    assert [(t.name, t.position) for t in g.path_tolerance] == [(n, 0.2) for n in names]
    assert [t.position for t in g.goal_tolerance] == pytest.approx([0.01, 0.02])
    assert (g.goal_time_tolerance.sec, g.goal_time_tolerance.nanosec) == (0, 500_000_000)
    assert (g.control_mode, g.preemption, g.gains.profile) == (IMPEDANCE, LATEST_WINS, 1)
    assert g.speed_scale == pytest.approx(0.5)
    assert (g.sender_id, bytes(g.token)) == (SENDER_ID, TOKEN)


# -- races at the end of an action ------------------------------------------------------------


def test_wait_for_server_after_end_action_is_not_in_action(arm_env):
    arm, _, _ = arm_env
    ctx = FakeCtx()
    arm.begin_action(ctx, None)
    client = arm._action("go_to_preset")
    arm.end_action()  # destroys the client
    with pytest.raises(NotInAction):
        arm._wait_for_server(client, ctx, HOLD_S)


def test_end_action_while_the_goal_is_accepted_cancels_it(arm_env):
    arm, _, fake = arm_env
    fake.hold_goals(HOLD_S)
    fake.delay_accept(0.5)  # end_action() runs between sending the goal and its acceptance
    arm.begin_action(FakeCtx(), None)
    result = []
    motion = threading.Thread(
        target=lambda: result.append(arm.move_joints([[0.3] * 7])), daemon=True
    )
    motion.start()
    deadline = time.monotonic() + HOLD_S
    while not fake.goals["execute_joint_trajectory"] and time.monotonic() < deadline:
        time.sleep(0.01)
    arm.end_action()
    assert fake.cancelled("execute_joint_trajectory")
    motion.join(HOLD_S)
    assert result == [False]
    assert arm.last_error == "action ended"
    assert arm._goal is None
