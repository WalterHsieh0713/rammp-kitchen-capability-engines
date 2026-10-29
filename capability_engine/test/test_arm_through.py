"""go_through / plan_through: several waypoints planned by cuRobo as one trajectory, run by the
driver as one /execute_joint_trajectory goal. The blending itself is RAMMP-CuRobo's
(core/rammp_curobo/blend.py, and its tests); here the arm's half: what it asks the planner, that
the plan reaches the driver unchanged, and that every failure refuses before anything moves."""

import numpy as np
import pytest
from fake_driver import FakeCtx
from rammp_arm import JOINTS, SENDER_ID, TOOL_Z
from rammp_curobo_interfaces.msg import Waypoint as WaypointMsg
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from capability_engine.arm import HOLD_LEVEL, IMPEDANCE, ArmInterface, Waypoint
from capability_engine.driver_state import ArmError

TOKEN = b"\x03" * 16
UP = [0.0, 0.0, 0.0, 1.0]
POSE = Waypoint([0.4, 0.1, 0.3], UP, blend_radius=0.02, hold=HOLD_LEVEL)
END = [0.3, 0.2, 0.1, -1.0, 0.0, 0.5, 0.0]


def _plan(fake, n=6):
    """A plan of n points from the fake's joints (zeros) to END, moving at every inner point,
    as a blended plan does, with the first waypoint's corner cut by 1.5 cm."""
    u = np.linspace(0.0, 1.0, n)[:, None]
    q = u * np.asarray(END)
    qd = np.sin(np.pi * u) * np.asarray(END)
    fake.plan_result(q.tolist(), velocities=qd.tolist(), corner_m=[0.015, 0.0])
    return q, qd


def _times(points):
    return [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points]


def test_go_through_runs_the_plan_as_one_goal(arm_env):
    arm, _, fake = arm_env
    q, qd = _plan(fake)
    fake.finish_at("execute_joint_trajectory", joints=END)
    arm.begin_action(FakeCtx(), TOKEN)
    assert arm.go_through([POSE, Waypoint(joints=END)], "through") is True

    (g,) = fake.goals["plan_through"]
    pose, joints = g.waypoints
    assert (pose.kind, joints.kind) == (WaypointMsg.POSE, WaypointMsg.JOINTS)
    p, o = pose.pose.position, pose.pose.orientation
    np.testing.assert_allclose([p.x, p.y, p.z, o.x, o.y, o.z, o.w], [0.4, 0.1, 0.3, *UP])
    assert (pose.hold, pose.blend_radius) == (HOLD_LEVEL, pytest.approx(0.02))
    np.testing.assert_allclose(joints.joints, END)
    assert (joints.straight, joints.blend_radius) == (False, 0.0)
    np.testing.assert_allclose(g.start_joints, [0.0] * 7)  # the measured joints

    (t,) = fake.goals["execute_joint_trajectory"]
    assert list(t.trajectory.joint_names) == JOINTS
    np.testing.assert_allclose([pt.positions for pt in t.trajectory.points], q)
    np.testing.assert_allclose([pt.velocities for pt in t.trajectory.points], qd)
    assert _times(t.trajectory.points) == pytest.approx(np.arange(6) * 0.1)
    assert [tol.position for tol in t.path_tolerance] == pytest.approx([0.35] * 7)
    assert t.speed_scale == pytest.approx(0.3)  # GO_TO_SPEED_SCALE
    assert (t.sender_id, bytes(t.token)) == (SENDER_ID, TOKEN)
    assert not fake.goals["go_to_ee_pose"] and not fake.goals["go_to_joint_config"]


def test_go_through_passes_the_driver_fields(arm_env):
    arm, _, fake = arm_env
    _plan(fake)
    fake.finish_at("execute_joint_trajectory", joints=END)
    arm.begin_action(FakeCtx(), None)
    straight = Waypoint(joints=END, straight=True)
    assert arm.go_through([straight], speed_scale=0.7, control_mode=IMPEDANCE, gains="soft")
    assert fake.goals["plan_through"][-1].waypoints[0].straight
    t = fake.goals["execute_joint_trajectory"][-1]
    assert (t.speed_scale, t.control_mode) == (pytest.approx(0.7), IMPEDANCE)


def test_go_through_checks_where_the_arm_ended(arm_env):
    arm, _, fake = arm_env
    _plan(fake)
    fake.finish_at("execute_joint_trajectory", flange=([0.4, 0.1, 0.315 - TOOL_Z], UP))
    arm.begin_action(FakeCtx(), None)
    assert arm.go_through([Waypoint(joints=END, blend_radius=0.05), POSE]) is False
    assert "tool 1.5 cm from the goal" in arm.last_error


def test_a_failed_plan_refuses_before_anything_moves(arm_env):
    arm, _, fake = arm_env  # no plan_result: the fake's planner fails
    arm.begin_action(FakeCtx(), None)
    assert arm.go_through([POSE, Waypoint(joints=END)], "through") is False
    assert "through: no plan" in arm.last_error and "nothing sent" in arm.last_error
    assert fake.goals["plan_through"] and not fake.goals["execute_joint_trajectory"]


def test_a_driver_failure_is_a_refusal(arm_env):
    arm, _, fake = arm_env
    _plan(fake)
    fake.set_result("execute_joint_trajectory", -4)
    arm.begin_action(FakeCtx(), None)
    assert arm.go_through([Waypoint(joints=END)], "through") is False
    assert "through ended PATH_TOLERANCE_VIOLATED" in arm.last_error


def test_no_waypoints_and_bad_gains_raise_before_planning(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    with pytest.raises(ArmError, match="no waypoints"):
        arm.go_through([])
    with pytest.raises(ArmError, match="gains"):
        arm.go_through([Waypoint(joints=END)], gains="bogus")
    assert not fake.goals["plan_through"]


def test_plan_through_returns_the_corners(arm_env):
    arm, _, fake = arm_env
    q, _ = _plan(fake)
    arm.begin_action(FakeCtx(), None)
    plan = arm.plan_through([POSE, Waypoint(joints=END)], [0.0] * 7)
    assert plan.success and plan.corner_m == pytest.approx([0.015, 0.0])
    np.testing.assert_allclose(plan.q, q)


def test_mock_go_through_ends_on_the_last_waypoint(ros):
    node = Node("arm_mock_through")
    arm = ArmInterface(node, ReentrantCallbackGroup(), mock=True)
    try:
        arm.begin_action(FakeCtx(), None)
        assert arm.go_through([POSE, Waypoint(joints=END)]) is True
        np.testing.assert_allclose(arm.raw_joints(), END)
    finally:
        arm.end_action()
        node.destroy_node()
