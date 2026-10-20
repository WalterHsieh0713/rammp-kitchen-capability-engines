"""RammpArm.go_to_ee_pose: every cuRobo move is slowed by GO_TO_SPEED_SCALE.

cuRobo plans at the arm's full speed: on 2026-10-14 a 51 cm staging move from HOME planned at
1.16 s and hit the microwave (cuRobo's world does not contain it) before anyone could reach the
e-stop. Runs go_to_ee_pose against a stand-in action client that keeps the goal it was sent.
"""

from types import SimpleNamespace

import numpy as np
import pytest

ra = pytest.importorskip("rammp_button_press.rammp_arm")  # needs rclpy + rammp interfaces
from builtin_interfaces.msg import Time  # noqa: E402


class Done:
    def __init__(self, value):
        self.value = value

    def done(self):
        return True

    def result(self):
        return self.value


class GoToClient:
    """Accepts any goal, keeps it, and reports success."""

    def __init__(self):
        self.goals = []

    def wait_for_server(self, timeout_sec=None):
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        res = SimpleNamespace(result=SimpleNamespace(error_code=0, error_string=""))
        return Done(SimpleNamespace(accepted=True, get_result_async=lambda: Done(res), cancel_goal_async=lambda: None))


def arm_at(p_now, execute=True):
    arm = ra.RammpArm.__new__(ra.RammpArm)  # no ROS: only what go_to_ee_pose touches
    arm.execute, arm.cancel_requested, arm.sent_goals, arm._active_gh = execute, False, 0, None
    arm._token = bytes(16)
    arm.frame = "arm_mount"
    arm.kin = SimpleNamespace(base_pos=np.zeros(3))
    arm.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)))
    arm._goto = GoToClient()
    arm.pos = np.asarray(p_now, dtype=float)
    arm.ee_pos = lambda: arm.pos
    return arm


def test_every_curobo_move_is_sent_slowed_down():
    arm = arm_at([0.577, 0.001, 0.433])
    target = np.array([0.463, -0.346, 0.20])

    def arrive(goal, feedback_callback=None):
        arm.pos = target  # the driver got there
        return GoToClient.send_goal_async(arm._goto, goal, feedback_callback)

    arm._goto.send_goal_async = arrive
    logged = []
    arm.go_to_ee_pose(target, np.eye(3), "to staging", logged.append)
    (goal,) = arm._goto.goals
    assert goal.speed_scale == ra.GO_TO_SPEED_SCALE
    assert goal.target.header.frame_id == "arm_mount"  # the nightly driver refuses any other frame
    assert logged[-1]["speed_scale"] == ra.GO_TO_SPEED_SCALE


def test_the_speed_scale_is_full_speed_and_one_the_driver_accepts():
    # full speed at the user's request (2026-10-14, after the straight retreat from the panel went
    # in); the driver refuses (does not clamp) a scale outside [0.01, 1]
    assert ra.GO_TO_SPEED_SCALE == 1.0


def test_a_dry_run_still_sends_nothing():
    arm = arm_at([0.577, 0.001, 0.433], execute=False)
    arm.go_to_ee_pose(np.array([0.463, -0.346, 0.20]), np.eye(3), "to staging", lambda rec: None)
    assert arm._goto.goals == []
