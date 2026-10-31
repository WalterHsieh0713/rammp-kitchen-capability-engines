import importlib.util
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fake_driver import FakeCtx, counts
from rammp_arm import TOOL_Z
from rammp_arm_interfaces.msg import ImpedanceGains
from scipy.spatial.transform import Rotation as R

from capability_engine import sim
from capability_engine.cornell_commands import (
    CartesianTrajectoryCommand,
    CloseGripperCommand,
    JointCommand,
    OpenGripperCommand,
)
from capability_engine.driver_state import ControlTaken
from capability_engine.engine import Cancelled

# tests/arm/conftest.py holds the goldens and the Cornell sim. Loaded by path under its own name:
# `from conftest import ...` reaches whichever conftest pytest loaded last.
_spec = importlib.util.spec_from_file_location(
    "arm_goldens", Path(__file__).parents[2] / "tests" / "arm" / "conftest.py"
)
_goldens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_goldens)
G, cornell_sim = _goldens.G, _goldens.cornell_sim

HOLD_S = 5.0  # how long the fake runs a goal in the tests that need one running
SEIZE_AFTER_S = 0.3  # when another sender takes control, or the arm e-stops, mid-goal
STATUS_LATE_S = 0.1  # how long after the goal halts the e-stopped /control_status arrives
PAUSE_S = 0.3  # how long a paused action is held before resume


def pt_time(p):
    return p.time_from_start.sec + p.time_from_start.nanosec * 1e-9


def fake_logs(arm):
    return arm.last_error


def cases(kind):
    return [c for c in G["send"] if c["kind"] == kind]


def assert_goal_matches(g, case):
    assert list(g.trajectory.joint_names) == [f"arm_joint_{i}" for i in range(1, 8)]
    np.testing.assert_allclose([pt_time(p) for p in g.trajectory.points], case["t"], atol=1e-6)
    np.testing.assert_allclose([p.positions for p in g.trajectory.points], case["q"], atol=1e-6)
    np.testing.assert_allclose([p.velocities for p in g.trajectory.points], case["qd"], atol=1e-6)
    assert g.control_mode == (1 if case["impedance"] else 0)
    if case["impedance"]:
        assert g.gains.profile == ImpedanceGains.PROFILE_CUSTOM
        np.testing.assert_allclose(g.gains.custom.kq, case["impedance"]["kq"])
        assert g.gains.custom.zeta == pytest.approx(case["impedance"]["zeta"])
        np.testing.assert_allclose(g.gains.custom.torque_limit, case["impedance"]["torque_limit"])
    assert all(
        t.position == pytest.approx(case["impedance"]["path_tol"] if case["impedance"] else 0.15)
        for t in g.path_tolerance
    )


def test_move_joints_sends_golden_trajectory(arm_env):
    arm, _, fake = arm_env
    assert cases("move_joints")
    for case in cases("move_joints"):
        fake.set_joints(case["q0"])
        arm.begin_action(FakeCtx(), None)
        assert arm.move_joints(
            case["knots"], case["cart_lengths"], case["speed_scale"], case["impedance"]
        )
        assert_goal_matches(fake.goals["execute_joint_trajectory"][-1], case)
        arm.end_action()


def set_tool_pose(fake, q, ee):
    """Joints `q`, and the flange placed so the arm reads tool pose `ee` (the golden's frame)."""
    ee = np.asarray(ee, float)
    fake.set_joints(q)
    fake.set_flange(ee[:3] - R.from_quat(ee[3:]).apply([0.0, 0.0, TOOL_Z]), ee[3:])


def test_move_cartesian_sends_golden(arm_env):
    arm, _, fake = arm_env
    cornell_sim()  # the process's one sim: the arm's make_sim() returns this copy
    assert cases("move_cartesian")
    for case in cases("move_cartesian"):
        ee0 = np.asarray(case["ee0"], float)
        set_tool_pose(fake, case["q0"], ee0)
        arm.begin_action(FakeCtx(), None)
        np.testing.assert_allclose(arm.get_state()["ee_pos"], ee0, atol=1e-12)
        traj = [(pos, quat) for pos, quat in case["traj"]]
        assert arm.move_cartesian(traj, case["impedance"], case["speed_scale"])
        assert_goal_matches(fake.goals["execute_joint_trajectory"][-1], case)
        arm.end_action()


def test_token_attached_when_given(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), b"\x01" * 16)
    assert arm.move_joints([[0.3] * 7]) is True
    assert bytes(fake.goals["execute_joint_trajectory"][-1].token) == b"\x01" * 16


def test_no_token_leaves_the_field_default(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    assert arm.move_joints([[0.3] * 7])
    assert not any(fake.goals["execute_joint_trajectory"][-1].token)


def test_goal_rejected_without_token(arm_env):
    arm, _, fake = arm_env
    fake.enforce_arbitration()
    arm.begin_action(FakeCtx(), None)
    t0 = time.monotonic()
    assert arm.move_joints([[0.3] * 7]) is False
    assert time.monotonic() - t0 < 2.0
    assert "no control token" in fake_logs(arm)
    assert not fake.goals["execute_joint_trajectory"]  # nothing sent


def test_goal_not_accepted_returns_false(arm_env):
    arm, _, fake = arm_env
    fake.accept("execute_joint_trajectory", False)
    arm.begin_action(FakeCtx(), None)
    assert arm.move_joints([[0.3] * 7]) is False
    assert "REJECTED" in fake_logs(arm)


def test_failed_goal_returns_false(arm_env):
    arm, _, fake = arm_env
    fake.set_result("execute_joint_trajectory", -4)
    arm.begin_action(FakeCtx(), None)
    assert arm.move_joints([[0.3] * 7]) is False
    assert "PATH_TOLERANCE_VIOLATED" in fake_logs(arm)


def test_short_path_sends_nothing(arm_env):
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), None)
    assert arm.move_joints([[0.0] * 7]) is True  # the fake's joints are already there
    assert not fake.goals["execute_joint_trajectory"]


def test_monitor_cancel_counts_as_success(arm_env):
    arm, _, fake = arm_env
    fake.set_result_after_cancel("execute_joint_trajectory", -6)
    fake.hold_goals(HOLD_S)  # the monitor must see a running goal
    arm.begin_action(FakeCtx(), None)
    assert arm.move_joints([[0.3] * 7], monitor=lambda: True) is True
    assert arm.monitor_stopped
    assert fake.cancelled("execute_joint_trajectory")


def test_raising_monitor_cancels_the_goal(arm_env):
    arm, node, fake = arm_env
    fake.hold_goals(HOLD_S)  # without a cancel the goal would outlive the call
    base = counts(node)
    arm.begin_action(FakeCtx(), None)

    def monitor():
        raise RuntimeError("monitor failed")

    with pytest.raises(RuntimeError, match="monitor failed"):
        arm.move_joints([[0.3] * 7], monitor=monitor)
    assert fake.cancelled("execute_joint_trajectory")  # by the motion, not end_action()
    arm.end_action()
    assert counts(node) == base


def test_cancel_during_move_joints(arm_env):
    arm, _, fake = arm_env
    ctx = FakeCtx()
    fake.hold_goals(HOLD_S)
    arm.begin_action(ctx, None)
    threading.Timer(0.3, ctx.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        arm.move_joints([[0.3] * 7])
    assert time.monotonic() - t0 < 0.8
    assert fake.cancelled("execute_joint_trajectory")


def test_cancelled_before_sending_sends_nothing(arm_env):
    arm, _, fake = arm_env
    ctx = FakeCtx()
    ctx.cancel()
    arm.begin_action(ctx, None)
    with pytest.raises(Cancelled):
        arm.move_joints([[0.3] * 7])
    with pytest.raises(Cancelled):
        arm.move_cartesian([([0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0])])
    assert not fake.goals["execute_joint_trajectory"]


def test_halted_with_token_is_control_taken(arm_env):
    arm, _, fake = arm_env
    fake.hold_goals(HOLD_S)
    arm.begin_action(FakeCtx(), b"\x01" * 16)
    threading.Timer(SEIZE_AFTER_S, fake.seize, args=("teleop",)).start()
    with pytest.raises(ControlTaken, match="teleop"):
        arm.move_joints([[0.3] * 7])


def test_estopped_before_sending_returns_false(arm_env):
    arm, _, fake = arm_env
    fake.engage_estop()
    arm.begin_action(FakeCtx(), b"\x01" * 16)
    assert arm.move_joints([[0.3] * 7]) is False
    assert "arm is E-STOPPED -- clear it first" in fake_logs(arm)
    assert not fake.goals["execute_joint_trajectory"]


@pytest.mark.parametrize(
    "status_delay_s", [0.0, STATUS_LATE_S], ids=["status_first", "result_first"]
)
def test_estop_mid_goal_with_token_returns_false(arm_env, status_delay_s):
    arm, _, fake = arm_env
    fake.hold_goals(HOLD_S)
    arm.begin_action(FakeCtx(), b"\x01" * 16)
    threading.Timer(SEIZE_AFTER_S, fake.engage_estop, args=(status_delay_s,)).start()
    assert arm.move_joints([[0.3] * 7]) is False  # RammpArm's failure, not ControlTaken
    assert "trajectory ended HALTED" in fake_logs(arm)
    assert arm._driver.token == b"\x01" * 16  # an e-stop is not a takeover


def test_halted_without_token_returns_false(arm_env):
    arm, _, fake = arm_env
    fake.set_result("execute_joint_trajectory", -9)
    arm.begin_action(FakeCtx(), None)
    assert arm.move_joints([[0.3] * 7]) is False
    assert "HALTED" in fake_logs(arm)


def test_end_action_cancels_a_running_goal(arm_env):
    arm, _, fake = arm_env
    fake.hold_goals(HOLD_S)
    arm.begin_action(FakeCtx(), None)
    result = []
    motion = threading.Thread(
        target=lambda: result.append(arm.move_joints([[0.3] * 7])), daemon=True
    )
    motion.start()
    deadline = time.monotonic() + HOLD_S
    while not fake.goals["execute_joint_trajectory"] and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)  # the goal is accepted and running
    arm.end_action()
    assert fake.cancelled("execute_joint_trajectory")
    motion.join(HOLD_S)
    assert not motion.is_alive()
    assert result == [False]
    assert arm.last_error == "action ended"


def test_pause_holds_the_send_until_resume(arm_env):
    arm, _, fake = arm_env
    ctx = FakeCtx()
    ctx.pause()
    arm.begin_action(ctx, None)
    result = []
    motion = threading.Thread(
        target=lambda: result.append(arm.move_joints([[0.3] * 7])), daemon=True
    )
    motion.start()
    time.sleep(PAUSE_S)
    assert motion.is_alive()
    assert not fake.goals["execute_joint_trajectory"]
    ctx.resume()
    motion.join(HOLD_S)
    assert result == [True]
    assert len(fake.goals["execute_joint_trajectory"]) == 1


def test_ik_refusal_sets_last_error(arm_env):
    arm, _, fake = arm_env
    cornell_sim()
    case = next(c for c in G["cartesian_knots"] if c["knots"] is None)
    set_tool_pose(fake, case["q_start"], case["ee_start"])
    arm.begin_action(FakeCtx(), None)
    assert arm.move_cartesian([(pos, quat) for pos, quat in case["traj"]]) is False
    assert "IK err" in fake_logs(arm) and fake_logs(arm).endswith("-- nothing sent.")
    assert not fake.goals["execute_joint_trajectory"]


def test_last_path_records_the_tool_point(arm_env):
    arm, _, fake = arm_env
    fake.set_flange([0.5, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0])
    fake.hold_goals(0.3)
    arm.begin_action(FakeCtx(), None)
    assert arm.move_joints([[0.3] * 7])
    assert len(arm.last_path) >= 5  # ~50 Hz for 0.3 s
    t, x, y, z = arm.last_path[-1]
    assert 0.0 < t < 1.0
    np.testing.assert_allclose([x, y, z], [0.5, 0.0, 0.4 + TOOL_Z])
    assert [p[0] for p in arm.last_path] == sorted(p[0] for p in arm.last_path)


def test_keep_out_sends_nothing(arm_env):
    arm, _, fake = arm_env
    cornell_sim()
    sim.set_keep_out(lambda rb, q: "in the wheelchair" if q[0] > 0.2 else None)
    try:
        arm.begin_action(FakeCtx(), None)
        assert arm.move_joints([[0.3] * 7]) is False
        assert "keep-out zone (in the wheelchair)" in fake_logs(arm)
        assert not fake.goals["execute_joint_trajectory"]
    finally:
        sim.set_keep_out(None)


def test_execute_command_dispatch(arm_env):
    arm, _, _ = arm_env
    calls = []
    arm.move_joints = lambda knots: calls.append(("move_joints", knots)) or True
    arm.move_cartesian = lambda traj: calls.append(("move_cartesian", traj)) or True
    arm.gripper = lambda position: calls.append(("gripper", position)) or True
    arm.begin_action(FakeCtx(), None)
    traj = [([0.5, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0])]
    assert arm.execute_command(JointCommand(pos=[0.1] * 7))
    assert arm.execute_command(CartesianTrajectoryCommand(traj))
    assert arm.execute_command(OpenGripperCommand())
    assert arm.execute_command(CloseGripperCommand())
    (_, knots), (_, poses), *grips = calls
    np.testing.assert_allclose(knots, [[0.1] * 7])
    np.testing.assert_allclose(poses[0][0], traj[0][0])
    np.testing.assert_allclose(poses[0][1], traj[0][1])
    assert grips == [("gripper", 0.0), ("gripper", 1.0)]
    with pytest.raises(TypeError, match="unsupported command"):
        arm.execute_command(object())


def test_taken_before_the_first_command_raises(arm_env):
    """Another sender takes the arm after the action's first arm call (a state read) but before
    its first command: the command raises ControlTaken, not just a refusal from the driver."""
    arm, _, fake = arm_env
    arm.begin_action(FakeCtx(), b"\x01" * 16)
    arm.get_state()

    def owner():
        latest = arm._latest.get("/control_status")
        return None if latest is None else latest[0].owner_id

    deadline = time.monotonic() + HOLD_S
    while owner() is None:  # the baseline the takeover is judged against has arrived
        assert time.monotonic() < deadline, "no /control_status"
        time.sleep(0.01)
    fake.seize("teleop")
    # Seen before the command: a takeover still in flight then is the driver's to refuse.
    while owner() != "teleop":
        assert time.monotonic() < deadline, "the takeover's /control_status never arrived"
        time.sleep(0.01)
    with pytest.raises(ControlTaken, match="teleop"):
        arm.move_joints([[0.3] * 7])
