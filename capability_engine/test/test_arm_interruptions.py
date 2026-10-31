"""Every arm operation against every way it can be cut short. Afterwards, whatever happened:
nothing the action started is left running at the driver, no session is left open, the action's
entities are gone, and the next action works. (Issue #24: one test for the whole space, not one
per bug found.)"""

import importlib.util
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fake_driver import FakeCtx, counts
from rammp_arm import TOOL_Z
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.publisher import Publisher
from scipy.spatial.transform import Rotation as R

from capability_engine import arm as arm_module
from capability_engine.driver_state import DriverState, NotInAction

# The goldens and the Cornell sim, from tests/arm/conftest.py (loaded by path: see test_arm_motion).
_spec = importlib.util.spec_from_file_location(
    "arm_goldens", Path(__file__).parents[2] / "tests" / "arm" / "conftest.py"
)
_goldens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_goldens)
G, cornell_sim = _goldens.G, _goldens.cornell_sim

HOLD_S = 5.0  # the fake runs every goal this long, so each is still running when interrupted
INTERRUPT_AFTER_S = 0.3  # when the timed interruptions land
ANSWER_WAIT_S = 0.3  # GOAL_ACCEPT_S and SERVICE_CALL_S in the "no answer" cases
LATE_ANSWER_S = 0.6  # the driver answers then: after the call gave up, within end_action()'s wait
SETTLE_S = 2.0  # how long the driver may take to see a goal cancelled
TWIST = np.ones(6) * 0.01


def trajectory(arm):
    return arm.move_joints([[0.3] * 7])


def go_to(arm):
    return arm.go_to_preset("home")


def go_to_many(arm):
    return arm.go_to_ee_poses(
        [
            arm_module.PoseTarget([0.4, 0.1, 0.3], [0.0, 0.0, 0.0, 1.0], blend_radius=0.05),
            arm_module.PoseTarget([0.5, 0.1, 0.2], [0.0, 0.0, 0.0, 1.0]),
        ]
    )


def go_to_many_joints(arm):
    return arm.go_to_joint_configs(
        [arm_module.JointTarget([0.1] * 7, blend_radius=0.05), arm_module.JointTarget([0.2] * 7)]
    )


def plan(arm):
    return arm.plan_to_joints([0.1] * 7, [0.0] * 7)


def stream(arm):
    with arm.stream("ee_twist") as s:
        while True:
            s.send(TWIST)
            time.sleep(0.02)


def gripper(arm):
    return arm.gripper(1.0, timeout_s=HOLD_S)


GOALS = {
    trajectory: "execute_joint_trajectory",
    go_to: "go_to_preset",
    go_to_many: "go_to_ee_poses",
    go_to_many_joints: "go_to_joint_configs",
    plan: "plan_to_joints",
}
OPERATIONS = [trajectory, go_to, go_to_many, go_to_many_joints, plan, stream, gripper]


def cancel(arm, ctx, fake):
    threading.Timer(INTERRUPT_AFTER_S, ctx.cancel).start()


def end_action(arm, ctx, fake):
    threading.Timer(INTERRUPT_AFTER_S, arm.end_action).start()


def estop(arm, ctx, fake):
    threading.Timer(INTERRUPT_AFTER_S, fake.engage_estop).start()


def no_answer(arm, ctx, fake):
    """The driver answers only after the call has given up, accepting: it must be undone."""
    fake.delay_accept(LATE_ANSWER_S)
    fake.delay_open_stream(LATE_ANSWER_S)


def refused(arm, ctx, fake):
    """The driver says no: nothing to undo, and nothing of anyone else's to touch."""
    for name in GOALS.values():
        fake.accept(name, False)
    fake.delay_open_stream(0.0, accept=False)


INTERRUPTIONS = [cancel, end_action, estop, no_answer, refused]
# The gripper streams setpoints with no goal or session: the driver cannot fail to answer it.
NOT_APPLICABLE = {(gripper, no_answer), (gripper, refused)}


@pytest.mark.parametrize("interrupt", INTERRUPTIONS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("operation", OPERATIONS, ids=lambda f: f.__name__)
def test_an_interrupted_operation_leaves_nothing_behind(arm_env, monkeypatch, operation, interrupt):
    if (operation, interrupt) in NOT_APPLICABLE:
        pytest.skip("the gripper has nothing for the driver to answer")
    arm, node, fake = arm_env
    monkeypatch.setattr(arm_module, "GOAL_ACCEPT_S", ANSWER_WAIT_S)
    monkeypatch.setattr(arm_module, "SERVICE_CALL_S", ANSWER_WAIT_S)
    fake.hold_goals(HOLD_S)
    fake.plan_result([[0.0] * 7, [0.1] * 7])
    base = counts(node)
    ctx = FakeCtx()
    arm.begin_action(ctx, None)
    interrupt(arm, ctx, fake)
    t0 = time.monotonic()
    try:
        operation(arm)
    except Exception:  # noqa: BLE001 -- each interruption ends it its own way; the after matters
        pass
    assert time.monotonic() - t0 < HOLD_S, "the interruption did not cut the operation short"
    arm.end_action()

    deadline = time.monotonic() + SETTLE_S
    while fake.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake.running() == 0, "a goal the action started is still running at the driver"
    assert not fake.session_open(), "a streaming session the action opened is still open"
    assert counts(node) == base, "the action's entities were not all destroyed"
    assert not arm._goals

    if interrupt is estop:
        return  # a latched e-stop is the driver's state, not the action's
    fake.hold_goals(0.0)
    fake.delay_accept(0.0)
    fake.delay_open_stream(0.0)
    for name in GOALS.values():
        fake.accept(name, True)
    arm.begin_action(FakeCtx(), None)
    assert arm.move_joints([[0.3] * 7]), arm.last_error


# -- end_action() from another thread, at each point along a RammpArm motion's path ------------

CARTESIAN = next(c for c in G["send"] if c["kind"] == "move_cartesian")
GOAL_MOTIONS = {
    "move_joints": lambda arm: arm.move_joints([[0.3] * 7]),
    "move_cartesian": lambda arm: arm.move_cartesian(CARTESIAN["traj"]),
    "go_to_ee_pose": lambda arm: arm.go_to_ee_pose([0.4, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0]),
    "go_to_joint_config": lambda arm: arm.go_to_joint_config([0.1] * 7),
    "go_to_preset": lambda arm: arm.go_to_preset("home"),
    "go_to_ee_poses": go_to_many,
    "go_to_joint_configs": go_to_many_joints,
}
# Where the end lands: (class, method, after it returns). end_goal comes after the result.
LANDINGS = {
    "after_server_wait": (arm_module.ArmInterface, "_wait_for_server", True),
    "begin_goal": (DriverState, "begin_goal", False),
    "send_goal_async": (ActionClient, "send_goal_async", False),
    "get_result_async": (ClientGoalHandle, "get_result_async", False),
    "end_goal": (DriverState, "end_goal", False),
}
ENDER_HOLD_S = 0.3  # how long the motion waits for the ender, which may wait for its guard
GOAL_S = 0.5  # how long the fake runs each goal: still running at get_result_async


def end_at(monkeypatch, arm, cls, name, after):
    """end_action(), from another thread, the first time the calling thread reaches cls.name."""
    caller, ender = threading.current_thread(), threading.Thread(target=arm.end_action)
    original = getattr(cls, name)

    def landing(*args, **kwargs):
        result = original(*args, **kwargs) if after else None
        if threading.current_thread() is caller and ender.ident is None:
            ender.start()
            ender.join(ENDER_HOLD_S)
        return result if after else original(*args, **kwargs)

    monkeypatch.setattr(cls, name, landing)
    return ender


def run_cut_short(arm_env, monkeypatch, motion, landing):
    """motion(arm), its action ended at `landing`: its result, once nothing is left behind."""
    arm, node, fake = arm_env
    fake.hold_goals(GOAL_S)
    cornell_sim()  # move_cartesian's IK; the fake's tool pose is the golden case's start
    ee0 = np.asarray(CARTESIAN["ee0"], float)
    fake.set_joints(CARTESIAN["q0"])
    fake.set_flange(ee0[:3] - R.from_quat(ee0[3:]).apply([0.0, 0.0, TOOL_Z]), ee0[3:])
    base = counts(node)
    arm.begin_action(FakeCtx(), None)
    ender = end_at(monkeypatch, arm, *landing)
    result = motion(arm)
    assert ender.ident is not None, "the end never landed"
    ender.join(SETTLE_S + HOLD_S)
    deadline = time.monotonic() + SETTLE_S
    while fake.running() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake.running() == 0 and not fake.session_open()
    assert counts(node) == base
    return result, arm


@pytest.mark.parametrize("landing", LANDINGS)
@pytest.mark.parametrize("motion", GOAL_MOTIONS)
def test_a_motion_cut_short_by_the_end_of_its_action_returns_false(
    arm_env, monkeypatch, motion, landing
):
    """RammpArm's motions return False whenever their action ends after they started (at
    end_goal the goal has already finished: True or False, never an exception)."""
    result, arm = run_cut_short(arm_env, monkeypatch, GOAL_MOTIONS[motion], LANDINGS[landing])
    if landing == "end_goal":
        assert isinstance(result, bool)
    else:
        assert result is False and arm.last_error == arm_module.ACTION_ENDED_MESSAGE


def test_a_motion_outside_an_action_still_raises(arm_env):
    arm, _, _ = arm_env
    with pytest.raises(NotInAction):
        arm.move_joints([[0.3] * 7])


def test_a_motion_misusing_a_closed_stream_still_raises(arm_env):
    """Only the end of the action turns NotInAction into False: a monitor that sends on a
    stream closed earlier in the same action raises it, as RammpArm's callers expect."""
    arm, _, fake = arm_env
    fake.hold_goals(GOAL_S)
    arm.begin_action(FakeCtx(), None)
    with arm.stream("ee_twist") as s:
        pass
    with pytest.raises(NotInAction, match="the stream is closed"):
        arm.move_joints([[0.3] * 7], monitor=lambda: s.send(TWIST))


def test_a_gripper_cut_short_by_the_end_of_its_action_returns_false(arm_env, monkeypatch):
    landing = (Publisher, "publish", False)
    result, _ = run_cut_short(arm_env, monkeypatch, lambda arm: arm.gripper(1.0), landing)
    assert result is False
