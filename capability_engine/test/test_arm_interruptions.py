"""Every arm operation against every way it can be cut short. Afterwards, whatever happened:
nothing the action started is left running at the driver, no session is left open, the action's
entities are gone, and the next action works. (Issue #24: one test for the whole space, not one
per bug found.)"""

import threading
import time

import numpy as np
import pytest
from fake_driver import FakeCtx, counts

from capability_engine import arm as arm_module

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


def plan(arm):
    return arm.plan_to_joints([0.1] * 7, [0.0] * 7)


def stream(arm):
    with arm.stream("ee_twist") as s:
        while True:
            s.send(TWIST)
            time.sleep(0.02)


def gripper(arm):
    return arm.gripper(1.0, timeout_s=HOLD_S)


GOALS = {trajectory: "execute_joint_trajectory", go_to: "go_to_preset", plan: "plan_to_joints"}
OPERATIONS = [trajectory, go_to, plan, stream, gripper]


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
