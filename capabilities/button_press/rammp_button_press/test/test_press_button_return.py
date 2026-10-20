"""Leaving the panel: back straight out before any cuRobo move, and never cuRobo right at the panel.

2026-10-14: the return to start handed the whole move to cuRobo with the fingertips 2 cm from the
button; cuRobo's world has no microwave, and its path swept the gripper into it. Runs Run's moves
against a stand-in arm that records what would be sent (no ROS, nothing moves).
"""

import argparse

import numpy as np
import pytest

pb = pytest.importorskip("rammp_button_press.press_button")  # needs rclpy + rammp interfaces
from rammp_button_press import Abort  # noqa: E402

N_OUT = np.array([-0.993, 0.104, 0.053]) / np.linalg.norm([-0.993, 0.104, 0.053])  # out of the panel, toward the arm
SPOT = np.array([0.675, -0.317, 0.077])  # the pre-press spot
HOME = np.array([0.577, 0.001, 0.433])


class FakeArm:
    def __init__(self, p):
        self.p, self.R = np.asarray(p, float), np.eye(3)

    def ee_pose(self):
        return self.p, self.R


def run_at(p, execute=True, fail=()):
    """A Run whose straight moves and cuRobo moves are recorded; straight legs named in `fail` abort."""
    run = pb.Run.__new__(pb.Run)
    run.args = argparse.Namespace(execute=execute, presses=1, push_mm=1.0, stop_nm=4.0)
    run.arm = FakeArm(p)
    run.sent, run.speeds = [], {}

    def move(path, name, track_abort_m=pb.TRACK_ABORT_M, cart_speed=pb.CART_SPEED):
        if not execute:
            return
        run.sent.append(("straight", name, path))
        run.speeds[name] = cart_speed
        if name in fail:
            run.arm.p = np.asarray(path[0][0], float)  # stopped partway along the leg
            raise Abort(f"{name}: ended 1.2 cm from the target -- HOLDING HERE")
        run.arm.p = np.asarray(path[-1][0], float)

    def long_move(p, Rm, name):
        run.sent.append(("curobo", name, p))
        run.arm.p = np.asarray(p, float)

    run.move, run.long_move, run.log = move, long_move, lambda rec: None
    run.reanchor_seed = lambda why: run.sent.append(("reseed", why, None))
    return run


def test_back_to_start_backs_straight_out_first():
    run = run_at(SPOT)
    pb.back_to_start(run, HOME, np.eye(3), N_OUT)
    kinds = [(k, n) for k, n, _ in run.sent]
    assert kinds == [("straight", "back out from the panel"), ("curobo", "back to start")]
    path = run.sent[0][2]
    end = path[-1][0]
    assert np.allclose(end, SPOT + pb.RETREAT_OUT_M * N_OUT)  # straight out along the normal
    assert all(np.allclose(Rm, np.eye(3)) for _, Rm in path)  # without turning the wrist
    assert all(abs(float((p - SPOT) @ N_OUT) - np.linalg.norm(p - SPOT)) < 1e-9 for p, _ in path)
    assert np.allclose(run.sent[1][2], HOME)


def test_the_retreat_clears_the_panel_like_staging():
    assert pb.RETREAT_OUT_M >= 0.10  # cuRobo starts well clear of it


def test_a_bail_from_the_approach_backs_out_before_curobo():
    run = run_at(SPOT + 0.05 * N_OUT)
    run.go_back(HOME, np.eye(3), "move aborted", n_out=N_OUT)
    assert [(k, n) for k, n, _ in run.sent] == [("straight", "back out from the panel"), ("curobo", "bail")]


def test_a_bail_far_from_the_panel_goes_straight_to_curobo():
    run = run_at(SPOT + 0.30 * N_OUT)
    run.go_back(HOME, np.eye(3), "move aborted")
    assert [(k, n) for k, n, _ in run.sent] == [("curobo", "bail")]


def test_an_aborted_press_returns_to_the_spot_in_a_straight_line():
    run = run_at(SPOT)

    def press_contact(p_spot, n_out, name, log, push, stop_nm):
        run.arm.p = SPOT - 0.008 * N_OUT  # stopped 8 mm in
        raise Abort(f"{name}: no touch within 30 mm -- session closed, HOLDING HERE")

    run.arm.press_contact = press_contact
    with pytest.raises(Abort, match="press 1"):
        run.press(SPOT, np.eye(3), N_OUT)
    assert [k for k, _, _ in run.sent if k != "reseed"] == ["straight"]  # back to the spot; never cuRobo at the panel
    assert run.sent[-2][0] == "reseed"  # the leg back is checked from the joints the press left
    assert np.allclose(run.sent[-1][2][-1][0], SPOT)


def test_a_dry_run_leaves_without_moving():
    run = run_at(SPOT, execute=False)
    run.retreat(N_OUT)
    assert run.sent == []


# ---- the way in: cuRobo only to a lead-in point, then straight in (2026-10-15: cuRobo's path to the
# staging pose hit the top of the microwave on the raised counter)


def test_staging_goes_to_the_lead_in_point_then_straight_in():
    p_st = SPOT + 0.15 * N_OUT
    run = run_at(HOME)
    pb.go_to_staging(run, p_st, np.eye(3), N_OUT, 0.20)
    assert [(k, n) for k, n, _ in run.sent] == [
        ("curobo", "to the lead-in point"),
        ("reseed", "at the lead-in point"),
        ("straight", "straight in to staging"),
    ]  # checked from cuRobo's end
    assert np.allclose(run.sent[0][2], p_st + 0.20 * N_OUT)  # 20 cm further out than staging
    path = run.sent[2][2]
    assert np.allclose(path[-1][0], p_st)
    assert all(abs(float((p - p_st) @ N_OUT) - np.linalg.norm(p - p_st)) < 1e-9 for p, _ in path)  # on the normal
    assert run.speeds["straight in to staging"] == pb.LEAD_IN_SPEED_MPS > pb.CART_SPEED  # the quick leg


def test_lead_in_zero_is_one_curobo_move():
    run = run_at(HOME)
    pb.go_to_staging(run, SPOT + 0.15 * N_OUT, np.eye(3), N_OUT, 0.0)
    assert [(k, n) for k, n, _ in run.sent] == [("curobo", "to staging")]


def test_lead_in_defaults_to_10_cm():
    assert pb.build_parser().parse_args([]).lead_in == pb.STAGE_LEAD_IN_M == 0.10
    assert pb.build_parser().parse_args(["--lead-in", "0"]).lead_in == 0.0


class LeadArm:
    """IK reaches a lead-in point only up to `reach` m out from staging (2026-10-15: 10 cm put the
    point beside the arm's base, 7 mm short)."""

    def __init__(self, p_st, reach):
        self.p_st, self.reach = p_st, reach
        self.kin = self

    def joints(self):
        return np.zeros(7)

    def ee_pose(self):
        return HOME, np.eye(3)

    def solve_translation(self, q, d, posture=None, rot=None):
        out = float((HOME + d - self.p_st) @ N_OUT)
        return np.zeros(7), (0.0 if out <= self.reach + 1e-9 else 0.007), None, 0.0, []

    def precheck_poses(self, *a, **k):
        return 0.0, 0.0, None, []


@pytest.mark.parametrize("reach,used", [(0.20, 0.10), (0.06, 0.05), (0.02, 0.0)])
def test_the_lead_in_shrinks_to_what_the_arm_can_reach(reach, used):
    p_st = SPOT + 0.27 * N_OUT
    run = run_at(HOME)
    run.arm, run.seed_posture = LeadArm(p_st, reach), None
    lead, p_lead, leg = pb.pick_lead_in(run, p_st, np.eye(3), N_OUT, 0.10)
    assert abs(lead - used) < 1e-9 and np.allclose(p_lead, p_st + used * N_OUT)
    assert (leg is None) == (used == 0.0)
