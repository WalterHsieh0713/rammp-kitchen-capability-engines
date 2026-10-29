"""Offline checks for place_container's planning (no arm, no ROS), ported from RAMMP-table's
test_place_bowl.py. The sim tests load the flatbot model (30 deg mount) from the arm test image."""

from pathlib import Path

import numpy as np
import pytest
from capability_engine import sim
from rammp_arm import TOOL_Z
from scipy.spatial.transform import Rotation

from place_container import plan

FLATBOT_URDF = Path("/fixtures/urdf/flatbot.urdf")
TABLE_Z = 0.198  # the flatbot's table, `mark`ed 2026-10-26 (arm_mount)
SPOT = [0.764, -0.248]  # the sticky note the container was placed on that day
needs_model = pytest.mark.skipif(not FLATBOT_URDF.exists(), reason="no flatbot model (run in the arm test image)")


@pytest.fixture(scope="module")
def model():
    sim.reset_for_tests()
    sim.make_sim(str(FLATBOT_URDF))
    yield plan.Model()
    sim.reset_for_tests()


def holding_pose(model, spot):
    """A plausible start: container held near the spot, a little high and back, slightly off-square."""
    d = plan.approach_direction(spot)
    pos = plan.tool_above_spot(spot, TABLE_Z, 0.25, model.reach) - 0.05 * d
    quat = (Rotation.from_quat(plan.hand_orientation(d)) * Rotation.from_euler("xz", [3, 4], degrees=True)).as_quat()
    for seed in ([0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725], [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]):
        q = model.ik(pos, quat, seed)
        if q is not None:
            return q
    return None


# -- pure geometry ------------------------------------------------------------------------------


def test_container_sits_in_front_of_the_tool_frame():
    assert plan.hold_reach() == pytest.approx(0.055 + 0.17955 - TOOL_Z)  # radius + tool -> fingertip
    assert plan.RETREAT > plan.RETREAT_MIN > plan.LIP_WIDTH  # the fingertips clear the lip


def test_hand_points_down_at_the_pitch_with_the_camera_on_the_right():
    for yaw in np.radians([0, 37, -18, 90]):
        d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        R = Rotation.from_quat(plan.hand_orientation(d, pitch_deg=5.0)).as_matrix()
        assert np.degrees(np.arcsin(-R[2, 2])) == pytest.approx(5.0)  # tool z 5 deg below horizontal
        assert np.allclose(R[:2, 2] / np.linalg.norm(R[:2, 2]), d[:2])  # ... along the heading
        assert R[:, 1] @ np.cross(d, [0, 0, 1]) > 0.99  # camera side (+y) on the right
        assert plan.bowl_tilt_deg(R, 5.0) < 1e-6
        assert not plan.camera_on_left(R)


def test_tool_sits_behind_the_container():
    pos = plan.tool_above_spot(SPOT, TABLE_Z, 0.15, 0.1)
    d = plan.approach_direction(SPOT)
    assert np.linalg.norm(pos[:2] - SPOT) == pytest.approx(0.1)
    assert np.allclose((np.array(SPOT) - pos[:2]) / 0.1, d[:2])
    assert pos[2] == pytest.approx(TABLE_Z + 0.15)


def test_hold_point_is_on_the_line_below_the_contact():
    line_q = [np.full(7, i * 0.01) for i in range(5)]
    line_z = [0.40, 0.39, 0.38, 0.37, 0.36]
    assert plan.hold_point(line_q, line_z, 0.41) is None  # above the start: hold where it is
    np.testing.assert_allclose(plan.hold_point(line_q, line_z, 0.385), np.full(7, 0.015))
    np.testing.assert_allclose(plan.hold_point(line_q, line_z, 0.30), line_q[-1])  # past the end


def test_vz_contact_catches_the_folding_wrist():
    """The flatbot's first lowering (descent_20261026_125221): -2 cm/s down, then ~-0.5 cm/s while
    the wrist folded about the fingertips. vz never got back to 0."""
    trace = [0.0] * 5 + [-0.02] * 50 + [-0.005, -0.004, -0.006, -0.005] * 20
    late, now = plan.VzContact(stop=0.0), plan.VzContact(stop=-0.01)
    assert not any(late.update(v) for v in trace)
    hits = [now.update(v) for v in trace]
    assert hits.index(True) == 56  # the 2nd message after the touch


def test_vz_contact_needs_the_hand_to_have_moved_down():
    c = plan.VzContact(stop=-0.01)
    assert not any(c.update(v) for v in [0.0, -0.002, 0.001, 0.0, 0.0])  # settling, never moving


# -- on the flatbot model ---------------------------------------------------------------------


@needs_model
def test_a_spot_plan_goes_straight_down_in_arm_mount(model):
    q_start = holding_pose(model, SPOT)
    assert q_start is not None
    moves, (zs, qs), info = plan.plan_place(model, q_start, SPOT, TABLE_Z)
    assert [n for n, _ in moves if not n.endswith("'")][-1] == "transit"
    top = model.fk(qs[0])
    assert top[2, 3] == pytest.approx(TABLE_Z + plan.SAFE_HEIGHT, abs=plan.IK_TOL)
    centre = top[:3, 3] + model.reach * plan.approach_direction(SPOT)
    assert np.linalg.norm(centre[:2] - SPOT) < plan.IK_TOL  # container centre over the spot
    for q in qs:  # straight down (true vertical: arm_mount), container held as it should be
        T = model.fk(q)
        assert np.linalg.norm(T[:2, 3] - top[:2, 3]) < plan.IK_TOL
        assert plan.bowl_tilt_deg(T[:3, :3], model.pitch) <= plan.MAX_LOWER_TILT_DEG
    assert zs[0] - zs[-1] <= plan.MAX_DROP + 1e-9


@needs_model
def test_a_big_turn_with_the_container_in_hand_is_refused(model):
    d = plan.approach_direction(SPOT)
    pos = plan.tool_above_spot(SPOT, TABLE_Z, 0.25, model.reach)
    on_top = (Rotation.from_quat(plan.hand_orientation(d)) * Rotation.from_euler("z", 90, degrees=True)).as_quat()
    q = model.ik(pos, on_top, [0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725])
    assert q is not None
    with pytest.raises(plan.Refused, match="deg from holding"):
        plan.plan_place(model, q, SPOT, TABLE_Z)


@needs_model
def test_retreat_goes_straight_back_level(model):
    q_start = holding_pose(model, SPOT)
    line = plan.plan_retreat(model, q_start)
    a, b = model.fk(line[0]), model.fk(line[-1])
    back = b[:3, 3] - a[:3, 3]
    assert np.linalg.norm(back) == pytest.approx(plan.RETREAT, abs=plan.IK_TOL)
    assert abs(back[2]) < plan.IK_TOL  # level
    assert back[:2] @ a[:2, 2] < 0  # opposite the hand's heading


@needs_model
def test_retreat_takes_a_shorter_line_past_the_lip_and_refuses_less(model, monkeypatch):
    q_start = holding_pose(model, SPOT)
    ik = model.ik
    for allowed, ok in ((16, True), (8, False)):  # of 20 ~4 mm steps: 6.35 cm (>= RETREAT_MIN), 3.2 cm
        calls = {"n": 0}

        def limited(*args, _allowed=allowed, _calls=calls, **kwargs):
            _calls["n"] += 1
            return ik(*args, **kwargs) if _calls["n"] <= _allowed else None

        monkeypatch.setattr(model, "ik", limited)
        notes = []
        if ok:
            line = plan.plan_retreat(model, q_start, log=notes.append)
            assert len(line) == allowed + 1 and notes[0].startswith("retreat: only")
            back = np.linalg.norm(model.fk(line[-1])[:3, 3] - model.fk(line[0])[:3, 3])
            assert plan.RETREAT_MIN <= back < plan.RETREAT
        else:
            with pytest.raises(plan.Refused, match="can't go straight back"):
                plan.plan_retreat(model, q_start)
    monkeypatch.setattr(model, "ik", ik)
