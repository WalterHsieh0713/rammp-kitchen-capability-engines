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
# Kinematic inputs only: the flatbot's table `mark`ed 2026-10-26 (arm_mount) and the sticky note the
# container was placed on that day. The arm moved 0.2032 m forward on the chair on 2026-11-07, so the
# real table is no longer here; the arm's kinematics in arm_mount are unchanged.
TABLE_Z = 0.198
SPOT = [0.764, -0.248]
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


def test_vz_contact_ignores_the_sag_before_it_is_armed():
    """No settle: the arm sags ~1 cm as it goes compliant (down, then still) before the descent
    gets going. Unarmed, that is forgotten; armed, the same trace is a contact."""
    sag = [-0.02] * 10 + [0.0] * 10
    c = plan.VzContact(stop=-0.01)
    assert not any(c.update(v, armed=False) for v in sag)
    assert not c.moved and c.count == 0
    armed = plan.VzContact(stop=-0.01)
    assert any(armed.update(v) for v in sag)  # what it would have read


def test_contact_arms_after_the_ignored_drop():
    length = 0.30
    assert not plan.contact_armed(0.0, length, 0.03)
    u = plan.minjerk_time(0.03 / length)
    assert not plan.contact_armed(u - 0.01, length, 0.03) and plan.contact_armed(u + 0.01, length, 0.03)
    assert plan.contact_armed(plan.minjerk_time(0.5) + 0.01, 0.04, 0.03)  # a short line: half of it
    # a line that starts just above the table: armed CONTACT_ARM_BEFORE above it, not after it
    assert plan.arm_distance(0.16, 0.08, meet=0.06 / 0.16) == pytest.approx(0.05)
    assert plan.arm_distance(0.30, 0.03, meet=0.79) == pytest.approx(0.03)  # the long line: unchanged
    line_z = [0.50, 0.40, 0.30]
    assert plan.commanded_z(line_z, 0.0) == pytest.approx(0.50)
    assert plan.commanded_z(line_z, 0.5) == pytest.approx(0.40)
    assert plan.commanded_z(line_z, 1.0) == pytest.approx(0.30)


def test_lowering_speed_and_contact_threshold_must_fit():
    plan.check_lowering(0.02, -0.01, 0.16)  # the tuned values
    plan.check_lowering(0.04, -0.02, 0.16)  # faster, threshold scaled
    with pytest.raises(plan.Refused, match="free motion would read as contact"):
        plan.check_lowering(0.01, -0.01, 0.16)  # slower, threshold not scaled
    with pytest.raises(plan.Refused, match="peak the line down"):
        plan.check_lowering(0.12, -0.05, 0.16)
    with pytest.raises(plan.Refused, match="above 0"):
        plan.check_lowering(0.0, -0.01, 0.16)


def test_descent_passes_the_table_at_the_tuned_speed():
    """move_joints times the line minimum-jerk; the scale puts 2 cm/s where the table should be."""
    cart_speed, lower = 0.16, 0.02
    for meet in (0.4, 0.69, 0.8):  # 0.69: 30 cm above the table to 10 cm below where the bottom meets it
        scale = plan.descent_speed_scale(meet, lower, cart_speed)
        peak = cart_speed * scale  # joint_trajectory: peak speed = CART_SPEED x speed_scale
        u = plan.minjerk_time(meet)
        assert plan.minjerk_distance(u) == pytest.approx(meet, abs=1e-6)
        assert peak * plan.minjerk_speed(u) / plan.minjerk_speed(0.5) == pytest.approx(lower, rel=1e-3)
        assert peak <= plan.MAX_PEAK * lower


def test_contact_counts_only_while_the_command_is_moving():
    length, lower = 0.32, 0.02
    scale = plan.descent_speed_scale(0.69, lower, 0.16)
    duration = 1.875 * length / (0.16 * scale)  # joint_trajectory's duration for a Cartesian line
    gate = [plan.contact_gate(u, length, duration, lower) for u in np.linspace(0, 1, 101)]
    assert not gate[0] and not gate[10] and not gate[95] and not gate[100]  # slow start and end
    assert gate[50] and plan.contact_gate(plan.minjerk_time(0.69), length, duration, lower)  # at the table
    open_ = [plan.minjerk_distance(u) for u, g in zip(np.linspace(0, 1, 101), gate, strict=True) if g]
    assert min(open_) < 0.2 and max(open_) > 0.8  # the table may be ~15 cm higher or ~3 cm lower than marked


class HeightModel:
    """A stand-in model whose tool z is joint 1 (m): enough for the blend radii."""

    def fk(self, q, link="tool"):
        T = np.eye(4)
        T[2, 3] = float(q[0])
        return T


def test_reset_cuts_the_turn_only_once_the_fingers_are_clear_of_the_container():
    m, q_now = HeightModel(), np.zeros(7)
    high = [("up", np.full(7, 0.20)), ("start", np.full(7, 0.25))]  # risen 20 cm: clear
    assert plan.reset_radii(m, q_now, high, 0.03) == [0.03, 0.0]
    low = [("up", np.full(7, 0.06)), ("start", np.full(7, 0.06))]  # 6 cm: fingers still at the lip
    assert plan.reset_radii(m, q_now, low, 0.03) == [0.0, 0.0]
    split = [("up'", np.full(7, 0.05)), ("up", np.full(7, 0.20)), ("start", np.full(7, 0.2))]
    assert plan.reset_radii(m, q_now, split, 0.03) == [0.03, 0.03, 0.0]  # on the line up: free
    assert plan.reset_radii(m, q_now, [("start", np.ones(7))], 0.03) == [0.0]


def test_blend_radii_are_small_below_the_carrying_height_and_zero_at_the_end():
    low, high = {"pos": [0.5, 0.0, TABLE_Z + 0.12]}, {"pos": [0.5, 0.0, TABLE_Z + plan.SAFE_HEIGHT]}
    assert plan.blend_radii([low, high, high], TABLE_Z, 0.03, 0.05) == [0.03, 0.05, 0.0]
    assert plan.blend_radii([high], TABLE_Z, 0.03, 0.05) == [0.0]
    assert plan.blend_radii([], TABLE_Z, 0.03, 0.05) == []


CARRY = ["adjust", "lift", "transit"]
WHERE = {"transit": [0.45, -0.20, 0.50]}


def test_a_failed_leg_is_named_and_placed():
    msg = plan.explain_failure(
        "  RammpArm: adjust -> lift -> transit ended PLANNING_FAILED leg 3/3: IK_FAIL: no solution", CARRY, WHERE
    )
    assert msg.startswith("no arm pose reaches the transit leg's target (tool to x=0.45 y=-0.20 z=0.50 m")
    assert "0.49 m from the base axis" in msg and "Nothing moved." in msg and "Next: " in msg
    assert msg.endswith("[driver: adjust -> lift -> transit ended PLANNING_FAILED leg 3/3: IK_FAIL: no solution]")


def test_collisions_with_the_chair_say_what_to_move():
    start = plan.explain_failure(
        "ended PLANNING_FAILED leg 1/3: INVALID_START_STATE_WORLD_COLLISION [start collides with person/torso]", CARRY
    )
    assert start.startswith("cuRobo won't plan from where the arm is") and "move the arm clear" in start
    goal = plan.explain_failure("ended PLANNING_FAILED leg 3/3: goal collides with person/knees", CARRY, WHERE)
    assert "transit's target is inside the chair's collision volumes" in goal and "move the spot" in goal
    back = plan.explain_failure("ended PLANNING_FAILED leg 2/2: goal collides with seat/backrest", ["up", "start"])
    assert "saved start position is inside them" in back and "reset with save" in back


def test_other_failures_point_at_the_right_fix():
    def text(e, legs=CARRY):
        return plan.explain_failure(e, legs)

    assert "adjust_mode:=joints" in text("ended PLANNING_FAILED leg 2/3: CONSTRAINT_VIOLATED tilt 4 deg")
    assert "could not be joined" in text("ended PLANNING_FAILED blend: DT_EXCEPTION")
    assert "found no smooth path for the lift leg" in text("ended PLANNING_FAILED leg 2/3: TRAJOPT_FAIL")
    assert "stopped short" in text("transit: tool 2.3 cm from the goal", ["transit"])
    assert "e-stop" in text("ended HALTED")
    assert "not running" in text("adjust: no action server (the go_to_* actions need the cuRobo planner)")
    assert "busy" in text("ended PLANNING_FAILED leg 1/3: planner busy -- one plan at a time")
    assert "arm node is up" in text("StateTimeout: no /joint_states, /ee_state or /gripper_state", ["place"])
    assert text("something new", ["lower"]).startswith("the lower leg did not complete.")
    assert "smooth" not in text("velocity threshold exceeded", ["lower"])  # 'HOLD' inside a word: not a hold


# -- on the flatbot model ---------------------------------------------------------------------


def test_the_approach_takes_the_line_down_to_lower_from():
    zs = np.round(np.arange(0.50, 0.18 - 1e-9, -plan.PATH_STEP), 6)
    qs = [np.full(7, z) for z in zs]  # stand-ins: each q tagged by its z
    table_z, lower_from = 0.20, 0.06
    approach, (zs2, qs2) = plan.split_approach((zs, qs), table_z, lower_from)
    end = table_z + plan.BOWL_DEPTH + lower_from
    assert approach[-1][0] == "approach" and all(n == "approach'" for n, _ in approach[:-1])
    # the approach ends, and the slow line starts, at `end` (to the 2 mm step, at or above it)
    assert end - 1e-9 <= approach[-1][1][0] <= end + plan.PATH_STEP + 1e-9
    assert zs2[0] == approach[-1][1][0]
    assert np.all(np.diff([q[0] for _, q in approach]) < 0)  # going down
    assert np.all(np.diff([zs[0]] + [q[0] for _, q in approach]) >= -plan.APPROACH_EVERY - 1e-9)  # every 4 cm
    off, (zs3, _) = plan.split_approach((zs, qs), table_z, 0.0)
    assert off == [] and len(zs3) == len(zs)  # off: all of it slow


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
    # the carrying motion's targets: adjust unheld, then each held target keeps the previous quat
    # where its hold needs it (lift), and ends on the top of the line down
    wps = info["waypoints"]
    assert [w["name"] for w in wps][0] == "adjust" and wps[-1]["name"] == "transit"
    assert wps[0]["hold"] == "none" and all(w["hold"] == "level" for w in wps[1:])
    for a, b in zip(wps, wps[1:], strict=False):
        if b["name"] == "lift":
            np.testing.assert_allclose(b["quat"], a["quat"])
    np.testing.assert_allclose(wps[-1]["pos"], top[:3, 3], atol=plan.IK_TOL)


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
        else:  # the swing gets no IK either: refused, saying both
            with pytest.raises(plan.Refused, match="can't back out.*swing left"):
                plan.plan_retreat(model, q_start)
    monkeypatch.setattr(model, "ik", ik)


@needs_model
def test_retreat_swings_out_where_the_line_back_runs_out(model, monkeypatch):
    q_start = holding_pose(model, SPOT)
    T0, tip0 = model.fk(q_start), model.fk(q_start, "tip")
    heading = np.array([T0[0, 2], T0[1, 2], 0.0]) / np.hypot(T0[0, 2], T0[1, 2])
    centre = tip0[:2, 3] + plan.BOWL_RADIUS * heading[:2]
    ik, start_rot = model.ik, Rotation.from_matrix(T0[:3, :3])

    def no_straight_line(pos, quat, *args, **kwargs):  # the straight line keeps the orientation
        if (Rotation.from_quat(quat) * start_rot.inv()).magnitude() < 1e-6:
            return None
        return ik(pos, quat, *args, **kwargs)

    monkeypatch.setattr(model, "ik", no_straight_line)
    notes = []
    line = plan.plan_retreat(model, q_start, log=notes.append)
    assert "swings out" in notes[0] and "WARNING" in notes[0]  # no straight part: still at the wall
    T1, tip1 = model.fk(line[-1]), model.fk(line[-1], "tip")
    yaw = np.degrees(np.arctan2(np.cross(T0[:2, 2], T1[:2, 2]), T0[:2, 2] @ T1[:2, 2]))
    assert abs(yaw) == pytest.approx(plan.SWING_DEG, abs=2.0)  # turned, seen from above
    assert abs(T1[2, 3] - T0[2, 3]) < plan.IK_TOL  # at the same height: still straddling the lip
    assert np.hypot(*(tip1[:2, 3] - centre)) >= plan.BOWL_RADIUS + plan.LIP_WIDTH  # past the lip
    flange0, flange1 = model.fk(q_start, "flange"), model.fk(line[-1], "flange")
    assert np.linalg.norm(flange1[:3, 3] - flange0[:3, 3]) < 0.01  # the wrist stays put


@needs_model
def test_retreat_backs_off_the_wall_before_it_swings(model, monkeypatch):
    q_start = holding_pose(model, SPOT)
    T0 = model.fk(q_start)
    ik, start_rot, back = model.ik, Rotation.from_matrix(T0[:3, :3]), -T0[:2, 2] / np.hypot(*T0[:2, 2])

    def short_straight_line(pos, quat, *args, **kwargs):  # the straight line fits only 2 cm
        straight = (Rotation.from_quat(quat) * start_rot.inv()).magnitude() < 1e-6
        if straight and (np.asarray(pos)[:2] - T0[:2, 3]) @ back > 0.02 + 1e-6:
            return None
        return ik(pos, quat, *args, **kwargs)

    monkeypatch.setattr(model, "ik", short_straight_line)
    notes = []
    line = plan.plan_retreat(model, q_start, log=notes.append)
    assert "backs 2.0 cm, then the hand swings out" in notes[0] and "WARNING" not in notes[0]
    turned = [
        np.degrees((Rotation.from_matrix(model.fk(q)[:3, :3]) * start_rot.inv()).magnitude()) for q in line
    ]
    first_turn = next(k for k, d in enumerate(turned) if d > 0.5)
    tip_back = (model.fk(line[first_turn - 1], "tip")[:2, 3] - model.fk(q_start, "tip")[:2, 3]) @ back
    assert tip_back == pytest.approx(0.02, abs=plan.IK_TOL)  # straight back 2 cm, THEN the turn


@needs_model
def test_the_contact_moves_are_checked_at_their_ends(model):
    q_start = holding_pose(model, SPOT)
    _, descent, _ = plan.plan_place(model, q_start, SPOT, TABLE_Z)
    checks = plan.contact_zone_checks(model, descent, TABLE_Z)
    assert [c[0] for c in checks] == ["lower", "retreat"]
    np.testing.assert_allclose(checks[0][1], descent[1][0])
    np.testing.assert_allclose(checks[0][2], descent[1][-1])
    meet = model.fk(checks[1][1])[2, 3]
    assert meet == pytest.approx(TABLE_Z + plan.BOWL_DEPTH, abs=plan.PATH_STEP)
