"""SheppyArm's constants and pure path helpers reproduce the goldens."""

import numpy as np
import pytest
import sheppy_arm
from conftest import G, cornell_sim


def test_cartesian_knots_match_goldens():
    sim = cornell_sim()
    for case in G["cartesian_knots"]:
        knots, lengths = sheppy_arm.cartesian_knots(
            sim, case["ee_start"], case["q_start"], case["traj"]
        )
        if case["knots"] is None:
            assert knots is None
        else:
            np.testing.assert_allclose(knots, case["knots"], atol=1e-6)
            np.testing.assert_allclose(lengths, case["lengths"], atol=1e-9)


def test_joint_trajectory_matches_goldens():
    for case in G["joint_trajectory"]:
        t, q, qd = sheppy_arm.joint_trajectory(
            case["q0"], case["knots"], case["cart_lengths"], case["speed_scale"]
        )
        np.testing.assert_allclose(t, case["t"], atol=1e-6)
        np.testing.assert_allclose(q, case["q"], atol=1e-6)
        np.testing.assert_allclose(qd, case["qd"], atol=1e-6)


def test_speed_constants_read_at_call_time():
    case = G["joint_trajectory"][0]
    t1, _, _ = sheppy_arm.joint_trajectory(case["q0"], case["knots"])
    sheppy_arm.JOINT_SPEED *= 0.5
    try:
        t2, _, _ = sheppy_arm.joint_trajectory(case["q0"], case["knots"])
    finally:
        sheppy_arm.JOINT_SPEED *= 2
    assert t2[-1] > t1[-1]


def test_contact_speed_scale():
    assert sheppy_arm.contact_speed_scale() == pytest.approx(min(1.0, 0.09 / 0.16, 0.5 / 0.85))
