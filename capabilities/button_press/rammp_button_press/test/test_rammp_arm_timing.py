"""RammpArm's trajectory timing (pure functions; needs rclpy + rammp interfaces importable)."""

import numpy as np
import pytest

rammp_arm = pytest.importorskip("rammp_button_press.rammp_arm")


def test_unwrap_goes_the_short_way():
    q0 = np.zeros(7)
    q0[0] = 3.1
    k = np.zeros(7)
    k[0] = -3.1  # J1 continuous: +0.083 rad, not -6.2
    out = rammp_arm.unwrap_from(q0, [k])[0]
    assert out[0] == pytest.approx(3.1 + (2 * np.pi - 6.2))


def test_min_jerk_endpoints_and_rest():
    path = np.array([np.zeros(7), np.full(7, 0.1)])
    t, q, qd = rammp_arm.min_jerk_samples(path, np.array([0.0, 1.0]), 2.0)
    assert np.allclose(q[0], 0) and np.allclose(q[-1], 0.1)
    assert np.allclose(qd[0], 0) and np.allclose(qd[-1], 0)
    assert np.all(np.diff(t) <= rammp_arm.DT + 1e-9)


@pytest.mark.parametrize("dist", [0.01, 0.3])
def test_leg_respects_speed_caps(dist):
    """Peak tool speed <= CART_SPEED and peak joint speed <= JOINT_SPEED (linear joint/Cartesian map)."""
    R0 = np.eye(3)
    n = max(1, int(round(dist / 0.01)))
    poses = [(np.array([0.5 + dist * (i + 1) / n, 0.0, 0.3]), R0) for i in range(n)]
    knots = [np.full(7, 0.002 * (i + 1)) for i in range(n)]  # 0.2 deg per cm
    path, s, dur = rammp_arm.leg_timing(np.zeros(7), knots, poses, np.array([0.5, 0.0, 0.3]), R0)
    t, q, qd = rammp_arm.min_jerk_samples(path, s, dur)
    s_t = np.interp(q[:, 0], path[:, 0], s)  # arc length along the leg vs time
    assert np.max(np.gradient(s_t, t)) <= rammp_arm.CART_SPEED * 1.02
    assert np.max(np.abs(qd)) <= rammp_arm.JOINT_SPEED * 1.02
    assert dur >= rammp_arm.MIN_DURATION_S


def test_stream_setpoints_come_faster_than_the_session_timeout():
    assert 1.0 / rammp_arm.STREAM_RATE_HZ < rammp_arm.STREAM_TIMEOUT_S  # the contact press never lapses


def test_a_faster_leg_is_shorter_and_respects_its_own_cap():
    """The lead-in leg asks for more than CART_SPEED; the joint cap still holds."""
    R0 = np.eye(3)
    poses = [(np.array([0.5 + 0.2 * (i + 1) / 20, 0.0, 0.3]), R0) for i in range(20)]
    knots = [np.full(7, 0.002 * (i + 1)) for i in range(20)]
    _, _, slow = rammp_arm.leg_timing(np.zeros(7), knots, poses, np.array([0.5, 0.0, 0.3]), R0)
    path, s, fast = rammp_arm.leg_timing(np.zeros(7), knots, poses, np.array([0.5, 0.0, 0.3]), R0, cart_speed=0.15)
    assert fast < slow
    t, q, qd = rammp_arm.min_jerk_samples(path, s, fast)
    s_t = np.interp(q[:, 0], path[:, 0], s)
    assert np.max(np.gradient(s_t, t)) <= 0.15 * 1.02
    assert np.max(np.abs(qd)) <= rammp_arm.JOINT_SPEED * 1.02
