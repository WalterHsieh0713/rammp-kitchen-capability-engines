"""Tests for the panel frame the pre-press spot is stored in (pure numpy)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rammp_button_press.panel_frame import (
    from_panel,
    panel_frame,
    pose_path,
    rot_angle_deg,
    rot_from_panel,
    rot_to_panel,
    to_panel,
)


def test_frame_is_right_handed_orthonormal():
    _, R = panel_frame([0.5, -0.3, 0.2], [-0.9, 0.1, 0.2])
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert R[2, 1] > 0  # panel y has a positive world-up component


def test_round_trip():
    o, R = panel_frame([0.56, -0.31, 0.21], [-0.8, 0.12, 0.6])
    p = np.array([0.53, -0.30, 0.20])
    np.testing.assert_allclose(from_panel(to_panel(p, o, R), o, R), p, atol=1e-12)


def test_offset_follows_a_moved_panel():
    # Teach against one panel pose, replay against the panel shifted and yawed 10 deg.
    o1, R1 = panel_frame([0.56, -0.31, 0.21], [-1.0, 0.0, 0.0])
    ee = o1 + np.array([-0.02, 0.0, 0.0])  # 2 cm out from the panel along its normal
    local = to_panel(ee, o1, R1)
    np.testing.assert_allclose(local, [0, 0, 0.02], atol=1e-12)
    yaw = np.radians(10)
    n2 = [-np.cos(yaw), -np.sin(yaw), 0.0]
    o2, R2 = panel_frame([0.58, -0.28, 0.21], n2)
    np.testing.assert_allclose(from_panel(local, o2, R2), o2 + 0.02 * np.asarray(n2), atol=1e-12)


def test_vertical_normal_rejected():
    with pytest.raises(ValueError):
        panel_frame([0, 0, 0], [0, 0, 1])


def test_rotation_round_trip():
    _, R = panel_frame([0.56, -0.31, 0.21], [-0.8, 0.12, 0.3])
    Rb = Rotation.from_euler("xyz", [10, -70, 35], degrees=True).as_matrix()
    np.testing.assert_allclose(rot_from_panel(rot_to_panel(Rb, R), R), Rb, atol=1e-12)


def test_one_offset_serves_every_button():
    # press_button stores ONE tool pose in the panel frame; the frame's origin is whichever
    # button is targeted, so two buttons on the same panel get goals exactly their spacing apart.
    n = [-0.95, 0.1, 0.0]
    offset = np.array([-0.007, -0.0425, -0.007])
    rot_local = Rotation.from_quat([-0.04047, 0.98014, -0.19117, 0.03365]).as_matrix()
    b1 = np.array([0.543, -0.304, 0.238])
    _, R1 = panel_frame(b1, n)
    b2 = b1 + R1 @ np.array([-0.03, -0.025, 0.0])  # 3 cm left, 2.5 cm below, same panel
    _, R2 = panel_frame(b2, n)
    np.testing.assert_allclose(R1, R2, atol=1e-12)
    g1, g2 = from_panel(offset, b1, R1), from_panel(offset, b2, R2)
    np.testing.assert_allclose(g2 - g1, b2 - b1, atol=1e-12)
    np.testing.assert_allclose(rot_from_panel(rot_local, R1), rot_from_panel(rot_local, R2), atol=1e-12)


def test_pose_path_limits_and_endpoint():
    R0 = np.eye(3)
    R1 = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    p0, p1 = np.zeros(3), np.array([0.1, 0.05, 0.0])
    path = pose_path(p0, R0, p1, R1, max_step_m=0.01, max_step_deg=4.0)
    np.testing.assert_allclose(path[-1][0], p1, atol=1e-12)
    np.testing.assert_allclose(path[-1][1], R1, atol=1e-12)
    prev_p, prev_R = p0, R0
    for p, Rm in path:
        assert np.linalg.norm(p - prev_p) <= 0.01 + 1e-12
        assert rot_angle_deg(prev_R, Rm) <= 4.0 + 1e-9
        prev_p, prev_R = p, Rm
    # A pure turn is chunked by angle, a pure move by distance.
    assert len(pose_path(p0, R0, p0, R1, 0.01, 4.0)) == 8
    assert len(pose_path(p0, R0, p1, R0, 0.01, 4.0)) == 12


def test_rot_angle():
    R1 = Rotation.from_euler("x", 25, degrees=True).as_matrix()
    assert rot_angle_deg(np.eye(3), R1) == pytest.approx(25.0)
    assert rot_angle_deg(R1, R1) == pytest.approx(0.0, abs=1e-6)
