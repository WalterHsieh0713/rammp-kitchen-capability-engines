"""kinematics.Kinematics: the lightweight URDF reproduces the full feeding sim, and the gates hold."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rammp_button_press import Abort
from rammp_button_press.kinematics import MAX_JOINT_STEP_DEG, Kinematics
from rammp_button_press.panel_frame import pose_path

# A press-like posture (tool ~0.5 m out, fingers forward); FK checked below to be inside the gates.
Q_PRESS = np.array([0.0, 0.35, -3.14, -2.0, 0.0, -0.9, 1.57])


@pytest.fixture(scope="module")
def kin():
    k = Kinematics()
    yield k
    k.close()


def test_tool_frame_is_flange_plus_120mm(kin):
    """tool_frame = end_effector_link + 0.120 m along its z: what RammpArm adds to /ee_state."""
    import pybullet as p

    q = np.array([0.3, -0.4, 2.0, -1.5, 0.7, -1.0, 0.2])
    kin._set_sim(q)
    flange = p.getLinkState(
        kin.robot_id, kin.end_effector_id - 1, computeForwardKinematics=True, physicsClientId=kin.cid
    )
    tool, _ = kin.fk(q)
    want = np.asarray(flange[4]) + Rotation.from_quat(flange[5]).apply([0, 0, 0.120])
    assert np.allclose(tool, want, atol=1e-6)


def test_matches_full_feeding_sim(kin):
    """Same FK and IK as feeding_deployment's FeedingDeploymentPyBulletSimulator (base offset removed)."""
    sd = pytest.importorskip("feeding_deployment.simulation.scene_description")
    sim_mod = pytest.importorskip("feeding_deployment.simulation.simulator")
    from pathlib import Path

    import pybullet as p

    cfg = Path(sd.__file__).parent / "configs" / "vention.yaml"
    scene = sd.create_scene_description_from_config(str(cfg), "skewer")
    rb = sim_mod.FeedingDeploymentPyBulletSimulator(scene, use_gui=False).robot
    base = np.asarray(scene.robot_base_pose.position)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-2.2, 2.2, 7)
        for i, j in enumerate([1, 2, 3, 4, 5, 6, 7]):
            p.resetJointState(rb.robot_id, j, q[i], physicsClientId=rb.physics_client_id)
        ls = p.getLinkState(rb.robot_id, rb.end_effector_id, physicsClientId=rb.physics_client_id)
        pos, quat = kin.fk(q)
        assert np.allclose(np.asarray(ls[4]) - base, pos, atol=1e-6)
        assert (Rotation.from_quat(ls[5]) * Rotation.from_quat(quat).inv()).magnitude() < 1e-6


def test_press_stroke_passes_and_returns_knots(kin):
    p0, quat = kin.fk(Q_PRESS)
    R0 = Rotation.from_quat(quat).as_matrix()
    p1 = p0 + np.array([0.01, 0.0, 0.0])  # 1 cm along +x, at 3 mm spacing like the press
    path = pose_path(p0, R0, p1, R0, 0.003, 4.0)
    err, jump, (q_end, p_end, _), knots = kin.precheck_poses(
        path, "stroke", lambda r: None, (Q_PRESS, p0, R0), posture=Q_PRESS
    )
    assert len(knots) == len(path)
    assert err < 0.001 and jump < MAX_JOINT_STEP_DEG
    assert np.allclose(kin.fk(q_end)[0], p1, atol=0.001)
    assert np.allclose(p_end, p1)


def test_joint_limit_gate(kin):
    assert kin.limit_failures(Q_PRESS) == []
    q = Q_PRESS.copy()
    q[5] = np.radians(-128.5)  # J6 limit is -127.8 deg (2.23 rad): within the margin
    assert any("J6" in b for b in kin.limit_failures(q))
    q = Q_PRESS.copy()
    q[0] = 3.1  # J1 is continuous: no limit
    assert kin.limit_failures(q) == []


def test_unreachable_target_aborts(kin):
    p0, quat = kin.fk(Q_PRESS)
    R0 = Rotation.from_quat(quat).as_matrix()
    far = p0 + np.array([0.5, 0.0, 0.0])
    with pytest.raises(Abort, match="pre-check failed"):
        kin.precheck_poses([(far, R0)], "too far", lambda r: None, (Q_PRESS, p0, R0))


def test_a_tilted_mount_plans_in_the_mount_frame(kin):
    """flatbot: the arm base pitched 30 deg in arm_mount. FK is the level FK turned by the mount, and
    IK of a pose given in the mount frame lands on the same joints."""
    tilted = Kinematics(base_rpy_deg=(0.0, 30.0, 0.0))
    try:
        mount = Rotation.from_euler("y", 30.0, degrees=True)
        p_level, q_level = kin.fk(Q_PRESS)
        p_tilt, q_tilt = tilted.fk(Q_PRESS)
        assert np.allclose(p_tilt, mount.apply(p_level), atol=1e-6)
        assert np.allclose(
            (Rotation.from_quat(q_tilt) * (mount * Rotation.from_quat(q_level)).inv()).magnitude(), 0.0, atol=1e-6
        )
        q, err, _, _, bad = tilted.solve_translation(Q_PRESS, [0.0, 0.0, 0.05], posture=Q_PRESS)
        assert err < 1e-3 and not bad
        assert np.allclose(tilted.fk(q)[0], p_tilt + [0.0, 0.0, 0.05], atol=1e-3)
    finally:
        tilted.close()
