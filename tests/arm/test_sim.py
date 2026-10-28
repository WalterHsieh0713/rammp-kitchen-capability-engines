from pathlib import Path

import numpy as np
import pybullet as p
import pytest
from conftest import G, add_box, cornell_sim

from capability_engine import sim


def test_solve_ik_matches_goldens():
    scene, rb = cornell_sim()
    for case in G["solve_ik"]:
        q, err = sim.solve_ik(scene, rb, case["pos"], case["quat"], case["seed"])
        np.testing.assert_allclose(q, case["joints"], atol=1e-6)
        assert abs(err - case["err"]) < 1e-6


def test_goldens_hold_through_one_shared_copy_with_door_boxes_between():
    scene, rb = cornell_sim()
    for i, case in enumerate(G["solve_ik"]):
        add_box(rb, center=[0.8, 0.1 * i, 0.3], half=[0.02, 0.2, 0.2])  # as callers' door models do
        q, _ = sim.solve_ik(scene, rb, case["pos"], case["quat"], case["seed"])
        np.testing.assert_allclose(q, case["joints"], atol=1e-6)


def test_one_copy_per_process():
    sim.reset_for_tests()
    sim.set_model_provider(lambda: Path("/fixtures/urdf/flatbot.urdf").read_text())
    assert sim.make_sim() is sim.make_sim()


def test_explicit_urdf_refused_while_live():
    sim.reset_for_tests()
    sim.set_model_provider(lambda: Path("/fixtures/urdf/flatbot.urdf").read_text())
    with pytest.raises(sim.SimModelError, match="driver"):
        sim.make_sim("/fixtures/cornell/robot.urdf")


def test_missing_collision_mesh_fails(tmp_path):
    # the fixtures name their meshes by absolute file:// path, so break that prefix
    bad = (
        Path("/fixtures/urdf/dojo.urdf")
        .read_text()
        .replace("/opt/ros/humble/share/kortex_description", "/nope")
    )
    (tmp_path / "bad.urdf").write_text(bad)
    sim.reset_for_tests()
    with pytest.raises(sim.SimModelError, match="mesh"):
        sim.make_sim(str(tmp_path / "bad.urdf"))


def test_solve_ik_flatbot_fk_round_trip():
    sim.reset_for_tests()
    scene, rb = sim.make_sim("/fixtures/urdf/flatbot.urdf")
    target = ([0.55, -0.10, 0.35], [0.0, 0.7071068, 0.0, 0.7071068])
    q, err = sim.solve_ik(scene, rb, *target, [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57])
    assert err < sim.MAX_IK_ERR
    assert [p.getJointInfo(rb.robot_id, j)[1].decode() for j in sim.ARM] == [
        f"arm_joint_{i}" for i in range(1, 8)
    ]


def test_keep_out_hook_reaches_solve_ik():
    scene, rb = cornell_sim()
    case = G["solve_ik"][0]
    sim.set_keep_out(lambda rb, q: "in the zone")
    _, err = sim.solve_ik(scene, rb, case["pos"], case["quat"], case["seed"])
    assert err == sim.KEEP_OUT_ERR
    sim.set_keep_out(None)


def _tiny_urdf(tmp_path, joint_fmt="joint_{}", mesh=None):
    """A 7-joint chain of bare links (an optional collision mesh on the first), ending in `tool`."""
    geometry = (
        f'<collision><geometry><mesh filename="{mesh}"/></geometry></collision>' if mesh else ""
    )
    links = "".join(f'<link name="l{i}">{geometry if i == 0 else ""}</link>' for i in range(8))
    joints = "".join(
        f'<joint name="{joint_fmt.format(i + 1)}" type="revolute"><parent link="l{i}"/>'
        f'<child link="l{i + 1}"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3" effort="1" velocity="1"/>'
        "</joint>"
        for i in range(7)
    )
    tool = (
        '<link name="tool"/><joint name="tool_joint" type="fixed">'
        '<parent link="l7"/><child link="tool"/></joint>'
    )
    path = tmp_path / "tiny.urdf"
    path.write_text(f'<robot name="tiny">{links}{joints}{tool}</robot>')
    return str(path)


def test_wrap_joints_and_short_delta():
    np.testing.assert_allclose(
        sim.wrap_joints([np.pi + 0.1, 0.0, -np.pi - 0.1, 1.0, 0, 0, 0])[[0, 2]],
        [-np.pi + 0.1, np.pi - 0.1],
        atol=1e-12,
    )
    assert sim.wrap_joints([0, 4.0, 0, 4.0, 0, 4.0, 0])[[1, 3, 5]].tolist() == [
        4.0,
        4.0,
        4.0,
    ]  # limited joints
    d = sim.short_delta([np.radians(179)] + [0] * 6, [np.radians(-179)] + [0] * 6)
    assert np.degrees(d[0]) == pytest.approx(2.0)


def test_continuous_ok_refuses_a_near_half_turn():
    q0 = np.zeros(7)
    assert sim.continuous_ok(
        np.radians([179, 0, 0, 0, 0, 0, 0]), np.radians([-179, 0, 0, 0, 0, 0, 0])
    )
    assert sim.continuous_ok(q0, np.radians([0, 0, 169, 0, 0, 0, 0]))
    assert not sim.continuous_ok(q0, np.radians([0, 0, 171, 0, 0, 0, 0]))


def test_joint_n_names_are_the_arm_fallback(tmp_path):
    sim.reset_for_tests()
    _, rb = sim.make_sim(_tiny_urdf(tmp_path), ee_link="tool")
    assert [p.getJointInfo(rb.robot_id, j)[1].decode() for j in sim.ARM] == [
        f"joint_{i}" for i in range(1, 8)
    ]


def test_model_without_arm_joints_fails(tmp_path):
    sim.reset_for_tests()
    with pytest.raises(sim.SimModelError, match="arm_joint_1"):
        sim.make_sim(_tiny_urdf(tmp_path, joint_fmt="wrist_{}"), ee_link="tool")


def test_initial_joints_length_mismatch_fails(tmp_path):
    sim.reset_for_tests()
    with pytest.raises(sim.SimModelError, match="initial_joints"):
        sim.make_sim(_tiny_urdf(tmp_path), ee_link="tool", initial_joints=[0.0] * 3)


def test_uninstalled_package_mesh_names_the_package(tmp_path):
    sim.reset_for_tests()
    with pytest.raises(sim.SimModelError, match="no_such_pkg"):
        sim.make_sim(_tiny_urdf(tmp_path, mesh="package://no_such_pkg/m.stl"), ee_link="tool")


def test_later_calls_with_different_arguments_are_refused(tmp_path):
    scene, rb = cornell_sim()
    cornell = "/fixtures/cornell/robot.urdf"
    assert sim.make_sim() == (scene, rb)
    assert sim.make_sim(cornell, base_pose=scene.robot_base_pose, ee_link="tool_frame") == (
        scene,
        rb,
    )
    with pytest.raises(sim.SimModelError, match="urdf_path"):
        sim.make_sim(_tiny_urdf(tmp_path), ee_link="tool")
    with pytest.raises(sim.SimModelError, match="base_pose"):
        sim.make_sim(cornell, base_pose=sim.Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    with pytest.raises(sim.SimModelError, match="ee_link"):
        sim.make_sim(ee_link="camera_link")


def _tip_gap(rb):
    """Distance between the two gripper fingertip links at the sim's current joint state."""
    pos = [
        np.array(p.getLinkState(rb.robot_id, j, physicsClientId=rb.physics_client_id)[4])
        for j in range(p.getNumJoints(rb.robot_id, physicsClientId=rb.physics_client_id))
        if p.getJointInfo(rb.robot_id, j, physicsClientId=rb.physics_client_id)[12].decode()
        in ("arm_robotiq_85_left_finger_tip_link", "arm_robotiq_85_right_finger_tip_link")
    ]
    return float(np.linalg.norm(pos[0] - pos[1]))


def test_clearance_open_and_closed_on_the_driver_model():
    sim.reset_for_tests()
    _, rb = sim.make_sim("/fixtures/urdf/dojo.urdf")
    q = np.zeros(7)
    for j, jj in enumerate(sim.ARM):
        p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=rb.physics_client_id)
    tip = p.getLinkState(rb.robot_id, rb.end_effector_id, physicsClientId=rb.physics_client_id)[4]
    box = add_box(rb, center=[tip[0], tip[1], tip[2] + 0.12], half=[0.2, 0.2, 0.01])
    assert set(sim.FINGER_CLOSE_SIGN.values()) == {1.0, -1.0}
    d_open, what_open = sim.clearance(rb, q, [(box, "box")])
    gap_open = _tip_gap(rb)
    d_closed, what_closed = sim.clearance(rb, q, [(box, "box")], closed=True)
    gap_closed = _tip_gap(rb)
    assert gap_closed < gap_open - 0.01  # the fingertips moved toward each other
    assert d_open < sim.CLEARANCE_QUERY_M and d_closed < sim.CLEARANCE_QUERY_M
    assert what_open.endswith("vs box") and what_closed.endswith("vs box")
    assert d_open != d_closed


def test_default_ee_link_sentinel_applies_only_to_ee_link():
    cornell_sim()
    with pytest.raises(sim.SimModelError, match="urdf_path"):
        sim.make_sim(sim.DEFAULT_EE_LINK)  # a path equal to the sentinel is still a path


def test_base_pose_and_initial_joints_compare_by_value():
    scene, rb = cornell_sim()
    pose = scene.robot_base_pose
    as_numpy = sim.Pose(np.array(pose.position), np.array(pose.orientation))
    joints = G["meta"]["initial_joints"]
    for initial in (joints, tuple(joints), np.array(joints)):
        assert sim.make_sim(base_pose=as_numpy, initial_joints=initial) == (scene, rb)
    moved = sim.Pose(np.array(pose.position) + 0.1, np.array(pose.orientation))
    with pytest.raises(sim.SimModelError, match="base_pose"):
        sim.make_sim(base_pose=moved)
    with pytest.raises(sim.SimModelError, match="initial_joints"):
        sim.make_sim(initial_joints=np.array(joints) + 0.1)
    with pytest.raises(sim.SimModelError, match="initial_joints"):
        sim.make_sim(initial_joints=joints[:3])


def test_model_provider_refused_after_an_explicit_path():
    sim.reset_for_tests()
    sim.make_sim("/fixtures/urdf/flatbot.urdf")

    def provider():
        return Path("/fixtures/urdf/flatbot.urdf").read_text()

    with pytest.raises(sim.SimModelError, match="flatbot.urdf"):
        sim.set_model_provider(provider)
    assert not sim.has_model_provider()
    sim.set_model_provider(None)  # offline stays allowed
    sim.reset_for_tests()
    sim.set_model_provider(provider)  # no copy yet: fine
    assert sim.has_model_provider()
    sim.reset_for_tests()
