from pathlib import Path

import numpy as np
import pybullet as p
import pytest
from capability_engine import sim
from conftest import G, add_box, cornell_sim


def test_solve_ik_matches_goldens():
    scene, rb = cornell_sim()
    for case in G["solve_ik"]:
        q, err = sim.solve_ik(scene, rb, case["pos"], case["quat"], case["seed"])
        np.testing.assert_allclose(q, case["joints"], atol=1e-6)
        assert abs(err - case["err"]) < 1e-6


def test_goldens_hold_through_one_shared_copy_with_door_boxes_between():
    scene, rb = cornell_sim()
    for i, case in enumerate(G["solve_ik"]):
        add_box(rb, center=[0.8, 0.1 * i, 0.3], half=[0.02, 0.2, 0.2])   # as callers' door models do
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
    bad = Path("/fixtures/urdf/dojo.urdf").read_text().replace("/opt/ros/humble/share/kortex_description", "/nope")
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
    assert [p.getJointInfo(rb.robot_id, j)[1].decode() for j in sim.ARM] == [f"arm_joint_{i}" for i in range(1, 8)]


def test_keep_out_hook_reaches_solve_ik():
    scene, rb = cornell_sim()
    case = G["solve_ik"][0]
    sim.set_keep_out(lambda rb, q: "in the zone")
    _, err = sim.solve_ik(scene, rb, case["pos"], case["quat"], case["seed"])
    assert err == sim.KEEP_OUT_ERR
    sim.set_keep_out(None)
