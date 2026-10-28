"""Shared fixtures for the self.arm tests: the SheppyArm goldens and the Cornell sim they were recorded on."""
import json
from pathlib import Path

import pybullet as p
from capability_engine import sim
from capability_engine.sim import Pose

FIXTURES = Path(__file__).parent / "fixtures"
G = json.load(open(FIXTURES / "sheppy_goldens.json"))


def cornell_sim():
    """The Cornell kinova-gen3 sim the goldens were recorded on: its ee link, base pose and initial joints."""
    meta = G["meta"]
    pose = meta["robot_base_pose"]
    sim.reset_for_tests()
    return sim.make_sim("/fixtures/cornell/robot.urdf",
                        base_pose=Pose(tuple(pose["position"]), tuple(pose["orientation_xyzw"])),
                        ee_link=meta["ee_link"], initial_joints=meta["initial_joints"])


def add_box(rb, center, half):
    """A static collision box in `rb`'s sim, as callers' door models add; returns its body id."""
    c = rb.physics_client_id
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half), physicsClientId=c)
    return p.createMultiBody(0, shape, basePosition=list(center), physicsClientId=c)
