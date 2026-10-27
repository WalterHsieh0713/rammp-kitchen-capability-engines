"""Record golden outputs from the students' SheppyArm code, unmodified.

"Exact" for `self.arm` means reproducing what the students' `microwave/sheppy_arm.py` and
`microwave/microwave_common.py` (feeding-deployment, microwave-wip@7bc328ae) compute. This
tool runs their code and writes the inputs and outputs to a JSON fixture; the tests replay them.

Run it in an environment with the students' package (`feeding_deployment`, `pybullet_helpers`)
installed, with cwd at their repo root (their `make_sim` opens a relative config path):

    cd feeding-deployment
    python /path/to/scripts/record_sheppy_goldens.py --out sheppy_goldens.json

Their `sheppy_arm` imports ROS at module top, none of which the recorded paths touch, so those
modules are stubbed before the import.
"""

import argparse
import importlib.metadata
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

# Modules `sheppy_arm.py` imports at the top that the recorded code paths never use.
ROS_STUBS = [
    "rclpy", "rclpy.action", "rclpy.executors", "rclpy.node", "rclpy.qos",
    "builtin_interfaces", "builtin_interfaces.msg",
    "control_msgs", "control_msgs.msg",
    "geometry_msgs", "geometry_msgs.msg",
    "sensor_msgs", "sensor_msgs.msg",
    "trajectory_msgs", "trajectory_msgs.msg",
    "rammp_arm_interfaces", "rammp_arm_interfaces.action", "rammp_arm_interfaces.msg",
    "rammp_common_interfaces", "rammp_common_interfaces.msg",
]  # fmt: skip

# Seed joints for every case unless stated otherwise (the Cornell home pose, rounded).
Q0 = [0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]
DOOR_RADIUS_M = 0.35
DOOR_ARC_DEG = 60.0
DOOR_WAYPOINT_SPACING_M = 0.02
# The straight approach, and the offsets of the other tool poses from the seed pose.
APPROACH_M = [0.15, 0.0, 0.0]
TURN_MOVE_M = [0.08, 0.06, 0.0]  # a 10 cm move, with a 30 degree wrist turn
J6_FAIL_MOVE_M = [0.10, 0.0, 0.0]  # with a 135 degree turn about z: J6 reaches ~117 deg > 115
ARC_START_M = [0.18, 0.0, 0.0]  # the door arc starts here; the hinge is DOOR_RADIUS_M to -y


def load_students_code():
    """Import their modules (cwd = their repo root, `microwave/` on sys.path)."""
    for name in ROS_STUBS:
        sys.modules[name] = MagicMock()
    sys.path.insert(0, str(Path.cwd() / "microwave"))
    import microwave_common
    import sheppy_arm

    return microwave_common, sheppy_arm


def lst(x):
    return np.asarray(x, dtype=float).tolist()


def tool_pose(mc, scene, rb, q):
    """The end-effector link pose in the robot base frame at joints `q`, as [x y z qx qy qz qw]."""
    p = mc.p
    for j, jj in enumerate(mc.ARM):
        p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=rb.physics_client_id)
    ls = p.getLinkState(rb.robot_id, rb.end_effector_id, physicsClientId=rb.physics_client_id)
    rel = mc.multiply_poses(scene.robot_base_pose.invert(), mc.Pose(ls[4], ls[5]))
    return lst([*rel.position, *rel.orientation])


def robot_meta(mc, scene, rb, commit):
    p = mc.p
    infos = [
        p.getJointInfo(rb.robot_id, i, physicsClientId=rb.physics_client_id)
        for i in range(p.getNumJoints(rb.robot_id, physicsClientId=rb.physics_client_id))
    ]
    flags = p.URDF_USE_INERTIA_FROM_FILE
    if rb.self_collision_link_names:
        flags |= p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    base = scene.robot_base_pose
    ee = infos[rb.end_effector_id]
    return {
        "commit": commit,
        "python": sys.version.split()[0],
        "pybullet": importlib.metadata.version("pybullet-arm64"),
        "pybullet_dist": "pybullet-arm64",
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "robot_name": scene.robot_name,
        "robot_urdf": str(Path(scene.robot_urdf_path).relative_to(Path.cwd())),
        "robot_base_pose": {
            "position": lst(base.position),
            "orientation_xyzw": lst(base.orientation),
        },
        "load_urdf": {
            "useFixedBase": True,
            "flags": int(flags),
            "flags_meaning": "URDF_USE_INERTIA_FROM_FILE"
            + (
                " | URDF_USE_SELF_COLLISION | URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS"
                if rb.self_collision_link_names
                else ""
            ),
        },
        "ee_joint": ee[1].decode(),
        "ee_link": ee[12].decode(),
        "ee_link_index": int(rb.end_effector_id),
        "arm_joint_indices": list(mc.ARM),
        "arm_joint_names": [infos[i][1].decode() for i in mc.ARM],
        "joints": [
            {
                "index": i[0],
                "name": i[1].decode(),
                "type": i[2],
                "lower": i[8],
                "upper": i[9],
                "child_link": i[12].decode(),
            }
            for i in infos
        ],  # fmt: skip
        "initial_joints": lst(scene.initial_joints),
        "ik_call": "p.calculateInverseKinematics(robot, ee_index, pos, quat, maxNumIterations=200)",
    }


def make_arm(sa, q0, ee0):
    """A SheppyArm without ROS: state set by hand, `_send` capturing what would be sent."""
    arm = sa.SheppyArm.__new__(sa.SheppyArm)
    arm._lock = threading.Lock()
    arm._js = (np.array(q0, float), np.zeros(7), np.zeros(7))
    arm._ee = None if ee0 is None else np.array(ee0, float)
    arm._sim = None
    arm._status = SimpleNamespace(estopped=False, arbitration_enabled=False)
    sent = []

    def capture(t, q, qd, impedance=None, monitor=None):
        sent.append((t, q, qd, impedance))
        return True

    arm._send = capture
    return arm, sent


def record(out):
    mc, sa = load_students_code()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    scene, rb = mc.make_sim()
    sim = (scene, rb)
    ee0 = tool_pose(mc, scene, rb, Q0)
    pos0, quat0 = np.array(ee0[:3]), np.array(ee0[3:])
    turn = mc.R.from_quat(quat0)

    def quat_about_z(deg):
        return lst((mc.R.from_euler("z", deg, degrees=True) * turn).as_quat())

    def wp(pose):
        return [lst(pose.position), lst(pose.orientation)]

    # solve_ik: the handle region 0.75-0.80 m out, then two targets out of reach.
    ik_targets = [
        ([0.75, 0.0, 0.30], quat0), ([0.78, 0.05, 0.35], quat0), ([0.80, -0.05, 0.25], quat0),
        ([0.76, 0.10, 0.20], quat0), ([1.30, 0.0, 0.30], quat0), ([0.80, 0.0, 1.20], quat0),
    ]  # fmt: skip
    solve_ik = []
    for pos, quat in ik_targets:
        joints, err = mc.solve_ik(scene, rb, pos, quat, Q0)
        solve_ik.append({"pos": lst(pos), "quat": lst(quat), "seed": Q0,
                         "joints": lst(joints), "err": err})  # fmt: skip

    start = mc.Pose(tuple(pos0 + ARC_START_M), tuple(quat0))
    hinge = (start.position[0], start.position[1] - DOOR_RADIUS_M, start.position[2])
    arc_wps = mc.door_arc_waypoints(
        start, hinge, DOOR_RADIUS_M * np.radians(DOOR_ARC_DEG), DOOR_WAYPOINT_SPACING_M
    )
    approach = [[lst(pos0 + APPROACH_M), lst(quat0)]]
    arc = [wp(start)] + [wp(w) for w in arc_wps]
    knot_trajs = {
        "approach_15cm": approach,
        "move_10cm_turn_30deg": [[lst(pos0 + TURN_MOVE_M), quat_about_z(30.0)]],
        "door_arc_60deg": arc,
        "fails_j6_guard": [[lst(pos0 + J6_FAIL_MOVE_M), quat_about_z(135.0)]],
    }
    cartesian_knots = []
    knots_by_name = {}
    for name, traj in knot_trajs.items():
        knots, lengths = sa.cartesian_knots(sim, np.array(ee0), np.array(Q0), traj)
        knots_by_name[name] = (knots, lengths)
        cartesian_knots.append({
            "name": name, "ee_start": ee0, "q_start": Q0, "traj": traj,
            "knots": None if knots is None else [lst(k) for k in knots],
            "lengths": None if lengths is None else lst(lengths),
        })  # fmt: skip
    assert knots_by_name["door_arc_60deg"][0] is not None
    assert knots_by_name["fails_j6_guard"][0] is None

    approach_knots, approach_lengths = knots_by_name["approach_15cm"]
    contact = sa.contact_speed_scale()
    move_joints_cases = [
        ("one_knot", [solve_ik[0]["joints"]], None, 1.0, None),
        ("approach_knots_cart_lengths", approach_knots, approach_lengths, 1.0, None),
        ("contact_speed_scale", approach_knots, approach_lengths, contact, None),
        ("impedance", [solve_ik[1]["joints"], solve_ik[2]["joints"]], None, 1.0, sa.IMPEDANCE),
    ]
    joint_trajectory, send = [], []
    for name, knots, lengths, speed_scale, impedance in move_joints_cases:
        arm, sent = make_arm(sa, Q0, None)
        assert arm.move_joints(
            knots, cart_lengths=lengths, speed_scale=speed_scale, impedance=impedance
        )
        (t, q, qd, imp), = sent  # fmt: skip
        inputs = {
            "q0": Q0, "knots": [lst(k) for k in knots],
            "cart_lengths": None if lengths is None else lst(lengths),
            "speed_scale": speed_scale,
        }  # fmt: skip
        joint_trajectory.append({"name": name, **inputs, "t": lst(t), "q": lst(q), "qd": lst(qd)})
        send.append({
            "name": name, "kind": "move_joints", **inputs,
            "impedance": imp, "t": lst(t), "q": lst(q), "qd": lst(qd),
        })  # fmt: skip

    move_cartesian_cases = [
        ("approach", approach, None, 1.0),
        ("door_arc", arc, sa.IMPEDANCE, contact),
    ]
    for name, traj, impedance, speed_scale in move_cartesian_cases:
        arm, sent = make_arm(sa, Q0, ee0)
        assert arm.move_cartesian(traj, impedance=impedance, speed_scale=speed_scale)
        (t, q, qd, imp), = sent  # fmt: skip
        send.append({
            "name": name, "kind": "move_cartesian", "q0": Q0, "ee0": ee0, "traj": traj,
            "impedance": imp, "speed_scale": speed_scale,
            "t": lst(t), "q": lst(q), "qd": lst(qd),
        })  # fmt: skip

    goldens = {
        "meta": robot_meta(mc, scene, rb, commit),
        "solve_ik": solve_ik,
        "cartesian_knots": cartesian_knots,
        "joint_trajectory": joint_trajectory,
        "send": send,
    }
    Path(out).write_text(json.dumps(goldens) + "\n")
    n = sum(len(goldens[k]) for k in goldens if k != "meta")
    print(f"wrote {n} cases to {out}: " + ", ".join(
        f"{k}={len(goldens[k])}" for k in goldens if k != "meta"))  # fmt: skip
    print(f"meta.ee_link={goldens['meta']['ee_link']} commit={commit}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    record(ap.parse_args().out)
