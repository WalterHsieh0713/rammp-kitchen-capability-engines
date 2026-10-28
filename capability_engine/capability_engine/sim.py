"""The one PyBullet copy of the arm per process, and the planning helpers on it.

Ported from the students' microwave_common.py@7bc328ae. No rclpy: this imports offline.
"""

import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pybullet as p

MAX_IK_ERR = 0.02  # IK position error (m) above which a solution counts as unreachable
MAX_STEP_JUMP_DEG = 20.0
J6_GUARD_DEG = 115.0
# J4's configured soft limit is +-147.8 deg (09-07: a command past it fails METHOD_FAILED; 09-28:
# a Cartesian grasp path drove J4 from -145 into it and Kortex aborted). PyBullet only knows the
# URDF's wider limit, so every sim check gates J4 with a margin.
J4_GUARD_DEG = 144.0
# J1/J3/J5/J7 spin freely; every commanded angle is wrapped to [-180, 180). 09-29 hardware
# test (tools/wrap_test.py, J7 and J3): a JointCommand across +-180 takes the SHORT way, so a
# step is judged (and interpolated for clearance checks) by its short-way change -- 179 -> -179
# is 2 deg. Only a short-way change near 180 (direction ambiguous) is refused.
CONTINUOUS = [0, 2, 4, 6]
MAX_CONTINUOUS_DELTA_DEG = 170.0
KEEP_OUT_ERR = 9.0  # solve_ik's error for a solution in the keep-out zone: fails every IK gate
IK_MAX_ITERATIONS = 200  # PyBullet IK iterations per solve
CLEARANCE_QUERY_M = 0.5  # clearance() ignores bodies farther than this from every link
FINGERTIP_OPEN_RAD = 0.0  # finger joint angle with the gripper open
# Robotiq 2F-85 linkage: one closing angle, mirrored on the inner-finger joints. Keyed by joint
# name; the Cornell names are the sim URDF the goldens were recorded on, the arm_robotiq_85_*
# names are the driver's.
FINGER_CLOSE_SIGN_BY_NAME = {
    "finger_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_finger_joint": -1.0,
    "right_inner_knuckle_joint": 1.0,
    "arm_robotiq_85_left_knuckle_joint": 1.0,
    "arm_robotiq_85_right_knuckle_joint": -1.0,
    "arm_robotiq_85_left_inner_knuckle_joint": 1.0,
    "arm_robotiq_85_right_inner_knuckle_joint": -1.0,
    "arm_robotiq_85_left_finger_tip_joint": -1.0,
    "arm_robotiq_85_right_finger_tip_joint": 1.0,
}
FINGER_CLOSED_RAD = 0.7  # a little short of the 0.8 full close -> slightly wide, i.e. conservative

ARM: list = []  # arm joint indices in the loaded model (filled by make_sim)
FINGER_JOINTS: list = []  # gripper joint indices in the loaded model (filled by make_sim)
FINGER_CLOSE_SIGN: dict = {}  # finger joint index -> sign of its closing angle (filled by make_sim)


class SimModelError(Exception):
    pass


class Pose(NamedTuple):
    position: tuple
    orientation: tuple  # xyzw


class Scene(NamedTuple):
    robot_base_pose: Pose


class SimRobot(NamedTuple):
    robot_id: int
    end_effector_id: int
    physics_client_id: int


_SIM: tuple | None = None  # (Scene, SimRobot): the one copy per process
_SIM_ARGS: dict | None = None  # the make_sim arguments the copy was built with
_MODEL_PROVIDER: Callable[[], str] | None = None
_KEEP_OUT: Callable | None = None


def set_model_provider(fn):
    """`fn() -> URDF text` of the model the driver is running (the live arm).

    None: offline, give make_sim a path.
    """
    global _MODEL_PROVIDER
    _MODEL_PROVIDER = fn


def set_keep_out(fn):
    """Fridge only: `fn(rb, joints) -> message if the arm is in a keep-out zone, else None`.

    The zone is the wheelchair. None (the microwave): no such check anywhere.
    """
    global _KEEP_OUT
    _KEEP_OUT = fn


def reset_for_tests():
    global _SIM, _SIM_ARGS, _MODEL_PROVIDER, _KEEP_OUT
    if _SIM is not None:
        p.disconnect(physicsClientId=_SIM[1].physics_client_id)
    _SIM, _SIM_ARGS, _MODEL_PROVIDER, _KEEP_OUT = None, None, None, None
    ARM.clear()
    FINGER_JOINTS.clear()
    FINGER_CLOSE_SIGN.clear()


def multiply_poses(a, b):
    pos, orn = p.multiplyTransforms(a.position, a.orientation, b.position, b.orientation)
    return Pose(tuple(pos), tuple(orn))


def _resolve_meshes(urdf_text, urdf_dir):
    """The URDF with every mesh filename made an absolute path.

    package://<pkg>/x -> the installed package, file://x -> x, relative x -> beside the URDF.
    Also puts one element per line (see below).
    """

    def path(m):
        name = m.group(1)
        if name.startswith("package://"):
            from ament_index_python.packages import (
                PackageNotFoundError,
                get_package_share_directory,
            )

            pkg, _, rest = name[len("package://") :].partition("/")
            try:
                return f'filename="{Path(get_package_share_directory(pkg)) / rest}"'
            except PackageNotFoundError:
                raise SimModelError(f"mesh package {pkg!r} is not installed ({name})") from None
        if name.startswith("file://"):
            name = name[len("file://") :]
        if urdf_dir is not None and not Path(name).is_absolute():
            name = str(urdf_dir / name)
        return f'filename="{name}"'

    resolved = re.sub(r'filename="([^"]*)"', path, urdf_text)
    # PyBullet reads the URDF line by line into a fixed buffer and mis-parses a long line; xacro
    # writes a single one.
    return re.sub(r">\s*<", ">\n<", resolved)


def _check_collision_meshes(urdf_text, robot_id, client):
    """Raise SimModelError naming every link whose <collision> mesh PyBullet did not load."""
    missing = []
    for link in ET.fromstring(urdf_text).iter("link"):
        wanted = len(link.findall("collision/geometry/mesh"))
        if not wanted:
            continue
        name = link.get("name")
        index = next(
            (
                j
                for j in range(p.getNumJoints(robot_id, physicsClientId=client))
                if p.getJointInfo(robot_id, j, physicsClientId=client)[12].decode() == name
            ),
            -1,
        )
        if index == -1 and p.getBodyInfo(robot_id, physicsClientId=client)[0].decode() != name:
            continue
        loaded = [
            s
            for s in p.getCollisionShapeData(robot_id, index, physicsClientId=client)
            if s[2] == p.GEOM_MESH
        ]
        if len(loaded) < wanted:
            missing.append(name)
    if missing:
        raise SimModelError(f"collision mesh did not load for link(s): {', '.join(missing)}")


def _find(client, robot_id, names):
    by_name = {
        p.getJointInfo(robot_id, j, physicsClientId=client)[1].decode(): j
        for j in range(p.getNumJoints(robot_id, physicsClientId=client))
    }
    return by_name, [by_name[n] for n in names if n in by_name]


DEFAULT_EE_LINK = "arm_tool_frame"


def make_sim(urdf_path=None, base_pose=None, ee_link=DEFAULT_EE_LINK, initial_joints=None):
    """The process's one PyBullet copy of the arm: (Scene, SimRobot).

    The model is the driver's (set_model_provider) or, offline, `urdf_path`; the arm sits at
    `base_pose` (its mount). `initial_joints` (movable joints, in index order) sets the starting
    state; default is the arm at zeros with the gripper open.
    """
    global _SIM, _SIM_ARGS
    if urdf_path is not None and _MODEL_PROVIDER is not None:
        raise SimModelError(
            "a driver model is live: self.arm simulates the driver's URDF, not an explicit path"
        )
    args = {
        "urdf_path": urdf_path,
        "base_pose": base_pose,
        "ee_link": ee_link,
        "initial_joints": None if initial_joints is None else list(initial_joints),
    }
    if _SIM is not None:
        # Arguments left at their defaults mean "the copy as built"; explicit ones must match it.
        differ = [
            k
            for k, v in args.items()
            if v is not None and v != DEFAULT_EE_LINK and v != _SIM_ARGS[k]
        ]
        if differ:
            raise SimModelError(
                f"the sim already exists with different {', '.join(differ)}: "
                "one copy per process"
            )
        return _SIM
    if urdf_path is None and _MODEL_PROVIDER is None:
        raise SimModelError("no model: give a URDF path offline")
    if urdf_path is not None:
        urdf_text = _resolve_meshes(Path(urdf_path).read_text(), Path(urdf_path).resolve().parent)
    else:
        urdf_text = _resolve_meshes(_MODEL_PROVIDER(), None)
    base_pose = base_pose or Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    client = p.connect(p.DIRECT)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "robot.urdf"
            f.write_text(urdf_text)
            try:
                robot_id = p.loadURDF(
                    str(f),
                    basePosition=list(base_pose.position),
                    baseOrientation=list(base_pose.orientation),
                    useFixedBase=True,
                    physicsClientId=client,
                    flags=p.URDF_USE_INERTIA_FROM_FILE
                    | p.URDF_USE_SELF_COLLISION
                    | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                )
            except p.error as e:
                raise SimModelError(f"PyBullet could not load the model (mesh?): {e}") from e
        _check_collision_meshes(urdf_text, robot_id, client)
        n = p.getNumJoints(robot_id, physicsClientId=client)
        link_index = {
            p.getJointInfo(robot_id, j, physicsClientId=client)[12].decode(): j for j in range(n)
        }
        if ee_link not in link_index:
            raise SimModelError(f"end-effector link {ee_link!r} not in the model")
        arm = None
        for fmt in ("arm_joint_{}", "joint_{}"):
            _, arm = _find(client, robot_id, [fmt.format(i) for i in range(1, 8)])
            if len(arm) == 7:
                break
        else:
            raise SimModelError("model has no arm_joint_1..7 (or joint_1..7)")
        names = {
            j: p.getJointInfo(robot_id, j, physicsClientId=client)[1].decode() for j in range(n)
        }
        fingers = [j for j in range(n) if names[j] in FINGER_CLOSE_SIGN_BY_NAME]
        movable = [
            j
            for j in range(n)
            if p.getJointInfo(robot_id, j, physicsClientId=client)[2] != p.JOINT_FIXED
        ]
        if initial_joints is not None:
            if len(initial_joints) != len(movable):
                raise SimModelError(
                    f"initial_joints has {len(initial_joints)} values, "
                    f"model has {len(movable)} joints"
                )
            state = dict(zip(movable, initial_joints, strict=False))
        else:
            state = {j: 0.0 for j in movable}
        for j, q in state.items():
            p.resetJointState(robot_id, j, float(q), physicsClientId=client)
    except BaseException:
        p.disconnect(physicsClientId=client)
        raise
    ARM[:] = arm
    FINGER_JOINTS[:] = fingers
    FINGER_CLOSE_SIGN.clear()
    FINGER_CLOSE_SIGN.update({j: FINGER_CLOSE_SIGN_BY_NAME[names[j]] for j in fingers})
    _SIM_ARGS = {**args, "base_pose": base_pose}
    _SIM = (Scene(base_pose), SimRobot(robot_id, link_index[ee_link], client))
    return _SIM


def wrap_joints(q):
    q = np.asarray(q, dtype=float).copy()
    q[CONTINUOUS] = (q[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return q


def short_delta(q_from, q_to):
    """q_to - q_from with the free-spinning joints taken the short way round (what Kortex does)."""
    d = np.asarray(q_to, dtype=float) - np.asarray(q_from, dtype=float)
    d[CONTINUOUS] = (d[CONTINUOUS] + np.pi) % (2 * np.pi) - np.pi
    return d


def continuous_ok(q_from, q_to):
    d = np.degrees(np.abs(short_delta(q_from, q_to)[CONTINUOUS]))
    return bool(np.all(d <= MAX_CONTINUOUS_DELTA_DEG))


def keep_out_message(rb, q):
    """Why joints `q` are in the keep-out zone, or None.

    Always None unless a fridge run switched it on.
    """
    return None if _KEEP_OUT is None else _KEEP_OUT(rb, q)


def solve_ik(scene, rb, pos, quat, seed_joints):
    """PyBullet IK seeded from `seed_joints`; returns (joints, position error in m).

    A solution in the keep-out zone (fridge runs only, set_keep_out) returns KEEP_OUT_ERR as its
    error.
    """
    for j, jj in enumerate(ARM):
        p.resetJointState(
            rb.robot_id, jj, float(seed_joints[j]), physicsClientId=rb.physics_client_id
        )
    wpose = multiply_poses(scene.robot_base_pose, Pose(tuple(pos), tuple(quat)))
    sol = p.calculateInverseKinematics(
        rb.robot_id,
        rb.end_effector_id,
        list(wpose.position),
        list(wpose.orientation),
        physicsClientId=rb.physics_client_id,
        maxNumIterations=IK_MAX_ITERATIONS,
    )
    joints = wrap_joints([sol[k] for k in range(7)])
    for j, jj in enumerate(ARM):
        p.resetJointState(rb.robot_id, jj, float(joints[j]), physicsClientId=rb.physics_client_id)
    ls = p.getLinkState(rb.robot_id, rb.end_effector_id, physicsClientId=rb.physics_client_id)
    err = float(np.linalg.norm(np.asarray(ls[4]) - np.asarray(wpose.position)))
    if _KEEP_OUT is not None and _KEEP_OUT(rb, joints) is not None:
        return joints, KEEP_OUT_ERR
    return joints, err


def clearance(rb, q, bodies, closed=False):
    """Smallest distance from any robot link to the door/body at joints q, with the gripper
    OPEN (default, the widest hand) or `closed`."""
    c = rb.physics_client_id
    for j, jj in enumerate(ARM):
        p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=c)
    for jj in FINGER_JOINTS:
        p.resetJointState(
            rb.robot_id,
            jj,
            FINGER_CLOSE_SIGN[jj] * FINGER_CLOSED_RAD if closed else FINGERTIP_OPEN_RAD,
            physicsClientId=c,
        )
    best = (9.0, "")
    for bid, name in bodies:
        for pt in p.getClosestPoints(rb.robot_id, bid, CLEARANCE_QUERY_M, physicsClientId=c):
            if pt[8] < best[0]:
                ln = (
                    p.getJointInfo(rb.robot_id, pt[3], physicsClientId=c)[12].decode()
                    if pt[3] >= 0
                    else "base"
                )
                best = (float(pt[8]), f"{ln} vs {name}")
    return best
