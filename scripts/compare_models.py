"""Report how the driver's arm model differs from the students' Cornell model.

Runs inside the test container (docker/test.Dockerfile; /repo, /fixtures). From the repo root:

    make test-arm ARM_CMD="python3 /repo/scripts/compare_models.py"

For every golden input it runs solve_ik and cartesian_knots on the Cornell model and on the
driver's dojo model and prints the joint differences, the tool-frame offset between the two
models, and whether flatbot (dojo behind a mounting rotation) is dojo up to that rotation. It is
a report: it asserts nothing and tunes nothing. Lines marked FINDING exceed a threshold below.
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pybullet as p
import sheppy_arm

from capability_engine import sim
from capability_engine.sim import Pose

REPO = Path(__file__).resolve().parent.parent
GOLDENS = REPO / "tests/arm/fixtures/sheppy_goldens.json"
CORNELL_URDF = "/fixtures/cornell/robot.urdf"
DRIVER_URDF = "/fixtures/urdf/{}.urdf"
CORNELL_EE, DRIVER_EE = "tool_frame", "arm_tool_frame"
MOUNT_JOINT = "arm_base_joint"  # arm_mount -> arm_base_link in the driver models

TOOL_OFFSET_MM = 1.0  # tool-frame position offset above which it is a finding
TOOL_OFFSET_DEG = 0.1  # tool-frame orientation offset above which it is a finding
JOINT_DIFF_DEG = 1.0  # max joint difference between two models above which it is a finding
LINK_MISMATCH_MM = 0.1  # flatbot link position mismatch (vs dojo through the mount) finding
LINK_MISMATCH_DEG = 0.01  # flatbot link orientation mismatch finding


def out(text=""):
    sys.stdout.write(text + "\n")


def flag(bad):
    return "  FINDING" if bad else ""


def mount_pose(urdf_path):
    """The (position, quaternion xyzw) of arm_base_link in arm_mount, from the URDF."""
    root = ET.parse(urdf_path).getroot()
    origin = next(j for j in root.iter("joint") if j.get("name") == MOUNT_JOINT).find("origin")
    xyz = [float(v) for v in origin.get("xyz").split()]
    rpy = [float(v) for v in origin.get("rpy").split()]
    return tuple(xyz), tuple(p.getQuaternionFromEuler(rpy))


def to_mount(mount, pos, quat):
    """A pose in arm_base_link expressed in arm_mount."""
    q = p.multiplyTransforms(mount[0], mount[1], list(pos), list(quat))
    return np.array(q[0]), np.array(q[1])


def pose_gap(a, b):
    """(mm, deg) between poses a and b, each (position, quaternion xyzw)."""
    mm = 1000 * float(np.linalg.norm(np.asarray(a[0]) - np.asarray(b[0])))
    x, y, z, w = p.getDifferenceQuaternion(list(a[1]), list(b[1]))
    return mm, float(np.degrees(2 * np.arctan2(np.linalg.norm([x, y, z]), abs(w))))


def link_poses(rb, q):
    """{link name: (position, quaternion)} of every link but the root, world frame, at arm q."""
    c = rb.physics_client_id
    for j, jj in enumerate(sim.ARM):
        p.resetJointState(rb.robot_id, jj, float(q[j]), physicsClientId=c)
    poses = {}
    for i in range(p.getNumJoints(rb.robot_id, physicsClientId=c)):
        name = p.getJointInfo(rb.robot_id, i, physicsClientId=c)[12].decode()
        ls = p.getLinkState(rb.robot_id, i, physicsClientId=c)
        poses[name] = (np.array(ls[4]), np.array(ls[5]))
    return poses


def run_model(urdf, base_pose, ee, ik_cases, knot_cases, qs, mount=None):
    """IK / knots for the goldens (poses go through `mount` if given), tool poses, link poses."""
    sim.reset_for_tests()
    scene, rb = sim.make_sim(urdf, base_pose=base_pose, ee_link=ee)

    def xf(pos, quat):
        return to_mount(mount, pos, quat) if mount else (np.asarray(pos), np.asarray(quat))

    res = {"ik": [], "knots": [], "tool": [], "links": []}
    for c in ik_cases:
        pos, quat = xf(c["pos"], c["quat"])
        res["ik"].append(sim.solve_ik(scene, rb, pos, quat, c["seed"]))
    for c in knot_cases:
        pos, quat = xf(c["ee_start"][:3], c["ee_start"][3:7])
        start = np.concatenate([pos, quat])
        traj = [xf(pp, qq) for pp, qq in c["traj"]]
        res["knots"].append(
            sheppy_arm.cartesian_knots((scene, rb), start, c["q_start"], traj, log=lambda m: None)
        )
    for q in qs:
        res["tool"].append(sim.tool_pose(scene, rb, q))
        res["links"].append(link_poses(rb, q))
    return res


def main():
    G = json.load(open(GOLDENS))
    meta = G["meta"]
    cornell_base = Pose(
        tuple(meta["robot_base_pose"]["position"]),
        tuple(meta["robot_base_pose"]["orientation_xyzw"]),
    )
    identity = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    ik_cases, knot_cases = G["solve_ik"], G["cartesian_knots"]
    q0 = meta["initial_joints"][:7]
    qs = [q0] + [c["q_start"] for c in knot_cases] + [c["seed"] for c in ik_cases]
    qs += [k for c in G["joint_trajectory"] for k in c["knots"][:2]]
    qs = [np.asarray(q, float)[:7] for q in qs]

    runs = {}
    runs["cornell"] = run_model(CORNELL_URDF, cornell_base, CORNELL_EE, ik_cases, knot_cases, qs)
    runs["dojo"] = run_model(
        DRIVER_URDF.format("dojo"), identity, DRIVER_EE, ik_cases, knot_cases, qs
    )
    mounts = {n: mount_pose(DRIVER_URDF.format(n)) for n in ("dojo", "flatbot", "gen_1_5")}
    for n in ("flatbot", "gen_1_5"):
        runs[n] = run_model(
            DRIVER_URDF.format(n), identity, DRIVER_EE, ik_cases, knot_cases, qs, mounts[n]
        )

    out("Thresholds (a FINDING is reported, never tuned away):")
    out(f"  TOOL_OFFSET_MM={TOOL_OFFSET_MM}  TOOL_OFFSET_DEG={TOOL_OFFSET_DEG}")
    out(f"  JOINT_DIFF_DEG={JOINT_DIFF_DEG}  MAX_IK_ERR={sim.MAX_IK_ERR} m (IK refusal)")
    out(f"  LINK_MISMATCH_MM={LINK_MISMATCH_MM}  LINK_MISMATCH_DEG={LINK_MISMATCH_DEG}")
    out("Mount of arm_base_link in arm_mount (from each URDF):")
    for n, (pos, quat) in mounts.items():
        rpy = np.degrees(p.getEulerFromQuaternion(quat))
        out(f"  {n:8s} xyz={tuple(round(v, 4) for v in pos)} rpy_deg={np.round(rpy, 3).tolist()}")

    def jdiff(a, b):
        return float(np.degrees(np.max(np.abs(sim.short_delta(a, b)))))

    cor, doj = runs["cornell"], runs["dojo"]
    out("\n== solve_ik: Cornell vs dojo (same base-frame targets, same seeds) ==")
    out("case  maxJointDiff_deg  cornell_err_m  cornell_refused  dojo_err_m  dojo_refused")
    for i, ((qc, ec), (qd, ed)) in enumerate(zip(cor["ik"], doj["ik"], strict=False)):
        d = jdiff(qc, qd)
        out(
            f"{i:4d}  {d:16.3f}  {ec:13.5f}  {str(ec > sim.MAX_IK_ERR):15s}  {ed:10.5f}  "
            f"{str(ed > sim.MAX_IK_ERR):12s}{flag(d > JOINT_DIFF_DEG)}"
        )

    out("\n== cartesian_knots: Cornell vs dojo ==")
    out("name                      knots(c/d)  maxJointDiff_deg  cornell_refused  dojo_refused")
    for c, (kc, _), (kd, _) in zip(knot_cases, cor["knots"], doj["knots"], strict=False):
        nc, nd = (0 if kc is None else len(kc)), (0 if kd is None else len(kd))
        if kc is None or kd is None or nc != nd:
            d, text = None, "n/a"
        else:
            d = max(jdiff(a, b) for a, b in zip(kc, kd, strict=False))
            text = f"{d:.3f}"
        bad = d is None and (kc is None) != (kd is None) or (d or 0) > JOINT_DIFF_DEG
        out(
            f"{c['name']:24s}  {nc:3d}/{nd:<3d}    {text:>16s}  {str(kc is None):15s}  "
            f"{str(kd is None):12s}{flag(bad)}"
        )

    out("\n== tool-frame offset, dojo vs Cornell, in the base frame (over joint states) ==")
    gaps = [pose_gap(a, b) for a, b in zip(cor["tool"], doj["tool"], strict=False)]
    out(f"at q0 (initial joints): {gaps[0][0]:.3f} mm, {gaps[0][1]:.3f} deg")
    wmm, wdeg = max(g[0] for g in gaps), max(g[1] for g in gaps)
    out(f"worst over {len(gaps)} joint states: {wmm:.3f} mm, {wdeg:.3f} deg")
    out(f"tool frames agree: {wmm <= TOOL_OFFSET_MM and wdeg <= TOOL_OFFSET_DEG}")
    if wmm > TOOL_OFFSET_MM or wdeg > TOOL_OFFSET_DEG:
        out("FINDING: tool frame offset exceeds threshold")

    for n in ("flatbot", "gen_1_5"):
        r = runs[n]
        out(f"\n== {n}: targets rotated through the mount; link poses vs dojo through the mount ==")
        d_ik = max(jdiff(a[0], b[0]) for a, b in zip(r["ik"], doj["ik"], strict=False))
        kd = [
            jdiff(a, b)
            for (ka, _), (kb, _) in zip(r["knots"], doj["knots"], strict=False)
            if ka is not None and kb is not None and len(ka) == len(kb)
            for a, b in zip(ka, kb, strict=False)
        ]
        refusals = [
            (ka is None) == (kb is None)
            for (ka, _), (kb, _) in zip(r["knots"], doj["knots"], strict=False)
        ]
        d_k = max(kd, default=0.0)
        out(f"solve_ik joints vs dojo: max {d_ik:.3f} deg{flag(d_ik > JOINT_DIFF_DEG)}")
        out(f"cartesian_knots joints vs dojo: max {d_k:.3f} deg; refusals agree: {all(refusals)}")
        worst = (0.0, 0.0, "", -1)
        for k in range(len(qs)):
            for name, pose in r["links"][k].items():
                want = to_mount(mounts[n], *doj["links"][k][name])
                mm, deg = pose_gap(pose, want)
                if mm + deg > worst[0] + worst[1]:
                    worst = (mm, deg, name, k)
        bad = worst[0] > LINK_MISMATCH_MM or worst[1] > LINK_MISMATCH_DEG
        out(
            f"worst link mismatch over {len(qs)} joint states: {worst[2]} "
            f"{worst[0]:.4f} mm, {worst[1]:.4f} deg{flag(bad)}"
        )
        out(f"link poses match dojo up to the mount: {not bad}")


if __name__ == "__main__":
    main()
