"""Simulated arm for planning: PyBullet FK/IK and every motion safety gate. No ROS, no real arm.

Before anything moves, press_button plans each leg here: every tool pose along the path is
solved to joint angles and checked against the gates. RammpArm then sends exactly those joint
angles to the driver, so the arm follows the path that was checked.

Gates (any failure raises Abort and nothing is commanded):
  - IK position error <= MAX_IK_ERR_M, orientation error <= MAX_IK_ROT_ERR_DEG
  - joint step between consecutive points <= MAX_JOINT_STEP_DEG
  - tool within MAX_REACH_M of the base, height inside Z_RANGE_M
  - J2/J4/J6 at least JOINT_LIMIT_MARGIN_DEG inside their URDF limits

The robot is a mesh-free Gen3 URDF (urdf/gen3_tool_frame_kinematic.urdf) ending at Kinova's
tool_frame, with its base at the origin, turned by base_rpy_deg (sim world == the run's frame). IK, seeding and gate
values are unchanged from ``feeding_deployment.button_press.arm.Arm``; the joint-limit gate is
new (Kortex used to do the IK). test/test_kinematics.py checks it against the full feeding sim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from rammp_button_press import Abort
from rammp_button_press.geometry import max_joint_delta_deg

# ---- motion gates (same family as the grasp scripts; tighter where the moves are smaller)
MAX_IK_ERR_M = 0.005
MAX_IK_ROT_ERR_DEG = 1.0  # orientation miss allowed when a move also turns the wrist
SEED_GOOD_M = 0.001  # posture seed is honoured unquestioned below this; see solve_translation
MAX_JOINT_STEP_DEG = 10.0
MAX_REACH_M = 1.2  # existing rig constant -- do not raise without asking
Z_RANGE_M = (0.0, 1.2)  # tool-point height above the arm base. Was (0.12, 0.75), tuned to the rchi
# lab's microwave height (button at 0.171); widened 2026-10-13 at the user's
# request so the microwave's position is not assumed. What is left is a TABLE
# guard, not a placement rule: on the RAMMP bench the table top is at
# z = -0.027 (rammp_box_opening world_bench.yaml), so 0.0 keeps the tool
# point 2.7 cm above it. The top is just MAX_REACH_M (no separate ceiling).
# Revolute joints (J2/J4/J6) must stay this far inside their URDF limits (= kinova_gen3_node's
# gen3_7dof_2f85.urdf: J2 +-2.41, J4 +-2.66, J6 +-2.23 rad). New with the RAMMP port (see the
# module docstring). The arm repo's runbook (2026-09-01) saw a run abort with J4 at -2.660.
JOINT_LIMIT_MARGIN_DEG = 2.0

JOINTS = [f"joint_{i}" for i in range(1, 8)]
EE_LINK = "tool_frame"
URDF_NAME = "gen3_tool_frame_kinematic.urdf"


def _find_urdf() -> Path:
    """Source checkout (urdf/ next to the package dir) or colcon install (share/rammp_button_press/urdf)."""
    here = Path(__file__).resolve().parent
    candidates = [here.parent / "urdf" / URDF_NAME]
    try:
        from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

        candidates.append(Path(get_package_share_directory("rammp_button_press")) / "urdf" / URDF_NAME)
    except Exception:  # noqa: BLE001 -- not installed / no ament: the source path is all there is
        pass
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"{URDF_NAME} not found in {[str(c) for c in candidates]}")


class Kinematics:
    """Headless PyBullet copy of the arm: FK, the seeded IK and the gates."""

    def __init__(self, urdf: str | Path | None = None, base_rpy_deg=(0.0, 0.0, 0.0)):
        """base_rpy_deg: the arm base's orientation in the frame the run works in (a tilted mount:
        flatbot's arm_mount has it at 0 30 0). The sim world is then that frame, and FK, IK and the
        reach / z gates are all in it; the base stays at the origin."""
        self.cid = p.connect(p.DIRECT)
        base_quat = Rotation.from_euler("xyz", base_rpy_deg, degrees=True).as_quat()
        self.robot_id = p.loadURDF(
            str(urdf or _find_urdf()), useFixedBase=True, baseOrientation=list(base_quat), physicsClientId=self.cid
        )
        by_name, self.limits = {}, []
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self.cid)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            by_name[info[1].decode()] = j
            if info[12].decode() == EE_LINK:
                self.end_effector_id = j
        self.arm = [by_name[n] for n in JOINTS]
        for j in self.arm:
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.cid)
            revolute = info[2] == p.JOINT_REVOLUTE and info[8] < info[9]
            self.limits.append((info[8], info[9]) if revolute else None)
        self.base_pos = np.zeros(3)

    def close(self):
        p.disconnect(physicsClientId=self.cid)

    # -- kinematics ---------------------------------------------------------------------------
    def _set_sim(self, q):
        for i, jj in enumerate(self.arm):
            p.resetJointState(self.robot_id, jj, float(q[i]), physicsClientId=self.cid)

    def fk(self, q):
        """(base-frame position, quaternion xyzw) of the tool frame for joint vector q."""
        self._set_sim(q)
        ls = p.getLinkState(
            self.robot_id, self.end_effector_id, computeForwardKinematics=True, physicsClientId=self.cid
        )
        return np.asarray(ls[4], dtype=float), np.asarray(ls[5], dtype=float)

    def limit_failures(self, q) -> list[str]:
        """One message per revolute joint within JOINT_LIMIT_MARGIN_DEG of its limit (empty = ok)."""
        bad = []
        m = np.radians(JOINT_LIMIT_MARGIN_DEG)
        for i, lim in enumerate(self.limits):
            if lim is not None and not (lim[0] + m <= q[i] <= lim[1] - m):
                bad.append(
                    f"J{i + 1} {np.degrees(q[i]):.1f} deg within {JOINT_LIMIT_MARGIN_DEG:.0f} deg of its "
                    f"limit [{np.degrees(lim[0]):.0f}, {np.degrees(lim[1]):.0f}]"
                )
        return bad

    def solve_translation(self, q_cur, d_base, posture=None, rot=None):
        """IK for one step: move the tool by `d_base` (and turn it by `rot`, 3x3 base frame) from q_cur.

        The step is applied to the sim's own FK of q_cur, not the driver's pose, so a constant
        sim-vs-driver tool offset cancels out. The IK is seeded from `posture` (a fixed anchor,
        refreshed by Run.reanchor_seed), not q_cur: the Gen3 is redundant, and re-seeding from
        each new pose let the joints drift along the null space until the joint-step gate
        tripped (2026-09-21). The anchor's solution is used outright if it is within
        SEED_GOOD_M; otherwise q_cur is tried too and the more accurate solution wins.

        Returns (q, ik_err_m, target_xyz, joint_step_deg, gate_failures).
        """
        pos, quat = self.fk(q_cur)
        target = pos + np.asarray(d_base, dtype=float)
        if rot is not None:
            quat = Rotation.from_matrix(np.asarray(rot, dtype=float) @ Rotation.from_quat(quat).as_matrix()).as_quat()
        seeds = [posture, q_cur] if posture is not None else [q_cur]
        q = None
        err = None
        rerr = 0.0
        for seed in seeds:
            self._set_sim(seed)
            sol = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_id,
                list(target),
                list(quat),
                maxNumIterations=400,
                residualThreshold=1e-5,
                physicsClientId=self.cid,
            )
            q_try = np.asarray(sol[:7], dtype=float)
            got, got_q = self.fk(q_try)
            err_try = float(np.linalg.norm(got - target))
            # Orientation is only scored when this move turns the wrist; the translation-only
            # path keeps its original behaviour exactly.
            rerr_try = (
                float(np.degrees((Rotation.from_quat(got_q) * Rotation.from_quat(quat).inv()).magnitude()))
                if rot is not None
                else 0.0
            )
            better = q is None or (rerr_try <= MAX_IK_ROT_ERR_DEG, -err_try) > (rerr <= MAX_IK_ROT_ERR_DEG, -err)
            if better:
                q, err, rerr = q_try, err_try, rerr_try
            # Stop at the posture seed only when it is GOOD (<= SEED_GOOD_M), not merely under the
            # 5 mm gate: a stale anchor's ~3 mm answer once hid the better q_cur solution while
            # the corrections asked for were only 1-4 mm (2026-09-21).
            if err <= SEED_GOOD_M and rerr <= MAX_IK_ROT_ERR_DEG:
                break
        bad = []
        tb = target - self.base_pos
        if np.linalg.norm(tb) > MAX_REACH_M:
            bad.append(f"reach {np.linalg.norm(tb):.3f} > {MAX_REACH_M}")
        if not (Z_RANGE_M[0] <= tb[2] <= Z_RANGE_M[1]):
            bad.append(f"z {tb[2]:.3f} outside {Z_RANGE_M}")
        if err > MAX_IK_ERR_M:
            bad.append(f"IK err {err*100:.2f} cm > {MAX_IK_ERR_M*100:.1f}")
        if rerr > MAX_IK_ROT_ERR_DEG:
            bad.append(f"IK orientation err {rerr:.1f} deg > {MAX_IK_ROT_ERR_DEG}")
        jump = max_joint_delta_deg(q, q_cur)
        if jump > MAX_JOINT_STEP_DEG:
            bad.append(f"joint jump {jump:.1f} deg > {MAX_JOINT_STEP_DEG}")
        bad += self.limit_failures(q)
        return q, err, target, jump, bad

    def precheck_poses(self, poses, name, log, start, posture=None):
        """Check a whole path of (xyz, R) tool poses in the sim before anything moves.

        Solves each point in turn from `start` = (q, xyz, R); pass one leg's end state as the
        next leg's start to plan several back to back. Raises Abort at the first point that
        fails a gate. Returns (worst IK error m, worst joint step deg, end state, joint solution
        of every point)."""
        q, prev_p, prev_R = start
        worst_err, worst_jump, knots = 0.0, 0.0, []
        for i, (pt, Rt) in enumerate(poses):
            q, err, _, jump, bad = self.solve_translation(q, pt - prev_p, posture=posture, rot=Rt @ prev_R.T)
            worst_err, worst_jump = max(worst_err, err), max(worst_jump, jump)
            if bad:
                log({"step": name, "precheck_failed_at": i + 1, "n_points": len(poses), "gates": bad})
                raise Abort(
                    f"{name}: pre-check failed at point {i + 1}/{len(poses)}: {'; '.join(bad)}" " -- arm not commanded"
                )
            knots.append(q)
            prev_p, prev_R = pt, Rt
        return worst_err, worst_jump, (q, prev_p, prev_R), knots
