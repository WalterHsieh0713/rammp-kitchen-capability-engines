#!/usr/bin/env python3
"""Main entry point: find the target button, plan and check every move, then press it and come back.

DRY RUN BY DEFAULT: nothing moves without ``--execute``, and someone stands at the e-stop for
every execute run. Needs the sheppy profile `arm-camera`.

    ros2 run rammp_button_press press_button                        # plan only, START/+30SEC
    ros2 run rammp_button_press press_button --target timer_clock   # another button
    ros2 run rammp_button_press press_button --execute              # move the arm

Steps:
  1. preflight   driver up, not e-stopped, control available, tool frame consistent, gripper
                 closed, camera_info and tf available
  2. first look  a rough button position + panel normal from the first source that works: the
                 fixed scene camera (scene_finder; default, --far scene) or the wrist camera's
                 domes (dome_pattern). This sets the PANEL FRAME: origin = the button, z = out
                 of the panel, y = up
  3. stage       always after a scene-camera estimate, otherwise when the camera is further
                 than CLOSE_VIEW_M: move (cuRobo) to a pose in front of the button and re-detect
                 the domes from ~25 cm. The press is only ever planned from the domes
  4. plan        pre-press tool pose = PREPRESS_EE_OFFSET_M / PREPRESS_EE_QUAT_PANEL in the panel
                 frame; the approach and every press stroke are solved and gated in the sim
                 before anything moves
  5. go          straight line to the pre-press spot, wrist turning to face the panel
  6. press       --presses times: in along the normal until the wrist torques feel the button,
                 --push-mm further (or until --stop-nm), then straight back to the spot
  7. return      back to the start pose (cuRobo), unless --no-return

A dry run that needs staging stops after checking the staging goal: the rest can only be planned
from the re-detect. Any failed check raises Abort and the arm holds; a failed move first backs
off to its last safe pose (the start, or the pre-press spot during a press). The pre-press offset
is stored relative to the button, so one measurement (measure_prepress_offset) serves every button.

--interactive (with --execute; Stops, decide.py): the run waits at the staging pose and again at
the pre-press spot. space goes on (at the spot: one press, then wait again); q goes back to the
start (back_to_start). A failure while waiting there holds the arm until q. While the staging stop
waits, the re-detect and the press plan already run in the background (LookAhead; nothing moves),
so space there goes straight to the pre-press spot.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

from rammp_button_press import Abort, dome_pattern, say, scene_finder
from rammp_button_press.decide import BACK, GO, TerminalKeys
from rammp_button_press.geometry import PlaneFitError
from rammp_button_press.kinematics import MAX_IK_ERR_M
from rammp_button_press.panel_frame import (
    describe,
    from_panel,
    panel_frame,
    pose_path,
    rot_angle_deg,
    rot_from_panel,
    to_panel,
)
from rammp_button_press.perception import (
    DEPTH_TOPIC,
    FRESH_S,
    SCENE_COLOR_FRAME,
    SCENE_DEPTH_FRAME,
    SCENE_DEPTH_TOPIC,
    SCENE_NS,
    YOLO_FRESH_S,
    YOLO_MIN_SCORE,
    YOLO_TOPIC,
    Perception,
    frame_of,
    scene_frame_from_msgs,
    to_bgr_depth,
    why_no_fit,
    yolo_box,
)
from rammp_button_press.rammp_arm import (
    CART_SPEED,
    CONTACT_MAX_TOUCH_M,
    CONTACT_PUSH_M,
    CONTACT_STOP_NM,
    CONTACT_TOUCH_NM,
    GRIPPER_CLOSED_ABOVE,
    TRACK_ABORT_M,
    RammpArm,
)

# ---- the pre-press spot, in the panel frame (x right, y up, z out of the panel; origin = button)
# Kinova tool-frame position and orientation with the LEFT fingertip STANDOFF_M in front of the
# button. MEASURED 2026-10-03 on timer_clock, current fingers: panel frame from the dome layout
# at 30.9 cm, then the fingertip hand-placed 2 cm in front of the button (tool z 10 deg off
# straight-in). Re-measure with scripts/button_press/measure_prepress_offset.py if the fingers or
# camera mount change. (The 09-27 placeholder it replaced had the tool 4.2 cm below the button,
# taught from a biased ~40 cm view -- it would have missed.)
# RAMMP rig (Jetson, 2026-10-12): re-measured here. The rchi-cpu-5 values (-0.0049, -0.0114,
# -0.0131 / -0.04947, 0.99522, -0.08381, -0.00846) absorb that rig's hand-eye bias and put the
# fingertip ~7 cm INTO the panel on this one. Reference: the fingertip hand-placed at start_30s's
# pre-press spot, tool (0.7128, 0.0474, 0.1554) in base_link, tool z 23 deg off straight-in; an
# attended 1.0 cm push along the normal from there clicked start_30s. Expressed in the panel frame
# from the CLOSE re-detect at the staging pose (23 cm) -- that is the frame press_button plans the
# approach in. A first version expressed in the far (33.5 cm) frame put the run's spot 1.4 cm too
# low and 0.5 cm too far out (missed); the close view reads start_30s ~0.6 cm lower than the far
# one. Fingertip reads ~3 cm out of the detected plane here (detection short in depth); the offset
# absorbs it. Verified on start_30s only.
# HERL bench (Jetson abra, D405 wrist camera, cream microwave), 2026-10-13: re-measured with
# measure_prepress_offset on start_30s from a 32.5 cm view (frame wrist_d405_color_optical, the
# mount from rammp_box_opening's camera_d405_wrist.yaml), the fingertip hand-placed in front of the
# button. Replaces the 10-12 values (-0.0050, 0.0339, 0.0849 / 0.12271, 0.97220, -0.19812,
# -0.02295). NOT yet re-expressed in the close staging re-detect's frame (see above), and not yet
# confirmed by a press.
# y lowered 0.5 cm (-0.0024 -> -0.0074) at the user's request, 2026-10-13: press lower on the button.
PREPRESS_EE_OFFSET_M = np.array([0.0026, -0.0074, 0.0456])
PREPRESS_EE_QUAT_PANEL = np.array([0.08162, 0.98750, -0.11196, 0.07514])  # xyzw
STANDOFF_M = 0.02
# Fingertip = tool frame + FINGERTIP_PAST_TOOL_M along the tool z. Was 0.05955 (robot.urdf
# finger_tip_joint 0.17955 m from the flange - tool_frame 0.120; the microwave's touch check used
# 0.062). On the HERL bench that put the hand-placed pre-press spot's fingertip 1.2 cm INSIDE the
# detected panel, 3.2 cm from where it was held. Set 2026-10-13, at the user's request, so that
# hand-placed spot reads as the intended STANDOFF_M (tool 4.56 cm out, tool z 15 deg off
# straight-in: (0.0456 - 0.02) / 0.964). It absorbs this rig's whole depth disagreement (finger
# length, camera mount, depth) in one number and assumes the hand-placed gap really was 2 cm --
# a --touch measurement would pin it down.
FINGERTIP_PAST_TOOL_M = 0.0265
MIN_PREPRESS_FINGERTIP_CLEAR_M = 0.01  # pre-press fingertip must be at least this far out of the fitted plane
# ---- refusals before anything moves
# No cap on the start -> staging / pre-press distance (removed 2026-10-12 at the user's request; was
# 0.40 m): every point of the straight line is still gated (reach, z band, joint limits, IK, joint
# step). Nothing checks collisions along it -- watch long moves.
MAX_GOTO_ROT_DEG = 45.0  # wrist turn on the way
# Where a leg may end relative to its target before the run holds. The approach and the press keep
# TRACK_ABORT_M (1 cm). Staging, bail and back-to-start only need to end near their target: the
# re-detect works from wherever staging actually stops. Long moves (50 cm, 2026-10-12) ended
# 0.5-1.05 cm off with the driver reporting success.
LOOSE_TRACK_ABORT_M = 0.05
# ---- far start: stage closer and re-detect before planning the approach
# From ~40 cm the detection is biased: on 2026-10-03 the same, unmoved button came out 1.2 cm
# lower from 39 cm than from 20 cm (normal z 0.154 vs 0.112) -- a button-width miss, and it put
# the goal under the Z floor. 2026-09-27 saw 2-3 cm at 40 cm. 17-22 cm is the validated range.
CLOSE_VIEW_M = 0.25  # camera -> button beyond this: stage first
# (No "jump check" comparing the staging re-detect with the far estimate: removed 2026-10-12. A
# wrong dome assignment shifts the button ~2 cm, but correct far estimates were off by 0.7-4.2 cm
# (front 34 cm .. side 50 cm), so the shift cannot tell them apart; it stopped a correct run on a
# rotated microwave. dome_pattern's blob size check and the frame agreement are the guards.)
STAGE_OUT_M = 0.10  # staging pose = pre-press pose this much further out: camera ~25.5 cm, in
# the 23-26 cm range every good re-detect came from. Was 0.08 (~23.5 cm):
# a far estimate that read the panel deep put the camera at 21.8 cm, under
# dome_pattern's MIN_RANGE_M 0.22, and the re-detect found nothing (2026-10-12)
RETRY_BACKOFF_M = 0.03  # re-detect at staging failed: back straight out this far and try once more
# Leaving the panel: cuRobo's world has no microwave. 2026-10-14 the return to start handed the
# whole move to cuRobo with the fingertips 2 cm from the button, and its path swept the gripper
# into the microwave. So before any cuRobo move that starts at the panel (back to start, bail
# from the approach), back straight out this far along the panel normal first -- the approach
# in reverse, sim-gated and at this package's own speed -- and back to the pre-press spot after an
# aborted stroke in a straight line, never through cuRobo.
RETREAT_OUT_M = 0.15
# The wrist camera: box-opening's D405 mount. Was "camera_color_optical_frame" (this package's own hand-eye
# node); 2026-10-15 the restarted arm driver began publishing the Kinova vision module's frames under that
# name, pointing 90 deg sideways, and a run without --camera-frame put the button 57 cm off.
CAMERA_FRAME = "wrist_d405_color_optical"
# ... and the same on the way in: cuRobo only takes the arm to a lead-in point this much further out
# than the staging pose (along the panel normal), then a straight sim-checked line goes the rest of the
# way in. With the microwave on the raised counter (top ~0.29 m), cuRobo's own path to the staging
# pose swung the arm into the microwave's top (2026-10-15). --lead-in 0 = the old single cuRobo move.
STAGE_LEAD_IN_M = 0.10  # was 0.20 (2026-10-15); with staging at 0.35 that put the lead-in ~60 cm out
MAX_PUSH_MM = 10.0  # --push-mm bound (the old box-opening travel was 8)
LEAD_IN_SPEED_MPS = 0.15  # the straight lead-in leg (free air, away from the microwave): 2.5 x CART_SPEED
# (no settle sleep: the re-detect only takes frames captured after the arm stopped -- see `after`)
# ---- scene camera (far/first look; see scene_finder.py): the fixed Orbbec Gemini finds the
# microwave's front wherever it stands; good to a few cm, for staging only.
SCENE_FRAMES = 5  # frames looked at (~0.4-0.7 s each on the Jetson)
SCENE_MIN_FITS = 4  # ... of which this many must find the front
SCENE_MIN_FITS_YOLO = 3  # ... when the YOLO node's box narrowed the search (it already found the microwave)
SCENE_MAX_SPREAD_M = 0.02  # and agree on the button to within this
SCENE_MAX_NORMAL_SPREAD_DEG = 5.0  # and on the normal to within this
SCENE_TIMEOUT_S = 8.0
SCENE_STAGE_OUT_M = 0.08  # staging = pre-press this much further out: the fingertip ~10 cm in front of the
# panel (the pre-press fingertip is ~2 cm out), by the CORRECTED scene estimate
# (SCENE_NORMAL_CORRECTION_M). The wrist camera is then ~25 cm away (domes: >= 22).
# History 2026-10-15: 0.15 / 0.25 / 0.35 / 0.27 against the uncorrected estimate.
SCENE_NORMAL_CORRECTION_M = 0.09  # the scene estimate is moved this far toward the arm along its normal: in every
# good run 2026-10-15 the wrist found the button 9-15 cm CLOSER than the scene camera
# (14-15 cm until 15:40, ~9 cm since 16:34). The smallest is used, so what is left
# of the error only ever leaves the fingertip further out than planned, never closer.
SCENE_MIN_BUTTON_Z_M = 0.10  # a scene estimate is raised to at least this before staging. The scene z is off
# by -3..+5 cm (10-14: 0.10-0.13 vs the wrist's ~0.08; 10-15: 0.05), and staging
# at 0.045 was IK_FAIL (flange near the table) while every 10-14 staging at >= 0.10
# reached and the wrist re-detect found the domes. Too high is harmless, too low is not
DOME_FRAMES = 7  # frames looked at per measurement
DOME_MIN_FITS = 5  # ... of which this many must fit all 5 domes
DOME_MAX_SPREAD_PX = 3.0  # and agree on the target pixel to within this, or within
DOME_SPREAD_FRAC = 0.3  # this x the dome spacing if larger. A dome's highlight can split and
# flicker between two spots ~0.25 spacing apart (rotated microwave,
# 2026-10-12: 12-13 px of a 47 px spacing, still arm); a wrong dome
# assignment is >= 1 spacing away, so it is still refused
# ---- motion
# Panel front face is near-vertical; a fitted normal with |z| above this is a bad fit.
MAX_PANEL_NORMAL_Z = 0.5
# Paths are chunked so no point moves more than RAY_STEP_MAX_M or turns more than
# ROT_STEP_MAX_DEG: how many joint degrees a centimetre costs depends on the posture (~5 deg/cm
# at the 2026-09-21 close standoff), and a merely LONG move must not trip the 10 deg joint gate.
# Press strokes use the finer SMOOTH_PRESS_SPACING_M below instead.
RAY_STEP_MAX_M = 0.01
ROT_STEP_MAX_DEG = 4.0
# Press strokes use finer spacing (was arm.SMOOTH_PRESS_SPACING_M, for Kortex's final-waypoint taper).
# Under RAMMP the driver interpolates linearly in JOINT space between the pre-checked points, so
# denser points keep the 1 cm stroke closer to the straight line along the normal.
SMOOTH_PRESS_SPACING_M = 0.003


class Run:
    """One press session: the perception node, the arm (RammpArm) and a JSONL run log in --log-dir.

    Also used by measure_prepress_offset."""

    def __init__(self, args):
        self.args = args
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_f = None
        self._log_lock = threading.Lock()  # LookAhead logs from its own thread
        # The arm first: if its driver is not up it raises Abort, and has cleaned up after itself
        # (press_server retries until it is).
        self.arm = RammpArm(
            args.execute,
            acquire_control=args.acquire_control,
            frame=args.arm_frame,
            joint_prefix=args.joint_prefix,
            mount_rpy_deg=args.mount_rpy_deg,
        )
        self.per = Perception(
            args.arm_frame,
            args.camera_frame,
            args.camera_ns,
            scene_ns=args.scene_ns if getattr(args, "far", "wrist") == "scene" else None,
            depth_topic=args.depth_topic,
            scene_depth_topic=args.scene_depth_topic,
        )
        self.start_log()  # only once connected: press_server retries Run until the driver is up
        # The IK seed is anchored to a posture to stop the redundant arm sliding along its
        # self-motion manifold (see Arm.solve_translation). Set by preflight, refreshed by
        # reanchor_seed() after a big move.
        self.seed_posture = None

    def start_log(self):
        """A new JSONL run log (press_server: one per goal)."""
        if self._log_f is not None:
            self._log_f.close()
        self.log_path = str(self.log_dir / f"press_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        self._log_f = open(self.log_path, "a")  # noqa: SIM115

    def log(self, rec):
        """Append one record (plus a timestamp) to the run log."""
        rec = dict(rec, t=time.time())
        with self._log_lock:
            self._log_f.write(json.dumps(rec) + "\n")
            self._log_f.flush()

    def reanchor_seed(self, why):
        """Re-anchor the IK seed posture to the current joints, after a big move.

        A seed far from the arm gives poor IK: a stale one 10.9 cm away added 2-4 mm of error to
        1-4 mm corrections (2026-09-21)."""
        say(f"  IK seed re-anchored to the current posture ({why})")
        if not self.args.execute:
            return  # nothing moved in a dry run, so the preflight anchor is still current
        self.seed_posture = self.arm.joints()

    def preflight(self, require_ready=True):
        """Check the arm, gripper, camera and tf before anything moves; Abort lists every problem.

        On an execute run it also closes an open gripper and takes control if arbitration is
        enforced. Saves the start pose to <log-dir>/start_joints.json. require_ready=False
        (measure_prepress_offset) checks only what reading the arm needs."""
        say("\n== preflight ==")
        a = self.args
        st = self.arm.state()
        grip = float(st.get("gripper_pos", -1))
        arm_problems, info = self.arm.status()
        for line in info:
            say(f"  {line}")
        say(
            f"  gripper_pos    : {grip:.3f}  ({'closed' if grip > GRIPPER_CLOSED_ABOVE else 'NOT closed'})"
            f"{'' if self.arm.gripper_present() else '  -- gripper NOT present'}"
        )
        say(
            f"  tool xyz       : {np.round(st['ee_pos'][:3], 3)}  "
            f"joints (deg) {np.round(np.degrees(st['position']), 1)}"
        )
        # Speed: no Kortex preset under RAMMP -- RammpArm times every leg (CART_SPEED / JOINT_SPEED).
        # measure_prepress_offset never moves, so it only needs the read-side checks.
        problems = (
            list(arm_problems)
            if require_ready
            else [pr for pr in arm_problems if pr.startswith(("no ", "tool frame", "/joint_states", "/ee_state"))]
        )
        if grip <= GRIPPER_CLOSED_ABOVE and require_ready and self.arm.gripper_present():
            # Preflight closes it (added 2026-10-13 at the user's request). Only once everything
            # read so far is fine, and only on an execute run: a dry run moves nothing.
            if not a.execute:
                say("  gripper        : open -- an --execute run closes it here (dry run: left as it is)")
            elif problems:
                problems.append("gripper is open (not closed: fix the problems above first)")
            else:
                say("  gripper        : open -- closing it (keep clear of the fingers)")
                try:
                    grip = self.arm.close_gripper()
                    say(f"  gripper_pos    : {grip:.3f}  (closed)")
                except Abort as e:
                    problems.append(str(e))
        elif grip <= GRIPPER_CLOSED_ABOVE:
            problems.append("gripper must be CLOSED (this presses with the closed fingertips)")
        say("  waiting for camera_info ...")
        if self.per.wait("info", 5, None) is None:
            problems.append("no camera_info")
        try:
            R_bc = self.per.cam_rotation_in_base()
            say(f"  tf {a.arm_frame} <- {a.camera_frame}: ok (camera +z in base = {np.round(R_bc[:, 2], 3)})")
        except Abort as e:
            problems.append(str(e))

        start_joints = self.arm.joints()
        self.seed_posture = start_joints
        (self.log_dir / "start_joints.json").write_text(
            json.dumps({"joints": start_joints.tolist(), "ee_pos": list(map(float, st["ee_pos"])), "t": time.time()})
        )
        say(f"  start joints saved -> {self.log_dir / 'start_joints.json'}")
        self.log({"stage": "preflight", "status": info, "gripper": grip, "problems": problems})
        if problems:
            for pr in problems:
                say(f"  PREFLIGHT FAIL: {pr}")
            raise Abort("preflight failed -- nothing commanded")
        if self.args.execute and require_ready:
            self.arm.take_control()  # no-op unless arbitration is enforced
        say("  preflight OK")

    def pose_path(self, p0, R0, p1, R1, spacing=RAY_STEP_MAX_M):
        """Straight-line (xyz, R) points from one tool pose to another, chunked so that no
        point moves more than `spacing` or turns more than ROT_STEP_MAX_DEG."""
        return pose_path(p0, R0, p1, R1, spacing, ROT_STEP_MAX_DEG)

    def move(self, path, name, track_abort_m=TRACK_ABORT_M, cart_speed=CART_SPEED):
        """Execute one straight leg: re-check `path` in the sim from the arm's actual pose, then
        send it (RammpArm.smooth_poses). Dry run: does nothing (press_button already checked it)."""
        if not self.args.execute:
            return
        p_now, R_now = self.arm.ee_pose()
        start = (self.arm.joints(), p_now, R_now)
        _, _, _, knots = self.arm.precheck_poses(path, name, self.log, posture=self.seed_posture, start=start)
        self.arm.smooth_poses(path, knots, start, name, self.log, track_abort_m=track_abort_m, cart_speed=cart_speed)

    def go_back(self, p0, R0, why, n_out=None):
        """After an aborted move: go back to (p0, R0) with a long (cuRobo) move, unless already
        there or in a dry run. Near the panel (n_out given), back straight out from it first
        (retreat). If that fails too, the arm holds."""
        p_now, R_now = self.arm.ee_pose()
        if not self.args.execute or (np.linalg.norm(p_now - p0) < 0.005 and rot_angle_deg(R_now, R0) < 2.0):
            return
        say(f"  {why} -- going back to the start first")
        if n_out is not None:
            self.retreat(n_out)
        self.long_move(p0, R0, "bail")

    def retreat(self, n_out, name="back out from the panel"):
        """Straight out along the panel normal n_out, RETREAT_OUT_M, keeping the wrist as it is:
        the way cuRobo may start from (its world has no microwave). Dry run: nothing."""
        if not self.args.execute:
            return
        p, Rm = self.arm.ee_pose()
        self.move(
            self.pose_path(p, Rm, p + RETREAT_OUT_M * np.asarray(n_out, dtype=float), Rm),
            name,
            track_abort_m=LOOSE_TRACK_ABORT_M,
        )

    def back_to_spot(self, p_spot, R_spot, why):
        """After an aborted press stroke: straight back to the pre-press spot (never cuRobo, right
        at the panel), unless already there or in a dry run. If that fails too, the arm holds."""
        p_now, R_now = self.arm.ee_pose()
        if not self.args.execute or (np.linalg.norm(p_now - p_spot) < 0.002 and rot_angle_deg(R_now, R_spot) < 2.0):
            return
        say(f"  {why} -- straight back to the pre-press spot")
        # The contact press runs on the driver's own Cartesian controller, which lets the
        # joints drift along the null space: seeded from the old posture, the first point of this
        # leg was a "17 deg joint jump" and the arm was left holding on the button (2026-10-15).
        self.reanchor_seed("after the press")
        self.move(
            self.pose_path(p_now, R_now, p_spot, R_spot, spacing=SMOOTH_PRESS_SPACING_M),
            "back to the spot",
            track_abort_m=LOOSE_TRACK_ABORT_M,
        )

    def long_move(self, p, Rm, name):
        """A move that need not be straight (staging, back to start, bail): the driver's
        go_to_ee_pose (cuRobo plans and the driver executes). Ends within LOOSE_TRACK_ABORT_M."""
        return self.arm.go_to_ee_pose(p, Rm, name, self.log, track_abort_m=LOOSE_TRACK_ABORT_M)

    def press(self, p_spot, R_spot, n_out, count=None, first=1):
        """Press `count` (default --presses) times from the pre-press spot, numbering the presses
        from `first`: in until the touch, --push-mm past it (RammpArm.press_contact), then straight
        back to the spot. If a press aborts, go back to the spot and hold."""
        a = self.args
        count = a.presses if count is None else count
        how = f"in until it touches, then up to {a.push_mm:.0f} mm further (or {a.stop_nm:.0f} N.m)"
        say(f"\n== press {f'x{count}' if first == 1 else first}: {how} and back ==")
        for i in range(first, first + count):
            try:
                res = self.arm.press_contact(
                    p_spot, n_out, f"press {i} (contact)", self.log, push=a.push_mm / 1000.0, stop_nm=a.stop_nm
                )
            except Abort as e:
                self.back_to_spot(p_spot, R_spot, f"press aborted ({e})")
                raise
            depth = res["depth_m"] if res else None
            self.log({"stage": "press_normal", "press": i, "depth": depth, "mode": "contact"})
            self.back_to_spot(p_spot, R_spot, f"press {i} done {depth*1000:.1f} mm in" if res else "press done")


def publish_pose(pub, frame, p, Rm):
    """Publish a tool pose as a PoseStamped (the planned pre-press goal, for viewing in RViz)."""
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
    q = Rotation.from_matrix(Rm).as_quat()
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
    pub.publish(msg)


def prepress_pose(origin, R, out=0.0):
    """Tool (xyz, R) for the stored pre-press spot, `out` metres further out along the normal."""
    p = from_panel(PREPRESS_EE_OFFSET_M + np.array([0.0, 0.0, out]), origin, R)
    return p, rot_from_panel(Rotation.from_quat(PREPRESS_EE_QUAT_PANEL).as_matrix(), R)


def stamp_ns(msg) -> int:
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def measure_panel_frame_domes(run: Run, target: str, after: int | None = None):
    """Precise panel frame from the dome detector: the measurement the press is planned from.

    Runs dome_pattern.detect on DOME_FRAMES frames, drops odd ones, and needs DOME_MIN_FITS to
    agree on the target pixel; then fits the panel plane and moves it into the arm base frame.
    `after` (ns, ROS clock): ignore frames captured before it -- pass the moment the arm stopped.
    Returns (origin = button xyz, R = panel axes), arm base frame. Raises Abort with a hint
    (move closer / back, panel in view?) if the detection fails or is unstable.
    """
    per = run.per
    if target not in dome_pattern.NAMES:
        raise Abort(f"dome detector knows {dome_pattern.NAMES}, not {target!r}")
    fx, fy, cx, cy = per.intrinsics()
    fits, seen, last, t0, reasons, prev = [], 0, None, time.monotonic(), [], None
    while seen < DOME_FRAMES and time.monotonic() - t0 < 6.0:
        color, depth = per.get("color", FRESH_S), per.get("depth", FRESH_S)
        if (
            color is None
            or depth is None
            or color is last
            or (after is not None and min(stamp_ns(color), stamp_ns(depth)) < after)
        ):
            time.sleep(0.03)  # wait for the next frame
            continue
        last = color
        seen += 1
        bgr, d = to_bgr_depth(per.bridge, color, depth)
        fit = dome_pattern.detect(bgr, d, fx, hint=prev)
        if fit is None and prev is not None:
            fit = dome_pattern.detect(bgr, d, fx)  # the hint lost it: search the whole frame
        prev = fit or prev
        if fit is None:
            reasons.append(why_no_fit(bgr, d, fx))
            continue
        try:
            n, dd, _, _ = dome_pattern.panel_plane(bgr, d, fit, (fx, fy, cx, cy))
        except PlaneFitError:
            continue
        fits.append((np.array(fit.px[target]), n, dd, fit, color.header.stamp))
    pxs = np.array([f[0] for f in fits]) if fits else np.zeros((0, 2))
    px = np.median(pxs, axis=0) if fits else None
    # Drop single odd frames (a highlight flicker, a frame grabbed mid-settle) instead of failing
    # on them; a camera that is actually moving leaves too few frames agreeing and still aborts.
    tol = max(DOME_MAX_SPREAD_PX, DOME_SPREAD_FRAC * float(np.median([f[3].s_px for f in fits]))) if fits else 0.0
    keep = [f for f, p in zip(fits, pxs, strict=False) if np.linalg.norm(p - px) <= tol] if fits else []
    say(f"  dome layout: {len(fits)}/{seen} frames fit all 5 domes, {len(keep)} agree within {tol:.0f} px")
    run.log({"dome_frames": {"target_px": pxs.tolist(), "seen": seen, "agree": len(keep)}})
    if not fits:
        why = max(set(reasons), key=reasons.count) if reasons else "no frames"
        raise Abort(
            f"dome detector: nothing in {seen} frames ({why}) -- "
            + (
                "move the camera back"
                if why.startswith("TOO CLOSE")
                else "move the camera closer"
                if why.startswith("TOO FAR")
                else "is the panel in view?"
            )
        )
    if len(keep) < DOME_MIN_FITS:
        raise Abort(
            f"dome detector: only {len(keep)}/{seen} frames agree on the {target} pixel (need {DOME_MIN_FITS}; "
            f"within {tol:.0f} px; pixels {np.round(pxs).astype(int).tolist()}) -- the camera or arm moving, "
            "or the detection is unstable from this view"
        )
    fits = keep
    pxs = np.array([f[0] for f in fits])
    px = np.median(pxs, axis=0)
    spread = float(np.max(np.linalg.norm(pxs - px, axis=1)))
    n = np.mean([f[1] for f in fits], axis=0)
    n /= np.linalg.norm(n)
    dd = float(np.median([f[2] for f in fits]))
    x_cam = dome_pattern.button_xyz_cam(px, n, dd, (fx, fy, cx, cy))
    t_bc, R_bc = per.cam_pose_at(fits[-1][4])  # the camera where the last kept frame was taken
    button = t_bc + R_bc @ x_cam
    n_base = R_bc @ n
    if abs(float(n_base[2])) > MAX_PANEL_NORMAL_Z:
        raise Abort(f"dome detector: panel normal {np.round(n_base, 3)} is not near-horizontal -- bad plane fit")
    f0 = fits[0][3]
    say(
        f"  {target} pixel {np.round(px, 1)} (spread {spread:.1f} px), dome pitch {f0.s_mm:.1f} mm, "
        f"{np.linalg.norm(x_cam)*100:.1f} cm from the camera"
    )
    say(f"  button xyz (base) {np.round(button, 3)}   panel normal (out) {np.round(n_base, 3)}")
    origin, R = panel_frame(button, n_base)
    run.log(
        {
            "panel_frame_domes": {
                "origin": origin.tolist(),
                "R": R.tolist(),
                "px": px.tolist(),
                "fits": len(fits),
                "seen": seen,
                "spread_px": spread,
            }
        }
    )
    return origin, R


def combine_scene_fits(fits, seen: int, reasons, min_fits: int = SCENE_MIN_FITS):
    """Combine the scene finder's per-frame fits into (button xyz, unit normal, frames kept).

    Keeps the frames within SCENE_MAX_SPREAD_M (button) and SCENE_MAX_NORMAL_SPREAD_DEG (normal)
    of the others and takes their median. Raises Abort, with the finder's own reason, when fewer
    than min_fits frames found the front or agree."""
    if seen == 0:
        raise Abort(f"scene camera: no frames in {SCENE_TIMEOUT_S:.0f} s -- is sheppy `scene_camera` up?")
    if len(fits) < min_fits:
        why = max(set(reasons), key=reasons.count) if reasons else "no reason given"
        raise Abort(f"scene camera: found the microwave in {len(fits)}/{seen} frames (need {min_fits}): {why}")
    buttons = np.array([f.button for f in fits])
    normals = np.array([f.n for f in fits])
    med = np.median(buttons, axis=0)
    n_mean = normals.mean(axis=0)
    n_mean /= np.linalg.norm(n_mean)
    ang = np.degrees(np.arccos(np.clip(normals @ n_mean, -1.0, 1.0)))
    keep = (np.linalg.norm(buttons - med, axis=1) <= SCENE_MAX_SPREAD_M) & (ang <= SCENE_MAX_NORMAL_SPREAD_DEG)
    if keep.sum() < min_fits:
        raise Abort(
            f"scene camera: unstable -- only {int(keep.sum())}/{seen} frames agree on the button within "
            f"{SCENE_MAX_SPREAD_M*100:.0f} cm and the normal within {SCENE_MAX_NORMAL_SPREAD_DEG:.0f} deg "
            f"(buttons {np.round(buttons, 3).tolist()})"
        )
    n = normals[keep].mean(axis=0)
    return np.median(buttons[keep], axis=0), n / np.linalg.norm(n), int(keep.sum())


def measure_panel_frame_scene(run: Run, target: str, use_yolo: bool = True):
    """Rough panel frame from the fixed scene camera (scene_finder), over SCENE_FRAMES frames.

    Each frame's search is narrowed to the YOLO node's microwave box when one is fresh (use_yolo;
    perception.yolo_box), else it covers the whole view. Same return as measure_panel_frame_domes,
    but good for staging only. Does not move the arm.

    The frames are searched side by side, one thread each: one after another they took ~1 s each
    (5 recorded frames: 4.3-5.1 s one by one, 2.0 s side by side, same fits; 2026-10-16)."""
    per = run.per
    T_c = T_d = None
    tool = run.arm.ee_pos()
    boxes, searches = [], []
    fits, reasons, seen, last, t0 = [], [], 0, (None, None), time.monotonic()
    with ThreadPoolExecutor(SCENE_FRAMES) as pool:
        while seen < SCENE_FRAMES and time.monotonic() - t0 < SCENE_TIMEOUT_S:
            c, d = per.get("scene_color", FRESH_S), per.get("scene_depth", FRESH_S)
            ci, di = per.get("scene_cinfo"), per.get("scene_dinfo")
            if c is None or d is None or ci is None or di is None or c is last[0] or d is last[1]:
                time.sleep(0.03)  # wait for a new colour AND depth frame
                continue
            if T_c is None:
                # each image's own optical frame; the Gemini's names if a message carries none
                T_c = per.transform(frame_of(c, SCENE_COLOR_FRAME))
                T_d = per.transform(frame_of(d, SCENE_DEPTH_FRAME))
            last = (c, d)
            seen += 1
            box = yolo_box(per.get("yolo", YOLO_FRESH_S), frame=frame_of(c, SCENE_COLOR_FRAME)) if use_yolo else None
            boxes.append(None if box is None else [round(v, 3) for v in box])
            searches.append(
                pool.submit(
                    scene_finder.find_microwave,
                    **scene_frame_from_msgs(per.bridge, c, d, ci, di, T_c, T_d),
                    target=target,
                    tool_xyz=tool,
                    arm_base=run.arm.kin.base_pos,
                    roi=None if box is None else box[:4],
                )
            )
    for search in searches:  # in frame order
        fit, why = search.result()
        if fit is None:
            reasons.append(why)
        else:
            fits.append(fit)
    n_box = sum(b is not None for b in boxes)
    if not use_yolo:
        say("  scene camera: --no-yolo, searching the whole view")
    elif n_box:
        sc = max(b[4] for b in boxes if b is not None)
        say(f"  scene camera: YOLO microwave box (score {sc:.2f}) narrowed the search in {n_box}/{seen} frames")
    else:
        say(
            f"  scene camera: no YOLO microwave box on {YOLO_TOPIC} (none fresh, score >= {YOLO_MIN_SCORE}) -- "
            "searching the whole view"
        )
    run.log({"stage": "scene_yolo", "use_yolo": use_yolo, "boxes": boxes})
    button, n_base, kept = combine_scene_fits(
        fits, seen, reasons, min_fits=SCENE_MIN_FITS_YOLO if n_box * 2 > seen else SCENE_MIN_FITS
    )
    if abs(float(n_base[2])) > MAX_PANEL_NORMAL_Z:
        raise Abort(f"scene camera: front normal {np.round(n_base, 3)} is not near-horizontal -- bad fit")
    f0 = fits[0]
    front = {
        k: np.median([getattr(f, k) for f in fits], axis=0).tolist() for k in ("centre", "corner_br", "ex", "ey", "n")
    }
    front.update(
        width_m=float(np.median([f.width_m for f in fits])), height_m=float(np.median([f.height_m for f in fits]))
    )
    say(
        f"  scene camera: microwave front in {len(fits)}/{seen} frames, {kept} agree; "
        f"{f0.width_m*100:.0f} x {f0.height_m*100:.0f} cm, window {f0.window_dark:.2f}, "
        f"{np.linalg.norm(button - T_c[:3, 3])*100:.0f} cm from the scene camera"
    )
    say(f"  button xyz (base) {np.round(button, 3)}  (ROUGH: scene camera)   panel normal (out) {np.round(n_base, 3)}")
    origin, R = panel_frame(button, n_base)
    # the front (median over the frames) is what tools/scene_offset_from_logs.py refines FACE_CORNER_TO_START_MM from
    run.log(
        {
            "panel_frame_scene": {
                "target": target,
                "origin": origin.tolist(),
                "R": R.tolist(),
                "fits": len(fits),
                "kept": kept,
                "seen": seen,
                "reasons": reasons,
                "front": front,
            }
        }
    )
    return origin, R


def first_look(run: Run, a):
    """First panel-frame estimate: returns (origin, R, source).

    Tries "scene" (the fixed scene camera; only with --far scene), then "wrist" (the domes). A
    scene failure is printed and the wrist camera tried; a wrist failure aborts."""
    if a.far == "scene":
        try:
            origin, R = measure_panel_frame_scene(run, a.target, use_yolo=getattr(a, "yolo", True))
            origin = origin + SCENE_NORMAL_CORRECTION_M * R[:, 2]
            say(
                f"  scene estimate moved {SCENE_NORMAL_CORRECTION_M*100:.0f} cm toward the arm (its known depth error) "
                f"-> button xyz (base) {np.round(origin, 3)}"
            )
            run.log({"stage": "scene_normal_correction", "m": SCENE_NORMAL_CORRECTION_M, "origin": origin.tolist()})
            if origin[2] < SCENE_MIN_BUTTON_Z_M:
                say(
                    f"  scene estimate z {origin[2]:.3f} is below {SCENE_MIN_BUTTON_Z_M:.2f} m: staging from "
                    f"{SCENE_MIN_BUTTON_Z_M:.2f} (too low for cuRobo; the wrist re-detect fixes the height)"
                )
                run.log({"stage": "scene_min_z", "z_was": float(origin[2]), "z_now": SCENE_MIN_BUTTON_Z_M})
                origin = np.array([origin[0], origin[1], SCENE_MIN_BUTTON_Z_M])
            return origin, R, "scene"
        except Abort as e:
            say(f"  {e}")
            say("  -> falling back to the wrist camera")
            run.log({"stage": "scene_fallback", "why": str(e)})
        finally:
            run.per.stop_scene()  # one look only: its frames would starve the wrist camera's
    origin, R = measure_panel_frame_domes(run, a.target)
    return origin, R, "wrist"


def staging_plan(source: str, cam_d: float):
    """Decide whether to stage before planning the press.

    Returns (how far out from the pre-press spot to stage, or None = plan from here; whether the
    MAX_GOTO_ROT_DEG wrist-turn limit applies to that move). A scene estimate always stages, with
    no turn limit (the arm may start facing anywhere). Wrist estimates stage only from beyond
    CLOSE_VIEW_M."""
    if source == "scene":
        return SCENE_STAGE_OUT_M, False
    if cam_d <= CLOSE_VIEW_M:
        return None, True
    return STAGE_OUT_M, True


def pick_lead_in(run: Run, p_st, R_st, n_out, wanted: float):
    """The longest lead-in of `wanted`, half of it, or none whose point the arm can reach and whose
    straight leg in passes the sim gates (checked before anything moves, from an IK solution at the
    lead-in point; cuRobo's own may differ, and run.move re-checks from it). Returns
    (lead m, lead-in point, straight leg poses or None). 2026-10-15: with the microwave 12 cm
    closer the full 10 cm put the point beside the arm's base, 7 mm short in IK, and the run aborted."""
    arm = run.arm
    q_now = arm.joints()
    p_now, R_now = arm.ee_pose()
    why = []
    for lead in (wanted, wanted / 2.0):
        if lead <= 0.005:
            break
        p_lead = np.asarray(p_st) + lead * np.asarray(n_out, dtype=float)
        leg = run.pose_path(p_lead, R_st, p_st, R_st)
        q_lead, err, _, _, _ = arm.kin.solve_translation(
            q_now, p_lead - p_now, posture=run.seed_posture, rot=R_st @ R_now.T
        )
        if err > MAX_IK_ERR_M:
            why.append(
                f"{lead*100:.0f} cm: lead-in point {np.round(p_lead, 3)} has no IK solution ({err*1000:.0f} mm off)"
            )
            continue
        try:
            arm.precheck_poses(leg, "lead-in", run.log, posture=run.seed_posture, start=(q_lead, p_lead, R_st))
        except Abort as e:
            why.append(f"{lead*100:.0f} cm: {e}")
            continue
        if why:
            say("  lead-in shortened -- " + "; ".join(why))
        say(
            f"  staging via a lead-in point {lead*100:.0f} cm further out {np.round(p_lead, 3)}: cuRobo to there, "
            "then straight in (checked in the sim)"
        )
        return lead, p_lead, leg
    if why:
        say("  no lead-in: cuRobo goes straight to the staging pose -- " + "; ".join(why))
    run.log({"stage": "lead_in", "wanted_m": wanted, "used_m": 0.0, "why": why})
    return 0.0, np.asarray(p_st), None


def go_to_staging(run: Run, p_st, R_st, n_out, lead: float):
    """Execute the staging move: cuRobo to the lead-in point `lead` m further out along n_out, then
    a straight sim-checked line in (STAGE_LEAD_IN_M); lead 0 = cuRobo all the way."""
    if lead <= 0:
        run.long_move(p_st, R_st, "to staging")
        return
    p_lead = np.asarray(p_st) + lead * np.asarray(n_out, dtype=float)
    run.long_move(p_lead, R_st, "to the lead-in point")
    # cuRobo picks its own elbow branch: check the straight leg from where it ended up, not from the
    # run's start posture (a 176 deg "jump" at the first point, 2026-10-15)
    run.reanchor_seed("at the lead-in point")
    p_now, R_now = run.arm.ee_pose()
    run.move(
        run.pose_path(p_now, R_now, p_st, R_st),
        "straight in to staging",
        track_abort_m=LOOSE_TRACK_ABORT_M,
        cart_speed=LEAD_IN_SPEED_MPS,
    )


def fingertip_out_of_panel(p_tool, R_tool, origin, R) -> float:
    """Signed distance (m) of the fingertip in front of the fitted panel plane (negative = inside)."""
    tip = np.asarray(p_tool) + np.asarray(R_tool)[:, 2] * FINGERTIP_PAST_TOOL_M
    return float(to_panel(tip, origin, R)[2])


def check_fingertip(p_tool, R_tool, origin, R, what):
    """Abort if this tool pose puts the fingertip less than MIN_PREPRESS_FINGERTIP_CLEAR_M in front
    of the panel: the sign of a PREPRESS_EE_OFFSET_M measured on another rig."""
    out = fingertip_out_of_panel(p_tool, R_tool, origin, R)
    say(f"  fingertip at the {what}: {out*100:+.1f} cm out of the fitted panel plane")
    if out < MIN_PREPRESS_FINGERTIP_CLEAR_M:
        raise Abort(
            f"the stored pre-press offset puts the fingertip {-out*100:.1f} cm INTO the panel at the {what} "
            f"(need >= {MIN_PREPRESS_FINGERTIP_CLEAR_M*100:.0f} cm in front) -- PREPRESS_EE_OFFSET_M is from "
            "another rig/calibration; re-measure it here with measure_prepress_offset"
        )


def check_move(p_from, R_from, p_to, R_to, what):
    """Abort if moving between these poses would turn the wrist more than MAX_GOTO_ROT_DEG."""
    turn = rot_angle_deg(R_from, R_to)
    if turn > MAX_GOTO_ROT_DEG:
        raise Abort(
            f"wrist would turn {turn:.0f} deg to the {what} (> {MAX_GOTO_ROT_DEG:.0f}) -- start facing the panel"
        )


def redetect_at_staging(run: Run, a, origin, R, source: str, stopped: int, first=None):
    """Step 3's re-detect: the domes from the staging pose, once more from RETRY_BACKOFF_M further out
    if that fails. `origin`, `R`: the far estimate staging was planned from; `stopped`: when the arm
    stopped (ns); `first`: the first try, already measured (LookAhead.measured: returns its (origin, R)
    or raises its Abort), None = measure it here. Returns the close (origin, R) the press is planned from."""
    arm = run.arm
    far = origin
    try:
        try:
            origin, R = first() if first is not None else measure_panel_frame_domes(run, a.target, after=stopped)
        except Abort as e:
            # Once: back straight out along the (far) panel normal and look again -- the far
            # estimate can put staging closer than planned (21.8 cm, 2026-10-12).
            say(f"  re-detect failed ({e}); backing out {RETRY_BACKOFF_M*100:.0f} cm and trying once more")
            p_s, R_s = arm.ee_pose()
            run.move(
                run.pose_path(p_s, R_s, p_s + RETRY_BACKOFF_M * R[:, 2], R_s),
                "back out to re-detect",
                track_abort_m=LOOSE_TRACK_ABORT_M,
            )
            stopped = run.per.get_clock().now().nanoseconds
            origin, R = measure_panel_frame_domes(run, a.target, after=stopped)
    except Abort as e:
        if source != "scene":
            raise
        raise Abort(
            f"{e} -- staged from the scene camera's estimate {np.round(far, 3)}: if the domes are out "
            "of the wrist camera's view, the scene camera's calibration (box-opening's camera_scene.yaml) "
            "or FACE_CORNER_TO_START_MM is off"
        ) from None
    moved = float(np.linalg.norm(origin - far))
    say(f"  button moved {np.round((origin - far) * 100, 1)} cm vs the {source} estimate ({moved*100:.1f} cm)")
    run.log(
        {
            "stage": "staging_redetect",
            "source": source,
            "far_button_xyz": far.tolist(),
            "button_xyz": origin.tolist(),
            "moved_cm": moved * 100,
        }
    )
    return origin, R


def plan_press(run: Run, a, pub, p0, origin, R, presses: int, back: bool):
    """Step 4: the pre-press spot from the (close) panel frame, and every leg -- the approach, `presses`
    strokes in and out, and (back) the way out and back to the start p0 -- solved and gated in the sim
    before anything moves. Returns (approach path, stroke in, stroke out, spot xyz, spot R)."""
    arm = run.arm
    n_out = R[:, 2]
    p_goal, R_goal = prepress_pose(origin, R)
    p_now, R_now = arm.ee_pose()
    say(f"  pre-press tool pose: {describe(PREPRESS_EE_OFFSET_M)}")
    say(f"  tool xyz now  {np.round(p_now, 3)}")
    say(
        f"  tool xyz goal {np.round(p_goal, 3)}   ({np.linalg.norm(p_goal - p_now)*100:.1f} cm straight line, "
        f"wrist turns {rot_angle_deg(R_now, R_goal):.1f} deg)"
    )
    publish_pose(pub, a.arm_frame, p_goal, R_goal)
    check_fingertip(p_goal, R_goal, origin, R, "pre-press spot")
    run.log(
        {
            "stage": "press_button_plan",
            "target": a.target,
            "button_xyz": origin.tolist(),
            "panel_R": R.tolist(),
            "ee_start": p0.tolist(),
            "ee_now": p_now.tolist(),
            "ee_goal": p_goal.tolist(),
        }
    )
    check_move(p_now, R_now, p_goal, R_goal, "pre-press spot")

    # Plan every leg back to back in the sim before anything moves.
    say("\n== plan (sim pre-check of every leg) ==")
    go = run.pose_path(p_now, R_now, p_goal, R_goal)
    # the press ends wherever the button is, so gate the deepest it may go
    p_in = p_goal - (CONTACT_MAX_TOUCH_M + a.push_mm / 1000.0) * n_out
    stroke_in = run.pose_path(p_goal, R_goal, p_in, R_goal, spacing=SMOOTH_PRESS_SPACING_M)
    stroke_out = run.pose_path(p_in, R_goal, p_goal, R_goal, spacing=SMOOTH_PRESS_SPACING_M)
    legs = [("to pre-press", go)]
    for i in range(presses):
        legs += [(f"press {i + 1} in", stroke_in), (f"press {i + 1} out", stroke_out)]
    if back:
        legs.append(("back out from the panel", run.pose_path(p_goal, R_goal, p_goal + RETREAT_OUT_M * n_out, R_goal)))
    state = None
    for name, path in legs:
        err, jump, state, _ = arm.precheck_poses(path, name, run.log, posture=run.seed_posture, start=state)
        say(f"  {name:16s} {len(path):3d} pts  IK <= {err*1000:.1f} mm  joint step <= {jump:.1f} deg  ok")
    if back:
        # go_to_ee_pose cannot plan without moving, so only the goal is checked here; the driver
        # plans the way back from the real joints when it runs.
        arm.check_goal_pose(p0, "back to start (check)")
        say(
            f"  back to start    go_to_ee_pose  goal ok (from {RETREAT_OUT_M*100:.0f} cm out; path planned by cuRobo "
            "in the driver, not previewed)"
        )
    return go, stroke_in, stroke_out, p_goal, R_goal


class LookAhead:
    """--interactive, at the staging pose: the first re-detect (measure_panel_frame_domes: the camera
    only) and then plan_press (the sim only) on a background thread while the staging stop waits for
    space, so space goes straight to the pre-press spot. Nothing in it moves the arm: a failed
    re-detect's back-out-and-retry (redetect_at_staging) still runs only after space. Anything it
    raises is kept and raised where the run asks for the result (measured / planned)."""

    def __init__(self, run: Run, a, pub, p0, stopped: int, presses: int, back: bool):
        self._measured = self._planned = None
        self._thread = threading.Thread(
            target=self._work, args=(run, a, pub, p0, stopped, presses, back), name="look-ahead", daemon=True
        )
        self._thread.start()

    def _work(self, run, a, pub, p0, stopped, presses, back):
        say("\n== re-detect and plan from the staging pose (background) ==")
        try:
            origin, R = measure_panel_frame_domes(run, a.target, after=stopped)
        except BaseException as e:  # Abort is a SystemExit
            self._measured = e
            say(f"  re-detect failed ({e}) -- space backs out {RETRY_BACKOFF_M*100:.0f} cm and tries once more")
            return
        self._measured = (origin, R)
        try:
            self._planned = plan_press(run, a, pub, p0, origin, R, presses, back)
        except BaseException as e:
            self._planned = e
            say(f"  plan refused: {e}")
            return
        say("  ready -- space goes straight to the pre-press spot")

    def wait(self):
        """Until the thread is done (it uses the camera and the sim)."""
        if self._thread.is_alive():
            say("  (waiting for the background re-detect and plan)")
        self._thread.join()

    def measured(self):
        """The re-detect's (origin, R); raises what it raised."""
        self.wait()
        if isinstance(self._measured, BaseException):
            raise self._measured
        return self._measured

    def planned(self):
        """plan_press's result from the measured frame; raises what it raised. None when the re-detect
        failed (the retry's frame is then planned in the foreground)."""
        self.wait()
        if isinstance(self._measured, BaseException):
            return None
        if isinstance(self._planned, BaseException):
            raise self._planned
        return self._planned


def press_button(run: Run, a, pub, decide=None) -> int:
    """The whole run, steps 1-7 of the module docstring. Returns 0; raises Abort on any failed check.

    decide (--interactive; see decide.py): asked at the staging pose and at the pre-press spot, GO or
    BACK (Stops). None, or a dry run: the one-shot run, no stops."""
    arm = run.arm
    run.preflight()
    p0, R0 = arm.ee_pose()
    stops = Stops(run, decide if a.execute else None, p0, R0)
    presses, back = (1, True) if stops.on else (a.presses, not a.no_return)
    ahead = None  # LookAhead: the staging stop's background re-detect and plan

    say(f"\n== detect: {a.target} ==")
    origin, R, source = first_look(run, a)
    cam_d = float(np.linalg.norm(origin - run.per.cam_position_in_base()))
    stage_out, limit_turn = staging_plan(source, cam_d)
    if source == "scene" and a.stage_out is not None:
        stage_out = a.stage_out
    if stage_out is not None:
        # Go to a staging pose in front of the (rough) button, re-detect the domes from there, and
        # plan everything else from that close measurement.
        check_fingertip(*prepress_pose(origin, R), origin, R, "pre-press spot (far estimate)")
        p_st, R_st = prepress_pose(origin, R, out=stage_out)
        if source == "scene":
            say(
                f"  scene camera estimate: staging {stage_out*100:.0f} cm out from the pre-press spot "
                "(cuRobo; wrist turn not limited for this move)"
            )
        else:
            say(f"  camera is {cam_d*100:.0f} cm from the button (> {CLOSE_VIEW_M*100:.0f}): staging closer first")
        if limit_turn:
            check_move(p0, R0, p_st, R_st, "staging pose")
        lead, p_lead, lead_leg = pick_lead_in(run, p_st, R_st, R[:, 2], max(0.0, a.lead_in))
        if not a.execute:  # dry run: goal gates only (go_to_ee_pose has no plan-only mode)
            arm.go_to_ee_pose(p_lead, R_st, "to the lead-in point" if lead_leg else "to staging", run.log)
            say(
                "\ndry run -- the staging leg passed the gates. The approach and press are planned only after "
                "re-detecting at the staging pose, so they need --execute (or start within "
                f"{CLOSE_VIEW_M*100:.0f} cm with --far wrist to dry-run the whole thing)"
            )
            return 0
        say("\n== go to the staging pose ==")
        try:
            go_to_staging(run, p_st, R_st, R[:, 2], lead)
        except Abort as e:
            run.go_back(p0, R0, f"move aborted ({e})")
            raise
        stopped = run.per.get_clock().now().nanoseconds  # smooth_poses returned: the arm is still
        run.reanchor_seed("at the staging pose")
        if stops.on:
            ahead = LookAhead(run, a, pub, p0, stopped, presses, back)
        if not stops.at_staging(R[:, 2], ahead):
            return 0
        if ahead is None:
            say("\n== re-detect from the staging pose ==")
        with stops.holding(R[:, 2]):
            origin, R = redetect_at_staging(
                run, a, origin, R, source, stopped, first=None if ahead is None else ahead.measured
            )
    n_out = R[:, 2]
    with stops.holding(n_out):
        planned = None if ahead is None else ahead.planned()
        go, _, _, p_goal, R_goal = planned or plan_press(run, a, pub, p0, origin, R, presses, back)
    if not a.execute:
        say("\ndry run -- every leg passed the gates; add --execute to move")
        return 0

    say("\n== go to the pre-press spot ==")
    try:
        run.move(go, "to pre-press")
    except Abort as e:
        run.go_back(p0, R0, f"move aborted ({e})", n_out=n_out)
        raise

    if stops.on:
        n = stops.at_button(lambda i: run.press(p_goal, R_goal, n_out, count=1, first=i), n_out)
        say(f"\ndone ({n} press{'' if n == 1 else 'es'}, back at the start).")
        return 0

    if a.presses:
        run.press(p_goal, R_goal, n_out)  # backs itself out on abort

    if a.no_return:
        say("\ndone (--no-return: holding at the pre-press spot).")
        return 0
    say("\n== back to start ==")
    back_to_start(run, p0, R0, n_out)
    say("\ndone.")
    return 0


class Stops:
    """--interactive: the run's two stops, asked through `decide` (decide.py) and logged.

    At the staging pose and at the pre-press spot the arm holds until the answer: GO goes on, BACK
    goes back to the start (back_to_start: straight out from the panel first, then cuRobo). An Abort
    inside `holding` (the arm away from the start) holds too, until q, then goes back. decide None
    (the one-shot run, a dry run): every stop passes straight through."""

    def __init__(self, run: Run, decide, p0, R0):
        self.run, self.decide, self.p0, self.R0 = run, decide, p0, R0

    @property
    def on(self) -> bool:
        return self.decide is not None

    def ask(self, stop: str, prompt: str, allowed=(GO, BACK)) -> str:
        choice = self.decide(stop, prompt, allowed)
        self.run.log({"stage": "decision", "stop": stop, "choice": choice})
        return choice

    def go_back(self, n_out, why: str):
        """Back to the start. Never inside `holding`: a failure here holds and ends the run."""
        say(f"\n== back to start ({why}) ==")
        back_to_start(self.run, self.p0, self.R0, n_out)

    def at_staging(self, n_out, ahead: LookAhead | None = None) -> bool:
        """Stop 1, at the staging pose: True = go on; False = went back to the start (the run is over).
        `ahead`: the background re-detect and plan, waited for before the way back (it uses the sim)."""
        if not self.on:
            return True
        say("\n== STOP at the staging pose: check the wrist camera (Foxglove) ==")
        if self.ask("staging", "space: go to the button   q: go back to the start") == GO:
            return True
        if ahead is not None:
            ahead.wait()
        self.go_back(n_out, "q at the staging pose")
        say("\ndone (went back from the staging pose).")
        return False

    def at_button(self, press, n_out) -> int:
        """Stop 2, at the pre-press spot: each GO is one press(i), i = 1, 2, ...; BACK goes back to the
        start. Returns the number of presses."""
        n = 0
        say("\n== STOP at the button ==")
        with self.holding(n_out):
            while self.ask("button", f"space: press{' again' if n else ''}   q: go back to the start") == GO:
                n += 1
                press(n)
        self.go_back(n_out, "q at the button")
        return n

    @contextmanager
    def holding(self, n_out):
        """An Abort in this block, with the arm away from the start: say why, hold until q, go back,
        then end the run (the Abort, re-raised). At the start, or not interactive: re-raised as is."""
        try:
            yield
        except Abort as e:
            if not self.on:
                raise
            p, Rm = self.run.arm.ee_pose()
            if np.linalg.norm(p - self.p0) < 0.005 and rot_angle_deg(Rm, self.R0) < 2.0:
                raise
            say(f"\nSTOPPED: {e}\n  the arm is holding here")
            self.run.log({"stage": "abort_at_stop", "why": str(e)})
            self.ask("failed", "q: go back to the start   (Ctrl-C: leave the arm holding here)", allowed=(BACK,))
            self.go_back(n_out, "q after a failure")
            raise Abort(f"{e} -- went back to the start") from None


def back_to_start(run: Run, p0, R0, n_out):
    """From the panel back to the start pose: straight out RETREAT_OUT_M along the panel normal
    first, then cuRobo (its world has no microwave: never let it start right at the panel)."""
    run.retreat(n_out)
    run.long_move(p0, R0, "back to start")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--target",
        default="start_30s",
        choices=dome_pattern.NAMES,
        help="which button (default start_30s = START/+30SEC). The stored pre-press spot is "
        "relative to the target button, so it is the same for every one.",
    )
    ap.add_argument("--execute", action="store_true", help="actually move the arm (default: dry run)")
    ap.add_argument("--presses", type=int, default=1, help="0 = go to the spot and come back without pressing")
    ap.add_argument("--no-return", action="store_true", help="stay at the pre-press spot at the end")
    ap.add_argument("--log-dir", default=str(Path.home() / "press_logs"))
    ap.add_argument(
        "--arm-frame",
        default="base_link",
        help="the frame the run works in: /ee_state's and go_to_ee_pose's (base_link on the 1.1.0 driver, "
        "arm_mount on the nightly). Its z must point up: the panel frame and the scene finder take +z as up",
    )
    ap.add_argument(
        "--joint-prefix",
        default="",
        help="prefix of the driver's joint names (the nightly's are arm_joint_1..7: 'arm_')",
    )
    ap.add_argument(
        "--mount-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("R", "P", "Y"),
        help="the arm base's orientation in --arm-frame (the robot's mount.yaml rpy_deg; flatbot 0 30 0). The "
        "sim plans in --arm-frame with it; preflight's tool-frame check refuses a wrong one",
    )
    ap.add_argument(
        "--camera-frame",
        default=CAMERA_FRAME,
        help=f"the wrist camera's optical frame in tf (default {CAMERA_FRAME}, box-opening's D405 mount)",
    )
    ap.add_argument("--camera-ns", default="/wrist_camera", help="wrist camera topic namespace")
    ap.add_argument(
        "--depth-topic",
        default=DEPTH_TOPIC,
        help="wrist depth, aligned to colour, relative to --camera-ns (the OAK's: depth/image_rect_raw)",
    )
    ap.add_argument(
        "--far",
        choices=("scene", "wrist"),
        default="scene",
        help="first look: the fixed scene camera (default; falls back to the wrist camera when it "
        "finds nothing) or the wrist camera only (the behaviour before 2026-10-14)",
    )
    ap.add_argument("--scene-ns", default=SCENE_NS, help="scene camera topic namespace")
    ap.add_argument(
        "--scene-depth-topic",
        default=SCENE_DEPTH_TOPIC,
        help="scene depth, relative to --scene-ns (the OAK's: depth/image_rect_raw)",
    )
    ap.add_argument(
        "--no-yolo",
        dest="yolo",
        action="store_false",
        help=f"scene camera: search the whole view, not just the YOLO node's microwave box ({YOLO_TOPIC})",
    )
    ap.add_argument(
        "--push-mm",
        type=float,
        default=CONTACT_PUSH_M * 1000.0,
        metavar="MM",
        help=f"contact press: push at most MM past the touch (default {CONTACT_PUSH_M*1000:.0f})",
    )
    ap.add_argument(
        "--stop-nm",
        type=float,
        default=CONTACT_STOP_NM,
        metavar="NM",
        help=f"contact press: stop the push when a wrist torque is NM off its baseline (default "
        f"{CONTACT_STOP_NM:g}; the touch is {CONTACT_TOUCH_NM:g})",
    )
    ap.add_argument(
        "--lead-in",
        type=float,
        default=STAGE_LEAD_IN_M,
        metavar="M",
        help=f"cuRobo goes only to a point M metres further out than the staging pose (default "
        f"{STAGE_LEAD_IN_M}), then a straight line goes in; 0 = cuRobo all the way",
    )
    ap.add_argument(
        "--stage-out",
        type=float,
        default=None,
        metavar="M",
        help=f"scene estimate: stage M metres out from the pre-press spot (default {SCENE_STAGE_OUT_M}); "
        "further out = the wrist camera sees more of the panel",
    )
    ap.add_argument(
        "--acquire-control",
        action="store_true",
        help="under enforced arbitration, acquire a token (only if nobody holds the arm)",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="with --execute: wait at the staging pose and at the button for a key in this terminal "
        "(space: go on / press once, q: go back to the start); --presses is ignored, --no-return "
        "refused. A dry run has no stops",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and run press_button. Returns the exit code: 0 done, 2 aborted (arm holding)."""
    a = build_parser().parse_args(argv)
    if a.presses < 0:
        say("--presses must be >= 0")
        return 2
    if not 0 < a.push_mm <= MAX_PUSH_MM:
        say(f"--push-mm must be in (0, {MAX_PUSH_MM:.0f}]")
        return 2
    if not CONTACT_TOUCH_NM < a.stop_nm <= 10.0:
        say(f"--stop-nm must be above the touch ({CONTACT_TOUCH_NM:g}) and at most 10")
        return 2
    decide = None
    if a.interactive:
        if a.no_return:
            say("--no-return does not go with --interactive: q at the button is the way back")
            return 2
        if a.presses != 1:
            say(f"--interactive: --presses {a.presses} ignored -- each space at the button is one press")
        if not a.execute:
            say("--interactive: dry run -- no stops (nothing moves)")
        else:
            decide = TerminalKeys()
            try:
                decide.check()
            except Abort as e:
                say(f"ABORT: {e}")
                return 2
    rclpy.init()
    run = None
    try:
        try:
            run = Run(a)
            pub = run.per.create_publisher(PoseStamped, "/button_press/target_pose", 1)
            return press_button(run, a, pub, decide=decide)
        except Abort as e:
            say(f"\nABORT: {e}")
            if run is not None:
                run.log({"stage": "abort", "why": str(e)})
            return 2
        except KeyboardInterrupt:
            if decide is None:
                raise
            say(
                "\nCtrl-C: run ended. Nothing more is sent to the arm (a move already sent finishes; "
                "the e-stop stops it)"
            )
            if run is not None:
                run.log({"stage": "abort", "why": "Ctrl-C"})
            return 130
    finally:
        if run is not None:
            run.arm.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
