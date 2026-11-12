"""The place_container capability: put the held food container down at a spot on the table.

Ported from RAMMP-table (place_bowl.py + kinova_arm.py), where it placed the container on a marked
sticky note on the flatbot (2026-10-26). Everything is in arm_mount (x forward, y left, z true up).

Actions:
  place_here  the UI's one tap (offered on a `table`; no params): place for real at the marked spot and
           end back at the start. Every action shows its phases and any error on the UI panel and is
           recorded (a rosbag + run.json, bag.py).
  mark     the closed fingertips touching the target -> saved as the spot (its z = the table). No motion.
  place    the container, gripped by its lip, to the spot ({x, y, table_z}, or the marked one):
             1. adjust: square the hand in place to the nearest hold (level, pointing `pitch_deg` down);
             2. lift to SAFE_HEIGHT above the table, if lower; 3. transit to above the spot; 3b. the
             approach straight down to `lower_from` above where the container meets the table
             -- 1-3b are ONE go_to_joint_configs motion through our planned joints (cuRobo, around the
             chair's collision volumes, blended at the waypoints; `carry_targets` pose: go_to_ee_poses;
             `blend` false: separate moves as before);
             4. the rest of the straight line down in the driver's IMPEDANCE mode
             until contact, then hold `hold_press` below it (lower.py); 5. open, drop `release_drop`;
             6. back out: straight back, or a short way back and then a turn of the hand that swings
             the fingertips off the lip (plan.plan_swing).
           The arm is left in impedance, held where the retreat ended.
  retreat  only step 6 (after a place stopped with the gripper open).
  reset    back to the start position (where the last place started, or `save`d): straight up,
           then to the start's joints, one cuRobo motion (go_to_joint_configs, around the chair's
           collision volumes); gripper closed.

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`.
Otherwise every goal is a dry run that plans and checks and returns the plan. Someone stands at the
e-stop whenever allow_execute is on.

    ros2 launch place_container place_container.launch.py use_orchestrator:=false allow_execute:=true
    make action CAP=place_container ACTION=mark
    make action CAP=place_container ACTION=place PARAMS='{"execute": false}'
"""

import contextlib
import json
import time
from pathlib import Path

import numpy as np
import rammp_arm
from capability_engine import Action, ArmError, Cancelled, CapabilityEngine, EngineError, run
from capability_engine.arm import HOLD_LEVEL, HOLD_NONE, JointTarget, PoseTarget
from rammp_arm import _unwrap_from
from scipy.spatial.transform import Rotation

from . import lower, plan, ui

# The arm moved 8 in (0.2032 m) forward on the flatbot's chair (rammp-deployments #156, merged
# 2026-11-07 17:56 UTC). Everything here is in arm_mount, which moved with the arm: the arm's own
# geometry (flange_to_tip, pitch, reach as kinematics) still holds, but a spot or start position saved
# before then was saved relative to a chair and table that are now 0.2032 m nearer in x.
ARM_MOVED = 1794074180.0  # 2026-11-07T17:56:20Z
# deg: after the carry, the planned line down is used if the arm is within this of its first joints
# (lower() starts the line from the measured joints, so this much is a smooth start, not a jump)
REPLAN_DEG = 1.0

# ROS parameter -> default: the flatbot's values (2026-10-26, unless noted).
PARAMS = {
    "allow_execute": False,
    # flange face to the closed fingertips (m): sets where the container centre lands
    "flange_to_tip": plan.FLANGE_TO_TIP,
    # the hand points this many deg down while placing, so the container touches first
    "pitch_deg": plan.PLACE_PITCH_DEG,
    # contact: vz (m/s, + up) back up to this after moving down; 0.0 came 4.5 s late on the flatbot
    "vz_stop": -0.01,
    # the hold after contact, this far (m) below the contact point, while the gripper opens
    "hold_press": 0.003,
    # after the gripper opens, the hand goes this far (m) straight down before the retreat, which then
    # doesn't lift: the open fingers leave the lip downward (lifting first dragged the container,
    # 2026-11-10). 0.0 = off (the retreat lifts by hold_press as before)
    "release_drop": 0.01,
    # held in impedance at the start this long (s) before going down; must end still. 0 = no pause:
    # the arm goes compliant as the descent starts (contact_ignore_m covers its sag)
    "settle_s": 0.0,  # flatbot 2026-11-10: contact_ignore_m covers the sag; 3.0 added ~4 s
    # contact is ignored until the descent's command has gone this far down (m): the sag when the arm
    # goes compliant must not read as moving down and being stopped
    "contact_ignore_m": 0.08,  # 0.03 armed mid catch-up: a false contact (flatbot 2026-11-10)
    # contact more than this (m) above where the bottom meets the table = not the table
    "early_stop": 0.03,
    # joint impedance for the settle, lowering, hold, retreat and reset, one set so the stiffness never
    # changes mid-place (move_joints takes custom gains, not the driver's named profiles). Default: the
    # driver's documented starting point (rammp_arm.IMPEDANCE), NOT the "medium" profile RAMMP-table
    # used -- compare the settle's sag in the recording (medium: 1.0 cm on the flatbot, 2026-10-26)
    "impedance_kq": list(rammp_arm.IMPEDANCE["kq"]),
    "impedance_zeta": rammp_arm.IMPEDANCE["zeta"],
    "impedance_torque_limit": list(rammp_arm.IMPEDANCE["torque_limit"]),
    # the slow impedance lowering starts this far (m) above where the container's bottom meets the table;
    # the carrying motion takes the hand straight down to there first (position mode, fast). 0.0 = the
    # whole way down from the carrying height slow, as before (~14 s at the flatbot's spot, 2026-11-10)
    "lower_from": 0.06,
    # speeds. The lowering stays at the 2 cm/s its contact test was tuned at; the rest is 2x RAMMP-table's
    # (2026-10-29): the hand's speed where the table should be (m/s)
    "lower_speed": 0.02,
    # speed_scale of the carrying moves (go_to_ee_pose; was 0.2)
    "go_to_speed_scale": 0.6,  # flatbot 2026-11-10 (was 0.4)
    # speed_scale of the squaring turn with blend false (move_joints; was 0.5)
    "adjust_speed_scale": 1.0,
    # adjust -> lift -> transit as ONE go_to_ee_poses motion (cuRobo, blended at the waypoints, all at
    # go_to_speed_scale); false = separate moves as on 2026-10-29 (adjust on move_joints)
    "blend": True,
    # with blend: "pose" = the squaring turn is the motion's first target (unheld: it changes the tilt);
    # "joints" = a go_to_joint_config to our checked joints first (if cuRobo swings the hand for the
    # pose, as it did 2026-10-08), then lift -> transit blended
    "adjust_mode": "pose",
    # with blend: "joints" = the carrying motion goes through OUR planned joints (go_to_joint_configs,
    # blended; plan_segment keeps each leg level, above the table and under 120 deg a joint), so the arm
    # keeps the start's posture; "pose" = hand poses, cuRobo picks the joints -- on 2026-11-10 it chose
    # another posture at the spot (J1 +101, J3 -147 deg from the start) and reset could not undo it
    "carry_targets": "joints",
    # the retreat's peak speed (m/s; was 0.03)
    "retreat_speed": 0.10,  # flatbot 2026-11-10 (was 0.06)
    # speed_scale of reset's joint moves with reset_mode legacy (move_joints; was ~0.5)
    "reset_speed_scale": 1.0,
    # reset: "curobo" = one go_to_joint_configs motion (up, then the start), planned around the chair's
    # collision volumes, at go_to_speed_scale; "legacy" = the impedance move_joints moves (no chair check)
    "reset_mode": "curobo",
    # blend_radius (m) where a blended motion passes a waypoint near the table or the container (the
    # table is not in cuRobo's world), and up at the carrying height
    "blend_radius_near": 0.03,
    "blend_radius_far": 0.05,
    # before lowering, ask cuRobo whether the ends of the moves it doesn't plan (the line down, the
    # retreat) are clear of the chair's collision volumes
    "curobo_check": True,
    # spot distance from the arm base axis (m) where the whole place plans with the full descent
    # (flatbot, table z 0.198: test_place_bowl.py --sweep, 2026-10-29). Arm kinematics only: the sweep
    # had no chair collision volumes, so re-measure it with the planner's. Outside it the plan decides
    "reach": [0.70, 1.10],
    # where the line down can't reach the table in the arm's posture, switch to another posture that can
    # (go_to_joint_config: a bigger move, cuRobo in joint space, no level hold); off = refuse instead
    "allow_reconfigure": False,
    # the squaring turn may correct at most this much (deg)
    "max_adjust_deg": plan.MAX_ADJUST_DEG,
    # a marked spot older than this (min) is refused: the chair may have moved
    "max_spot_age_min": 30.0,
    # the arm's tool frame vs the model's (m): further apart = wrong model or tool configuration
    "tool_frame_tol": 0.015,
    # where the spot, the start position and the descent recordings are kept
    "state_dir": "/config/place_container",
    # a spot or start position saved before this (unix s) is refused: the arm has moved on the chair
    # since (default: ARM_MOVED)
    "state_valid_after": ARM_MOVED,
    # a failure is shown on the user's UI panel with an OK button; the action waits this long (s) for
    # the OK before it ends with the error (the panel closes when the action ends). 0 = don't wait
    "ui_error_wait_s": 60.0,
    # after letting go: "blended" = drop, back out (and swing) and go back to the start as ONE
    # go_to_joint_configs motion in impedance (cuRobo, blended), so place ends at the start; "legacy"
    # = the separate drop / back_out moves of 2026-11-10, ending by the container (then reset)
    "return_mode": "blended",
    # speed_scale of that motion's legs by the container (the drop and the retreat); the way up and
    # over to the start goes at go_to_speed_scale
    "return_speed_scale": 0.3,
    # a rosbag + run.json for every action run, in <state_dir>/runs/ (bag.py); false = none
    "record_bag": True,
    # arm state / scene camera frames per second in the bag
    "bag_state_hz": 25.0,
    "bag_frame_hz": 1.0,
    # the scene camera's colour, camera_info and depth in the bag (context; this capability uses no camera)
    "bag_camera": True,
    "bag_camera_topics": [
        "/scene_camera/color/image_raw",
        "/scene_camera/color/camera_info",
        "/scene_camera/depth/image_rect_raw",
    ],
    # mock only: place fails at this phase (carry, lower, open, return), to try the UI's error panel
    # from the UI (which sends no params); "" = none. A goal's own mock_fail wins
    "mock_fail": "",
}
EXECUTE = {"type": "object", "properties": {"execute": {"type": "boolean", "default": False}}}
PLACE = {
    "type": "object",
    "properties": {
        **EXECUTE["properties"],
        # the container centre (m, arm_mount); all three, or none for the marked spot
        "x": {"type": "number"},
        "y": {"type": "number"},
        "table_z": {"type": "number"},
        # an empty-gripper test run: don't refuse an open gripper
        "allow_empty_gripper": {"type": "boolean", "default": False},
        # mock only: fail at this phase (carry, lower, open, return) to see the error on the UI
        "mock_fail": {"type": "string"},
    },
}
RESET = {
    "type": "object",
    "properties": {
        **EXECUTE["properties"],
        "save": {"type": "boolean", "default": False},
        # nothing in the (closed) gripper: skip the container-tilt check on the way back
        "empty": {"type": "boolean", "default": False},
        # empty hand only: allow a joint to turn more than 120 deg on the way back (one big cuRobo joint
        # move, around the chair's volumes) -- to recover from a different arm posture
        "big_turn": {"type": "boolean", "default": False},
    },
}


def _panel_open(ctx):
    """Whether the run's UI panel goal is still open (or opening): the engine's own run record (PR #42).
    True when it can't tell, so nothing is re-sent needlessly. From microwave (feature/14)."""
    run = getattr(ctx, "_run", None)
    if run is None or not hasattr(run, "ui_goal_handle"):
        return True
    return bool(run.ui_goal_handle is not None or getattr(run, "ui_opening", False))


class _PanelContext:
    """The action's ctx, with every progress() also put on the user's UI panel in words
    (ui.progress_panel). Everything else is ctx's own. `panel` False (standalone: no orchestrator,
    so no session token) skips the panel altogether instead of a logged no-op on every call."""

    def __init__(self, ctx, action_id, panel=True):
        self._ctx, self._action_id, self._panel = ctx, action_id, panel
        self.bag = None  # the run's RunBag while it records: each phase goes in it too

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def update_user_panel(self, **kwargs):
        return self._ctx.update_user_panel(**kwargs) if self._panel else False

    def progress(self, phase, pct, message="", pausable=True):
        self._ctx.progress(phase, pct, message, pausable)
        if self.bag is not None:
            self.bag.event("phase", dict(phase=phase, pct=pct, message=message))
        self.update_user_panel(panel_json=ui.progress_panel(self._action_id, phase, message))


class Capability(CapabilityEngine):
    display_name = "Place container"
    # The UI offers this engine's actions when the user taps a table (perception's class name)
    perception_classes = ["table"]
    required_resources = ["arm", "gripper"]
    actions = [
        # The UI's one tap (it sends no params): the place for real, at the marked spot, back to the start
        Action("place_here", ui.PLACE_HERE),
        Action("place", "Place the container on the table", params_schema=PLACE),
        Action("mark", "Save the fingertip position as the spot", params_schema={"type": "object"}),
        Action("retreat", "Back the hand straight out", params_schema=EXECUTE),
        Action("reset", "Back to the start position", params_schema=RESET),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("place_container", **node_kwargs)
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self.cfg["gains"] = {
            "kq": [float(v) for v in self.cfg["impedance_kq"]],
            "zeta": float(self.cfg["impedance_zeta"]),
            "torque_limit": [float(v) for v in self.cfg["impedance_torque_limit"]],
        }
        self.state_dir = Path(self.cfg["state_dir"])
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def execute(self, action_id, params, ctx):
        """Runs the action with the user's UI panel showing each phase; a failure is shown there too
        (with an OK button, waited for up to ui_error_wait_s) before the action ends with it.
        Standalone (no orchestrator, no session token) every panel call is a logged no-op."""
        ctx = _PanelContext(ctx, action_id, panel=self.orchestrated)
        first = {"place": "plan", "place_here": "plan", "reset": "reset", "retreat": "retreat"}
        first = first.get(action_id, action_id)
        while True:
            ctx.update_user_panel(panel_json=ui.progress_panel(action_id, first))
            try:
                with self._recording(action_id, params) as bag:
                    ctx.bag = bag
                    result = self._act(action_id, params, ctx)
                    if bag is not None:
                        bag.result = result  # into run.json when the bag closes
                break
            except Cancelled:
                raise
            except EngineError as e:
                # The UI's tap, refused before anything moved: the reason and the same button to try
                # again (once the user has fixed it). After the arm moved ("STOPPED ..."): Done only.
                # As microwave's UI flow (feature/14-microwave-close).
                if action_id == "place_here" and e.recoverable and "STOPPED" not in str(e):
                    if self._wait_for_button(ctx, ui.retry_panel(str(e)), ui.RETRY):
                        continue
                    raise
                self._show_error(ctx, action_id, str(e))
                raise
            except Exception as e:  # a bug: still say so on the UI, then let the engine report it
                self._show_error(ctx, action_id, f"Unexpected error ({type(e).__name__}): {e}")
                raise
        if bag is not None:
            self.get_logger().info(f"run recorded: {bag.run_dir} (bag + run.json)")
            result = {**result, "bag_path": str(bag.path), "recording_errors": list(bag.errors)}
        if action_id == "place_here":  # the UI's one tap: say it's done, until the user taps Done
            self._wait_for_button(ctx, ui.placed_panel(), ui.DONE)
        else:
            ctx.update_user_panel(panel_json=ui.done_panel(action_id, "Done"))
        return result

    def _recording(self, action_id, params):
        """The run's RunBag (a context manager), or a no-op one with record_bag off. A recording
        that can't start refuses the action before anything moves."""
        if not self.cfg["record_bag"]:
            return contextlib.nullcontext(None)
        from .bag import RecordingError, RunBag  # lazy: the planning tests need no rosbag2

        camera = tuple(self.cfg["bag_camera_topics"]) if self.cfg["bag_camera"] else ()
        try:
            return RunBag(
                self,
                self.state_dir,
                action_id,
                params,
                {k: v for k, v in self.cfg.items() if k != "gains"},
                camera,
                frame_hz=float(self.cfg["bag_frame_hz"]),
                state_hz=float(self.cfg["bag_state_hz"]),
            )
        except RecordingError as e:
            why = f"{e}. Next: check the disk and state_dir, or run with record_bag:=false"
            raise EngineError("recording", why, recoverable=True) from e

    def _show_error(self, ctx, action_id, message):
        """The failure on the user's panel with an OK button, waited for (see _wait_for_button) so it
        is read before the action ends and the panel closes."""
        self._wait_for_button(ctx, ui.error_panel(action_id, message), ui.OK)

    def _wait_for_button(self, ctx, panel, button_id):
        """`panel` on the UI until its `button_id` is tapped (True), or ui_error_wait_s passes or the
        action is cancelled (False). Pushed again only once the UI's goal for it has ended (the bridge
        dropped or aborted it): re-sending an open panel every second made it flash (microwave, 11-12).
        False at once if the panel can't be shown (standalone, no bridge)."""
        if not ctx.update_user_panel(panel_json=panel):
            return False
        deadline = time.monotonic() + float(self.cfg["ui_error_wait_s"])
        refresh = time.monotonic() + 1.0
        while time.monotonic() < deadline and not ctx.cancelled():
            if ui.is_button(ctx.poll_ui_event(), button_id):
                return True
            if time.monotonic() >= refresh:
                if not _panel_open(ctx):
                    ctx.update_user_panel(panel_json=panel)
                refresh = time.monotonic() + 1.0
            time.sleep(0.1)
        return False

    def _mock(self, action_id, params, ctx):
        """No hardware: place walks through its phases on the panel (ui.MOCK_PHASE_S each), so the
        UI can be tried in a sim; `mock_fail` (a phase; the goal's, else the node parameter) ends it
        there with a refusal instead."""
        fail = params.get("mock_fail") or self.cfg["mock_fail"]
        for phase in ui.MOCK_PHASES if action_id in ("place", "place_here") else ():
            ctx.checkpoint()
            ctx.progress(phase, 0.0)
            if phase == fail:
                raise EngineError(
                    "refused",
                    f"mock: a failure at '{phase}', as asked (mock_fail). Next: run again without mock_fail",
                    recoverable=True,
                )
            time.sleep(ui.MOCK_PHASE_S)
        return {"executed": False, "mock": True}

    def _act(self, action_id, params, ctx):
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            return self._mock(action_id, params, ctx)
        log = self.get_logger().info
        try:
            model = plan.Model(self.cfg["flange_to_tip"], self.cfg["pitch_deg"])
            if action_id == "mark":
                return self._mark(model, log)
            if action_id == "place_here":
                # The UI's tap: for real (it sends no params), at the marked spot, back to the start
                if not self.cfg["allow_execute"]:
                    raise plan.Refused(
                        "placing is switched off on this robot (allow_execute is false). Next: start "
                        "place_container with allow_execute:=true, with someone at the e-stop"
                    )
                result = self._place(model, {}, True, ctx, log)
                if self.cfg["return_mode"] != "blended":  # legacy: the separate reset brings it back
                    ctx.progress("reset", 0.0)
                    self._reset(model, {"execute": True, "empty": True}, True, log)
                return result
            if action_id == "place":
                return {**self._place(model, params, execute, ctx, log), "note": note}
            if action_id == "retreat":
                return {**self._retreat(model, execute, log), "note": note}
            return {**self._reset(model, params, execute, log), "note": note}
        except plan.Refused as e:
            raise EngineError("refused", str(e), recoverable=True) from e
        except ArmError as e:  # the driver or its state, outside this capability (Cancelled is not one)
            # (the engine logs every EngineError's message as an error: /rosout, and a run's bag)
            why = plan.explain_failure(f"{type(e).__name__}: {e}", [action_id])
            raise EngineError(type(e).__name__, why, recoverable=True) from e

    def availability(self, instance_id, world_state):
        """The UI is offered only place_here (one tap, no params). place / mark / retreat / reset stay
        callable directly: from the UI, with no params, they would only dry-run (microwave's UI actions
        do the same, feature/14-microwave-close)."""
        return [a for a in super().availability(instance_id, world_state) if a.action_id == "place_here"]

    # -- actions -------------------------------------------------------------------------------

    def _mark(self, model, log):
        tip = model.fk(self._joints(), "tip")[:3, 3]
        spot = dict(x=float(tip[0]), y=float(tip[1]), table_z=float(tip[2]), time=time.time(), source="mark")
        self._save("spot.json", spot)
        log(
            f"fingertip at x={tip[0]:.3f} y={tip[1]:.3f} z={tip[2]:.3f} m, "
            f"{np.hypot(*tip[:2]):.2f} m from the base axis"
        )
        return {"spot": [spot["x"], spot["y"]], "table_z": spot["table_z"]}

    def _place(self, model, params, execute, ctx, log):
        spot, table_z = self._spot(params, log)
        q_now = self._joints()
        self._check_tool_frame(model, q_now, log)
        gripper = float(self.arm.gripper_state().position)
        if gripper < 0.05 and not params.get("allow_empty_gripper", False):
            raise plan.Refused(
                f"the gripper is at {gripper:.2f} (0 = open; under 0.05 holds nothing): "
                "it isn't holding the container. "
                "Close it on the lip, or set allow_empty_gripper for an empty test run"
            )
        plan.check_lowering(self.cfg["lower_speed"], self.cfg["vz_stop"], rammp_arm.CART_SPEED)
        r = float(np.hypot(*spot))
        lo, hi = self.cfg["reach"]
        if not lo <= r <= hi:
            log(
                f"NOTE: the spot is {r:.2f} m from the arm's base axis, outside the reach param {lo:.2f}-{hi:.2f} m "
                "(measured without the chair volumes); the plan decides"
            )
        moves, descent, info = plan.plan_place(
            model,
            q_now,
            spot,
            table_z,
            max_adjust=self.cfg["max_adjust_deg"],
            allow_reconfigure=self.cfg["allow_reconfigure"],
            reach=(lo, hi),
            lower_from=self.cfg["lower_from"],
            log=log,
        )
        if info["reconfigure"]:
            log(
                "NOTE: the transit changes the arm's posture (cuRobo, joint space): a bigger move, the container's "
                "tilt is checked only along a straight joint path -- watch it"
            )
        steps = self._describe(model, moves, descent, table_z, log)
        log(
            f"adjust turns the hand {info['adjust_deg']:.1f} deg. cuRobo avoids the chair's collision volumes; the "
            "table and what is on it are NOT in its world -- keep the way to the spot clear"
        )
        # Before anything moves (no pause at the top): the ends of the line down and the retreat
        # against the chair's volumes, on the planned line (the one from where the arm ends is the same
        # line down, a few mm apart at most).
        self._check_contact_zone(model, descent, table_z, log)
        if not execute:
            if self.cfg["blend"] and self.cfg["curobo_check"]:
                if self.cfg["carry_targets"] == "joints":
                    self._check_carry_joints(moves, log)
                else:
                    self._check_carry(info["waypoints"], log)
            log("DRY RUN -- all checks passed, nothing commanded")
            return {"executed": False, "plan": steps}

        self._ready()
        self._save("start.json", dict(q=[float(v) for v in q_now], time=time.time()))
        if self.cfg["blend"] and self.cfg["carry_targets"] == "joints":
            self._carry_joints(model, moves, table_z, ctx, log)
        elif self.cfg["blend"]:
            self._carry(info["waypoints"], table_z, ctx, log)
        else:
            self._carry_separately(model, moves, ctx, log)
        q_top = self._joints()
        tilt = model.tilt(q_top)
        if tilt > 1.0:
            raise plan.Refused(
                f"STOPPED above the spot: the container is {tilt:.1f} deg off level after the carrying motion "
                "(max 1.0); "
                "nothing let go. Next: reset, then run again with adjust_mode:=joints or blend:=false"
            )
        # The arm ended on the planned joints (a joint-target carry does): the planned line down starts
        # here, use it -- planning it again (~60 IK solves) held the arm still above the spot. Otherwise
        # (pose targets: cuRobo chose the joints) the line again, from where the arm IS, or a line planned
        # from other joints would start with a jump between the two postures.
        off_deg = float(np.max(np.abs(plan.wrap_deg(q_top - np.asarray(descent[1][0])))))
        if off_deg > REPLAN_DEG:
            log(f"the arm ended {off_deg:.1f} deg from the planned joints: the line down planned again from there")
            descent = plan.plan_descent(model, q_top, info["floor_z"], log=log)
        if not plan.reaches(descent, table_z):
            raise plan.Refused(
                f"STOPPED above the spot: {plan.reach_refusal(descent, table_z, spot)} in the posture cuRobo "
                "chose -- nothing let go"
            )

        rec, cfg = lower.Recorder(), {**self.cfg, "table_z": table_z, "bowl_depth": plan.BOWL_DEPTH}
        try:
            ctx.progress("lower", 0.0)
            hold_q = lower.lower(self.arm, model, descent, cfg, rec, log)
            ctx.progress("open", 0.0)
            lower.let_go(self.arm, rec, log)
            drop = self.cfg["release_drop"]
            if self.cfg["return_mode"] == "blended":
                # the drop, the retreat (and swing) and the way back to the start: ONE blended motion
                ctx.progress("return", 0.0)
                moves = plan.return_moves(model, hold_q, q_now, drop=drop, log=log)
                lower.go_back(self.arm, model, moves, self.cfg, log)
                self.arm.gripper(1.0)  # ready for the next run, as reset leaves it
                where = "Back at the start position (position control)."
            else:
                time.sleep(0.5)
                hold_q = lower.sink(self.arm, model, hold_q, drop, self.cfg["gains"], rec, log)
                ctx.progress("retreat", 0.0)
                # with the drop, the retreat stays down (no lift while the fingers may still be by the lip)
                release = 0.0 if drop > 0 else self.cfg["hold_press"]
                lower.back_out(self.arm, model, hold_q, cfg, rec, log, release=release)
                where = "The arm is still in impedance mode, by the container."
        finally:
            path = rec.write(self.state_dir / "descents")
            if path is not None:
                log(f"recorded {len(rec.rows)} /ee_state messages -> {path} (plot: plot_descent.py)")
        log(f"Done: container placed, hand clear. {where}")
        return {"executed": True, "recording": str(path)}

    def _retreat(self, model, execute, log):
        gripper = float(self.arm.gripper_state().position)
        if gripper > lower.OPEN_STALLED:
            raise plan.Refused(
                f"the gripper is at {gripper:.2f}, still closed past {lower.OPEN_STALLED} -- "
                "backing out would drag what "
                "it holds. Open it, then retreat"
            )
        q = self._joints()
        line = plan.plan_retreat(model, q, log=log)
        end = model.fk(line[-1])[:3, 3]
        if not execute:
            return {"executed": False, "to": np.round(end, 3).tolist()}
        self._ready()
        lower.back_out(self.arm, model, q, self.cfg, lower.Recorder(), log)
        return {"executed": True, "to": np.round(end, 3).tolist()}

    def _reset(self, model, params, execute, log):
        if params.get("save", False):
            q = self._joints()
            self._save("start.json", dict(q=[float(v) for v in q], time=time.time()))
            T = model.fk(q)
            log(f"saved the start position: hand at x={T[0, 3]:.3f} y={T[1, 3]:.3f} z={T[2, 3]:.3f}")
            return {"saved": True}
        start = self._load("start.json", "no start position saved: run reset with save, or a place first")
        # Nothing in the hand (the gripper is open, or the goal says so): the way back may turn the hand.
        empty = float(self.arm.gripper_state().position) <= lower.OPEN_STALLED or bool(params.get("empty", False))
        big_turn = bool(params.get("big_turn", False))
        if big_turn and not empty:
            raise plan.Refused("big_turn is only for an empty hand: open the gripper or set empty, then reset")
        moves = plan.reset_moves(
            model, self._joints(), np.asarray(start["q"], dtype=float), empty=empty, big_turn=big_turn
        )
        steps = [dict(name=n, tool=np.round(model.fk(q)[:3, 3], 3).tolist()) for n, q in moves]
        if not execute:
            return {"executed": False, "plan": steps}
        self._ready()
        if self.cfg["reset_mode"] == "legacy":
            lower.joint_moves(self.arm, model, moves, self.cfg, log)
        else:
            # After a place the arm holds in impedance a few cm off its setpoint: leave it on the
            # current joints first. Joint targets on the measured joints' branch (J1/J3/J5/J7 spin
            # freely: a wrapped target could send one the long way round).
            lower.exit_impedance(self.arm, log)
            qs = _unwrap_from(self.arm.raw_joints(), [q for _, q in moves])
            radii = plan.reset_radii(model, self._joints(), moves, self.cfg["blend_radius_near"])
            targets = [
                JointTarget(q, blend_radius=r, speed_scale=self.cfg["go_to_speed_scale"])
                for q, r in zip(qs, radii, strict=True)
            ]
            names = " -> ".join(n for n, _ in moves)
            log(f"-> reset: {names} (go_to_joint_configs, speed x{self.cfg['go_to_speed_scale']}, blend {radii})")
            if not self.arm.go_to_joint_configs(targets, "reset"):
                legs = [n for n, _ in moves]
                where = {n: model.fk(q)[:3, 3] for n, q in moves}
                raise plan.Refused("STOPPED in reset: " + plan.explain_failure(self.arm.last_error, legs, where))
        self.arm.gripper(1.0)  # ready for the next run, up here clear of the container
        log("Back at the start position.")
        return {"executed": True, "plan": steps}

    # -- helpers -------------------------------------------------------------------------------

    def _joints(self):
        return np.asarray(self.arm.get_state()["position"], dtype=float)

    def _ready(self):
        why = self.arm.not_ready()
        if why is not None:
            raise plan.Refused(
                f"the arm can't be commanded now: {why}. Next: clear that (e-stop, control), then run again"
            )

    def _carry(self, waypoints, table_z, ctx, log):
        """adjust -> lift -> transit as ONE go_to_ee_poses motion: cuRobo plans every leg around the
        chair's collision volumes and blends them, so the arm doesn't stop at the waypoints."""
        speed = self.cfg["go_to_speed_scale"]
        poses = [w for w in waypoints if not w.get("joints")]
        if self.cfg["adjust_mode"] == "joints":
            # the squaring turn to our own checked joints first (a joint-space plan: no swing), then
            # the rest from there
            w = poses.pop(0)
            ctx.progress("adjust", 0.0)
            (q,) = _unwrap_from(self.arm.raw_joints(), [w["q"]])
            log(f"-> adjust (go_to_joint_config, speed x{speed})")
            if not self.arm.go_to_joint_config(q, "adjust", speed_scale=speed):
                raise plan.Refused(
                    "STOPPED: " + plan.explain_failure(self.arm.last_error, ["adjust"], {"adjust": w["pos"]})
                )
        if poses:
            radii = plan.blend_radii(poses, table_z, self.cfg["blend_radius_near"], self.cfg["blend_radius_far"])
            targets = [
                PoseTarget(
                    w["pos"],
                    w["quat"],
                    hold=HOLD_LEVEL if w["hold"] == "level" else HOLD_NONE,
                    blend_radius=r,
                    speed_scale=speed,
                )
                for w, r in zip(poses, radii, strict=True)
            ]
            names = [w["name"] for w in poses]
            label = " -> ".join(names)
            ctx.progress(names[0] if len(names) == 1 else "carry", 0.0)
            log(f"-> {label} (go_to_ee_poses, one motion, speed x{speed}, blend {radii} m)")
            if not self.arm.go_to_ee_poses(targets, label):
                where = {w["name"]: w["pos"] for w in poses}
                raise plan.Refused("STOPPED: " + plan.explain_failure(self.arm.last_error, names, where))
        for w in waypoints:
            if w.get("joints"):  # transit-reconfigure: to these exact joints, the posture that reaches
                ctx.progress(w["name"], 0.0)
                (q,) = _unwrap_from(self.arm.raw_joints(), [w["q"]])
                log(f"-> {w['name']} (go_to_joint_config, speed x{speed})")
                if not self.arm.go_to_joint_config(q, w["name"], speed_scale=speed):
                    raise plan.Refused(
                        "STOPPED: " + plan.explain_failure(self.arm.last_error, [w["name"]], {w["name"]: w["pos"]})
                    )

    def _carry_joints(self, model, moves, table_z, ctx, log):
        """adjust -> lift -> transit as ONE go_to_joint_configs motion through plan_place's joints
        (every leg checked level and above the table, on the start's posture): cuRobo plans around the
        chair's volumes and blends, but can't pick another posture."""
        speed = self.cfg["go_to_speed_scale"]
        names = [n for n, _ in moves]
        qs = _unwrap_from(self.arm.raw_joints(), [q for _, q in moves])
        where = {n: model.fk(q)[:3, 3] for n, q in moves}
        radii = plan.blend_radii(
            [dict(pos=where[n]) for n in names], table_z, self.cfg["blend_radius_near"], self.cfg["blend_radius_far"]
        )
        targets = [JointTarget(q, blend_radius=r, speed_scale=speed) for q, r in zip(qs, radii, strict=True)]
        label = " -> ".join(names)
        ctx.progress("carry", 0.0)
        log(f"-> {label} (go_to_joint_configs, one motion, speed x{speed}, blend {radii} m)")
        if not self.arm.go_to_joint_configs(targets, label):
            raise plan.Refused("STOPPED: " + plan.explain_failure(self.arm.last_error, names, where))

    def _check_carry_joints(self, moves, log):
        """Dry run: cuRobo plans each joint leg of the carrying motion (nothing moves), from the
        measured joints and then each planned one."""
        names = [n for n, _ in moves]
        start = self.arm.raw_joints()
        for k, q in enumerate(_unwrap_from(start, [q for _, q in moves])):
            try:
                res = self.arm.plan_to_joints(q, start)
            except ArmError as e:
                raise plan.Refused(
                    f"cuRobo check of {names[k]}: " + plan.explain_failure(f"{type(e).__name__}: {e}", [names[k]])
                ) from e
            if not res.success:
                why = plan.explain_failure(f"leg {k + 1}/{len(names)}: {res.message}", names)
                raise plan.Refused(f"DRY RUN, cuRobo check: {why}")
            log(f"cuRobo check: {names[k]} plans in joint space ({res.planning_time:.2f} s)")
            start = q

    def _carry_separately(self, model, moves, ctx, log):
        """blend false: each move on its own, as on 2026-10-29 (adjust on move_joints, the rest
        go_to_ee_pose level)."""
        for name, q in moves:
            ctx.progress(name, 0.0)
            if name.startswith("adjust"):
                # a small turn in place: our checked joint move (cuRobo swung the hand sideways and back
                # for it, 2026-10-08)
                ok = self.arm.move_joints([q], speed_scale=self.cfg["adjust_speed_scale"])
            elif name == "transit-reconfigure":
                # to these exact joints: the posture whose line down reaches the table
                log(f"-> {name} (go_to_joint_config, speed x{self.cfg['go_to_speed_scale']})")
                (q,) = _unwrap_from(self.arm.raw_joints(), [q])
                ok = self.arm.go_to_joint_config(q, name, speed_scale=self.cfg["go_to_speed_scale"])
            else:
                T = model.fk(q)
                log(f"-> {name} (go_to_ee_pose, level, speed x{self.cfg['go_to_speed_scale']})")
                ok = self.arm.go_to_ee_pose(
                    T[:3, 3],
                    Rotation.from_matrix(T[:3, :3]).as_quat(),
                    name,
                    speed_scale=self.cfg["go_to_speed_scale"],
                    orientation_hold=HOLD_LEVEL,
                )
            if not ok:
                where = {name: model.fk(q)[:3, 3]}
                raise plan.Refused("STOPPED: " + plan.explain_failure(self.arm.last_error, [name], where))

    def _check_carry(self, waypoints, log):
        """Dry run: cuRobo plans each leg of the carrying motion on its own (nothing moves), from our
        IK joints at the leg before, so a target in the chair's volumes or out of reach is refused
        before anything moves. The real motion plans them again from where the arm is."""
        names = [w["name"] for w in waypoints]
        where = {w["name"]: w["pos"] for w in waypoints}
        starts = _unwrap_from(self.arm.raw_joints(), [w["q"] for w in waypoints])
        start = self.arm.raw_joints()
        for k, w in enumerate(waypoints):
            joints = w.get("joints") or (k == 0 and self.cfg["adjust_mode"] == "joints")
            try:
                if joints:
                    res = self.arm.plan_to_joints(starts[k], start)
                else:
                    hold = HOLD_LEVEL if w["hold"] == "level" else HOLD_NONE
                    res = self.arm.plan_to_pose(w["pos"], w["quat"], start, hold)
            except ArmError as e:
                raise plan.Refused(
                    f"cuRobo check of {w['name']}: " + plan.explain_failure(f"{type(e).__name__}: {e}", [w["name"]])
                ) from e
            if not res.success:
                why = plan.explain_failure(f"leg {k + 1}/{len(names)}: {res.message}", names, where)
                raise plan.Refused(f"DRY RUN, cuRobo check: {why}")
            log(f"cuRobo check: {w['name']} plans ({res.planning_time:.2f} s)")
            start = starts[k]

    def _check_contact_zone(self, model, descent, table_z, log):
        """cuRobo plans (nothing moves) to the ends of the line down and the retreat, which run on
        move_joints and so are never checked against the chair's collision volumes otherwise."""
        if not self.cfg["curobo_check"]:
            log("curobo_check is off: the line down and the retreat are NOT checked against the chair")
            return
        for label, q_from, q_to in plan.contact_zone_checks(model, descent, table_z):
            try:
                res = self.arm.plan_to_joints(q_to, q_from)
            except ArmError as e:
                raise plan.Refused(
                    f"cuRobo check of {label}: " + plan.explain_failure(f"{type(e).__name__}: {e}", [label])
                ) from e
            if not res.success:
                where = {label: model.fk(q_to)[:3, 3]}
                raise plan.Refused(
                    f"cuRobo check of {label} (nothing moved): " + plan.explain_failure(res.message, [label], where)
                )
            log(f"cuRobo check: the end of {label} is clear of the chair's collision volumes")

    def _check_tool_frame(self, model, q, log):
        reported = np.asarray(self.arm.get_state()["ee_pos"][:3], dtype=float)
        modelled = model.fk(q)[:3, 3]
        off = float(np.linalg.norm(reported - modelled))
        log(
            f"tool frame: arm says {np.round(reported, 3).tolist()}, "
            f"model {np.round(modelled, 3).tolist()} ({off * 100:.1f} cm apart)"
        )
        if off > self.cfg["tool_frame_tol"]:
            raise plan.Refused(
                f"the arm reports its tool frame at {np.round(reported, 3).tolist()} m, the model at "
                f"{np.round(modelled, 3).tolist()}: {off * 100:.1f} cm apart (tool_frame_tol "
                f"{self.cfg['tool_frame_tol'] * 100:.1f} cm). Next: check the driver's tool configuration (Robotiq "
                "2F-85, arm.tool_z 0.12 m) and that the driver's URDF is flatbot's"
            )

    def _spot(self, params, log):
        given = [k for k in ("x", "y", "table_z") if k in params]
        if given and len(given) < 3:
            raise plan.Refused("give all of x, y and table_z, or none (the marked spot)")
        if given:
            return np.array([params["x"], params["y"]], dtype=float), float(params["table_z"])
        spot = self._load(
            "spot.json", "no marked spot: run mark with the fingertips on the target, or give x, y, table_z"
        )
        age = (time.time() - spot["time"]) / 60
        log(f"marked spot ({age:.0f} min old): x={spot['x']:.3f} y={spot['y']:.3f}, table z {spot['table_z']:.3f}")
        if age > self.cfg["max_spot_age_min"]:
            raise plan.Refused(
                f"the marked spot is {age:.0f} min old -- the chair or table may have moved; mark it again"
            )
        return np.array([spot["x"], spot["y"]], dtype=float), float(spot["table_z"])

    def _describe(self, model, moves, descent, table_z, log):
        steps = []
        for name, q in moves:
            T = model.fk(q)
            steps.append(dict(name=name, tool=np.round(T[:3, 3], 3).tolist(), above_table=round(T[2, 3] - table_z, 3)))
            log(
                f"{name:10s} tool {np.round(T[:3, 3], 3).tolist()}  {(T[2, 3] - table_z) * 100:5.1f} cm above the table"
            )
        zs = descent[0]
        log(
            f"lower      one move (~{self.cfg['lower_speed'] * 100:.0f} cm/s at the table) from z={zs[0]:.3f} "
            f"down to at most {zs[-1]:.3f} "
            f"({(zs[0] - zs[-1]) * 100:.0f} cm), stopping when it feels the table"
        )
        steps.append(dict(name="lower", from_z=round(float(zs[0]), 3), to_z=round(float(zs[-1]), 3)))
        return steps

    def _save(self, name, record):
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            (self.state_dir / name).write_text(json.dumps(record, indent=2))
        except OSError as e:
            raise plan.Refused(f"can't save {name} in {self.state_dir} ({e}) -- set state_dir") from e

    def _load(self, name, missing):
        path = self.state_dir / name
        if not path.exists():
            raise plan.Refused(missing)
        record = json.loads(path.read_text())
        saved = float(record.get("time", 0.0))
        if saved < float(self.cfg["state_valid_after"]):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(saved))
            raise plan.Refused(
                f"{path} was saved {when}, before the arm moved on the chair "
                f"(state_valid_after={self.cfg['state_valid_after']:.0f}) -- "
                + (
                    "run mark again"
                    if name == "spot.json"
                    else "put the arm at a start position and run reset with save"
                )
            )
        return record


def main():
    run(Capability)


if __name__ == "__main__":
    main()
