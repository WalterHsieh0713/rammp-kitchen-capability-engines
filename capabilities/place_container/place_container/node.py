"""The place_container capability: put the held food container down at a spot on the table.

Ported from RAMMP-table (place_bowl.py + kinova_arm.py), where it placed the container on a marked
sticky note on the flatbot (2026-10-26). Everything is in arm_mount (x forward, y left, z true up).

Actions:
  mark     the closed fingertips touching the target -> saved as the spot (its z = the table). No motion.
  place    the container, gripped by its lip, to the spot ({x, y, table_z}, or the marked one):
             1. adjust: square the hand in place to the nearest hold (level, pointing `pitch_deg` down);
             2. lift to SAFE_HEIGHT above the table, if lower; 3. transit to above the spot
             (go_to_ee_pose, HOLD_LEVEL); 4. the straight line down in the driver's IMPEDANCE mode
             until contact, then hold `hold_press` below it (lower.py); 5. open; 6. straight back.
           The arm is left in impedance, held where the retreat ended.
  retreat  only step 6 (after a place stopped with the gripper open).
  reset    back to the start position (where the last place started, or `save`d): straight up,
           then a joint move; gripper closed.

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets `execute`.
Otherwise every goal is a dry run that plans and checks and returns the plan. Someone stands at the
e-stop whenever allow_execute is on.

    ros2 launch place_container place_container.launch.py use_orchestrator:=false allow_execute:=true
    make action CAP=place_container ACTION=mark
    make action CAP=place_container ACTION=place PARAMS='{"execute": false}'
"""

import json
import time
from pathlib import Path

import numpy as np
import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run
from capability_engine.arm import HOLD_LEVEL
from scipy.spatial.transform import Rotation

from . import lower, plan

# ROS parameter -> default: the flatbot's values (2026-10-26).
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
    # held in impedance at the start this long (s) before going down; must end still
    "settle_s": 3.0,
    # contact more than this (m) above where the bottom meets the table = not the table
    "early_stop": 0.03,
    # joint impedance for the settle, lowering, hold, retreat and reset, one set so the stiffness never
    # changes mid-place (move_joints takes custom gains, not the driver's named profiles). Default: the
    # driver's documented starting point (rammp_arm.IMPEDANCE), NOT the "medium" profile RAMMP-table
    # used -- compare the settle's sag in the recording (medium: 1.0 cm on the flatbot, 2026-10-26)
    "impedance_kq": list(rammp_arm.IMPEDANCE["kq"]),
    "impedance_zeta": rammp_arm.IMPEDANCE["zeta"],
    "impedance_torque_limit": list(rammp_arm.IMPEDANCE["torque_limit"]),
    # speeds. The lowering stays at the 2 cm/s its contact test was tuned at; the rest is 2x RAMMP-table's
    # (2026-10-29): the hand's speed where the table should be (m/s)
    "lower_speed": 0.02,
    # speed_scale of the carrying moves (go_to_ee_pose; was 0.2)
    "go_to_speed_scale": 0.4,
    # speed_scale of the squaring turn (move_joints; was 0.5)
    "adjust_speed_scale": 1.0,
    # the retreat's peak speed (m/s; was 0.03)
    "retreat_speed": 0.06,
    # speed_scale of reset's joint moves (move_joints; was ~0.5)
    "reset_speed_scale": 1.0,
    # spot distance from the arm base axis (m) where the whole place plans with the full descent
    # (flatbot, table z 0.198: test_place_bowl.py --sweep); outside it the plan still decides
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
    },
}
RESET = {
    "type": "object",
    "properties": {
        **EXECUTE["properties"],
        "save": {"type": "boolean", "default": False},
        # nothing in the (closed) gripper: skip the container-tilt check on the way back
        "empty": {"type": "boolean", "default": False},
    },
}


class Capability(CapabilityEngine):
    display_name = "Place container"
    perception_classes = []
    required_resources = ["arm", "gripper"]
    actions = [
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
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing planned or moved")
            return {"executed": False, "mock": True}
        log = self.get_logger().info
        try:
            model = plan.Model(self.cfg["flange_to_tip"], self.cfg["pitch_deg"])
            if action_id == "mark":
                return self._mark(model, log)
            if action_id == "place":
                return {**self._place(model, params, execute, ctx, log), "note": note}
            if action_id == "retreat":
                return {**self._retreat(model, execute, log), "note": note}
            return {**self._reset(model, params, execute, log), "note": note}
        except plan.Refused as e:
            raise EngineError("refused", str(e), recoverable=True) from e

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
            raise plan.Refused("the gripper is fully open -- it isn't holding the container")
        r = float(np.hypot(*spot))
        lo, hi = self.cfg["reach"]
        if not lo <= r <= hi:
            log(f"the spot is {r:.2f} m from the base axis, outside the reach {lo:.2f}-{hi:.2f}; the plan decides")
        moves, descent, info = plan.plan_place(
            model,
            q_now,
            spot,
            table_z,
            max_adjust=self.cfg["max_adjust_deg"],
            allow_reconfigure=self.cfg["allow_reconfigure"],
            log=log,
        )
        if info["reconfigure"]:
            log(
                "NOTE: the transit changes the arm's posture (cuRobo, joint space): a bigger move, the container's "
                "tilt is checked only along a straight joint path -- watch it"
            )
        steps = self._describe(model, moves, descent, table_z, log)
        log(f"adjust turns the hand {info['adjust_deg']:.1f} deg; no obstacle check -- keep the spot clear")
        if not execute:
            log("DRY RUN -- all checks passed, nothing commanded")
            return {"executed": False, "plan": steps}

        self._ready()
        self._save("start.json", dict(q=[float(v) for v in q_now], time=time.time()))
        for name, q in moves:
            ctx.progress(name, 0.0)
            if name.startswith("adjust"):
                # a small turn in place: our checked joint move (cuRobo swung the hand sideways and back
                # for it, 2026-10-08)
                ok = self.arm.move_joints([q], speed_scale=self.cfg["adjust_speed_scale"])
            elif name == "transit-reconfigure":
                # to these exact joints: the posture whose line down reaches the table
                log(f"-> {name} (go_to_joint_config, speed x{self.cfg['go_to_speed_scale']})")
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
                raise plan.Refused(f"STOPPED: {name} did not complete ({self.arm.last_error})")
        q_top = self._joints()
        tilt = model.tilt(q_top)
        if tilt > 1.0:
            raise plan.Refused(f"STOPPED: the container is {tilt:.1f} deg off its hold after the moves")
        # The line down again, from where the arm IS: go_to_ee_pose lets cuRobo choose the joints, and a
        # line planned from other joints would start with a jump between the two postures.
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
            time.sleep(0.5)
            ctx.progress("retreat", 0.0)
            lower.back_out(self.arm, model, hold_q, cfg, rec, log, release=self.cfg["hold_press"])
        finally:
            path = rec.write(self.state_dir / "descents")
            if path is not None:
                log(f"recorded {len(rec.rows)} /ee_state messages -> {path} (plot: plot_descent.py)")
        log("Done: container placed, hand clear. The arm is still in impedance mode.")
        return {"executed": True, "recording": str(path)}

    def _retreat(self, model, execute, log):
        gripper = float(self.arm.gripper_state().position)
        if gripper > lower.OPEN_STALLED:
            raise plan.Refused("the gripper is still closed -- backing out would drag what it holds; open it first")
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
        moves = plan.reset_moves(model, self._joints(), np.asarray(start["q"], dtype=float), empty=empty)
        steps = [dict(name=n, tool=np.round(model.fk(q)[:3, 3], 3).tolist()) for n, q in moves]
        if not execute:
            return {"executed": False, "plan": steps}
        self._ready()
        lower.joint_moves(self.arm, model, moves, self.cfg, log)
        self.arm.gripper(1.0)  # ready for the next run, up here clear of the container
        log("Back at the start position.")
        return {"executed": True, "plan": steps}

    # -- helpers -------------------------------------------------------------------------------

    def _joints(self):
        return np.asarray(self.arm.get_state()["position"], dtype=float)

    def _ready(self):
        why = self.arm.not_ready()
        if why is not None:
            raise plan.Refused(why)

    def _check_tool_frame(self, model, q, log):
        reported = np.asarray(self.arm.get_state()["ee_pos"][:3], dtype=float)
        modelled = model.fk(q)[:3, 3]
        off = float(np.linalg.norm(reported - modelled))
        log(
            f"tool frame: arm says {np.round(reported, 3).tolist()}, "
            f"model {np.round(modelled, 3).tolist()} ({off * 100:.1f} cm apart)"
        )
        if off > self.cfg["tool_frame_tol"]:
            raise plan.Refused("the arm's tool frame doesn't match the model's -- check the tool configuration")

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
        return json.loads(path.read_text())


def main():
    run(Capability)


if __name__ == "__main__":
    main()
