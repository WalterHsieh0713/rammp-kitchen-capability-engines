"""The microwave capability: open and close the microwave door with the arm.

Action `open` (open_door.run): detect the handle from the wrist camera, grasp it, swing the door
open, let go and push it further open (the user's 10-07 push-open demo, replayed relative to the
door). Action `close` (close_door.run): push the door shut with the side of the hand -- no grasp,
the gripper untouched (it may hold a container) -- against the door the open saved. Every leg is
planned and sim-gated before it moves; any failed check stops the run with the arm holding where
it is (EngineError "refused").

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets
`execute`. Otherwise the goal is a dry run (detect, plan and gate every leg; nothing moves) and the
result says so. Someone stands at the e-stop whenever allow_execute is on.

    ros2 launch microwave microwave.launch.py use_orchestrator:=false
    make action CAP=microwave ACTION=open PARAMS='{"execute": false}'
    make action CAP=microwave ACTION=close PARAMS='{"execute": false, "route_over": true}'
"""

import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run

from . import close_door, open_door
from . import microwave_common as mc
from .perception import YOLO_MODEL, WristCamera

COMMON = {
    "execute": {"type": "boolean", "default": False},
    "speed": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 1.0},
}


def _schema(default):
    """JSON Schema type of a close setting from its default (None: a number or unset)."""
    if isinstance(default, bool):
        return {"type": "boolean", "default": default}
    if default is None:
        return {"type": ["number", "null"], "default": None}
    return {"type": "number", "default": default}


# ROS parameter -> default. Topics and paths are per robot (the fragment's params:).
PARAMS = {
    "allow_execute": False,
    "data_dir": "/microwave_data",
    "color_topic": "/wrist_camera/color/image_raw",
    "camera_info_topic": "/wrist_camera/color/camera_info",
    "depth_topic": "/wrist_camera/aligned_depth_to_color/image_raw",
    "camera_frame": "",
    "yolo_model": YOLO_MODEL,
}


class Capability(CapabilityEngine):
    display_name = "Microwave"
    perception_classes = ["microwave"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action(
            "open",
            "Open the door",
            params_schema={
                "type": "object",
                "properties": {
                    **COMMON,
                    "swing_deg": {"type": "integer", "minimum": 45, "maximum": 75},
                    "push_open": {"type": "boolean", "default": True},
                },
            },
        ),
        Action(
            "close",
            "Close the door",
            params_schema={
                "type": "object",
                "properties": {**COMMON, **{k: _schema(v) for k, v in close_door.CLOSE_DEFAULTS.items()}},
            },
        ),
    ]

    def __init__(self):
        super().__init__("microwave")
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        mc.configure_paths(self.cfg["data_dir"])
        open_door.HANDLE_DETECTOR_MODEL = self.cfg["yolo_model"]
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def execute(self, action_id, params, ctx):
        speed = float(params.get("speed", 1.0))
        if not 0.0 < speed <= 1.0:
            raise EngineError("bad_params", "'speed' must be in (0, 1]", recoverable=True)
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing detected or moved")
            return {"executed": False, "mock": True}

        ctx.checkpoint()
        # the camera's subscriptions live only while the action runs
        camera = WristCamera(
            self,
            self.cfg["color_topic"],
            self.cfg["camera_info_topic"],
            self.cfg["depth_topic"],
            rammp_arm.FRAME,
            self.cfg["camera_frame"],
        )
        mc.set_camera(camera)
        mc.set_output(self.get_logger().info)
        ctx.progress(action_id, 0.0, note or ("executing" if execute else "dry run"))
        log_args = {**params, "execute": execute}
        try:
            if action_id == "open":
                ok = open_door.run(
                    self.arm,
                    execute,
                    swing_deg=params.get("swing_deg"),
                    push=bool(params.get("push_open", True)),
                    speed=speed,
                    log_args=log_args,
                )
            else:
                settings = {k: v for k, v in params.items() if k in close_door.CLOSE_DEFAULTS}
                ok = close_door.run(self.arm, execute, speed=speed, log_args=log_args, **settings)
        except mc.Refused as e:
            raise EngineError("refused", f"{e} -- the arm holds where it is", recoverable=True) from e
        finally:
            mc.set_output(None)
            mc.set_camera(None)
            camera.close()
        if not ok:
            raise EngineError(
                "stopped", f"the {action_id} stopped short -- the arm holds where it is", recoverable=True
            )
        if execute:
            ctx.update_object(self.instance_id, opened=action_id == "open")
        return {"executed": execute, "note": note, "runs_dir": str(mc.RUNS_DIR)}


def main():
    run(Capability)


if __name__ == "__main__":
    main()
