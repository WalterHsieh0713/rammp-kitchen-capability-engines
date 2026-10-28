"""The microwave capability: open the microwave door with the arm.

Action `open` (open_door.run): detect the handle from the wrist camera, grasp it, swing the door
open, let go and push it further open (the user's 10-07 push-open demo, replayed relative to the
door), then park. Every leg is planned and sim-gated before it moves; any failed check stops the
run with the arm holding where it is (EngineError "refused").

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets
`execute`. Otherwise the goal is a dry run (detect, plan and gate every leg; nothing moves) and the
result says so. Someone stands at the e-stop whenever allow_execute is on.

    ros2 launch microwave microwave.launch.py use_orchestrator:=false
    make action CAP=microwave ACTION=open PARAMS='{"execute": false}'
"""

import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run

from . import microwave_common as mc
from . import open_door
from .perception import YOLO_MODEL, WristCamera

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
                    "execute": {"type": "boolean", "default": False},
                    "swing_deg": {"type": "integer", "minimum": 45, "maximum": 75},
                    "push_open": {"type": "boolean", "default": True},
                    "speed": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 1.0},
                },
            },
        )
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
            ctx.progress("open", 100.0, "mock: nothing detected or moved")
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
        ctx.progress("open", 0.0, note or ("executing" if execute else "dry run"))
        try:
            ok = open_door.run(
                self.arm,
                execute,
                swing_deg=params.get("swing_deg"),
                push=bool(params.get("push_open", True)),
                speed=speed,
                log_args={**params, "execute": execute},
            )
        except mc.Refused as e:
            raise EngineError("refused", f"{e} -- the arm holds where it is", recoverable=True) from e
        finally:
            mc.set_output(None)
            mc.set_camera(None)
            camera.close()
        if not ok:
            raise EngineError("stopped", "the open stopped short -- the arm holds where it is", recoverable=True)
        if execute:
            ctx.update_object(self.instance_id, opened=True)
        return {"executed": execute, "note": note, "runs_dir": str(mc.RUNS_DIR)}


def main():
    run(Capability)


if __name__ == "__main__":
    main()
