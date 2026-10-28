"""The grab_container capability: grab the food container (tupperware) by its sleeve ring.

Action `grab` (grasp.grab): detect the ring from the wrist camera (YOLO segmentation, N frames
must agree), then open, pre-grasp, grasp, close and raise a little, holding it. Every leg is a
cuRobo move (self.arm.go_to_ee_pose). Action `park`: the scan pose by a direct joint move, for
when cuRobo refuses the start state. Any failed check stops the run with the arm holding where it
is (EngineError "refused").

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets
`execute`. Otherwise the goal is a dry run (grab: detect and plan every leg; park: report the
move) and nothing moves. Someone stands at the e-stop whenever allow_execute is on.

    ros2 launch grab_container grab_container.launch.py use_orchestrator:=false
    make action CAP=grab_container ACTION=grab PARAMS='{"execute": false}'
"""

import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run

from . import grasp

# ROS parameter -> default. Topics, frames and paths are per robot (the fragment's params:).
PARAMS = {
    "allow_execute": False,
    "model_path": "/models/best.pt",
    "camera_ns": "/wrist_camera",
    "camera_frame": "camera_color_optical_frame",
}
EXECUTE = {"type": "object", "properties": {"execute": {"type": "boolean", "default": False}}}


class Capability(CapabilityEngine):
    display_name = "Grab container"
    perception_classes = ["container"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action("grab", "Grab the container", params_schema=EXECUTE),
        Action("park", "Park at the scan pose", params_schema=EXECUTE),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("grab_container", **node_kwargs)
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self.model = None  # the YOLO model, loaded on the first grab
        self.get_logger().info(
            f"allow_execute={self.cfg['allow_execute']}: "
            + ("the arm WILL move on execute goals" if self.cfg["allow_execute"] else "every goal is a dry run")
        )

    def execute(self, action_id, params, ctx):
        execute = bool(params.get("execute", False)) and self.cfg["allow_execute"]
        note = "allow_execute is false: ran as a DRY RUN" if params.get("execute") and not execute else ""
        if self.mock:
            ctx.progress(action_id, 100.0, "mock: nothing detected or moved")
            return {"executed": False, "mock": True}

        log = self.get_logger().info
        try:
            if action_id == "park":
                return {**grasp.park(self.arm, execute, log), "note": note}
            camera = self._camera()
            try:
                return {**grasp.grab(self.arm, camera, self._model(), execute, log, ctx.progress), "note": note}
            finally:
                camera.close()
        except grasp.Abort as e:
            raise EngineError("refused", f"{e} -- the arm holds where it is", recoverable=True) from e

    def _camera(self):
        """The wrist camera: its subscriptions live only while the action runs."""
        from .camera import Camera

        return Camera(self, rammp_arm.FRAME, self.cfg["camera_frame"], self.cfg["camera_ns"])

    def _model(self):
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.cfg["model_path"])
        return self.model


def main():
    run(Capability)


if __name__ == "__main__":
    main()
