"""The grab_container capability: grab the food container (tupperware) by its brim.

Action `grab`:
  0. home: a cuRobo move to `home_joints`, where every run starts.
  1. locate (scene.locate_container): the scene segmenter's food-container mask and the scene
     camera's depth give a coarse position.
  2. view (scene.view_pose): the wrist camera moves above the container, on the approach side.
  3. grasp (grasp.grab): the students' YOLO brim detector and a circle fit find the grasp point,
     then open, pre-grasp, grasp, close and raise a little, holding it.
The detector loads in the background from the start of the action, while the arm moves. Every move
is a cuRobo move (self.arm.go_to_*); the goal's `impedance` runs them all on the driver's joint
impedance controller (`impedance_gains`) instead of stiff position control. Action `park`: the scan
pose by a direct joint move, for when cuRobo refuses the start state. A failed check stops the run
with the arm holding where it is (EngineError "refused").

SAFETY: the arm moves only if the node runs with `allow_execute:=true` AND the goal sets
`execute`. Otherwise the goal is a dry run (grab: locate and plan the view move; park: report the
move). Someone stands at the e-stop whenever allow_execute is on.

    ros2 launch grab_container grab_container.launch.py use_orchestrator:=false
    make action CAP=grab_container ACTION=grab PARAMS='{"execute": false}'
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rammp_arm
from capability_engine import Action, CapabilityEngine, EngineError, run
from capability_engine.arm import IMPEDANCE

from . import grasp, scene

# ROS parameter -> default. Topics, frames and paths are per robot (the fragment's params:).
PARAMS = {
    "allow_execute": False,
    "model_path": "/models/best.pt",
    "color_topic": "/wrist_camera/color/image_raw",
    "camera_info_topic": "/wrist_camera/color/camera_info",
    "depth_topic": "/wrist_camera/depth/image_rect_raw",
    "camera_frame": "",
    "detections_topic": "/perception/object_detections",
    "scene_color_topic": "/scene_camera/color/image_raw",
    "scene_camera_info_topic": "/scene_camera/color/camera_info",
    "scene_depth_topic": "/scene_camera/depth/image_rect_raw",
    # the tool point along the approach from the brim's outer edge (m; negative: short of it)
    "grasp_depth_m": grasp.GRASP_DEPTH_M,
    # new frames whose brim edges must agree to AGREE_M (grasp.N_FRAMES)
    "detect_frames": grasp.N_FRAMES,
    # where grab starts (rad, arm_joint_1..7, as /joint_states reports them): the flatbot's, read
    # 2026-10-28 with the arm at home
    "home_joints": [0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725],
    # the driver's impedance profile for `impedance` goals: default, soft, medium or stiff
    "impedance_gains": "default",
}
EXECUTE = {"type": "object", "properties": {"execute": {"type": "boolean", "default": False}}}
GRAB = {"type": "object", "properties": {**EXECUTE["properties"], "impedance": {"type": "boolean", "default": False}}}


class Capability(CapabilityEngine):
    display_name = "Grab container"
    perception_classes = ["container"]
    required_resources = ["arm", "gripper"]
    actions = [
        Action("grab", "Grab the container", params_schema=GRAB),
        Action("park", "Park at the scan pose", params_schema=EXECUTE),
    ]

    def __init__(self, **node_kwargs):
        super().__init__("grab_container", **node_kwargs)
        self.cfg = {name: self.declare_parameter(name, default).value for name, default in PARAMS.items()}
        self.model = None  # the detector's loading (a Future), started by the first grab
        grasp.GRASP_DEPTH_M = self.cfg["grasp_depth_m"]
        grasp.N_FRAMES = self.cfg["detect_frames"]
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
        if action_id == "grab" and self.model is None:  # load it while the arm moves home and to the view
            self.model = ThreadPoolExecutor(1).submit(self._load_model)
        try:
            if action_id == "park":
                return {**grasp.park(self.arm, execute, log), "note": note}
            # the cameras' and detections' subscriptions live only while the action runs
            cfg = self.cfg
            motion = {"control_mode": IMPEDANCE, "gains": cfg["impedance_gains"]} if params.get("impedance") else {}
            wrist = self._camera(cfg["color_topic"], cfg["camera_info_topic"], cfg["depth_topic"], cfg["camera_frame"])
            scene_cam = self._camera(cfg["scene_color_topic"], cfg["scene_camera_info_topic"], cfg["scene_depth_topic"])
            detections = scene.SceneDetections(self, cfg["detections_topic"])
            try:
                return {**self._grab(wrist, scene_cam, detections, execute, log, ctx, motion), "note": note}
            finally:
                detections.close()
                scene_cam.close()
                wrist.close()
        except grasp.Abort as e:
            raise EngineError("refused", f"{e} -- the arm holds where it is", recoverable=True) from e

    def _grab(self, wrist, scene_cam, detections, execute, log, ctx, motion):
        why = self.arm.not_ready()
        if why is not None:
            raise grasp.Abort(why)
        home = self.cfg["home_joints"]
        if execute:
            ctx.progress("home", 0.0)
            if not self.arm.go_to_joint_config(home, "home", **motion):
                raise grasp.Abort(f"home: {self.arm.last_error} -- HOLDING HERE")
        else:
            plan = self.arm.plan_to_joints(home, self.arm.raw_joints())
            if not plan.success:
                raise grasp.Abort(f"home: cuRobo found no path ({plan.message}) -- arm not commanded")
            log(f"home             cuRobo {len(plan.t):3d} pts  {plan.t[-1]:.1f} s")
        ctx.progress("locate", 2.0)
        c, obj = scene.locate_container(detections, scene_cam)
        pos, quat = scene.view_pose(c, grasp.grasp_orientation(), scene.tool_T_cam(self.arm, wrist))
        log(f"{obj.instance_id} ({obj.score:.2f}) at {np.round(c, 3)}; view from {np.round(pos, 3)}")
        located = {"container": obj.instance_id, "container_point": c.tolist(), "view_pose": pos.tolist()}
        if not execute:
            plan = self.arm.plan_to_pose(pos, quat, home)  # from home, where the run will start
            if not plan.success:
                raise grasp.Abort(f"view: cuRobo found no path ({plan.message}) -- arm not commanded")
            log(f"view             cuRobo {len(plan.t):3d} pts  {plan.t[-1]:.1f} s")
            log("DRY: located and planned the view only (the grasp is detected from the view)")
            return {"executed": False, **located}
        ctx.progress("view", 5.0)
        if not self.arm.go_to_ee_pose(pos, quat, "view", **motion):
            raise grasp.Abort(f"view: {self.arm.last_error} -- HOLDING HERE")
        log(f"wrist image turned {90 * wrist.set_upright()} deg for the detector (world up = image up)")
        return {**located, **grasp.grab(self.arm, wrist, self._model(), True, log, ctx.progress, **motion)}

    def _camera(self, color_topic, info_topic, depth_topic, frame=""):
        from .camera import Camera

        return Camera(self, rammp_arm.FRAME, color_topic, info_topic, depth_topic, frame)

    def _model(self):
        try:
            return self.model.result()
        except Exception:
            self.model = None  # the next grab tries again
            raise

    def _load_model(self):
        """The detector, run once on each image shape it will see (turned or not): the first inference
        initialises CUDA and takes seconds."""
        from ultralytics import YOLO

        model = YOLO(self.cfg["model_path"])
        for shape in ((400, 640, 3), (640, 400, 3)):
            model.predict(np.zeros(shape, np.uint8), conf=grasp.CONF, retina_masks=True, verbose=False)
        return model


def main():
    run(Capability)


if __name__ == "__main__":
    main()
