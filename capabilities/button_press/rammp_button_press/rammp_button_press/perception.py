"""Camera input: one ROS node that keeps the latest camera frames and looks up tf. Never moves the arm.

Subscribes to the wrist RealSense under ``camera_ns`` (colour, depth aligned to colour, which
needs ``align_depth.enable: true``, and camera_info) and, with ``scene_ns``, the fixed Orbbec
scene camera (colour and depth NOT aligned: each has its own camera_info and optical frame).
Spins itself on a daemon thread; ``press_button.Run`` reads frames and transforms from it.

tf: the arm's state publisher plus the deployment's camera mounts (rammp-deployments' static_tf,
transforms/cameras.yaml on flatbot).

With the scene camera it also keeps the YOLO node's /yolo/detections (vision_msgs Detection2DArray,
boxes in the scene colour image): yolo_box() picks the microwave's, which narrows scene_finder's search.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data  # best effort: receives from reliable and best-effort publishers
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

from rammp_button_press import Abort

FRESH_S = 1.0  # a frame older than this is stale
SCENE_NS = "/scene_camera"
SCENE_COLOR_FRAME = "scene_camera_color_optical_frame"
YOLO_TOPIC = "/yolo/detections"
YOLO_CLASS = "microwave"
YOLO_MIN_SCORE = 0.5  # live 2026-10-15: microwave 0.89-0.90; the next class (refrigerator) 0.57
YOLO_FRESH_S = 3.0  # a box received longer ago than this is not used
SCENE_DEPTH_FRAME = "scene_camera_depth_optical_frame"
# Depth topics, relative to the camera's namespace. The RealSense's aligned depth and the Gemini's
# depth are the defaults; the OAKs (flatbot) publish depth/image_rect_raw for both.
DEPTH_TOPIC = "aligned_depth_to_color/image_raw"
SCENE_DEPTH_TOPIC = "depth/image_raw"


def frame_of(msg, default: str) -> str:
    """The optical frame a camera message names in its header, or `default` if it names none."""
    frame = getattr(getattr(msg, "header", None), "frame_id", "") or ""
    return frame.lstrip("/") or default


def to_bgr_depth(bridge: CvBridge, color_msg: Image, depth_msg: Image):
    """(bgr8 image, float32 depth in metres) from a colour + depth message pair (mm depth converted)."""
    bgr = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, "passthrough").astype(np.float32)
    if depth_msg.encoding in ("16UC1", "mono16"):  # the RealSense driver publishes millimetres
        depth /= 1000.0
    return bgr, depth


def scene_frame_from_msgs(
    bridge: CvBridge,
    color_msg: Image,
    depth_msg: Image,
    color_info: CameraInfo,
    depth_info: CameraInfo,
    T_base_color,
    T_base_depth,
) -> dict:
    """Bundle the scene camera's messages and its two base_link <- optical-frame transforms (4x4)
    into the keyword arguments scene_finder.find_microwave takes."""
    bgr = bridge.imgmsg_to_cv2(color_msg, "bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, "passthrough").astype(np.float32)
    if depth_msg.encoding in ("16UC1", "mono16"):
        depth /= 1000.0
    k_c, k_d = color_info.k, depth_info.k
    return {
        "bgr": bgr,
        "depth_m": depth,
        "color_K": (k_c[0], k_c[4], k_c[2], k_c[5]),
        "depth_K": (k_d[0], k_d[4], k_d[2], k_d[5]),
        "T_base_color": np.asarray(T_base_color, dtype=float),
        "T_base_depth": np.asarray(T_base_depth, dtype=float),
    }


def yolo_box(msg, frame=SCENE_COLOR_FRAME, cls=YOLO_CLASS, min_score=YOLO_MIN_SCORE):
    """(u0, v0, u1, v1, score) of the best `cls` detection in a Detection2DArray, in `frame`'s colour
    pixels, or None (no message, another frame, nothing of that class at min_score)."""
    if msg is None or msg.header.frame_id.lstrip("/") != frame:
        return None
    best = None
    for det in msg.detections:
        score = max((r.hypothesis.score for r in det.results if r.hypothesis.class_id == cls), default=-1.0)
        if score >= min_score and (best is None or score > best[4]):
            c, sx, sy = det.bbox.center.position, det.bbox.size_x, det.bbox.size_y
            best = (c.x - sx / 2, c.y - sy / 2, c.x + sx / 2, c.y + sy / 2, float(score))
    return best


def why_no_fit(bgr, depth_m, fx) -> str:
    """Why dome_pattern.detect() returned None on this frame: the layout was found but outside
    the detector's range (so the camera should move), or no 5-dome layout at all."""
    from rammp_button_press import dome_pattern as dp  # noqa: PLC0415

    fit = dp.fit_layout(dp.candidates(bgr, depth_m, fx), fx)
    if fit is None:
        return "no 5-dome layout"
    if fit.z < dp.MIN_RANGE_M:
        return f"TOO CLOSE: domes at {fit.z*100:.1f} cm (min {dp.MIN_RANGE_M*100:.0f})"
    if fit.z > dp.MAX_RANGE_M:
        return f"TOO FAR: domes at {fit.z*100:.0f} cm (max {dp.MAX_RANGE_M*100:.0f})"
    return "no 5-dome layout"


class Perception(Node):
    """Latest camera messages (get / wait) and tf lookups into the arm base frame."""

    def __init__(
        self,
        arm_frame: str,
        cam_frame: str,
        camera_ns: str = "/wrist_camera",
        scene_ns: str | None = None,
        depth_topic: str = DEPTH_TOPIC,
        scene_depth_topic: str = SCENE_DEPTH_TOPIC,
    ):
        """cam_frame "" = the wrist colour image's own header.frame_id. depth_topic and
        scene_depth_topic are relative to their camera's namespace; the wrist depth must be aligned
        to colour (pixel for pixel, same optical frame)."""
        super().__init__("rammp_button_press")
        self.arm_frame, self.cam_frame = arm_frame, cam_frame
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest: dict[str, tuple[float, object]] = {}
        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf, self)

        def keep(name):
            def cb(msg):
                with self.lock:
                    self.latest[name] = (time.monotonic(), msg)

            return cb

        ns = camera_ns.rstrip("/")
        self.create_subscription(CameraInfo, f"{ns}/color/camera_info", keep("info"), qos_profile_sensor_data)
        self.create_subscription(Image, f"{ns}/color/image_raw", keep("color"), qos_profile_sensor_data)
        self.create_subscription(Image, f"{ns}/{depth_topic.strip('/')}", keep("depth"), qos_profile_sensor_data)
        self._scene_subs = []
        if scene_ns:
            sns = scene_ns.rstrip("/")
            self._scene_subs = [
                self.create_subscription(Image, f"{sns}/color/image_raw", keep("scene_color"), qos_profile_sensor_data),
                self.create_subscription(
                    Image, f"{sns}/{scene_depth_topic.strip('/')}", keep("scene_depth"), qos_profile_sensor_data
                ),
                self.create_subscription(
                    CameraInfo, f"{sns}/color/camera_info", keep("scene_cinfo"), qos_profile_sensor_data
                ),
                self.create_subscription(
                    CameraInfo, f"{sns}/depth/camera_info", keep("scene_dinfo"), qos_profile_sensor_data
                ),
                self.create_subscription(Detection2DArray, YOLO_TOPIC, keep("yolo"), qos_profile_sensor_data),
            ]

        self._thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        self._thread.start()

    # -- accessors ---------------------------------------------------------------------------
    def get(self, name, max_age=None):
        """Latest message on `name` ("color", "depth", "info", "scene_*"), or None if there is none
        yet or it is older than max_age seconds."""
        with self.lock:
            item = self.latest.get(name)
        if item is None:
            return None
        t, v = item
        if max_age is not None and time.monotonic() - t > max_age:
            return None
        return v

    def wait(self, name, timeout, max_age=FRESH_S):
        """Like get(), but waits up to `timeout` seconds for one. None on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            v = self.get(name, max_age)
            if v is not None:
                return v
            time.sleep(0.05)
        return None

    def intrinsics(self):
        """(fx, fy, cx, cy) of the wrist colour camera. Raises Abort without camera_info."""
        info = self.get("info")
        if info is None:
            raise Abort("no camera_info")
        return info.k[0], info.k[4], info.k[2], info.k[5]

    def camera_frame(self) -> str:
        """The wrist camera's optical frame: cam_frame, or (cam_frame "") the colour image's own."""
        if self.cam_frame:
            return self.cam_frame
        color = self.get("color")
        if color is None or not color.header.frame_id:
            raise Abort("no wrist colour image yet to take the camera frame from -- is the wrist camera up?")
        return color.header.frame_id.lstrip("/")

    def _cam_in_base(self):
        frame = self.camera_frame()
        try:
            return self.tfbuf.lookup_transform(self.arm_frame, frame, Time(), timeout=Duration(seconds=2.0)).transform
        except Exception as e:  # noqa: BLE001
            raise Abort(
                f"tf {self.arm_frame} <- {frame} unavailable: {e}\n"
                "  sheppy arm_state_publisher / static_tf nodes up? /joint_states flowing? verify with\n"
                f"  ros2 run tf2_ros tf2_echo {self.arm_frame} {frame}"
            ) from e

    def stop_scene(self):
        """Drop the scene camera's subscriptions once the first look is done. Left open, its 1280x720
        colour + depth at 30 Hz, unpacked on this node's one spin thread, starved the wrist camera:
        the staging re-detect got 4 frames in 8 s instead of 7 in under 1 s (2026-10-14)."""
        for sub in self._scene_subs:
            self.destroy_subscription(sub)
        self._scene_subs = []
        with self.lock:
            for k in ("scene_color", "scene_depth", "scene_cinfo", "scene_dinfo", "yolo"):
                self.latest.pop(k, None)

    def transform(self, frame: str) -> np.ndarray:
        """4x4 arm base <- `frame` (latest tf). Raises Abort."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        try:
            tr = self.tfbuf.lookup_transform(self.arm_frame, frame, Time(), timeout=Duration(seconds=2.0)).transform
        except Exception as e:  # noqa: BLE001
            raise Abort(f"tf {self.arm_frame} <- {frame} unavailable: {type(e).__name__}") from None
        m = np.eye(4)
        m[:3, :3] = R.from_quat([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]).as_matrix()
        m[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
        return m

    def cam_rotation_in_base(self) -> np.ndarray:
        """3x3 rotation taking camera-frame directions into arm-base directions."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        q = self._cam_in_base().rotation
        return R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def cam_pose_at(self, stamp, timeout_s: float = 2.0):
        """(position, 3x3 rotation) of the camera in the arm base frame AT `stamp` (an image header
        stamp), from ONE tf lookup that waits for tf to reach that time. Raises Abort.

        Not "latest", and not two lookups: right after a fast move this node's tf buffer can still be
        working through the backlog, and the two "latest" lookups (rotation, then position) once came
        from different moments -- the rotation from mid-move, 85 deg off, and a button 57 cm wrong
        from a perfect dome fit (2026-10-15)."""
        from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

        frame = self.camera_frame()
        try:
            tr = self.tfbuf.lookup_transform(
                self.arm_frame, frame, Time.from_msg(stamp), timeout=Duration(seconds=timeout_s)
            ).transform
        except Exception as e:  # noqa: BLE001
            raise Abort(
                f"tf {self.arm_frame} <- {frame} at the image time unavailable: "
                f"{type(e).__name__} -- arm/tf publishing?"
            ) from None
        q, t = tr.rotation, tr.translation
        return np.array([t.x, t.y, t.z]), R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def cam_position_in_base(self) -> np.ndarray:
        """Camera optical-frame origin in the arm base frame (tf2), metres."""
        t = self._cam_in_base().translation
        return np.array([t.x, t.y, t.z])
