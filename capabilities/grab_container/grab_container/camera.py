"""The wrist camera: the latest colour, aligned depth and camera_info, and tf2.

The button press's Perception (rammp_button_press/perception.py), on the capability's own node
instead of a node and spin thread of its own, and made per action: the subscriptions and the tf
listener exist only between Camera() and close(). Read-only: nothing here commands anything.

set_upright() turns the frames by quarter turns so the world's up is the image's up: the ring
detector was trained on upright images, and the wrist camera rolls with the gripper. frame(),
intrinsics() and pose_in_base() then all describe the turned image, so a point back-projected
from it lands in the right place.
"""

import threading
import time

import numpy as np
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation

from .grasp import Abort

FRESH_S = 1.0  # a frame older than this is stale
# One quarter turn counterclockwise (np.rot90): the turned camera frame in the original one.
QUARTER = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class Camera:
    def __init__(self, node, arm_frame, color_topic, info_topic, depth_topic, cam_frame=""):
        """`cam_frame` "" = the colour image's own frame_id. Depth must be aligned to colour."""
        import tf2_ros
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image

        self.node, self.arm_frame, self.cam_frame = node, arm_frame, cam_frame
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = {}
        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf, node)
        self.turns = 0  # quarter turns counterclockwise applied to every frame (set_upright)
        self.stamp = None  # the colour stamp of the last frame()

        def keep(name):
            def cb(msg):
                with self.lock:
                    self.latest[name] = (time.monotonic(), msg)

            return cb

        qos = qos_profile_sensor_data  # best effort: receives from reliable and best-effort publishers
        self.subs = [
            node.create_subscription(CameraInfo, info_topic, keep("info"), qos),
            node.create_subscription(Image, color_topic, keep("color"), qos),
            node.create_subscription(Image, depth_topic, keep("depth"), qos),
        ]

    def close(self):
        for s in self.subs:
            self.node.destroy_subscription(s)
        self.tfl.unregister()

    def get(self, name, max_age=None):
        with self.lock:
            item = self.latest.get(name)
        if item is None:
            return None
        t, v = item
        if max_age is not None and time.monotonic() - t > max_age:
            return None
        return v

    def wait(self, name, timeout, max_age=FRESH_S):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            v = self.get(name, max_age)
            if v is not None:
                return v
            time.sleep(0.05)
        return None

    @staticmethod
    def stamp_of(msg):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def frame(self, timeout, after=None):
        """(bgr8 image, float32 depth in metres) of the latest colour + aligned depth, or None. `after`: a
        stamp (stamp_of, ns); only a colour image stamped later is taken. The taken one's is `self.stamp`."""
        deadline = time.monotonic() + timeout
        while True:
            color, depth = self.get("color", FRESH_S), self.get("depth", FRESH_S)
            if color is not None and depth is not None and (after is None or self.stamp_of(color) > after):
                break
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)
        self.stamp = self.stamp_of(color)
        bgr = self.bridge.imgmsg_to_cv2(color, "bgr8")
        dm = self.bridge.imgmsg_to_cv2(depth, "passthrough").astype(np.float32)
        if depth.encoding in ("16UC1", "mono16"):  # the OAK and RealSense drivers publish millimetres
            dm /= 1000.0
        return np.ascontiguousarray(np.rot90(bgr, self.turns)), np.ascontiguousarray(np.rot90(dm, self.turns))

    def intrinsics(self):
        info = self.get("info")
        if info is None:
            raise Abort("no camera_info")
        fx, fy, cx, cy, w = info.k[0], info.k[4], info.k[2], info.k[5], info.width
        for _ in range(self.turns):  # each quarter turn: u' = v, v' = (w - 1) - u
            fx, fy, cx, cy, w = fy, fx, cy, w - 1 - cx, info.height if w == info.width else info.width
        return fx, fy, cx, cy

    def to_base(self, p_cam):
        """A camera-frame point in the arm's base frame (tf2)."""
        R, t = self.pose_in_base()
        return R @ np.asarray(p_cam) + t

    def pose_in_base(self, stamp=None):
        """(R, t): the (turned) camera frame in the arm's base frame (tf2), now, or at `stamp` (ns, an
        image's stamp_of: where the camera was when it took that image)."""
        R, t = self._raw_pose_in_base(stamp)
        return R @ np.linalg.matrix_power(QUARTER, self.turns), t

    def set_upright(self):
        """Choose the quarter turns that put the world's up nearest the image's up (-y), from where
        the camera is now. Returns them."""
        R, _ = self._raw_pose_in_base()
        self.turns = max(range(4), key=lambda k: -(R @ np.linalg.matrix_power(QUARTER, k))[2, 1])
        return self.turns

    def _raw_pose_in_base(self, stamp=None):
        color = self.get("color")
        cam_frame = self.cam_frame or (color.header.frame_id if color is not None else "")
        at = Time() if stamp is None else Time(nanoseconds=stamp)
        try:
            tf = self.tfbuf.lookup_transform(self.arm_frame, cam_frame, at, timeout=Duration(seconds=2.0)).transform
        except Exception as e:  # noqa: BLE001 -- tf2's exceptions
            raise Abort(
                f"tf {self.arm_frame} <- {cam_frame!r} unavailable: {e} -- verify with "
                f"ros2 run tf2_ros tf2_echo {self.arm_frame} {cam_frame}"
            ) from e
        q, t = tf.rotation, tf.translation
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([t.x, t.y, t.z])
