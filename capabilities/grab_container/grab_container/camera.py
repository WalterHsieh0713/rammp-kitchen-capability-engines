"""The wrist camera: the latest colour, aligned depth and camera_info, and tf2.

The button press's Perception (rammp_button_press/perception.py), on the capability's own node
instead of a node and spin thread of its own, and made per action: the subscriptions and the tf
listener exist only between Camera() and close(). Read-only: nothing here commands anything.
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


class Camera:
    def __init__(self, node, arm_frame, cam_frame, camera_ns="/wrist_camera"):
        import tf2_ros
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image

        self.node, self.arm_frame, self.cam_frame = node, arm_frame, cam_frame
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = {}
        self.tfbuf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfbuf, node)

        def keep(name):
            def cb(msg):
                with self.lock:
                    self.latest[name] = (time.monotonic(), msg)

            return cb

        ns = camera_ns.rstrip("/")
        qos = qos_profile_sensor_data  # best effort: receives from reliable and best-effort publishers
        self.subs = [
            node.create_subscription(CameraInfo, f"{ns}/color/camera_info", keep("info"), qos),
            node.create_subscription(Image, f"{ns}/color/image_raw", keep("color"), qos),
            node.create_subscription(Image, f"{ns}/aligned_depth_to_color/image_raw", keep("depth"), qos),
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

    def frame(self, timeout):
        """(bgr8 image, float32 depth in metres) of the latest colour + aligned depth, or None."""
        color, depth = self.wait("color", timeout), self.wait("depth", timeout)
        if color is None or depth is None:
            return None
        bgr = self.bridge.imgmsg_to_cv2(color, "bgr8")
        dm = self.bridge.imgmsg_to_cv2(depth, "passthrough").astype(np.float32)
        if depth.encoding in ("16UC1", "mono16"):  # the RealSense driver publishes millimetres
            dm /= 1000.0
        return bgr, dm

    def intrinsics(self):
        info = self.get("info")
        if info is None:
            raise Abort("no camera_info")
        return info.k[0], info.k[4], info.k[2], info.k[5]

    def to_base(self, p_cam):
        """A camera-frame point in the arm's base frame (tf2)."""
        try:
            tf = self.tfbuf.lookup_transform(
                self.arm_frame, self.cam_frame, Time(), timeout=Duration(seconds=2.0)
            ).transform
        except Exception as e:  # noqa: BLE001 -- tf2's exceptions
            raise Abort(
                f"tf {self.arm_frame} <- {self.cam_frame} unavailable: {e} -- verify with "
                f"ros2 run tf2_ros tf2_echo {self.arm_frame} {self.cam_frame}"
            ) from e
        q, t = tf.rotation, tf.translation
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return R @ np.asarray(p_cam) + np.array([t.x, t.y, t.z])
