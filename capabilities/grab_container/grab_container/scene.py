"""Find the container in the scene camera, and the wrist view to detect its ring from.

The scene segmenter (rammp-perception's RF-DETR, /perception/object_detections) masks the whole
container; with the scene camera's aligned depth that gives a coarse 3D position. The ring itself is
too thin for the scene camera, so the arm then moves the wrist camera to a view above the container,
on the approach side, and the students' detector picks the grasp point from there.
"""

import threading
import time

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation

from .grasp import DEPTH_MAX, DEPTH_MIN, MIN_PTS, Abort

CONTAINER_CLASS = "food container"
MIN_SCORE = 0.5
FRESH_S = 1.0  # detections older than this are stale
# The wrist camera's view of the container: this far back along the approach (toward the arm) and
# this far above it, looking at it.
VIEW_BACK_M = 0.15
VIEW_UP_M = 0.25


class SceneDetections:
    """The latest /perception/object_detections, only while the action runs."""

    def __init__(self, node, topic):
        from rammp_perception_interfaces.msg import ObjectDetections

        self.node, self.lock, self.latest = node, threading.Lock(), None
        self.sub = node.create_subscription(ObjectDetections, topic, self._on_msg, qos_profile_sensor_data)

    def _on_msg(self, m):
        with self.lock:
            self.latest = (time.monotonic(), m)

    def close(self):
        self.node.destroy_subscription(self.sub)

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                item = self.latest
            if item is not None and time.monotonic() - item[0] <= FRESH_S:
                return item[1]
            time.sleep(0.05)
        return None


def locate_container(detections, camera, timeout=3.0):
    """The best-scoring container's mask points (median), in the arm's base frame, and its score."""
    m = detections.wait(timeout)
    if m is None:
        raise Abort("no /perception/object_detections -- is the scene segmenter up?")
    objs = [o for o in m.objects if o.perception_class == CONTAINER_CLASS and o.mask_id and o.score >= MIN_SCORE]
    if not objs:
        seen = sorted({o.perception_class for o in m.objects})
        raise Abort(f"no {CONTAINER_CLASS!r} >= {MIN_SCORE} in the scene camera (sees {seen})")
    obj = max(objs, key=lambda o: o.score)
    labels = camera.bridge.imgmsg_to_cv2(m.label_image, "passthrough")
    frame = camera.frame(2.0)
    if frame is None:
        raise Abort("no scene camera colour + depth")
    _, dm = frame
    fx, fy, cx, cy = camera.intrinsics()
    ys, xs = np.where(labels == obj.mask_id)
    z = dm[ys, xs]
    ok = (z > DEPTH_MIN) & (z < DEPTH_MAX)
    if ok.sum() < MIN_PTS:
        raise Abort(f"{obj.instance_id}: {int(ok.sum())} mask pixels with depth (< {MIN_PTS})")
    u, v, zz = xs[ok], ys[ok], z[ok]
    pts = np.stack([(u - cx) * zz / fx, (v - cy) * zz / fy, zz], axis=1)
    return camera.to_base(np.median(pts, axis=0)), obj


def view_pose(c, R_grasp, tool_T_cam):
    """The tool pose (position, quat xyzw) that puts the wrist camera VIEW_BACK_M back along the
    grasp's horizontal approach and VIEW_UP_M above container point `c`, looking at it with the image
    upright, then rolled 90 deg about the tool's own z toward the grasp's roll, so the wrist does not
    turn between view and pre-grasp, and shifted sideways to keep the container centred. The camera rolls with it:
    Camera.set_upright() turns the image back. `tool_T_cam`: the camera in the tool frame (R, t)."""
    approach = R_grasp[:, 2]
    back = np.array([approach[0], approach[1], 0.0])
    back /= np.linalg.norm(back)
    p_cam = np.asarray(c) - VIEW_BACK_M * back + np.array([0.0, 0.0, VIEW_UP_M])
    z = np.asarray(c) - p_cam
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])  # image right, horizontal
    x /= np.linalg.norm(x)
    R_cam = np.column_stack([x, np.cross(z, x), z])  # optical frame: z forward, y down in the image
    R_tc, t_tc = tool_T_cam
    R_tool = R_cam @ R_tc.T
    rolls = [R_tool @ Rotation.from_euler("z", s * 90, degrees=True).as_matrix() for s in (1, -1)]
    R_view = min(rolls, key=lambda R: Rotation.from_matrix(R_grasp.T @ R).magnitude())
    # the camera sits off the tool's axis, so the roll swings it: shift the tool back sideways
    # (along the image's horizontal, x), which keeps the container in the middle of the image left to
    # right. Only sideways: the whole swing would also push the tool ~6 cm further out, out of reach.
    swing = (R_view - R_tool) @ t_tc
    return p_cam - R_tool @ t_tc - np.dot(swing, x) * x, Rotation.from_matrix(R_view).as_quat()


def tool_T_cam(arm, camera):
    """The wrist camera in the tool frame: tf's camera pose and the arm's tool pose, now."""
    ee = arm.get_state()["ee_pos"]
    R_bt, t_bt = Rotation.from_quat(ee[3:7]).as_matrix(), np.asarray(ee[:3])
    R_bc, t_bc = camera.pose_in_base()
    return R_bt.T @ R_bc, R_bt.T @ (t_bc - t_bt)
