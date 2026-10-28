"""The wrist camera, and the handle detector that reads its frames.

WristCamera: the latest RGB-D frame and where the camera was, in the arm's base frame.

Replaces feeding_deployment's RealSenseROS2Interface (colour + aligned depth + camera_info,
synchronised) and TFInterface (tf2 lookup base <- camera) for this capability: same frame contents
(colour bgr8, depth float32 mm aligned to colour, K), on the capability's own node. Read-only:
nothing here commands anything.

The base frame is the arm's FRAME (`arm.frame`, arm_mount on the nightly driver): the frame
/ee_state reports and every goal is in, so detections come out in the same frame as the arm's
poses. (The Jetson scripts looked up `arm_base_link`, an alias of the 1.1.0 driver's base_link.)

HandleDetector: microwave handle + door face from one RGB-D frame, in the arm's base frame.

The handle-finding half of `AppliancePerception.detect_handle_and_placement`, copied as-is
(same thresholds, same order of operations) so the grasp constants tuned against it still hold:

1. YOLO box around the microwave (COCO 'microwave', else the classes it confuses it with).
2. Every box pixel with valid depth -> 3D point (camera frame).
3. RANSAC plane = the door face; reject it if its normal isn't horizontal (table/floor).
4. Points 0-7 cm in front of the plane -> DBSCAN -> the most vertical, elongated cluster.
5. Handle = the cluster's median (the middle of the bar).
6. Hinge = the door-face point on the edge farthest from the handle, nearest the handle's height.
   Reject if that edge is not clearly on the far side (the plane pulled in background).
7. Into the arm's base frame with the frame's base_T_cam (perception.WristCamera, tf2).

Dropped vs the original: the hinge/placement/top-of-appliance poses, overlays, RViz markers,
image logging, the camera flip (this rig's camera is upright) and HANDLE_DEPTH_CORR (0 here).

NOTE: YOLO (step 1) is a development stand-in only -- the actual demo will NOT use YOLO; a different
detection model will provide the microwave box. Don't tune the task around YOLO's quirks (e.g. the
red/blue swap in `_box`).
"""

import threading
import time

import numpy as np
import open3d as o3d
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from .microwave_common import say

TF_TIMEOUT_S = 2.0  # wait this long for base <- camera at the frame's stamp, then use the latest
SYNC_SLOP_S = 0.05  # colour, depth and camera_info within this of each other count as one frame


class WristCamera:
    def __init__(self, node, color_topic, info_topic, depth_topic, base_frame, camera_frame=""):
        """`camera_frame` "" = the colour image's own frame_id. Depth must be aligned to colour."""
        import message_filters
        import tf2_ros
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image

        self.node, self.base_frame, self.camera_frame = node, base_frame, camera_frame
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None
        self.tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        qos = qos_profile_sensor_data
        self._subs = [
            message_filters.Subscriber(node, Image, color_topic, qos_profile=qos),
            message_filters.Subscriber(node, CameraInfo, info_topic, qos_profile=qos),
            message_filters.Subscriber(node, Image, depth_topic, qos_profile=qos),
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(self._subs, 10, SYNC_SLOP_S)
        self._sync.registerCallback(self._on_frame)
        self.topics = (color_topic, info_topic, depth_topic)

    def close(self):
        """Drop the subscriptions and the tf listener (the node makes a camera per action)."""
        for s in self._subs:
            self.node.destroy_subscription(s.sub)
        self._tf_listener.unregister()

    def _on_frame(self, color, info, depth):
        with self._lock:
            self._latest = (color, info, depth)

    def wait_for_frames(self, timeout_s=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(0.05)
        return False

    def base_T_cam(self, frame_id, stamp):
        """4x4 base <- camera at `stamp` (else the latest available), or None."""
        target = self.camera_frame or frame_id
        for when in (Time.from_msg(stamp), Time()):
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, target, when, timeout=Duration(seconds=TF_TIMEOUT_S)
                ).transform
            except Exception:  # noqa: BLE001 -- tf2's exceptions; try the latest, then give up
                continue
            m = np.eye(4)
            m[:3, :3] = R.from_quat([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]).as_matrix()
            m[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
            return m
        return None

    def grab(self, timeout_s=10.0):
        """The latest frame: {rgb (bgr8), depth_mm (float32), K (3x3), base_T_cam (4x4 or None), time,
        frame_id}. Raises RuntimeError if no frame has come in within `timeout_s`."""
        if not self.wait_for_frames(timeout_s):
            raise RuntimeError(f"no RGB-D frames on {', '.join(self.topics)} (depth aligned to colour?)")
        with self._lock:
            color, info, depth = self._latest
        rgb = self._bridge.imgmsg_to_cv2(color, "bgr8")
        depth_mm = np.asarray(self._bridge.imgmsg_to_cv2(depth, "passthrough"), dtype=np.float32)
        return {
            "rgb": rgb,
            "depth_mm": depth_mm,
            "K": np.asarray(info.k, dtype=float).reshape(3, 3),
            "base_T_cam": self.base_T_cam(color.header.frame_id, color.header.stamp),
            "time": time.time(),
            "frame_id": self.camera_frame or color.header.frame_id,
        }


# ---------------------------------------------------------------------------
# Handle detector
# ---------------------------------------------------------------------------
COCO_MICROWAVE_CLASS_ID = 68
YOLO_MODEL = "yolo26s.pt"  # dev stand-in (the node's yolo_model parameter): the demo uses a different model
# COCO classes YOLO confuses this microwave with (bus 5, train 6, suitcase 28, oven 69,
# refrigerator 72), used only when it finds no 'microwave' at all.
YOLO_FALLBACK_CLASS_IDS = [5, 6, 28, 69, 72]
BOX_THRESHOLD = 0.3

DEPTH_RANGE_M = (0.05, 2.0)
PLANE_DIST_M = 0.02
HANDLE_PROTRUSION_MAX_M = 0.07  # how far in front of the door plane a handle can sit
DBSCAN_EPS_M, DBSCAN_MIN_SAMPLES = 0.02, 50
HANDLE_MIN_CLUSTER_FRACTION = 0.25  # clusters below this fraction of the biggest are noise
HANDLE_ELONGATION_CAP = 10.0  # so a very thin sliver can't dominate the score
MAX_DOOR_NORMAL_VERTICAL = 0.5  # door is vertical, so its normal must be ~horizontal
# fixed handle orientation the detector has always returned (the grasp yaws it to the face)
HANDLE_QUAT = np.array([-0.5, 0.5, 0.5, -0.5])


class HandleDetector:
    def __init__(self, model=YOLO_MODEL):
        from ultralytics import YOLO  # here, not at import: loads torch, which the mock never needs

        self._yolo = YOLO(model)

    def _box(self, rgb):
        """Highest-confidence microwave box (x1, y1, x2, y2), or None.
        Tries the image as given AND with red/blue swapped: this red microwave often scores far
        higher swapped (10-24: 0.20 vs 0.77, 0.49 vs 0.67 -- red + boxy reads as bus/train)."""
        res = None
        for img in (rgb, np.ascontiguousarray(rgb[:, :, ::-1])):
            r = self._yolo.predict(img, classes=[COCO_MICROWAVE_CLASS_ID], conf=BOX_THRESHOLD, verbose=False)[0]
            if len(r.boxes) and (res is None or float(r.boxes.conf.max()) > float(res.boxes.conf.max())):
                res = r
        if res is None:
            res = self._yolo.predict(rgb, classes=YOLO_FALLBACK_CLASS_IDS, conf=BOX_THRESHOLD, verbose=False)[0]
            if len(res.boxes) == 0:
                return None
            best = int(res.boxes.conf.argmax())
            say(
                f"  (YOLO fallback: no 'microwave'; using '{res.names[int(res.boxes.cls[best])]}' "
                f"{float(res.boxes.conf[best]):.2f})"
            )
        else:
            best = int(res.boxes.conf.argmax())
            say(f"  microwave {float(res.boxes.conf[best]):.2f}")
        return res.boxes.xyxy[best].cpu().numpy().astype(int)

    def detect(self, rgb, K, depth, base_T_cam):
        """`K` 3x3, `base_T_cam` 4x4 (arm base <- camera, or None). Returns dict(handle, quat, normal, hinge,
        door_z, door_mid) in the arm's base frame, or None.
        normal: horizontal door-face normal, out of the door toward the camera.
        hinge: door-face point on the edge farthest from the handle, nearest the handle's height
          (the swing pivots about the vertical line through it).
        door_z: (low, high) of the door face (1st/99th percentile height).
        door_mid: door-face point at the middle of its horizontal span."""
        if base_T_cam is None:
            say("no arm base <- camera transform")
            return None
        up_cam = base_T_cam[2, :3]  # base +z in the camera frame

        box = self._box(rgb)
        if box is None:
            say("  no microwave box")
            return None
        x1, y1, x2, y2 = box

        # box pixels -> camera-frame points (row-major, like the original's np.where on the mask)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[y1:y2, x1:x2] = True
        vs, us = np.where(mask)
        d = depth[vs, us] / 1000.0
        ok = ~np.isnan(d) & (d >= DEPTH_RANGE_M[0]) & (d <= DEPTH_RANGE_M[1])
        us, vs, d = us[ok], vs[ok], d[ok]
        if len(d) == 0:
            say("  no valid depth in the box")
            return None
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts = np.stack([d / fx * (us - cx), d / fy * (vs - cy), d], axis=1)

        # door plane
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        plane_model, inliers = pcd.segment_plane(distance_threshold=PLANE_DIST_M, ransac_n=3, num_iterations=500)
        plane_pts = pts[inliers]
        off_plane = pts[np.setdiff1d(np.arange(len(pts)), inliers)]
        normal, offset = np.asarray(plane_model[:3], float), float(plane_model[3])
        nn = np.linalg.norm(normal)
        normal, offset = normal / nn, offset / nn
        if np.dot(normal, plane_pts.mean(axis=0)) > 0:  # point toward the camera: "in front" is +
            normal, offset = -normal, -offset
        if abs(float(np.dot(normal, up_cam))) > MAX_DOOR_NORMAL_VERTICAL:
            say("  plane is not a door (normal not horizontal -- table/floor?)")
            return None

        # protruding points -> clusters -> the vertical, elongated one
        signed = off_plane @ normal + offset
        handle_pts = off_plane[(signed > 0.0) & (signed < HANDLE_PROTRUSION_MAX_M)]
        if len(handle_pts) == 0:
            say("  nothing protruding in front of the door plane")
            return None
        # open3d's DBSCAN (was sklearn's; same clusters, ~3x faster, one dependency fewer -- 10-07)
        labels = np.asarray(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(handle_pts)).cluster_dbscan(
                eps=DBSCAN_EPS_M, min_points=DBSCAN_MIN_SAMPLES, print_progress=False
            )
        )
        if not np.any(labels >= 0):
            say("  DBSCAN found no clusters")
            return None
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        best_label, best_score = None, -1.0
        for label in unique[counts >= HANDLE_MIN_CLUSTER_FRACTION * counts.max()]:
            cluster = handle_pts[labels == label]
            _, s, vt = np.linalg.svd(cluster - cluster.mean(axis=0), full_matrices=False)
            score = abs(float(vt[0] @ up_cam)) * min(float(s[0] / max(s[1], 1e-9)), HANDLE_ELONGATION_CAP)
            if score > best_score:
                best_label, best_score = label, score
        cluster = handle_pts[labels == best_label]
        handle_cam = np.median(cluster, axis=0)
        # 10-07: no longer "4 cm in from the cluster's max camera y" -- on the original upside-down camera that was
        # the TOP of the bar, on this upright one it is the BOTTOM end where the bar curves into the door.
        # The middle of the bar is right on either.

        # door edges along the plane's own horizontal axis (percentiles: RANSAC's edge points drift)
        axis = np.cross(normal, up_cam)
        axis /= np.linalg.norm(axis)
        proj = plane_pts @ axis
        h_proj = float(handle_cam @ axis)
        lo, hi = float(np.percentile(proj, 1)), float(np.percentile(proj, 99))
        edge = lo if abs(h_proj - lo) > abs(h_proj - hi) else hi
        strip = plane_pts[np.abs(proj - edge) < 0.02]
        hinge_cam = strip[int(np.argmin(np.linalg.norm(strip - handle_cam, axis=1)))]
        if abs(float(hinge_cam @ axis) - h_proj) < 0.5 * max(abs(h_proj - lo), abs(h_proj - hi)):
            say("  far door edge is on the handle's side -- plane includes background")
            return None

        # to arm_base_link
        Rb, tb = base_T_cam[:3, :3], base_T_cam[:3, 3]
        n_base = Rb @ normal
        n_base[2] = 0.0
        plane_base = plane_pts @ Rb.T + tb
        return {
            "handle": Rb @ handle_cam + tb,
            "quat": HANDLE_QUAT,
            "normal": n_base / np.linalg.norm(n_base),
            "hinge": Rb @ hinge_cam + tb,
            "door_z": (float(np.percentile(plane_base[:, 2], 1)), float(np.percentile(plane_base[:, 2], 99))),
            "door_mid": plane_base[int(np.argmin(np.abs(proj - 0.5 * (lo + hi))))].copy(),
        }
