"""The scene camera's messages -> one scene_finder frame (perception.scene_frame_from_msgs), without
spinning ROS: the Gemini publishes rgb8 colour and 16UC1 millimetre depth, each with its own
camera_info and optical frame."""

import inspect

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

from rammp_button_press import scene_finder
from rammp_button_press.perception import scene_frame_from_msgs


def info(fx, fy, cx, cy, w, h):
    m = CameraInfo()
    m.width, m.height = w, h
    m.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    return m


def test_scene_messages_become_a_finder_frame():
    br = CvBridge()
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[0, 0] = (200, 100, 50)
    depth_mm = np.full((2, 4), 1234, np.uint16)
    color, depth = br.cv2_to_imgmsg(rgb, "rgb8"), br.cv2_to_imgmsg(depth_mm, "16UC1")
    T_c, T_d = np.eye(4), np.eye(4)
    T_d[:3, 3] = (0.01, 0.02, 0.0)
    fr = scene_frame_from_msgs(
        br, color, depth, info(612.7, 612.8, 635.4, 361.0, 3, 2), info(410.1, 410.1, 426.0, 237.4, 4, 2), T_c, T_d
    )
    assert tuple(fr["bgr"][0, 0]) == (50, 100, 200)
    assert fr["depth_m"].dtype == np.float32 and abs(float(fr["depth_m"][1, 3]) - 1.234) < 1e-6
    assert fr["color_K"] == (612.7, 612.8, 635.4, 361.0)
    assert fr["depth_K"] == (410.1, 410.1, 426.0, 237.4)
    assert np.array_equal(fr["T_base_depth"], T_d)
    want = [
        p
        for p in inspect.signature(scene_finder.find_microwave).parameters.values()
        if p.default is inspect.Parameter.empty
    ]
    assert set(fr) == {p.name for p in want}  # exactly the frame find_microwave needs


def detections(frame, *dets):
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    m = Detection2DArray()
    m.header.frame_id = frame
    for cls, score, (cx, cy, sx, sy) in dets:
        d = Detection2D()
        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id, h.hypothesis.score = cls, score
        d.results.append(h)
        d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.size_x, d.bbox.size_y = map(float, (cx, cy, sx, sy))
        m.detections.append(d)
    return m


def test_yolo_box_is_the_best_microwave_in_pixel_corners():
    from rammp_button_press.perception import SCENE_COLOR_FRAME, yolo_box

    # live 2026-10-15
    m = detections(
        SCENE_COLOR_FRAME,
        ("refrigerator", 0.57, (200, 300, 300, 400)),
        ("microwave", 0.9015, (699.375, 335.625, 293.25, 190.75)),
        ("microwave", 0.6, (100, 100, 50, 50)),
    )
    u0, v0, u1, v1, score = yolo_box(m)
    assert np.allclose((u0, v0, u1, v1), (552.75, 240.25, 846.0, 431.0)) and abs(score - 0.9015) < 1e-6


def test_yolo_box_refuses_weak_other_frame_and_missing():
    from rammp_button_press.perception import SCENE_COLOR_FRAME, yolo_box

    assert yolo_box(None) is None
    assert yolo_box(detections(SCENE_COLOR_FRAME, ("microwave", 0.3, (699, 335, 293, 190)))) is None
    assert yolo_box(detections(SCENE_COLOR_FRAME, ("refrigerator", 0.9, (699, 335, 293, 190)))) is None
    assert yolo_box(detections("wrist_d405_color_optical", ("microwave", 0.9, (699, 335, 293, 190)))) is None


def test_stop_scene_drops_the_scene_subscriptions():
    import rclpy

    from rammp_button_press.perception import SCENE_NS, Perception

    rclpy.init()
    try:
        per = Perception("base_link", "camera_color_optical_frame", "/wrist_camera", scene_ns=SCENE_NS)
        topics = lambda: {s.topic_name for s in per.subscriptions}  # noqa: E731
        assert f"{SCENE_NS}/color/image_raw" in topics()
        assert "/yolo/detections" in topics()
        per.stop_scene()
        assert not any(t.startswith(SCENE_NS) for t in topics()) and "/yolo/detections" not in topics()
        assert "/wrist_camera/color/image_raw" in topics()
        per.stop_scene()  # twice is fine
        per.destroy_node()
    finally:
        rclpy.shutdown()


def test_the_camera_pose_is_taken_at_the_image_time_not_the_latest():
    """2026-10-15: right after a fast move, "latest" tf gave a rotation from mid-move (85 deg off) and the
    button came out 57 cm wrong. cam_pose_at(stamp) answers for the image's own time, in one lookup."""
    import rclpy
    from builtin_interfaces.msg import Time as TimeMsg
    from geometry_msgs.msg import TransformStamped
    from scipy.spatial.transform import Rotation

    from rammp_button_press.perception import Perception

    rclpy.init()
    try:
        per = Perception("base_link", "cam", "/wrist_camera")
        for sec, yaw, x in ((100, 0.0, 0.5), (101, 85.0, 0.1)):
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id = "base_link", "cam"
            tf.header.stamp = TimeMsg(sec=sec)
            tf.transform.translation.x = x
            q = Rotation.from_euler("z", yaw, degrees=True).as_quat()
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = q
            per.tfbuf.set_transform(tf, "test")
        p, Rm = per.cam_pose_at(TimeMsg(sec=100))
        assert np.allclose(p, [0.5, 0, 0]) and np.allclose(Rm, np.eye(3), atol=1e-9)
        p, Rm = per.cam_pose_at(TimeMsg(sec=101))
        assert np.allclose(p, [0.1, 0, 0]) and abs(np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])) - 85.0) < 1e-6
        per.destroy_node()
    finally:
        rclpy.shutdown()
