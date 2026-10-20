"""Live detection viewer: shows what the detectors see. Read-only, never touches the arm. Esc / q / Ctrl-C to close.

    ros2 run rammp_button_press view_detection [--target timer_clock] [--camera-ns /wrist_camera]
    ros2 run rammp_button_press view_detection --scene [--target start_30s]

Default: the wrist camera's dome detector. The 5 domes found are circled in magenta (the target
in green), with the measured dome pitch (should read ~19 mm), fit error and range. If nothing is
found it shows why: "TOO CLOSE" / "TOO FAR" (move the camera) or "no 5-dome layout".

--scene: the fixed scene camera's microwave finder instead. The front is outlined in green with
the rough target button ringed, plus its size, window/strip darkness and colour, or why nothing
was found. The YOLO node's microwave box (/yolo/detections), which narrows the search as in
press_button, is drawn in yellow; --no-yolo searches the whole view. Needs the scene camera's tf
(box-opening's launch).

--publish: no window; the drawn frame goes to Foxglove instead (JPEG, --hz a second, default 5), on
WRIST_TOPIC or SCENE_TOPIC, which box-opening's read-only bridge on port 8780 lets through and
foxglove/button_press.json shows. Works over ssh. Run it on Cyclone like the bridge, Ctrl-C to stop:

    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 run rammp_button_press view_detection --target start_30s --publish
"""

import argparse
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from rammp_button_press import Abort, say, scene_finder
from rammp_button_press import dome_pattern as dp
from rammp_button_press.perception import (
    DEPTH_TOPIC,
    SCENE_COLOR_FRAME,
    SCENE_DEPTH_FRAME,
    SCENE_DEPTH_TOPIC,
    SCENE_NS,
    YOLO_FRESH_S,
    Perception,
    frame_of,
    scene_frame_from_msgs,
    to_bgr_depth,
    why_no_fit,
    yolo_box,
)

# --publish: under box-opening's /rammp_box_opening/viz/ because its 8780 bridge offers nothing else of ours
# (its topic_whitelist is viz/topics.RECORD_REGEX). Its runs' bags record these too while a viewer publishes.
VIZ_NS = "/rammp_box_opening/viz/button_press"
WRIST_TOPIC = VIZ_NS + "/wrist/image/compressed"
SCENE_TOPIC = VIZ_NS + "/scene/image/compressed"


def to_compressed(vis, header, quality=80):
    """A drawn frame as a sensor_msgs/CompressedImage (JPEG), stamped and framed like the camera
    frame it was drawn on."""
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return CompressedImage(header=header, format="jpeg", data=buf.tobytes())


def show(win, vis, pub, header):
    """The drawn frame to the OpenCV window, or (--publish) to Foxglove."""
    if pub is None:
        cv2.imshow(win, vis)
    else:
        pub.publish(to_compressed(vis, header))


def draw_domes(bgr, fit, target, dt_ms, why="NO FIT"):
    """The dome detection (or why there is none) drawn on a copy of the wrist image."""
    vis = bgr.copy()
    if fit is None:
        cv2.putText(vis, f"domes: {why}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis
    cv2.polylines(vis, [fit.region.astype(np.int32)], True, (255, 200, 0), 1)
    r = max(4, int(0.3 * fit.s_px))
    for name in dp.NAMES:
        u, v = (int(round(c)) for c in fit.px[name])
        hit = name == target
        cv2.circle(vis, (u, v), r + (4 if hit else 0), (0, 255, 0) if hit else (255, 0, 255), 2)
        cv2.putText(
            vis, name, (u + r + 2, v - r), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0) if hit else (255, 255, 0), 1
        )
    cv2.putText(
        vis,
        f"domes: pitch {fit.s_mm:.1f} mm  err {fit.err:.2f}  z {fit.z*100:.0f} cm  {dt_ms:.0f} ms",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    return vis


def _px(p, K, T):
    """Pixel (u, v) of base-frame point p in a camera with intrinsics K and pose T (base <- camera)."""
    c = (np.asarray(p, dtype=float) - T[:3, 3]) @ T[:3, :3]
    return int(round(c[0] / c[2] * K[0] + K[2])), int(round(c[1] / c[2] * K[1] + K[3]))


def draw_scene(bgr, fit, why, color_K, T_base_color, dt_ms, box=None):
    """The scene finder's answer on a copy of the scene colour image (and the YOLO box it searched in)."""
    vis = bgr.copy()
    T = np.asarray(T_base_color, dtype=float)
    if box is not None:
        cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 2)
        cv2.putText(
            vis,
            f"YOLO microwave {box[4]:.2f}",
            (int(box[0]), max(12, int(box[1]) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )
    if fit is None:
        cv2.putText(vis, f"scene: {why}"[:150], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return vis
    corners = [
        fit.centre + sx * fit.width_m / 2 * fit.ex + sy * fit.height_m / 2 * fit.ey
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    cv2.polylines(vis, [np.array([_px(c, color_K, T) for c in corners], np.int32)], True, (0, 255, 0), 2)
    cv2.circle(vis, _px(fit.button, color_K, T), 10, (0, 255, 0), 2)
    lab = " ".join(f"{v:.0f}" for v in fit.lab) or "-"
    cv2.putText(
        vis,
        f"front {fit.width_m*100:.0f} x {fit.height_m*100:.0f} cm  window {fit.window_dark:.2f}  "
        f"strip {fit.strip_dark:.2f}  Lab {lab}  button {np.round(fit.button, 3)} "
        f"(ROUGH)  {dt_ms:.0f} ms",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    return vis


def view_scene(a):
    """--scene mode: run scene_finder on every new scene-camera frame and show the result."""
    # no rclpy signal handler: Ctrl-C is a plain KeyboardInterrupt and the ending below stops the spinning
    # before the context (rclpy's own handler shut the context first: a traceback from the spin thread)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    per = Perception(a.arm_frame, "", a.camera_ns, scene_ns=a.scene_ns, scene_depth_topic=a.scene_depth_topic)
    win = "microwave (scene camera)"
    pub = per.create_publisher(CompressedImage, SCENE_TOPIC, 1) if a.publish else None
    if pub is None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 540)
    last, T = (None, None), None
    sent = 0.0
    try:
        while True:
            c, d = per.get("scene_color"), per.get("scene_depth")
            ci, di = per.get("scene_cinfo"), per.get("scene_dinfo")
            due = pub is None or time.monotonic() - sent >= 1.0 / a.hz
            if None not in (c, d, ci, di) and c is not last[0] and d is not last[1] and due:
                last, sent = (c, d), time.monotonic()
                try:
                    T = T or (
                        per.transform(frame_of(c, SCENE_COLOR_FRAME)),
                        per.transform(frame_of(d, SCENE_DEPTH_FRAME)),
                    )
                except Abort as e:
                    say(f"{e} -- box-opening's launch up? retrying")
                if T is not None:
                    fr = scene_frame_from_msgs(per.bridge, c, d, ci, di, *T)
                    yolo = per.get("yolo", YOLO_FRESH_S)
                    box = yolo_box(yolo, frame=frame_of(c, SCENE_COLOR_FRAME)) if a.yolo else None
                    t = time.perf_counter()
                    fit, why = scene_finder.find_microwave(**fr, target=a.target, roi=None if box is None else box[:4])
                    show(
                        win,
                        draw_scene(fr["bgr"], fit, why, fr["color_K"], T[0], (time.perf_counter() - t) * 1000, box=box),
                        pub,
                        c.header,
                    )
            if pub is not None:
                time.sleep(0.015)
            elif (cv2.waitKey(15) & 0xFF) in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    if pub is None:
        cv2.destroyAllWindows()
    rclpy.get_global_executor().shutdown()  # Perception spins on it: its thread then ends with the context
    rclpy.try_shutdown()
    per._thread.join(timeout=2.0)  # still running at the interpreter's exit, it aborts the process
    per.destroy_node()


def main():
    """Default mode: run the dome detector on every new wrist-camera frame and show the result."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="timer_clock")
    ap.add_argument("--camera-ns", default="/wrist_camera")
    ap.add_argument(
        "--scene", action="store_true", help="the scene camera's microwave finder instead of the wrist domes"
    )
    ap.add_argument("--scene-ns", default=SCENE_NS)
    ap.add_argument("--depth-topic", default=DEPTH_TOPIC, help="wrist depth aligned to colour, under --camera-ns")
    ap.add_argument("--scene-depth-topic", default=SCENE_DEPTH_TOPIC, help="scene depth, under --scene-ns")
    ap.add_argument("--no-yolo", dest="yolo", action="store_false", help="--scene: ignore the YOLO microwave box")
    ap.add_argument("--arm-frame", default="base_link", help="z up: base_link (1.1.0 driver) or arm_mount (nightly)")
    ap.add_argument("--publish", action="store_true", help="no window: send the drawn frame to Foxglove instead")
    ap.add_argument("--hz", type=float, default=5.0, help="--publish: frames detected and sent a second")
    a = ap.parse_args()
    if a.scene:
        return view_scene(a)
    ns = a.camera_ns.rstrip("/")

    # no rclpy signal handler: Ctrl-C is a plain KeyboardInterrupt and the ending below stops the spinning
    # before the context (rclpy's own handler shut the context first: a traceback from the spin thread)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("detection_viewer")
    bridge = CvBridge()
    got = {}
    qos = qos_profile_sensor_data
    node.create_subscription(Image, f"{ns}/color/image_raw", lambda m: got.__setitem__("c", m), qos)
    node.create_subscription(Image, f"{ns}/{a.depth_topic.strip('/')}", lambda m: got.__setitem__("d", m), qos)
    node.create_subscription(CameraInfo, f"{ns}/color/camera_info", lambda m: got.__setitem__("i", m), qos)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    win = "button detection (dome layout)"
    pub = node.create_publisher(CompressedImage, WRIST_TOPIC, 1) if a.publish else None
    if pub is None:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 640, 480)
    last = prev = None
    sent = 0.0
    try:
        while True:
            due = pub is None or time.monotonic() - sent >= 1.0 / a.hz
            if all(k in got for k in ("c", "d", "i")) and got["c"] is not last and due:
                last, sent = got["c"], time.monotonic()
                bgr, depth = to_bgr_depth(bridge, last, got["d"])
                t = time.perf_counter()
                fit = dp.detect(bgr, depth, got["i"].k[0], hint=prev)
                if fit is None and prev is not None:
                    fit = dp.detect(bgr, depth, got["i"].k[0])
                prev = fit
                why = why_no_fit(bgr, depth, got["i"].k[0]) if fit is None else ""
                show(win, draw_domes(bgr, fit, a.target, (time.perf_counter() - t) * 1000, why), pub, last.header)
            if pub is not None:
                time.sleep(0.015)
                continue
            k = cv2.waitKey(15) & 0xFF
            # (no WND_PROP_VISIBLE check: this OpenCV 4.5.4/GTK3 build reports 0 for a shown window)
            if k in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    if pub is None:
        cv2.destroyAllWindows()
    executor.shutdown()
    spin.join(timeout=2.0)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
