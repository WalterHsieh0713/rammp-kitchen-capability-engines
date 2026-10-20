"""view_detection --publish: the drawn detection goes to Foxglove as a JPEG, on a topic box-opening's read-only
bridge (port 8780) lets through."""

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

vd = pytest.importorskip("rammp_button_press.view_detection")  # needs rclpy + cv_bridge
from std_msgs.msg import Header  # noqa: E402


def test_a_drawn_frame_goes_out_as_a_jpeg_stamped_like_its_camera_frame():
    vis = np.zeros((480, 640, 3), np.uint8)
    vis[100:200, 300:400] = (255, 0, 255)  # a magenta dome circle, say
    header = Header(frame_id="camera_color_optical_frame")
    header.stamp.sec, header.stamp.nanosec = 1792180570, 123
    msg = vd.to_compressed(vis, header)
    assert msg.format == "jpeg"
    assert (msg.header.frame_id, msg.header.stamp.sec, msg.header.stamp.nanosec) == (
        "camera_color_optical_frame",
        1792180570,
        123,
    )
    out = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    assert out.shape == vis.shape
    assert np.abs(out.astype(int) - vis.astype(int)).mean() < 3  # the drawing survives the compression


def test_the_topics_pass_box_openings_read_only_bridge():
    # the bridge offers only what box-opening records: everything under /rammp_box_opening/viz/ among it
    for topic in (vd.WRIST_TOPIC, vd.SCENE_TOPIC):
        assert topic.startswith("/rammp_box_opening/viz/") and topic.endswith("/compressed")
    try:
        from rammp_box_opening.viz.topics import RECORD_REGEX
    except ImportError:
        return
    assert all(re.fullmatch(RECORD_REGEX, t) for t in (vd.WRIST_TOPIC, vd.SCENE_TOPIC))


@pytest.mark.parametrize("mode", [[], ["--scene"]])
def test_ctrl_c_ends_a_publishing_viewer_cleanly(mode):
    # no camera on these names and a domain of its own: nothing arrives, nothing reaches the rig
    env = dict(
        os.environ,
        ROS_DOMAIN_ID="97",
        PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]),
    )
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from rammp_button_press.view_detection import main; main()",
            "--publish",
            "--camera-ns",
            "/no_camera",
            "--scene-ns",
            "/no_scene",
            *mode,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(4.0)
    p.send_signal(signal.SIGINT)
    out, _ = p.communicate(timeout=15)
    assert p.returncode == 0 and "Traceback" not in out, out
