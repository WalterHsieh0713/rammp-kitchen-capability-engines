"""view_detection --scene: the overlay drawn on the scene image (no window, no ROS spinning)."""

import numpy as np
import pytest

vd = pytest.importorskip("rammp_button_press.view_detection")  # needs rclpy + cv_bridge
from rammp_button_press.scene_finder import MicrowaveFit  # noqa: E402

K = (612.7, 612.7, 635.4, 361.0)
T = np.eye(4)  # camera at the base origin, looking along +z


def test_a_found_front_is_outlined_and_its_button_marked():
    fit = MicrowaveFit(
        n=np.array([0.0, 0.0, -1.0]),
        centre=np.array([0.0, 0.0, 1.0]),
        ex=np.array([1.0, 0.0, 0.0]),
        ey=np.array([0.0, -1.0, 0.0]),
        width_m=0.445,
        height_m=0.26,
        corner_br=np.array([0.2225, 0.13, 1.0]),
        button=np.array([0.20, 0.07, 1.0]),
        window_dark=0.7,
        strip_dark=0.1,
        lab=(164.0, 125.0, 135.0),
        px_box=(0, 0, 1, 1),
        n_points=1000,
    )
    img = np.zeros((720, 1280, 3), np.uint8)
    out = vd.draw_scene(img, fit, "", K, T, 400.0)
    assert out is not img and img.max() == 0  # the input frame is left alone
    u = int(round(0.20 * K[0] + K[2]))
    v = int(round(0.07 * K[1] + K[3]))
    assert out[v - 12 : v + 13, u - 12 : u + 13].any()  # the button is marked
    left = int(round(-0.2225 * K[0] + K[2]))
    assert out[361 - 5 : 361 + 5, left - 3 : left + 4].any()  # the front's left edge is drawn


def test_no_fit_shows_why():
    img = np.zeros((720, 1280, 3), np.uint8)
    out = vd.draw_scene(img, None, "no microwave front: size 20 x 25 cm", K, T, 400.0)
    assert out[:40].any() and not out[100:].any()  # only the reason, along the top


def test_the_yolo_box_is_drawn():
    img = np.zeros((720, 1280, 3), np.uint8)
    out = vd.draw_scene(img, None, "no microwave front", K, T, 400.0, box=(553.0, 240.0, 846.0, 431.0, 0.9))
    assert out[240, 600:800].any() and out[431, 600:800].any() and not out[300:400, 600:800].any()
