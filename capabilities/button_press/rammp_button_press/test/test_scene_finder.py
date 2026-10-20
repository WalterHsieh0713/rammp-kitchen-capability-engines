"""Tests for the scene-camera microwave finder (scene_finder): synthetic scenes ray-cast from a box
model of the cream Comfee seen by a Gemini-like camera pair, plus the recorded scene frames in
tools/frames/scene/ (record_scene_frames.py; `neg_*` = no usable microwave front in view)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from rammp_button_press import scene_finder as sf

UP = np.array([0.0, 0.0, 1.0])
COLOR_K, COLOR_WH = (612.7, 612.7, 635.4, 361.0), (1280, 720)
DEPTH_K, DEPTH_WH = (410.1, 410.1, 426.0, 237.4), (848, 480)
TABLE_Z = -0.01
FEET_M = 0.02
EYE = np.array([0.0, 0.58, 0.35])  # where the bench's scene camera stands (base_link)
LOOK_AT = np.array([0.75, -0.05, 0.10])
DEPTH_OFFSET = np.array([0.013, 0.02, 0.0])  # the Gemini's depth optical frame sits ~2 cm from the colour one
SCENE_REC = Path(__file__).resolve().parents[1] / "tools" / "frames" / "scene"


def lab_to_bgr(lab):
    return np.array(cv2.cvtColor(np.uint8([[lab]]), cv2.COLOR_LAB2BGR)[0, 0], dtype=np.uint8)


CREAM = lab_to_bgr((164, 125, 135))  # the HERL microwave's face under the bench lights
DARK = np.array([35, 35, 35], np.uint8)
TABLE = np.array([150, 152, 150], np.uint8)
WALL = np.array([235, 235, 235], np.uint8)


def rotz(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def look_at(eye, target):
    """Optical frame (x right, y down, z forward) at `eye` looking at `target`, as base <- optical."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, UP)
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    T[:3, 3] = eye
    return T


class Rect:
    """A flat rectangle: centre c, unit axes ax/ay with half-sizes hx/hy, painted by paint(s, q)."""

    def __init__(self, c, ax, hx, ay, hy, paint, nodepth=False):
        self.c, self.ax, self.hx, self.ay, self.hy = (
            np.asarray(c, float),
            np.asarray(ax, float),
            hx,
            np.asarray(ay, float),
            hy,
        )
        self.nodepth = nodepth  # the depth sensor returns nothing here (dark glass)
        self.paint = (
            paint if callable(paint) else (lambda s, q, col=np.asarray(paint, np.uint8): np.tile(col, (len(s), 1)))  # noqa: B008
        )

    def hit(self, eye, rays):
        n = np.cross(self.ax, self.ay)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.c - eye) @ n) / (rays @ n)
        p = eye + t[:, None] * rays
        s, q = (p - self.c) @ self.ax, (p - self.c) @ self.ay
        ok = np.isfinite(t) & (t > 0) & (np.abs(s) <= self.hx) & (np.abs(q) <= self.hy)
        return np.where(ok, t, np.inf), s, q


def comfee(f, yaw_deg, w=sf.FRONT_W_M, h=sf.FRONT_H_M, d=0.34, window="left", colour=CREAM, no_depth=None):
    """The microwave as 4 rectangles. f = bottom centre of the front face; yaw 0 = facing -x (the arm).
    no_depth=(q0, q1): a band across the whole front (heights from its centre) with no depth return."""
    n = rotz(yaw_deg) @ np.array([-1.0, 0.0, 0.0])
    ex = np.cross(UP, n)  # right, seen from the front (panel_frame's x)
    colour = np.asarray(colour, np.uint8)

    def front(s, q):
        out = np.tile(colour, (len(s), 1))
        if window:
            x0 = -w / 2 + 0.05 if window == "left" else w / 2 - 0.30
            out[(s > x0) & (s < x0 + 0.25) & (np.abs(q) < h / 2 - 0.035)] = DARK
        out[(s - (w / 2 - 0.035)) ** 2 + (q - (h / 2 - 0.045)) ** 2 < 0.0275**2] = DARK  # the round display
        return out

    gap = []
    if no_depth:
        q0, q1 = no_depth
        gap = [Rect(f + (h / 2 + (q0 + q1) / 2) * UP + 0.001 * n, ex, w / 2, UP, (q1 - q0) / 2, front, nodepth=True)]
    return gap + [
        Rect(f + h / 2 * UP, ex, w / 2, UP, h / 2, front),
        Rect(f + h * UP - d / 2 * n, ex, w / 2, -n, d / 2, colour),
        Rect(f - w / 2 * ex + h / 2 * UP - d / 2 * n, -n, d / 2, UP, h / 2, colour),
        Rect(f + w / 2 * ex + h / 2 * UP - d / 2 * n, -n, d / 2, UP, h / 2, colour),
    ]


def front_truth(f, yaw_deg, w=sf.FRONT_W_M):
    """(normal out, start_30s xyz) the finder should report for comfee(f, yaw)."""
    n = rotz(yaw_deg) @ np.array([-1.0, 0.0, 0.0])
    ex = np.cross(UP, n)
    corner = f + w / 2 * ex
    dx, dy = np.asarray(sf.FACE_CORNER_TO_START_MM) / 1000.0
    return n, corner + dx * ex + dy * UP


def render(prims, T, K, wh, noise_m=0.0, seed=0):
    fx, fy, cx, cy = K
    w, h = wh
    us, vs = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rays = np.stack([(us.ravel() - cx) / fx, (vs.ravel() - cy) / fy, np.ones(w * h)], axis=1) @ T[:3, :3].T
    best, col = np.full(w * h, np.inf), np.zeros((w * h, 3), np.uint8)
    blind = np.zeros(w * h, bool)
    for pr in prims:
        t, s, q = pr.hit(T[:3, 3], rays)
        m = t < best
        best[m] = t[m]
        col[m] = pr.paint(s[m], q[m])
        blind[m] = pr.nodepth
    depth = np.where(np.isfinite(best) & ~blind, best, 0.0)  # camera-z of every ray is 1, so t is the depth
    if noise_m:
        depth = np.where(depth > 0, depth + np.random.default_rng(seed).normal(0.0, noise_m, depth.shape), 0.0)
    depth = np.round(depth * 1000.0) / 1000.0  # the driver's millimetres
    return depth.reshape(h, w).astype(np.float32), col.reshape(h, w, 3)


def scene(microwaves=(), extra=(), noise_m=0.002):
    """Rendered scene: table, back wall, the given microwaves [(f, yaw, kwargs)] and extra rectangles."""
    prims = [
        Rect((0.5, 0.0, TABLE_Z), (1, 0, 0), 50, (0, 1, 0), 50, TABLE),
        Rect((2.5, 0.0, 0.0), (0, 1, 0), 50, UP, 50, WALL),
    ]
    for f, yaw, kw in microwaves:
        prims += comfee(np.asarray(f, float), yaw, **kw)
    prims += list(extra)
    T_c = look_at(EYE, LOOK_AT)
    T_d = look_at(EYE + DEPTH_OFFSET, LOOK_AT + DEPTH_OFFSET)
    depth, _ = render(prims, T_d, DEPTH_K, DEPTH_WH, noise_m)
    _, bgr = render(prims, T_c, COLOR_K, COLOR_WH)
    return dict(bgr=bgr, depth_m=depth, color_K=COLOR_K, depth_K=DEPTH_K, T_base_color=T_c, T_base_depth=T_d)


def at(x, y):
    return np.array([x, y, TABLE_Z + FEET_M])


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0))))


# ---- synthetic -------------------------------------------------------------------------------


def test_finds_the_front_face_and_the_start_button():
    f = at(0.80, -0.06)
    fit, why = sf.find_microwave(**scene([(f, 0.0, {})]))
    assert fit is not None, why
    n, button = front_truth(f, 0.0)
    assert angle_deg(fit.n, n) < 3.0
    assert abs(fit.width_m - sf.FRONT_W_M) < 0.03 and abs(fit.height_m - sf.FRONT_H_M) < 0.03
    assert np.linalg.norm(fit.button - button) < 0.02, (fit.button, button)


@pytest.mark.parametrize(
    "x,y,yaw", [(0.65, -0.25, 0.0), (0.90, 0.05, 0.0), (0.75, -0.15, 20.0), (0.80, -0.05, -20.0), (0.70, 0.10, 25.0)]
)
def test_finds_a_moved_and_turned_microwave(x, y, yaw):
    f = at(x, y)
    fit, why = sf.find_microwave(**scene([(f, yaw, {})]))
    assert fit is not None, why
    n, button = front_truth(f, yaw)
    assert angle_deg(fit.n, n) < 4.0
    assert np.linalg.norm(fit.button - button) < 0.025, (fit.button, button)


def test_a_steeply_seen_plane_keeps_its_normals():
    """Seen 70 deg off its normal a flat wall's depth changes several cm across the normal stencil;
    that is a slope, not an edge (the 2026-10-14 bug that lost turned fronts)."""
    wall = Rect(at(0.80, -0.06) + 0.3 * UP, np.cross(UP, rotz(20) @ [-1.0, 0, 0]), 0.5, UP, 0.3, CREAM)
    sc = scene(extra=[wall], noise_m=0.0)
    T = sc["T_base_depth"]
    X, d, valid = sf._points(sc["depth_m"], sc["depth_K"], T, (0.0, 0.0, 0.0))
    N, ok = sf._normals(X, d, valid, T[:3, 3])
    on_wall = valid & (np.abs((X - wall.c) @ np.cross(wall.ax, wall.ay)) < 0.002)
    assert on_wall.sum() > 500
    assert ok[on_wall].mean() > 0.8


def test_a_front_split_by_missing_depth_is_still_one_front():
    """Live, 2026-10-14 (spot 2): the dark glass returns no depth and the frame around it is too thin
    for clean normals, so in 2 of 5 frames the front came apart into a 44 x 20 and a 43 x 10 cm
    piece, each too small, and the run fell back to the wrist camera."""
    f = at(0.80, -0.06)
    fit, why = sf.find_microwave(**scene([(f, 0.0, {"no_depth": (0.02, 0.05)})]))
    assert fit is not None, why
    _, button = front_truth(f, 0.0)
    assert abs(fit.height_m - sf.FRONT_H_M) < 0.03
    assert np.linalg.norm(fit.button - button) < 0.02, (fit.button, button)


def test_a_front_split_by_the_whole_window_is_still_one_front():
    """Live, 2026-10-14 (spot 3): the glass left a no-depth band as tall as the window, and the
    front came apart into a 46 x 12 and a 43 x 8 cm piece 8 cm apart (2 of 5 frames)."""
    f = at(0.80, -0.06)
    fit, why = sf.find_microwave(**scene([(f, 0.0, {"no_depth": (-0.06, 0.06)})]))
    assert fit is not None, why
    _, button = front_truth(f, 0.0)
    assert abs(fit.height_m - sf.FRONT_H_M) < 0.03
    assert np.linalg.norm(fit.button - button) < 0.02, (fit.button, button)


def test_coplanar_boxes_stacked_too_tall_are_not_one_front():
    """Joining pieces across a gap is bounded by the front's size: two microwave-wide boxes one
    above the other, 40 cm apart, stay two candidates and neither is the front."""
    low, high = at(0.80, -0.06), at(0.80, -0.06) + np.array([0.0, 0.0, 0.40])
    fit, why = sf.find_microwave(
        **scene([(low, 0.0, {"h": 0.12, "window": None}), (high, 0.0, {"h": 0.12, "window": None})])
    )
    assert fit is None
    assert "size" in why


def detector_box(f, yaw):
    """What a detector like YOLO reports: the microwave's pixel box in the colour image."""
    depth, _ = render(comfee(np.asarray(f, float), yaw), look_at(EYE, LOOK_AT), COLOR_K, COLOR_WH)
    vs, us = np.nonzero(depth > 0)
    return float(us.min()), float(vs.min()), float(us.max()), float(vs.max())


def test_a_coplanar_neighbour_is_left_out_by_the_detector_box():
    """Live 2026-10-15: the front grew sideways into something flat beside it on the same plane
    (129 x 32 cm, then 141 x 16) and the size gate refused it in every frame; with the YOLO
    node's box the same frames found it (3/3)."""
    f = at(0.80, -0.06)
    board = Rect(
        f + np.array([0.0, -sf.FRONT_W_M / 2 - 0.04 - 0.15, sf.FRONT_H_M / 2]),
        (0, 1, 0),
        0.15,
        UP,
        sf.FRONT_H_M / 2,
        TABLE,
    )  # 30 cm wide, 4 cm right of the front, same plane
    sc = scene([(f, 0.0, {})], extra=[board])
    fit, why = sf.find_microwave(**sc)
    assert fit is None and "size" in why, why
    fit, why = sf.find_microwave(**sc, roi=detector_box(f, 0.0))
    assert fit is not None, why
    _, button = front_truth(f, 0.0)
    assert abs(fit.width_m - sf.FRONT_W_M) < 0.03
    assert np.linalg.norm(fit.button - button) < 0.025, (fit.button, button)


@pytest.mark.parametrize("kw", [{"colour": TABLE}, {"window": None}, {"window": "right"}, {"w": 0.54}])
def test_inside_the_detector_box_only_the_shape_is_checked(kw):
    """YOLO has already said it is the microwave: colour, window side and a size 15-30 % off do not
    refuse it there (they do without the box)."""
    f = at(0.80, -0.06)
    sc = scene([(f, 0.0, kw)])
    assert sf.find_microwave(**sc)[0] is None
    fit, why = sf.find_microwave(**sc, roi=detector_box(f, 0.0))
    assert fit is not None, why


def test_inside_the_detector_box_a_far_wrong_size_is_still_refused():
    f = at(0.80, -0.06)
    sc = scene([(f, 0.0, {"h": 0.12, "window": None})])  # half height: its corner would misplace the button
    fit, why = sf.find_microwave(**sc, roi=detector_box(f, 0.0))
    assert fit is None and "size" in why


def test_a_box_elsewhere_finds_nothing_there():
    """The box only narrows the search: a box on empty table finds no front, it never invents one."""
    fit, why = sf.find_microwave(**scene([(at(0.80, -0.06), 0.0, {})]), roi=(1000.0, 600.0, 1100.0, 700.0))
    assert fit is None and "inside the detector's box" in why


def test_other_buttons_are_offset_by_the_dome_layout():
    f = at(0.80, -0.06)
    sc = scene([(f, 0.0, {})])
    start, _ = sf.find_microwave(**sc)
    timer, why = sf.find_microwave(**sc, target="timer_clock")
    assert timer is not None, why
    d = np.subtract(sf.BUTTON_FROM_DISPLAY_MM["timer_clock"], sf.BUTTON_FROM_DISPLAY_MM["start_30s"]) / 1000.0
    want = start.button + d[0] * start.ex + d[1] * start.ey
    assert np.linalg.norm(timer.button - want) < 1e-6


def test_a_box_of_the_wrong_size_is_not_the_microwave():
    fit, why = sf.find_microwave(**scene([(at(0.80, -0.06), 0.0, {"w": 0.30, "h": 0.17})]))
    assert fit is None
    assert "size" in why


def test_a_front_without_a_window_is_not_the_microwave():
    fit, why = sf.find_microwave(**scene([(at(0.80, -0.06), 0.0, {"window": None})]))
    assert fit is None
    assert "window" in why


def test_a_window_on_the_right_is_not_this_microwave():
    fit, why = sf.find_microwave(**scene([(at(0.80, -0.06), 0.0, {"window": "right"})]))
    assert fit is None
    assert "window" in why


def test_a_grey_box_is_not_the_cream_microwave():
    fit, why = sf.find_microwave(**scene([(at(0.80, -0.06), 0.0, {"colour": (200, 200, 200)})]))
    assert fit is None
    assert "cream" in why


def test_two_microwaves_are_refused_as_ambiguous():
    fit, why = sf.find_microwave(**scene([(at(0.85, -0.20), 0.0, {}), (at(0.65, 0.45), 0.0, {})]))
    assert fit is None
    assert "ambiguous" in why


def test_a_front_facing_away_from_the_arm_is_refused():
    fit, why = sf.find_microwave(**scene([(at(0.30, 0.25), -90.0, {})]))
    assert fit is None
    assert "away" in why


def test_a_floating_box_is_not_standing_on_anything():
    f = at(0.80, -0.06) + np.array([0.0, 0.0, 0.10])
    fit, why = sf.find_microwave(**scene([(f, 0.0, {})]))
    assert fit is None
    assert "support" in why


def test_a_partly_hidden_front_is_refused_and_the_arm_is_named():
    occluder_c = EYE + 0.45 * (np.array([0.80, -0.06, 0.13]) - EYE)  # between the camera and the face
    side = np.cross(UP, occluder_c - EYE)
    side /= np.linalg.norm(side)
    arm = Rect(occluder_c, side, 0.06, UP, 0.12, (90, 90, 90))
    sc = scene([(at(0.80, -0.06), 0.0, {})], extra=[arm])
    fit, why = sf.find_microwave(**sc)
    assert fit is None
    assert "arm in the way" not in why
    fit, why = sf.find_microwave(**sc, tool_xyz=occluder_c)
    assert fit is None
    assert "arm in the way?" in why


def test_an_empty_scene_says_so():
    fit, why = sf.find_microwave(**scene())
    assert fit is None
    assert why


# ---- recorded (tools/frames/scene/) ----------------------------------------------------------


def recordings(negative):
    if not SCENE_REC.is_dir():
        return []
    return sorted(
        p for p in SCENE_REC.iterdir() if (p / "meta.json").is_file() and p.name.startswith("neg_") == negative
    )


def load(rec):
    m = json.loads((rec / "meta.json").read_text())
    frames = []
    for c in sorted(rec.glob("*_color.png")):
        d = cv2.imread(str(c).replace("_color.png", "_depth.png"), cv2.IMREAD_UNCHANGED)
        frames.append(
            dict(
                bgr=cv2.imread(str(c)),
                depth_m=d.astype(np.float32) / 1000.0,
                color_K=m["color_K"],
                depth_K=m["depth_K"],
                T_base_color=np.array(m["T_base_color"]),
                T_base_depth=np.array(m["T_base_depth"]),
            )
        )
    return m, frames


@pytest.mark.parametrize("rec", recordings(False) or [None], ids=lambda r: r.name if r else "none")
def test_recorded_microwave_is_found_in_every_frame(rec):
    if rec is None:
        pytest.skip("no recorded scene frames")
    _, frames = load(rec)
    fits = [sf.find_microwave(**fr) for fr in frames]
    assert all(f is not None for f, _ in fits), [why for f, why in fits if f is None]
    buttons = np.array([f.button for f, _ in fits])
    assert np.max(np.linalg.norm(buttons - np.median(buttons, axis=0), axis=1)) < 0.02
    assert all(abs(f.n[2]) < 0.2 for f, _ in fits)


@pytest.mark.parametrize("rec", recordings(True) or [None], ids=lambda r: r.name if r else "none")
def test_recorded_negative_finds_nothing(rec):
    if rec is None:
        pytest.skip("no recorded negative scene frames")
    m, frames = load(rec)
    for fr in frames:
        fit, why = sf.find_microwave(**fr, tool_xyz=m.get("tool_xyz"))
        assert fit is None, rec.name
        if rec.name.startswith("neg_arm_"):
            assert "arm in the way?" in why


# start_30s's height measured with a ruler: 12-13 cm above the counter, which the scene camera puts
# at z -0.018 (2026-10-14, spot 2). The rough button must be within 2.5 cm of it: the first
# FACE_CORNER_TO_START_MM (from a pixel picked in the scene image) put it 5 cm low, and that staging
# pose was too low for cuRobo to reach.
RULER_START_Z = {"spot2_arm_home": 0.107}


@pytest.mark.parametrize(
    "rec",
    [r for r in recordings(False) if r.name.rsplit("_", 2)[0] in RULER_START_Z] or [None],
    ids=lambda r: r.name if r else "none",
)
def test_recorded_button_height_matches_the_ruler(rec):
    if rec is None:
        pytest.skip("no recording with a ruler-measured button")
    _, frames = load(rec)
    want = RULER_START_Z[rec.name.rsplit("_", 2)[0]]
    for fr in frames:
        fit, why = sf.find_microwave(**fr)
        assert fit is not None, why
        assert abs(fit.button[2] - want) < 0.025, (fit.button, want)
