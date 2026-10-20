"""Tests for the far-range dome-layout detector, on synthetic panels (no camera)."""

import cv2
import numpy as np

from rammp_button_press import dome_pattern as dp

FX = 606.0
RED = (30, 30, 200)  # BGR


def panel(z=0.4, roll_deg=0.0, centre=(320, 260), distractors=True, pitch_mm=19.3):
    """Red panel with the 5 domes drawn at the true layout for depth z. Returns (bgr, depth, truth px)."""
    img = np.full((480, 640, 3), RED, np.uint8)
    s = pitch_mm / 1000 * FX / z
    a = np.radians(roll_deg)
    ex, ey = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    truth = {}
    for name, (u, v) in zip(dp.NAMES, dp.LAYOUT, strict=False):
        p = np.array(centre) + s * (u * ex + v * ey)
        truth[name] = p
        cv2.circle(img, tuple(int(round(c)) for c in p), max(2, int(0.25 * s)), (225, 225, 225), -1)
    if distractors:
        # label text: small bright marks between and above the domes, at a smaller pitch
        for k in range(6):
            p = np.array(centre) + s * ((-1.2 + 0.45 * k) * ex + (-0.55) * ey)
            cv2.rectangle(img, tuple(int(c) for c in p - 1), tuple(int(c) for c in p + 1), (230, 230, 230), -1)
    return img, np.full((480, 640), z, np.float32), truth


def check(fit, truth, tol_px=1.5):
    assert fit is not None
    for name in dp.NAMES:
        assert np.linalg.norm(np.array(fit.px[name]) - truth[name]) < tol_px, name


def test_finds_and_names_all_five():
    img, d, truth = panel()
    fit = dp.detect(img, d, FX)
    check(fit, truth)
    assert abs(fit.s_mm - 19.3) < 1.0


def test_far_and_mid_range():
    for z in (0.3, 0.5, 0.7):
        img, d, truth = panel(z=z)
        check(dp.detect(img, d, FX), truth)


def test_rolled_camera_still_names_correctly():
    # 3 domes on top vs 2 below fixes "up": upside down, timer_clock is still the right end of the 3-row.
    for roll in (25.0, 180.0):
        img, d, truth = panel(roll_deg=roll)
        check(dp.detect(img, d, FX), truth)


def test_wrong_metric_size_rejected():
    # Same picture, but depth says it is half as far: the pitch measures ~9.7 mm, not a Comfee panel.
    img, _, _ = panel(z=0.4)
    assert dp.detect(img, np.full((480, 640), 0.2, np.float32), FX) is None


def test_too_close_abstains():
    img, d, _ = panel(z=0.4)
    fit = dp.fit_layout(dp.candidates(img, d, FX), FX)
    assert fit is not None
    # identical geometry reported at 18 cm is below MIN_RANGE_M -> detect() abstains
    img2, d2, _ = panel(z=0.18, centre=(320, 200))
    assert dp.detect(img2, d2, FX) is None


def test_four_domes_is_not_enough():
    img, d, truth = panel(distractors=False)
    p = truth["stop_eco"]
    cv2.circle(img, tuple(int(c) for c in p), 12, RED, -1)  # paint one dome out
    assert dp.detect(img, d, FX) is None


def test_button_xyz_on_plane():
    intr = (FX, FX, 320.0, 240.0)
    n, dd = np.array([0.0, 0.0, -1.0]), -0.4  # fronto-parallel plane at z = 0.4 (n toward camera)
    x = dp.button_xyz_cam((320.0 + 60.6, 240.0), n, dd, intr)
    np.testing.assert_allclose(x, [0.04, 0.0, 0.4], atol=1e-9)


def _layout_px(z=0.3, centre=(320, 260), pitch_mm=19.3):
    s = pitch_mm / 1000 * FX / z
    return {name: np.array(centre) + s * np.array([u, v]) for name, (u, v) in zip(dp.NAMES, dp.LAYOUT, strict=False)}, s


def test_fit_with_two_specks_rejected():
    # Three real domes plus two label-text specks in the layout: the sizes don't match, so no fit
    # (2026-10-12: such a fit put start_30s on the Timer/Clock button).
    img = np.full((480, 640, 3), RED, np.uint8)
    pts, s = _layout_px()
    for name, p in pts.items():
        c = tuple(int(round(v)) for v in p)
        if name in ("power_level", "stop_eco"):
            cv2.rectangle(img, (c[0] - 1, c[1] - 1), (c[0] + 1, c[1] + 1), (230, 230, 230), -1)
        else:
            cv2.circle(img, c, int(0.25 * s), (225, 225, 225), -1)
    assert dp.detect(img, np.full((480, 640), 0.3, np.float32), FX) is None


def _tilted_panel(tilt_deg, z0=0.32, cream=True, highlight_px=(0, -3)):
    """A panel turned tilt_deg about its vertical axis, with matching depth. Cream style: bright
    panel, each dome a chrome disc inside a dark rim, with a bright highlight off its centre."""
    h, w = 480, 640
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    t = np.radians(tilt_deg)
    ex, ey = np.array([np.cos(t), 0.0, np.sin(t)]), np.array([0.0, 1.0, 0.0])
    n = np.cross(ex, ey)  # away from the camera
    c3 = np.array([0.0, 0.01, z0])
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([(us - cx) / FX, (vs - cy) / FX, np.ones_like(us, float)], axis=2)
    depth = (float(c3 @ n) / (rays @ n)).astype(np.float32)
    img = np.full((h, w, 3), (205, 225, 235) if cream else RED, np.uint8)
    s = 0.0193
    truth = {}
    for name, (lu, lv) in zip(dp.NAMES, dp.LAYOUT, strict=False):
        p3 = c3 + s * (lu * ex + lv * ey)
        p = FX * p3[:2] / p3[2] + (cx, cy)
        truth[name] = p
        r = 0.32 * s * FX / p3[2]
        axes = lambda k, r=r: (max(1, int(round(k * r * np.cos(t)))), max(1, int(round(k * r))))  # noqa: E731
        c = tuple(int(round(v)) for v in p)
        cv2.ellipse(img, c, axes(1.0), 0, 0, 360, (95, 95, 95), -1)  # dark rim
        cv2.ellipse(img, c, axes(0.8), 0, 0, 360, (150, 150, 150), -1)  # chrome
        cv2.circle(img, (c[0] + highlight_px[0], c[1] + highlight_px[1]), 2, (250, 250, 250), -1)
    return img, depth, truth


def test_oblique_panel_is_fitted_on_its_plane():
    # 2026-10-13: seen 37 deg off straight-on the flat-in-the-image layout missed FIT_TOL.
    for tilt in (0, 25, 40, -40):
        img, depth, truth = _tilted_panel(tilt)
        check(dp.detect(img, depth, FX), truth, tol_px=2.5)


def test_centres_come_from_the_rims_not_the_highlights():
    # The highlight sits 3 px above every dome's centre; the reported centres are the rims'.
    img, depth, truth = _tilted_panel(0)
    check(dp.detect(img, depth, FX), truth, tol_px=1.5)


def test_gripper_range_spots_are_ignored():
    # A perfect layout 10 cm away (the gripper's own fingers) is not even a candidate.
    img, depth, _ = _tilted_panel(0, z0=0.10)
    assert dp.candidates(img, depth, FX) == []


def test_hinted_fit_matches_full_search_on_recorded_frames():
    # The hint only narrows the search (HINT_PAIR_FRAC): on real frames (cream panel, 31-49 cm) a
    # hinted fit must give what the full search gives (to 0.1 spacing, ~2 px: the hint crops the
    # image), and must not be found for a hint that points somewhere else.
    import glob  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import pytest  # noqa: PLC0415

    rec = Path(__file__).resolve().parents[1] / "tools" / "frames" / "rec"
    dirs = sorted(glob.glob(f"{rec}/*/"))
    if not dirs:
        pytest.skip("no recorded frames")
    n = 0
    for d in dirs:
        fx = float(np.load(d + "K.npy")[0])
        files = sorted(glob.glob(d + "*_color.png"))[:3]
        frames = [(cv2.imread(c), np.load(c.replace("_color.png", "_depth.npy"))) for c in files]
        hint = next((f for f in (dp.detect(b, z, fx) for b, z in frames) if f is not None), None)
        if hint is None:
            continue
        for bgr, depth in frames:
            full, hinted = dp.detect(bgr, depth, fx), dp.detect(bgr, depth, fx, hint=hint)
            if full is None:
                continue
            assert hinted is not None, d
            for name in dp.NAMES:
                assert np.hypot(*np.subtract(full.px[name], hinted.px[name])) < 0.1 * full.s_px, (d, name)
            n += 1
        away = dp.DomeFit(
            px={k: (u + 3 * hint.s_px, v) for k, (u, v) in hint.px.items()},
            s_px=hint.s_px,
            s_mm=hint.s_mm,
            err=0.0,
            z=hint.z,
            region=hint.region,
            r_px=hint.r_px,
        )
        assert dp.fit_layout(dp.candidates(*frames[0], fx), fx, away) is None, d
    assert n >= 10
