"""First rough look from the fixed scene camera: finds the microwave's front by its shape and places the button on it.

Uses the Orbbec Gemini on /scene_camera; no ROS in this module. From ~1.2 m the buttons are only
a few pixels apart, so it never looks for them: it finds the cream microwave's front face in the
depth image and puts the button at a stored offset from the front's corner. Good to a few cm,
enough to stage the arm in front of the panel, where the wrist camera's dome detector takes
over. A press is never planned from this.

How it works, on one frame:
  1. points    depth pixels -> base_link points (the depth has its own intrinsics and optical
               frame; it is not aligned to colour), cropped to the arm's surroundings
  2. surfaces  surface normals; horizontal ones give the support heights (table, counter),
               vertical ones are grouped into flat patches, each grown to its whole face
  3. gates     a patch is the front only if it faces the arm, measures FRONT_W_M x FRONT_H_M,
               stands on a support, has the dark door window on its LEFT and the light control
               strip on its right (sampled in the colour image), and is cream (Lab)
  4. result    the front's bottom-right corner + FACE_CORNER_TO_START_MM -> start_30s; the other
               buttons via the dome layout offsets. Two passing fronts -> refused as ambiguous

roi: a detector's box around the microwave in the colour image (the YOLO node's /yolo/detections,
2026-10-15) narrows where steps 2-3 look -- depth points that project outside it (padded by
ROI_PAD_PX) are left out, as box-opening does with its OWL boxes. Without it the front once grew
sideways into a coplanar neighbour (129 x 32 cm, 2026-10-15). The detector has already said it is the
microwave, so inside its box the gates are loose: the front must face the arm and be roughly the
right size (ROI_SIZE_TOL -- the button is placed from its corner, so a half front would misplace it);
the support, window, strip and colour checks are skipped, and of several fronts the largest wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

UP = np.array([0.0, 0.0, 1.0])
# Button centres from the display's centre, on the panel, mm: (x right, y up) seen from the front.
# Median of 12 frames in 5 views (dome detector + panel plane, 2026-10-13); views differed by up to
# ~1 cm, mostly in x (the blob's centre shifts with its reflections).
BUTTON_FROM_DISPLAY_MM = {
    "power_level": (-19.4, -127.8),
    "wgt_time_defrost": (0.8, -128.3),
    "timer_clock": (20.4, -127.8),
    "stop_eco": (-9.1, -148.8),
    "start_30s": (10.4, -148.6),
}

# ---- the cream Comfee's front, seen from the front (x right, y up)
# AS THIS FINDER MEASURES IT on scene depth, 2026-10-14 (spot 1, ~1.2 m, ~60 deg off the normal,
# 5 frames: 0.440-0.447 x 0.255-0.265 m). Tape-measure the real front to confirm.
FRONT_W_M = 0.445
FRONT_H_M = 0.26
FRONT_SIZE_TOL = 0.15  # fraction: a partly hidden front fails here
# start_30s from the front's bottom-right corner as this finder measures it, mm, x right / y up.
# x: the button's pixel picked in the scene image (spot 1, 5 frames: -24..-28). y: a ruler, 2026-10-14
# (spot 2): the button 12-13 cm above the counter, the finder's corner 1.3 cm above it -> 112. The
# pixel-picked y (63) was 5 cm low (a lower button row), and that staging pose was too low for cuRobo.
# The scene image itself reads the dome rows ~2 cm below 112. Refine both from staged runs with
# tools/scene_offset_from_logs.py.
FACE_CORNER_TO_START_MM = (-26.0, 112.0)
# The door window and the control strip, as fractions of the front's width from its left edge
# (spot 1: the window spans ~0.11-0.65, the strip ~0.80-1.0 with the round display in it).
WINDOW_BAND = (0.15, 0.60)
STRIP_BAND = (0.82, 0.98)
WINDOW_GREY_MAX = 90  # a "dark" sample (window glass 35-45, cream face ~150)
WINDOW_MIN_DARK = 0.40  # spot 1: 0.69
STRIP_MAX_DARK = 0.25  # spot 1: 0.09 (the display)
SAMPLE_STEP_M = 0.005  # face-rectangle sampling in the colour image
# Cream, in OpenCV 8-bit Lab of the light face samples (spot 1: L 164, a 125, b 135; a grey or
# white surface reads b ~128).
CREAM_L_MIN = 110
CREAM_A_MAX_DEV = 10
CREAM_B_RANGE = (2, 25)  # b - 128
LIGHT_GREY_MIN = 110  # samples brighter than this are face, not window/display/labels
# ---- points and surfaces
DOWNSAMPLE = 2
MIN_DEPTH_M = 0.15
CROP_RADIUS_M = 1.2  # from the arm base, horizontally
CROP_Z_M = (-0.3, 0.9)
NORMAL_STEP = 3  # grid cells each side for the normal's differences (12 px at DOWNSAMPLE 2)
NORMAL_JUMP_M = 0.03  # depth this far off the line through the stencil's ends = an edge, no normal
HORIZ_DEG = 10.0
VERT_DEG = 25.0
GROUP_DEG = 10.0
GROUP_PLANE_M = 0.015
MIN_PATCH_POINTS = 150  # grid points (DOWNSAMPLE 2): ~0.1 m^2 at 1.2 m
GROW_DEG = 30.0  # a patch grows into points on its plane facing its way, or with no normal (edges)
GROW_Q_M = 0.03  # ... within its height band +- this
GROW_GAP_M = 0.06  # ... sideways across gaps up to this (the door handle cuts off the control strip)
MIN_SUPPORT_POINTS = 150
SUPPORT_GAP_M = 0.04  # front's bottom edge above the support: the feet (spot 1: ~2.5 cm)
SUPPORT_BELOW_M = 0.01
MAX_PATCHES = 12
SAME_FRONT_M = 0.05  # two passing fits closer than this are the same front
ROI_SIZE_TOL = 0.30  # fraction, inside a detector box (FRONT_SIZE_TOL without one)
ROI_PAD_PX = 8  # a detector box is widened this much each side (~1.5 cm at 1.2 m). 25 px let a
# coplanar neighbour 4 cm off the front back in (test_scene_finder)


@dataclass
class MicrowaveFit:
    """The microwave front found in one frame (base frame) and the rough target button on it."""

    n: np.ndarray  # front normal, base frame, out of the front (toward the camera and the arm side)
    centre: np.ndarray  # centre of the front's rectangle
    ex: np.ndarray  # right along the front, seen from the front (panel_frame's x)
    ey: np.ndarray  # up along the front (panel_frame's y)
    width_m: float
    height_m: float
    corner_br: np.ndarray  # bottom-right corner, seen from the front
    button: np.ndarray  # rough xyz of the target button (staging only)
    window_dark: float
    strip_dark: float
    lab: tuple
    px_box: tuple  # the front's box in the colour image (u0, v0, u1, v1)
    n_points: int


def _points(depth_m, K, T, arm_base):
    """Grid of base-frame points (h, w, 3) and its valid mask, from every DOWNSAMPLE-th depth pixel."""
    d = np.asarray(depth_m, dtype=np.float64)[::DOWNSAMPLE, ::DOWNSAMPLE]
    fx, fy, cx, cy = K
    h, w = d.shape
    us, vs = np.meshgrid(np.arange(w) * DOWNSAMPLE, np.arange(h) * DOWNSAMPLE)
    X = np.stack([(us - cx) * d / fx, (vs - cy) * d / fy, d], axis=2) @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
    rel = X - np.asarray(arm_base, dtype=float)
    valid = (
        (d > MIN_DEPTH_M)
        & (np.hypot(rel[..., 0], rel[..., 1]) < CROP_RADIUS_M)
        & (X[..., 2] > CROP_Z_M[0])
        & (X[..., 2] < CROP_Z_M[1])
    )
    return X, d, valid


def _normals(X, d, valid, eye):
    """Unit normals on the grid (toward the camera at `eye`) and where they are trustworthy."""
    k = NORMAL_STEP
    h, w = d.shape
    N = np.zeros_like(X)
    ok = np.zeros((h, w), bool)
    c = (slice(k, h - k), slice(k, w - k))
    dx = X[k : h - k, 2 * k :] - X[k : h - k, : w - 2 * k]
    dy = X[2 * k :, k : w - k] - X[: h - 2 * k, k : w - k]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=2)
    n = n / np.maximum(nn, 1e-12)[..., None]
    # An edge is depth that leaves the straight line through its neighbours, not a steep slope: a
    # front seen 60-70 deg off its normal changes 6-7 cm across the stencil and is still flat.
    dc = d[c]
    jump = np.maximum(
        np.abs(d[k : h - k, 2 * k :] + d[k : h - k, : w - 2 * k] - 2 * dc),
        np.abs(d[2 * k :, k : w - k] + d[: h - 2 * k, k : w - k] - 2 * dc),
    )
    good = (
        valid[c]
        & valid[k : h - k, 2 * k :]
        & valid[k : h - k, : w - 2 * k]
        & valid[2 * k :, k : w - k]
        & valid[: h - 2 * k, k : w - k]
        & (nn > 1e-12)
        & (jump < NORMAL_JUMP_M)
    )
    flip = np.einsum("ijk,ijk->ij", n, eye - X[c]) < 0
    n[flip] *= -1.0
    N[c], ok[c] = n, good
    return N, ok


def _supports(X, N, ok):
    """Heights (m) of the horizontal surfaces in view: tables, counters."""
    horiz = ok & (np.abs(N[..., 2]) > np.cos(np.radians(HORIZ_DEG)))
    z = X[horiz][:, 2]
    if len(z) < MIN_SUPPORT_POINTS:
        return [], horiz
    hist, edges = np.histogram(z, bins=np.arange(CROP_Z_M[0], CROP_Z_M[1] + 0.01, 0.01))
    out = []
    for i in np.argsort(hist)[::-1]:
        if hist[i] < MIN_SUPPORT_POINTS:
            break
        zi = float(np.median(z[(z >= edges[i] - 0.01) & (z < edges[i + 1] + 0.01)]))
        if all(abs(zi - s) > 0.03 for s in out):
            out.append(zi)
    return out, horiz


def _patches(X, N, ok, horiz):
    """Flat vertical patches: (grid mask, unit normal) each, largest facing groups first."""
    vert = ok & ~horiz & (np.abs(N[..., 2]) < np.sin(np.radians(VERT_DEG)))
    left = vert.copy()
    out = []
    for _ in range(MAX_PATCHES * 3):
        if left.sum() < MIN_PATCH_POINTS or len(out) >= MAX_PATCHES:
            break
        az = np.degrees(np.arctan2(N[..., 1], N[..., 0]))
        hist, edges = np.histogram(az[left], bins=72, range=(-180, 180))
        hist = hist + np.roll(hist, 1) + np.roll(hist, -1)  # 15 deg windows, wrapping
        az0 = float(edges[int(np.argmax(hist))] + 2.5)
        same = left & (np.abs((az - az0 + 180.0) % 360.0 - 180.0) < GROUP_DEG)
        if same.sum() < MIN_PATCH_POINTS:
            left &= ~same
            continue
        n0 = N[same].mean(axis=0)
        n0 /= np.linalg.norm(n0)
        off = X @ n0
        offs = off[same]
        oh, oe = np.histogram(offs, bins=np.arange(offs.min() - 0.01, offs.max() + 0.02, 0.01))
        peak = float(oe[int(np.argmax(oh))] + 0.005)
        plane = same & (np.abs(off - peak) < GROUP_PLANE_M)
        if plane.sum() >= MIN_PATCH_POINTS:
            n_lab, lab = cv2.connectedComponents(plane.astype(np.uint8), connectivity=8)
            for i in range(1, n_lab):
                m = lab == i
                if m.sum() >= MIN_PATCH_POINTS:
                    out.append((m, n0))
            left &= ~plane
        else:
            left &= ~same
    out.sort(key=lambda p: -int(p[0].sum()))
    return out


def _fit_plane(P, eye):
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    if n @ (eye - c) < 0:
        n = -n
    return c, n


def _project(P, K, T):
    """Base-frame points -> colour pixels (u, v) and whether they are in front of the camera."""
    T = np.asarray(T)
    C = (np.asarray(P, dtype=float) - T[:3, 3]) @ T[:3, :3]
    fx, fy, cx, cy = K
    z = C[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([C[:, 0] / z * fx + cx, C[:, 1] / z * fy + cy], axis=1)
    return uv, z > 0.05


def _front(X, valid, N, ok, horiz, patch, eye, arm_base):
    """Grow a patch sideways along its own plane; the front's rectangle.

    The clean-normal patch is only part of the front: its edges lose ~2 cm to the normal stencil,
    and the protruding door handle cuts the narrow control strip off it (2026-10-14, spot 1). So
    every depth point on the patch's plane (GROUP_PLANE_M), facing its way or without a normal,
    within its height band (+-GROW_Q_M) counts, along the plane as far as the occupied stretch
    runs with gaps no wider than GROW_GAP_M. The height band keeps the growth off the line where
    the plane cuts the table (a 106 cm "front" when growth was unbounded)."""
    m, _ = patch
    c, n = _fit_plane(X[m], eye)
    ey = UP - (UP @ n) * n
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, n)
    R = X - c
    s_all, q_all = R @ ex, R @ ey
    q_lo, q_hi = np.percentile(q_all[m], [1, 99])
    s_m = s_all[m]
    cand = (
        valid
        & ~horiz
        & (~ok | (N @ n > np.cos(np.radians(GROW_DEG))))
        & (np.abs(R @ n) < GROUP_PLANE_M)
        & (q_all > q_lo - GROW_Q_M)
        & (q_all < q_hi + GROW_Q_M)
        & (s_all > s_m.min() - FRONT_W_M)
        & (s_all < s_m.max() + FRONT_W_M)
    )
    edges = np.arange(s_m.min() - FRONT_W_M, s_m.max() + FRONT_W_M + 0.01, 0.01)
    occ = np.histogram(s_all[cand], bins=edges)[0] >= 3
    gap = int(round(GROW_GAP_M / 0.01))
    i0 = i1 = int(np.clip(np.searchsorted(edges, np.median(s_m)) - 1, 0, len(occ) - 1))
    while True:  # extend over occupied bins, jumping gaps <= GROW_GAP_M
        nxt = [j for j in range(i1 + 1, min(len(occ), i1 + gap + 2)) if occ[j]]
        if not nxt:
            break
        i1 = nxt[0]
    while True:
        nxt = [j for j in range(i0 - 1, max(-1, i0 - gap - 2), -1) if occ[j]]
        if not nxt:
            break
        i0 = nxt[0]
    face = (cand & (s_all >= edges[i0]) & (s_all < edges[i1 + 1])) | m
    return _measure(X, face, eye, arm_base)


def _measure(X, face, eye, arm_base):
    """The rectangle of the depth points in `face` (grid mask): plane, axes, extents, corner."""
    P = X[face]
    c, n = _fit_plane(P, eye)
    keep = np.abs((P - c) @ n) < 0.01
    if keep.sum() >= MIN_PATCH_POINTS:
        P = P[keep]
        c, n = _fit_plane(P, eye)
    ey = UP - (UP @ n) * n
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, n)
    s, q = (P - c) @ ex, (P - c) @ ey
    s0, s1 = np.percentile(s, [1, 99])
    q0, q1 = np.percentile(q, [1, 99])
    centre = c + (s0 + s1) / 2 * ex + (q0 + q1) / 2 * ey
    faces_arm = float((np.asarray(arm_base, dtype=float) - centre)[:2] @ n[:2]) > 0
    return dict(
        face=face,
        n=n,
        ex=ex,
        ey=ey,
        centre=centre,
        w=float(s1 - s0),
        h=float(q1 - q0),
        corner_br=c + s1 * ex + q0 * ey,
        bottom_z=float((c + q0 * ey)[2]),  # ex is horizontal
        faces_arm=faces_arm,
        n_points=int(len(P)),
    )


def _coplanar(X, a, b, eye) -> bool:
    """Do two fronts lie on one plane? Their normals within GROUP_DEG, and one plane fitted to both
    pieces' points within GROUP_PLANE_M of each (median). Not each piece's own plane: a thin strip's
    normal tilts. Live 2026-10-14 (spot 2): the 10 cm strip below the window tilted 5-8 deg down,
    putting the strip above 1.6-2.7 cm off its plane, while one plane fits both to 2.4 mm."""
    if float(a["n"] @ b["n"]) < np.cos(np.radians(GROUP_DEG)):
        return False
    c, n = _fit_plane(X[a["face"] | b["face"]], eye)
    return all(np.median(np.abs((X[f["face"]] - c) @ n)) <= GROUP_PLANE_M for f in (a, b))


def _merge_fronts(X, fronts, eye, arm_base):
    """Join fronts that are pieces of one face: on one plane (_coplanar), and either no more than
    GROW_GAP_M apart along it both ways, or lined up (overlapping by half the narrower piece) one
    above the other / side by side with the joined span no bigger than the size gate allows. Live
    2026-10-14: the glass reads the microwave's inside (or nothing) and the frame around it is too
    thin for clean normals, so the front came apart above and below the window -- 44 x 20 + 43 x
    10 cm (spot 2), 46 x 12 + 43 x 8 cm 8 cm apart (spot 3), 43 x 8 + 45 x 10 cm (spot 2, in about
    half the frames). Two different objects joined by mistake make a front too big for the size gate."""
    fronts = list(fronts)
    merged = True
    while merged:
        merged = False
        for i in range(len(fronts)):
            for j in range(i + 1, len(fronts)):
                a, b = fronts[i], fronts[j]
                if not _coplanar(X, a, b, eye):
                    continue
                d = b["centre"] - a["centre"]
                dx, dy = abs(float(d @ a["ex"])), abs(float(d @ a["ey"]))
                gap_x, gap_y = dx - (a["w"] + b["w"]) / 2, dy - (a["h"] + b["h"]) / 2
                span_x, span_y = dx + (a["w"] + b["w"]) / 2, dy + (a["h"] + b["h"]) / 2
                stacked = -gap_x >= 0.5 * min(a["w"], b["w"]) and span_y <= (1 + FRONT_SIZE_TOL) * FRONT_H_M
                abreast = -gap_y >= 0.5 * min(a["h"], b["h"]) and span_x <= (1 + FRONT_SIZE_TOL) * FRONT_W_M
                if (gap_x <= GROW_GAP_M and gap_y <= GROW_GAP_M) or stacked or abreast:
                    fronts[i] = _measure(X, a["face"] | b["face"], eye, arm_base)
                    del fronts[j]
                    merged = True
                    break
            if merged:
                break
    return fronts


def _sample_face(fr, bgr, color_K, T_base_color):
    """Grey + Lab of the front's rectangle in the colour image, on a SAMPLE_STEP_M grid.
    Returns (grey (rows, cols), lab (rows, cols, 3), inside mask, px box)."""
    xs = np.arange(-fr["w"] / 2, fr["w"] / 2, SAMPLE_STEP_M)
    ys = np.arange(-fr["h"] / 2, fr["h"] / 2, SAMPLE_STEP_M)
    S, Q = np.meshgrid(xs, ys)
    P = fr["centre"] + S[..., None] * fr["ex"] + Q[..., None] * fr["ey"]
    uv, front = _project(P.reshape(-1, 3), color_K, T_base_color)
    h, w = bgr.shape[:2]
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    ui = np.clip(np.round(uv[:, 0]), 0, w - 1).astype(int)
    vi = np.clip(np.round(uv[:, 1]), 0, h - 1).astype(int)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[vi, ui].astype(float)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[vi, ui].astype(float)
    box = (
        (float(uv[inside, 0].min()), float(uv[inside, 1].min()), float(uv[inside, 0].max()), float(uv[inside, 1].max()))
        if inside.any()
        else (0.0, 0.0, 0.0, 0.0)
    )
    shape = S.shape
    return grey.reshape(shape), lab.reshape(shape + (3,)), inside.reshape(shape), box


def _band(grey, inside, band):
    cols = grey.shape[1]
    sl = slice(int(band[0] * cols), max(int(band[0] * cols) + 1, int(band[1] * cols)))
    g, m = grey[:, sl], inside[:, sl]
    return float(np.mean(g[m] < WINDOW_GREY_MAX)) if m.any() else float("nan")


def _gates(fr, supports, bgr, color_K, T_base_color, strict=True):
    """(why it is not the microwave's front or "", window_dark, strip_dark, lab medians, px box).
    strict=False (inside a detector's box): facing the arm and a looser size only; the rest is
    still measured for the log but never refuses."""
    if not fr["faces_arm"]:
        return "faces away from the arm", np.nan, np.nan, (), (0, 0, 0, 0)
    tol = FRONT_SIZE_TOL if strict else ROI_SIZE_TOL
    if abs(fr["w"] - FRONT_W_M) > tol * FRONT_W_M or abs(fr["h"] - FRONT_H_M) > tol * FRONT_H_M:
        return (
            f"size {fr['w']*100:.0f} x {fr['h']*100:.0f} cm, not {FRONT_W_M*100:.0f} x {FRONT_H_M*100:.0f}",
            np.nan,
            np.nan,
            (),
            (0, 0, 0, 0),
        )
    if not strict:
        grey, lab, inside, box = _sample_face(fr, bgr, color_K, T_base_color)
        light = inside & (grey > LIGHT_GREY_MIN)
        lab_med = tuple(float(v) for v in np.median(lab[light], axis=0)) if light.any() else ()
        return "", _band(grey, inside, WINDOW_BAND), _band(grey, inside, STRIP_BAND), lab_med, box
    gaps = [fr["bottom_z"] - s for s in supports]
    if not any(-SUPPORT_BELOW_M <= g <= SUPPORT_GAP_M for g in gaps):
        return (
            f"no support under it (bottom edge at z {fr['bottom_z']:.2f} m; surfaces at "
            f"{[round(s, 2) for s in supports]})",
            np.nan,
            np.nan,
            (),
            (0, 0, 0, 0),
        )
    grey, lab, inside, box = _sample_face(fr, bgr, color_K, T_base_color)
    if inside.mean() < 0.5:
        return "front leaves the colour image", np.nan, np.nan, (), box
    win, strip = _band(grey, inside, WINDOW_BAND), _band(grey, inside, STRIP_BAND)
    mirrored = (
        _band(grey, inside, (1 - WINDOW_BAND[1], 1 - WINDOW_BAND[0])),
        _band(grey, inside, (1 - STRIP_BAND[1], 1 - STRIP_BAND[0])),
    )
    if not (win >= WINDOW_MIN_DARK and strip <= STRIP_MAX_DARK):
        if mirrored[0] >= WINDOW_MIN_DARK and mirrored[1] <= STRIP_MAX_DARK:
            return "door window on the wrong side (right)", win, strip, (), box
        return f"no door window on the left (dark {win:.2f}, strip {strip:.2f})", win, strip, (), box
    light = inside & (grey > LIGHT_GREY_MIN)
    if light.sum() < 20:
        return "too few light face samples to check the colour", win, strip, (), box
    L, a, b = np.median(lab[light], axis=0)
    if not (L >= CREAM_L_MIN and abs(a - 128) <= CREAM_A_MAX_DEV and CREAM_B_RANGE[0] <= b - 128 <= CREAM_B_RANGE[1]):
        return f"not cream (Lab {L:.0f} {a:.0f} {b:.0f})", win, strip, (L, a, b), box
    return "", win, strip, (float(L), float(a), float(b)), box


def _in_roi(X, roi, color_K, T_base_color, shape):
    """Grid mask: the points whose colour pixel falls inside `roi` (u0, v0, u1, v1) widened by ROI_PAD_PX."""
    uv, front = _project(X.reshape(-1, 3), color_K, T_base_color)
    u0, v0, u1, v1 = roi
    p = ROI_PAD_PX
    with np.errstate(invalid="ignore"):
        m = front & (uv[:, 0] >= u0 - p) & (uv[:, 0] <= u1 + p) & (uv[:, 1] >= v0 - p) & (uv[:, 1] <= v1 + p)
    return m.reshape(shape)


def find_microwave(
    bgr,
    depth_m,
    color_K,
    depth_K,
    T_base_color,
    T_base_depth,
    target="start_30s",
    tool_xyz=None,
    arm_base=(0.0, 0.0, 0.0),
    roi=None,
) -> tuple[MicrowaveFit | None, str]:
    """Find the cream microwave's front in one scene frame (the module's entry point).

    depth_m is in metres in the depth camera's OWN optical frame (T_base_depth: base <- depth
    optical, 4x4); bgr is the colour image (T_base_color). K = (fx, fy, cx, cy). If tool_xyz is
    given and the arm stands in front of a candidate front, the reason adds "arm in the way?".
    roi = (u0, v0, u1, v1) colour pixels: look for the front only there (see the module docstring).
    Returns (MicrowaveFit, "") or (None, why).
    """
    if target not in BUTTON_FROM_DISPLAY_MM:
        raise ValueError(f"scene finder knows {tuple(BUTTON_FROM_DISPLAY_MM)}, not {target!r}")
    T_bd = np.asarray(T_base_depth, dtype=float)
    eye = T_bd[:3, 3]
    X, d, valid = _points(depth_m, depth_K, T_bd, arm_base)
    N, ok = _normals(X, d, valid, eye)
    supports, horiz = _supports(X, N, ok)
    where = ""
    if roi is not None:
        inside = _in_roi(X, roi, color_K, T_base_color, d.shape)
        valid, ok = valid & inside, ok & inside
        where = " inside the detector's box"
    patches = _patches(X, N, ok, horiz)
    if not patches:
        return None, "no flat vertical surface in reach" + where
    fronts, used = [], np.zeros(d.shape, bool)
    for patch in patches:
        if (patch[0] & used).sum() > 0.5 * patch[0].sum():
            continue  # part of a front already grown
        fronts.append(_front(X, valid, N, ok, horiz, patch, eye, arm_base))
        used |= fronts[-1]["face"]
    passed, rejected = [], []
    for fr in _merge_fronts(X, fronts, eye, arm_base):
        why, win, strip, lab, box = _gates(fr, supports, bgr, color_K, T_base_color, strict=roi is None)
        fr.update(window_dark=win, strip_dark=strip, lab=lab, px_box=box)
        (rejected if why else passed).append((fr, why))
    uniq = []
    for fr, _ in passed:
        if all(np.linalg.norm(fr["centre"] - u["centre"]) > SAME_FRONT_M for u in uniq):
            uniq.append(fr)
    if len(uniq) > 1 and roi is not None:
        uniq = [max(uniq, key=lambda u: u["n_points"])]  # the detector named the microwave: its biggest front
    if len(uniq) > 1:
        return None, (
            f"ambiguous: {len(uniq)} microwave-like fronts, at "
            + ", ".join(str(np.round(u["centre"], 2)) for u in uniq)
        )
    if not uniq:
        why = f"no microwave front{where}: " + "; ".join(
            f"{fr['w']*100:.0f} x {fr['h']*100:.0f} cm at {np.round(fr['centre'], 2)}: {r}"
            for fr, r in sorted(rejected, key=lambda x: -x[0]["n_points"])[:3]
        )
        if tool_xyz is not None and _tool_in_a_front(tool_xyz, rejected, color_K, T_base_color):
            why += " -- arm in the way?"
        return None, why
    fr = uniq[0]
    dx, dy = np.asarray(FACE_CORNER_TO_START_MM, dtype=float) / 1000.0
    rel = (np.asarray(BUTTON_FROM_DISPLAY_MM[target]) - np.asarray(BUTTON_FROM_DISPLAY_MM["start_30s"])) / 1000.0
    button = fr["corner_br"] + (dx + rel[0]) * fr["ex"] + (dy + rel[1]) * fr["ey"]
    return MicrowaveFit(
        n=fr["n"],
        centre=fr["centre"],
        ex=fr["ex"],
        ey=fr["ey"],
        width_m=fr["w"],
        height_m=fr["h"],
        corner_br=fr["corner_br"],
        button=button,
        window_dark=fr["window_dark"],
        strip_dark=fr["strip_dark"],
        lab=fr["lab"],
        px_box=fr["px_box"],
        n_points=fr["n_points"],
    ), ""


def start_offset_mm(corner_br, ex, ey, button_xyz, target="start_30s"):
    """Inverse of find_microwave's placement, for refining FACE_CORNER_TO_START_MM from logs.

    Given a precise button xyz (the wrist camera's dome fix), returns start_30s's offset from the
    front's bottom-right corner, (x right, y up) mm: what FACE_CORNER_TO_START_MM should have
    been for that front."""
    rel = np.asarray(button_xyz, dtype=float) - np.asarray(corner_br, dtype=float)
    d = np.asarray(BUTTON_FROM_DISPLAY_MM[target]) - np.asarray(BUTTON_FROM_DISPLAY_MM["start_30s"])
    return float(rel @ np.asarray(ex, dtype=float) * 1000.0 - d[0]), float(
        rel @ np.asarray(ey, dtype=float) * 1000.0 - d[1]
    )


def _tool_in_a_front(tool_xyz, rejected, color_K, T_base_color) -> bool:
    """Does the tool project into the colour-image box of a candidate front that faces the arm?"""
    uv, front = _project(np.asarray(tool_xyz, dtype=float)[None, :], color_K, T_base_color)
    if not front[0]:
        return False
    u, v = uv[0]
    for fr, _ in rejected:
        if not fr["faces_arm"]:
            continue
        u0, v0, u1, v1 = fr["px_box"]
        if u1 <= u0:  # never sampled (failed before the colour check): box of its rectangle corners
            corners = [
                fr["centre"] + sx * fr["w"] / 2 * fr["ex"] + sy * fr["h"] / 2 * fr["ey"]
                for sx in (-1, 1)
                for sy in (-1, 1)
            ]
            cuv, _ = _project(np.array(corners), color_K, T_base_color)
            u0, v0 = cuv.min(axis=0)
            u1, v1 = cuv.max(axis=0)
        if u0 <= u <= u1 and v0 <= v <= v1:
            return True
    return False
