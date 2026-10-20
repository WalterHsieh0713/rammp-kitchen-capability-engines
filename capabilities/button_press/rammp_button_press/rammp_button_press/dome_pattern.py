"""Main button detector: finds the microwave's 5 chrome button domes by their 3+2 layout.

Uses the wrist camera's grey image + aligned depth only (no colour, no reference images, no ROS),
so any panel colour works. press_button uses it for the first look (~22-50 cm) and for the
measurement at the staging pose (~25 cm) that the press is planned from.

How it works:
  1. candidates  dome-like spots: a ring darker than its inside or the panel around it, tried
                 at several ring sizes, each kept only where that size fits a dome at that depth
  2. layout fit  each candidate pair is tried as the top-middle/top-right domes; the other 3 are
                 predicted from the known layout on the local panel plane (so oblique views
                 work). A fit must hit all 5, with similar sizes, at one depth, and measure
                 DOME_SPACING_MM apart; label text and highlights fail one of these. 3 domes on
                 top vs 2 below fixes "up", so a rolled camera still names the buttons right
  3. refine      the 5 centres are moved from the domes' off-centre highlights to their true
                 centres, using each dome's edge ring and dark rim
  4. 3D          the panel plane is fitted to depth around the domes (not on them: chrome gives
                 bad depth) and the target pixel's ray is intersected with it

Range: abstains closer than MIN_RANGE_M (a dome breaks into several highlights and the fit can
lock onto label text) and beyond MAX_RANGE_M. On the Jetson: ~0.25 s per frame from scratch,
~10-20 ms with `hint` (the previous frame's fit), so a 7-frame measurement takes ~0.35 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from rammp_button_press.geometry import fit_plane_inverse_depth, pixel_ray

# Physical top-row pitch on the Comfee panel (measured 19.3 mm at 39 cm, 2026-10-03).
DOME_SPACING_MM = (16.8, 21.8)
NAMES = ("power_level", "wgt_time_defrost", "timer_clock", "stop_eco", "start_30s")
# Dome centres in units of the top-row spacing, origin = top-middle dome, x right, y DOWN the panel.
LAYOUT = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [-0.58, 0.92], [0.42, 0.92]])
FIT_TOL = 0.2  # max dome miss, fraction of s (true fits: <= 0.17 on the test frames)
MIN_RANGE_M = 0.22  # closer than this, domes break into several highlights: abstain
MAX_RANGE_M = 1.2
# Blob size check, on the equivalent radius sqrt(area / pi). On 2026-10-12 (RAMMP Jetson), from a
# close view (~24 cm) a 3+2 fit using two specks in the "Timer/Clock" label (r 1.5 px) and three
# real domes (r 8.3-8.9 px) beat the true layout on 12/12 frames, putting start_30s on the
# Timer/Clock button: smallest/largest 0.17. True fits: 0.93 at 24 cm, 0.66 from 48 cm at a steep
# angle (where a dome is only its highlight, r 1.5-2.3 px). A size floor relative to the spacing
# (0.08 s) was tried and rejected those oblique real domes (0.061 s). Size only, no roundness.
MIN_DOME_SIZE_RATIO = 0.6  # the 5 spots are the same size: smallest >= this x the largest
# Dome candidates by SHAPE (grey image only): a dark ring of radius r with a brighter band outside.
NOMINAL_SPACING_M = 0.0198  # sets the ring sizes searched; the layout fit still MEASURES the spacing
RING_FRAC_OF_SPACING = (0.07, 0.32)  # dome ring radius / spacing: 0.18 close (24 cm), ~0.11 far (48 cm)
RING_RADII_PX = (
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    14,
    16,
)  # every size a dome takes from ~60 cm to ~22 cm (2 px: label specks pass as domes)
DOME_Z_SPREAD_M = 0.04  # the 5 domes are on one flat panel: their depths agree to this
RING_DARK = (0.8, 1.15)  # dark annulus, in units of r
RING_OUT = (1.3, 1.7)  # the panel just outside it
RING_IN = 0.5  # the dome's inside, up to this radius
RING_MIN_CONTRAST = 35.0  # grey levels the ring must be darker by (a 3 px speck in an 8 px ring: ~27)
RING_REF_LEVEL = 225.0  # the light panel's grey level RING_MIN_CONTRAST was set at
RING_LIGHT_PANEL = 120.0  # panel (just outside the ring) at least this grey = a light panel
RING_MIN_CONTRAST_SCALE = 0.8  # the least the threshold scales to (28), light panels only
RING_MIN_STD = 4.0  # floor on the local contrast the score is divided by
RING_MIN_SCORE = 0.7  # (brighter side - ring) / local std; true domes scored 1.2-2.0
RING_MAX = 60  # strongest peaks kept (the layout fit is O(n^2))
# Highlight -> dome centre (_edge_shift): a dome's edge ring is this much larger than the fitted
# spot and within this fraction of the spacing of it (cream panel 2026-10-13: highlights r 3-4 px,
# edges r 7 px, 4-5 px apart at a 28 px spacing).
EDGE_MIN_SIZE_RATIO = 1.5
EDGE_SEARCH_FRAC = 0.3
EDGE_MIN_DOMES = 4  # of the 5 (one edge may be lost to the label text beside it); fewer but not 0: no fit
EDGE_AGREE_FRAC = 0.12
# Nothing nearer than this is a dome candidate: the gripper's own fingertips are always in view
# ~10 cm from the camera, and on 2026-10-13 their screws and tape fitted the 3+2 layout ("TOO
# CLOSE: domes at 10.8 cm" while the panel was 55 cm away). The panel itself is refused below
# MIN_RANGE_M anyway; between the two a too-close panel is still found, so it is still reported.
MIN_CAND_RANGE_M = 0.15
# Local panel plane around a candidate (camera frame, from depth), so the layout is predicted ON
# the panel and projected, not laid out flat in the image: seen 37 deg off straight-on the rows
# close up by 20% and the flat prediction missed FIT_TOL (0/6 frames, 2026-10-13).
PLANE_WIN_FRAC = 1.6  # half-window, in dome spacings at the candidate's depth
PLANE_GRID = 8  # depth samples per half-window
PLANE_BAND_M = 0.08  # depths further than this from the candidate's are not its panel
PLANE_TRIM_M = 0.01  # refit once without samples this far off the first fit
PLANE_MIN_PTS = 30
PLANE_MAX_TILT_COS = 0.26  # a "panel" seen more than ~75 deg off straight-on is not trusted: flat fit
# Dome centre from its RIM (detect): the dark gap around a button is a circle of fixed size on the
# panel whatever the chrome reflects. Projected through the fitted plane (an ellipse when the view
# is oblique), scored as panel-just-outside minus rim, averaged round the circle.
RIM_FRAC = (0.27, 0.31, 0.35)  # rim radius / dome spacing tried (cream panel: ~0.32)
RIM_OUT = 1.35  # the panel ring, x the rim radius
RIM_ANGLES = 32
RIM_SEARCH_FRAC = 0.3  # centre searched this far (x spacing) around the fitted spot
RIM_MIN_CONTRAST = 12.0  # grey levels the rim must be darker than the panel round it
RIM_MIN_DOMES = 4  # of the 5 must show a rim, and agree (EDGE_AGREE_FRAC), or no change
# Region (layout units) whose depth pixels the panel plane is fitted to: the button block and the
# panel around it, not the knob/display above or the microwave's bottom edge.
PLANE_REGION = np.array([[-2.2, -1.2], [2.2, -1.2], [2.2, 2.0], [-2.2, 2.0]])
# With a hint (the previous frame's fit), the layout fit tries only pairs whose top-middle and
# top-right spots are this close (x the spacing) to the hint's: a fit anywhere else is refused by
# fit_layout's hint check anyway, but trying every pair made a hinted frame cost as much as a
# full one (~0.6 s per frame on the loaded Jetson, so ~5 s for the staging re-detect, 2026-10-15).
# Wide enough for the highlight -> centre shift (edge + rim, ~0.15-0.3 s) between the two.
HINT_PAIR_FRAC = 0.5


@dataclass
class DomeFit:
    """One frame's detection: the 5 dome pixels plus what the fit measured."""

    px: dict  # name -> (u, v)
    s_px: float
    s_mm: float
    err: float  # mean dome miss / s
    z: float  # median depth at the domes (m)
    region: np.ndarray  # 4x2 image polygon for the plane fit
    score: float = 0.0  # weakest of the 5 ring strengths (how dome-like the set is)
    r_px: float = 0.0  # ring radius the layout was found at
    plane: tuple = None  # (n, d, cx, cy): local panel plane n.X = d (camera frame, n away from the
    # camera) the layout was fitted on, and the image centre used for rays


def _box_sum(img, h):
    """Sum over the (2h+1)^2 square centred on every pixel (zero outside the image)."""
    k = 2 * int(h) + 1
    return cv2.boxFilter(img, cv2.CV_32F, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)


def _ring_scores(gray, gray2, r):
    """How much darker a SQUARE ring at radius r is than the dome's inside OR the panel outside
    it, over the local contrast. Squares (integral image: the same cost at every size) instead of
    circles: the formation checks are what make a match strict. Measured 2026-10-12 (grey levels):
    red panel inside 196 / ring 73 / panel 90; cream 196 / 149 / 225 -- whichever side is
    brighter counts."""
    h_in, h_r0, h_r1, h_o0, h_o1 = (
        max(0, int(round(f * r))) for f in (RING_IN, RING_DARK[0], RING_DARK[1], RING_OUT[0], RING_OUT[1])
    )
    h_r1 = max(h_r1, h_in + 1)
    h_r0 = min(h_r0, h_r1 - 1)
    h_o0 = max(h_o0, h_r1)
    h_o1 = max(h_o1, h_o0 + 1)  # every band at least one pixel wide
    S = {h: _box_sum(gray, h) for h in {h_in, h_r0, h_r1, h_o0, h_o1}}
    area = {h: float((2 * h + 1) ** 2) for h in S}
    mi = S[h_in] / area[h_in]
    mr = (S[h_r1] - S[h_r0]) / (area[h_r1] - area[h_r0])
    mo = (S[h_o1] - S[h_o0]) / (area[h_o1] - area[h_o0])
    mean = S[h_o1] / area[h_o1]
    std = np.sqrt(np.maximum(_box_sum(gray2, h_o1) / area[h_o1] - mean * mean, 1.0))
    diff = np.maximum(mi - mr, mo - mr)
    # A real ring is darker by a real amount (domes: 45-120 grey levels), not merely relative to a
    # flat background's tiny contrast -- a speck in a big ring on a flat panel would score high.
    # ... scaled down for a LIGHT panel in a dimmer exposure, where every contrast shrinks with the
    # image: at panel grey 160 instead of 225 two real domes read 33-34 and the layout was lost
    # (cream panel, 40 cm, 2026-10-13). A dark panel (red: ~90) keeps the full threshold -- there
    # the lower one let weak spots on the domes' edges join two label specks into a layout.
    scale = np.where(mo >= RING_LIGHT_PANEL, np.clip(mo / RING_REF_LEVEL, RING_MIN_CONTRAST_SCALE, 1.0), 1.0)
    need = RING_MIN_CONTRAST * scale
    return np.where(diff >= need, diff / np.maximum(std, RING_MIN_STD), 0.0).astype(np.float32)


def _local_plane(depth_m, u, v, z, fx, cx, cy):
    """(nx, ny, nz, d): plane n.X = d (camera frame, n pointing away from the camera) through the
    depth around pixel (u, v), by least squares on inverse depth (a plane is linear in 1/z; see
    geometry.fit_plane_inverse_depth). Falls back to the fronto-parallel plane at depth z."""
    flat = (0.0, 0.0, 1.0, float(z))
    h_img, w_img = depth_m.shape
    w = int(max(6, PLANE_WIN_FRAC * fx * NOMINAL_SPACING_M / z))
    step = max(1, w // PLANE_GRID)
    xs = np.arange(max(0, u - w), min(w_img, u + w + 1), step)
    ys = np.arange(max(0, v - w), min(h_img, v + w + 1), step)
    patch = depth_m[np.ix_(ys, xs)]
    U, V = np.meshgrid(xs, ys)
    ok = (patch > 0.1) & (patch < 2.0) & (np.abs(patch - z) < PLANE_BAND_M)
    if ok.sum() < PLANE_MIN_PTS:
        return flat
    A = np.stack([(U[ok] - cx) / fx, (V[ok] - cy) / fx, np.ones(int(ok.sum()))], axis=1)
    zz = patch[ok].astype(np.float64)
    m, *_ = np.linalg.lstsq(A, 1.0 / zz, rcond=None)
    keep = np.abs(zz - 1.0 / np.maximum(A @ m, 1e-6)) <= PLANE_TRIM_M
    if PLANE_MIN_PTS <= keep.sum() < len(zz):
        m, *_ = np.linalg.lstsq(A[keep], 1.0 / zz[keep], rcond=None)
    nrm = float(np.linalg.norm(m))
    if not np.isfinite(nrm) or nrm < 1e-9 or m[2] / nrm < PLANE_MAX_TILT_COS:
        return flat
    return (float(m[0] / nrm), float(m[1] / nrm), float(m[2] / nrm), 1.0 / nrm)


def candidates(bgr, depth_m, fx, hint=None):
    """Dome-shaped spots in the grey image: a ring darker than its inside or the panel around it.

    Each RING_RADII_PX size is kept only where it fits a dome at that pixel's depth, and every
    response at every size is kept (fit_layout picks a size-consistent set). `hint` (the previous
    frame's DomeFit) limits the search to around it.

    Returns [(u, v, z, r, score, nx, ny, nz, d, cx, cy)]: pixel, depth, ring radius, ring
    strength, the local panel plane n.X = d at the spot, and the image centre used for rays.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h_img, w_img = gray.shape
    cx, cy = (w_img - 1) / 2.0, (h_img - 1) / 2.0  # detect() is only given fx
    x_off = y_off = 0
    radii = RING_RADII_PX
    near = None
    if hint is not None:
        near = (np.asarray(hint.px[NAMES[1]]), HINT_PAIR_FRAC * hint.s_px)  # only these spots can be pair spot i
        pad = 1.5 * hint.s_px
        us = [p[0] for p in hint.px.values()]
        vs = [p[1] for p in hint.px.values()]
        x_off, y_off = int(max(0, min(us) - pad)), int(max(0, min(vs) - pad))
        x_end, y_end = int(min(w_img, max(us) + pad)), int(min(h_img, max(vs) + pad))
        gray, depth_m = gray[y_off:y_end, x_off:x_end], depth_m[y_off:y_end, x_off:x_end]
        h_img, w_img = gray.shape
        radii = [
            r for r in RING_RADII_PX if hint.r_px - 3 <= r <= 2.6 * hint.r_px + 1
        ]  # the spot's size up to its edge ring
    valid = depth_m[(depth_m > MIN_CAND_RANGE_M) & (depth_m < 2.0)]
    if valid.size < 100:
        return []
    z_lo, z_hi = (float(np.percentile(valid, q)) for q in (2, 98))
    r_lo = RING_FRAC_OF_SPACING[0] * fx * NOMINAL_SPACING_M / max(z_hi, MIN_RANGE_M)
    r_hi = RING_FRAC_OF_SPACING[1] * fx * NOMINAL_SPACING_M / max(z_lo, MIN_CAND_RANGE_M)
    gray2 = gray * gray
    out, planes = [], {}
    for r in radii:
        if not r_lo - 1 <= r <= r_hi + 1:  # no dome at any depth in view is this size
            continue
        nms = max(3, int(r / RING_FRAC_OF_SPACING[1] * 0.3) | 1)  # ~0.3 dome spacing at this size
        sc = _ring_scores(gray, gray2, r)
        peaks = (sc >= RING_MIN_SCORE) & (sc == cv2.dilate(sc, np.ones((nms, nms), np.uint8)))
        vs, us = np.nonzero(peaks)
        for i in np.argsort(-sc[vs, us])[:RING_MAX]:
            v, u = int(vs[i]), int(us[i])
            x0, x1 = int(max(0, u - 2 * r)), int(min(w_img, u + 2 * r + 1))
            y0, y1 = int(max(0, v - 2 * r)), int(min(h_img, v + 2 * r + 1))
            patch = depth_m[y0:y1, x0:x1]
            patch = patch[(patch > 0.1) & (patch < 2.0)]
            if patch.size < 5:
                continue
            z = float(np.median(patch))
            if z < MIN_CAND_RANGE_M:  # the gripper's own fingers
                continue
            frac = r / (fx * NOMINAL_SPACING_M / z)  # ring radius / dome spacing at ITS depth
            if not RING_FRAC_OF_SPACING[0] <= frac <= RING_FRAC_OF_SPACING[1]:
                continue
            if (u, v) not in planes:
                if near is not None and np.hypot(u + x_off - near[0][0], v + y_off - near[0][1]) >= near[1]:
                    planes[(u, v)] = (0.0, 0.0, 1.0, z)  # never pair spot i in a hinted fit: plane unused
                else:
                    planes[(u, v)] = _local_plane(depth_m, u, v, z, fx, cx - x_off, cy - y_off)
            out.append((float(u + x_off), float(v + y_off), z, float(r), float(sc[v, u]), *planes[(u, v)], cx, cy))
    # Every response is kept, at every ring size (2026-10-13, cream panel at HERL): giving each spot
    # ONE size made a dome whose edge ring is weakened by the label text next to it take its small
    # highlight's size, and the 5 then failed MIN_DOME_SIZE_RATIO -- 0/6 and 5/6 frames from the
    # same pose 90 min apart. fit_layout takes the size-consistent set.
    return out


def fit_layout(cands, fx, hint=None):
    """Best fit of the 5-dome LAYOUT to the candidate spots, or None.

    With `hint`, None unless the fit is the one the previous frame found (timer_clock within
    0.3 spacing of it)."""
    near = None if hint is None else (hint.px[NAMES[1]], hint.px[NAMES[2]], HINT_PAIR_FRAC * hint.s_px)
    fit = _fit_layout_one(cands, fx, near)
    if (
        fit is not None
        and hint is not None
        and np.hypot(*np.subtract(fit.px[NAMES[2]], hint.px[NAMES[2]])) >= 0.3 * hint.s_px
    ):
        return None
    return fit


def _fit_layout_one(cands, fx, near=None):
    """Try every candidate pair as the top-middle/top-right domes and keep the most dome-like
    5-dome set that passes every check (see fit_layout). `near` = (top-middle px, top-right px,
    radius px): only pairs within radius of those two (HINT_PAIR_FRAC)."""
    if len(cands) < 5:
        return None
    P = np.array([[c[0], c[1]] for c in cands])
    Z = np.array([c[2] for c in cands])
    Rb = np.array([c[3] for c in cands])
    Sc = np.array([c[4] if len(c) > 4 else 0.0 for c in cands])
    if len(cands[0]) >= 11:  # from candidates(): local planes + image centre
        N = np.array([c[5:8] for c in cands])
        Dp = np.array([c[8] for c in cands])
        c0 = np.array(cands[0][9:11])
    else:  # bare (u, v, z, r[, score]): a flat panel facing the camera
        N = np.tile([0.0, 0.0, 1.0], (len(cands), 1))
        Dp, c0 = Z.copy(), np.zeros(2)
    rays = np.column_stack([(P - c0) / fx, np.ones(len(P))])
    best = None
    idx, j_ok = range(len(P)), np.ones(len(P), bool)
    if near is not None:
        idx = np.nonzero(np.linalg.norm(P - np.asarray(near[0]), axis=1) < near[2])[0]
        j_ok = np.linalg.norm(P - np.asarray(near[1]), axis=1) < near[2]
    for i in idx:
        n = N[i]
        X = rays * (Dp[i] / (rays @ n))[:, None]  # every spot's ray on spot i's panel plane
        d3 = np.linalg.norm(X - X[i], axis=1)
        # Only pairs whose spacing measures like the top row, in metres ON the panel.
        for j in np.nonzero((d3 >= DOME_SPACING_MM[0] / 1000.0) & (d3 <= DOME_SPACING_MM[1] / 1000.0) & j_ok)[0]:
            s3 = float(d3[j])
            ex = (X[j] - X[i]) / s3
            ey = np.array(
                [n[1] * ex[2] - n[2] * ex[1], n[2] * ex[0] - n[0] * ex[2], n[0] * ex[1] - n[1] * ex[0]]
            )  # n x ex: panel "down" (np.cross is slow per call)
            pred3 = X[i] + s3 * (LAYOUT[:, :1] * ex + LAYOUT[:, 1:] * ey)
            pred = fx * pred3[:, :2] / pred3[:, 2:3] + c0
            s = fx * s3 / float(X[i, 2])  # the spacing in pixels at that depth, seen straight on
            dist = np.linalg.norm(P[None] - pred[:, None], axis=2)
            # Per predicted dome, of the spots within FIT_TOL take the one whose SIZE matches the
            # pair's own (a dome also carries small highlight spots); nearest breaks ties.
            r_ref = np.sqrt(Rb[i] * Rb[j])
            inside = dist < FIT_TOL * s
            if not inside.any(axis=1).all():
                continue
            cost = np.where(inside, np.abs(np.log(Rb[None] / r_ref)) + 0.01 * dist / s, np.inf)
            k = cost.argmin(1)
            e = dist[np.arange(5), k] / s
            if len(set(k)) < 5:
                continue
            r = Rb[k]
            if r.min() < MIN_DOME_SIZE_RATIO * r.max():
                continue
            if (np.abs(Z[k] - pred3[:, 2]) > DOME_Z_SPREAD_M).any():  # all 5 on that one panel
                continue
            # Small spots are either all dome highlights (their domes' edge rings around them) or
            # all whole domes seen small (no larger ring). Some with an edge and some without is
            # domes mixed with label specks: 3 domes + 2 specks, the 2026-10-12 wrong-button fit.
            n_edge = len(_edge_offsets(P[k], float(np.median(r)), s, P, Rb, Sc))
            if 0 < n_edge < EDGE_MIN_DOMES:
                continue
            # most dome-like set first (its weakest ring), geometry error to break ties
            key = (float(Sc[k].min()), -float(e.mean()))
            if best is None or key > best[0]:
                reg3 = X[i] + s3 * (PLANE_REGION[:, :1] * ex + PLANE_REGION[:, 1:] * ey)
                region = fx * reg3[:, :2] / reg3[:, 2:3] + c0
                best = (
                    key,
                    float(e.mean()),
                    P[k],
                    s,
                    s3 * 1000.0,
                    float(np.median(Z[k])),
                    region,
                    float(np.median(Rb[k])),
                    (n.copy(), float(Dp[i]), float(c0[0]), float(c0[1])),
                )
    if best is None:
        return None
    key, err, pts, s, s_mm, z, region, r_med, plane = best
    shift = _edge_shift(_edge_offsets(pts, r_med, s, P, Rb, Sc), s)
    pts, region = pts + shift, region + shift
    return DomeFit(
        px={n: tuple(map(float, p)) for n, p in zip(NAMES, pts, strict=False)},
        s_px=s,
        s_mm=s_mm,
        err=err,
        z=z,
        region=region,
        score=key[0],
        r_px=r_med,
        plane=plane,
    )


def _edge_offsets(pts, r_fit, s, P, Rb, Sc):
    """Per fitted spot that also answers at a clearly larger ring close by (its dome's edge,
    EDGE_MIN_SIZE_RATIO x the fitted size, within EDGE_SEARCH_FRAC of the spacing): the offset from
    the spot to that ring's centre."""
    offs = []
    for p in pts:
        d = np.linalg.norm(P - p, axis=1)
        near = np.nonzero((d < EDGE_SEARCH_FRAC * s) & (Rb >= EDGE_MIN_SIZE_RATIO * r_fit))[0]
        if near.size:
            j = near[np.lexsort((Sc[near], Rb[near]))[-1]]  # largest ring, strongest among those
            offs.append(P[j] - p)
    return np.array(offs).reshape(-1, 2)


def _edge_shift(offs, s, min_domes=None):
    """Image shift that moves a fitted set from the domes' highlights to their centres. The best
    set is often the 5 small highlights, which sit off-centre the same way on every dome (the light
    comes from one side: 4 px = 3 mm high at 31 cm, cream panel 2026-10-13); the dome's edge ring
    is centred on it. The shift is the median over the domes that show an edge, when they agree to
    EDGE_AGREE_FRAC of the spacing, applied to all 5 so the layout is kept."""
    if len(offs) < (EDGE_MIN_DOMES if min_domes is None else min_domes):
        return np.zeros(2)
    med = np.median(offs, axis=0)
    if (np.linalg.norm(offs - med, axis=1) > EDGE_AGREE_FRAC * s).any():
        return np.zeros(2)
    return med


def _rim_offsets(gray, fit, fx):
    """Per dome whose rim shows (>= RIM_MIN_CONTRAST): the offset from its fitted spot to the centre
    of its rim. The rim is a circle of radius RIM_FRAC x the spacing ON the panel plane, projected
    into the image; its score at a trial centre is the mean grey just outside it (the panel) minus
    the mean grey on it (the dark gap round the button), each averaged round the circle -- so the
    chrome's reflections inside the button do not enter. One rim size for all 5 (the best)."""
    n, d, cx, cy = fit.plane
    c0 = np.array([cx, cy])
    s3 = fit.s_mm / 1000.0
    a = np.cross(n, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    th = np.linspace(0.0, 2.0 * np.pi, RIM_ANGLES, endpoint=False)
    circle = np.cos(th)[:, None] * a + np.sin(th)[:, None] * b  # unit circle on the panel
    g = cv2.GaussianBlur(gray, (0, 0), 0.8)
    h_img, w_img = g.shape
    m = int(np.ceil(RIM_SEARCH_FRAC * fit.s_px))
    best = None
    for frac in RIM_FRAC:
        res = []
        for p in fit.px.values():
            p = np.asarray(p, dtype=float)
            ray = np.array([(p[0] - cx) / fx, (p[1] - cy) / fx, 1.0])
            C = ray * (d / float(ray @ n))
            p_int = np.round(p).astype(int)
            sums = []
            for k in (1.0, RIM_OUT):
                pts3 = C + k * frac * s3 * circle
                off = np.round(fx * pts3[:, :2] / pts3[:, 2:3] + c0 - p_int).astype(int)  # ellipse, px from p_int
                acc = np.zeros((2 * m + 1, 2 * m + 1), np.float32)
                for ox, oy in off:
                    x0, y0 = p_int[0] + ox - m, p_int[1] + oy - m
                    if x0 < 0 or y0 < 0 or x0 + 2 * m + 1 > w_img or y0 + 2 * m + 1 > h_img:
                        acc = None
                        break
                    acc += g[y0 : y0 + 2 * m + 1, x0 : x0 + 2 * m + 1]
                if acc is None:
                    break
                sums.append(acc / len(off))
            if len(sums) < 2:
                res.append((0.0, np.zeros(2)))
                continue
            S = sums[1] - sums[0]  # panel outside minus rim, per trial centre
            iy, ix = np.unravel_index(int(S.argmax()), S.shape)
            peak = float(S[iy, ix])
            y0, y1, x0, x1 = max(0, iy - 1), min(S.shape[0], iy + 2), max(0, ix - 1), min(S.shape[1], ix + 2)
            wgt = np.clip(S[y0:y1, x0:x1] - 0.6 * peak, 0.0, None)  # sub-pixel: centroid of the peak
            if peak > 0 and wgt.sum() > 0:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                ix_f, iy_f = float((wgt * xx).sum() / wgt.sum()), float((wgt * yy).sum() / wgt.sum())
            else:
                ix_f, iy_f = float(ix), float(iy)
            res.append((peak, np.array([ix_f - m, iy_f - m]) + p_int - p))
        total = float(np.sort([r[0] for r in res])[-RIM_MIN_DOMES:].sum())
        if best is None or total > best[0]:
            best = (total, res)
    return np.array([o for pk, o in best[1] if pk >= RIM_MIN_CONTRAST]).reshape(-1, 2)


def detect(bgr, depth_m, fx, hint=None):
    """Detect the 5 domes in one frame (the module's entry point).

    Returns a DomeFit with the centres refined to the domes' rims, or None if no layout fits or
    it is outside MIN_RANGE_M..MAX_RANGE_M. Pass the previous frame's fit as `hint` for a much
    faster local search.
    """
    fit = fit_layout(candidates(bgr, depth_m, fx, hint), fx, hint)
    if fit is None or not MIN_RANGE_M <= fit.z <= MAX_RANGE_M:
        return None
    if fit.plane is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shift = _edge_shift(_rim_offsets(gray, fit, fx), fit.s_px, RIM_MIN_DOMES)
        if shift.any():
            fit.px = {name: (u + float(shift[0]), v + float(shift[1])) for name, (u, v) in fit.px.items()}
            fit.region = fit.region + shift
    return fit


def panel_plane(bgr, depth_m, fit: DomeFit, intr):
    """Fit the panel plane n.X = d (camera frame) to depth in the button region, minus the domes.

    Returns geometry.fit_plane_inverse_depth's (n, d, median_depth, n_points)."""
    mask = np.zeros(depth_m.shape, np.uint8)
    cv2.fillConvexPoly(mask, fit.region.astype(np.int32), 1)
    for u, v in fit.px.values():  # not the domes: the depth sensor lies on chrome
        cv2.circle(mask, (int(round(u)), int(round(v))), int(np.ceil(0.35 * fit.s_px)), 0, -1)
    ok = (mask > 0) & (depth_m > 0.1) & (depth_m < 2.0)
    vs, us = np.where(ok)
    return fit_plane_inverse_depth(us, vs, depth_m[vs, us], *intr)


def button_xyz_cam(px, n, d, intr):
    """Camera-frame point where the button pixel's ray meets the plane n.X = d."""
    ray = pixel_ray(px, *intr)
    return ray * (d / float(np.dot(n, ray)))
