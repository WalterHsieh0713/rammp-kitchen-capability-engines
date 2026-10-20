"""Pure geometry helpers: panel plane fit from depth, pixel rays, joint-angle differences.

numpy only (no ROS, arm or PyBullet), so it is unit-testable off the robot.
"""

from __future__ import annotations

import numpy as np

# Drop depths this far off the median (background seen through gaps around the panel).
PLANE_Z_BAND_M = 0.05
# Refit after dropping points this far off the first fit.
PLANE_TRIM_M = 0.02
PLANE_MIN_POINTS = 200
PLANE_MAX_RESID_M = 0.01


class PlaneFitError(ValueError):
    """The depth pixels do not support a trustworthy plane."""


def fit_plane_inverse_depth(us, vs, z, fx, fy, cx, cy):
    """Fit the plane ``n.X = d`` (camera frame) to depth pixels ``(us, vs, z)``.

    Ordinary least squares on inverse depth (for a plane, 1/z is exactly linear in the pixel
    coordinates), not SVD on 3D points: the pixel coordinates are exact and all the noise is in
    z. At close range SVD mistook depth noise for tilt (61 deg instead of 19 deg, 2026-09-21).
    Background pixels far from the median depth are dropped first; then one refit without
    outliers.

    Returns ``(n, d, median_depth, n_points)`` with n pointing back toward the camera
    (``n[2] < 0``). Raises :class:`PlaneFitError` on too few points or a large residual.
    """
    us = np.asarray(us, dtype=float)
    vs = np.asarray(vs, dtype=float)
    zz = np.asarray(z, dtype=float)
    if len(zz) < PLANE_MIN_POINTS:
        raise PlaneFitError(f"only {len(zz)} valid panel depth pixels")
    up = (us - cx) / fx
    vp = (vs - cy) / fy
    # Background seen through gaps around the panel is only ~2% of the pixels but sits
    # at ~37 cm vs the panel's ~21 cm, and 1/z gives it enormous leverage. Drop it first.
    keep0 = np.abs(zz - np.median(zz)) <= PLANE_Z_BAND_M
    if keep0.sum() >= PLANE_MIN_POINTS:
        up, vp, zz = up[keep0], vp[keep0], zz[keep0]
    m = None
    for _ in range(2):
        A = np.stack([up, vp, np.ones_like(up)], axis=1)
        m, *_ = np.linalg.lstsq(A, 1.0 / zz, rcond=None)
        pred = 1.0 / (A @ m)
        keep = np.abs(zz - pred) <= PLANE_TRIM_M
        if keep.sum() < PLANE_MIN_POINTS or bool(keep.all()):
            break
        up, vp, zz = up[keep], vp[keep], zz[keep]
    A = np.stack([up, vp, np.ones_like(up)], axis=1)
    resid = float(np.std(zz - 1.0 / (A @ m)))
    nrm = float(np.linalg.norm(m))
    n, d_plane = m / nrm, 1.0 / nrm
    if n[2] > 0:  # make the normal point back toward the camera (-z); d flips with it
        n, d_plane = -n, -d_plane
    if resid > PLANE_MAX_RESID_M:
        raise PlaneFitError(f"residual {resid*100:.1f} cm -- not planar / bad depth")
    return n, d_plane, float(np.median(zz)), len(zz)


def pixel_ray(px, fx, fy, cx, cy) -> np.ndarray:
    """Unit ray (camera optical frame) through pixel ``px``."""
    r = np.array([(px[0] - cx) / fx, (px[1] - cy) / fy, 1.0])
    return r / np.linalg.norm(r)


def max_joint_delta_deg(q_a, q_b) -> float:
    """Largest per-joint difference in degrees, with angle wrap-around."""
    diff = np.asarray(q_a, dtype=float) - np.asarray(q_b, dtype=float)
    return float(np.degrees(np.max(np.abs((diff + np.pi) % (2 * np.pi) - np.pi))))
