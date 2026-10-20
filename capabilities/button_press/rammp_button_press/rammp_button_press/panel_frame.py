"""The panel frame: a coordinate frame centred on the target button.

The pre-press tool pose is stored in this frame, so it stays correct when the microwave is
moved. The frame is rebuilt from every detection:

  origin  the button's xyz in the arm base frame
  z       the panel normal, pointing OUT of the panel (toward the robot)
  y       world up (base +z) projected onto the panel; a microwave on a table does not roll
  x       y cross z (to the right, seen from the front)

Also converts points and rotations between the base and panel frames, and builds straight-line
pose paths. numpy only, so it is unit-testable off the robot.
"""

from __future__ import annotations

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def panel_frame(origin, normal_out) -> tuple[np.ndarray, np.ndarray]:
    """Build the panel frame at `origin` with z along `normal_out`.

    Returns (origin, R), R's columns = the panel x, y, z axes in the base frame. Raises
    ValueError if the normal is too close to vertical to define "up".
    """
    z = np.asarray(normal_out, dtype=float)
    z = z / np.linalg.norm(z)
    y = UP - np.dot(UP, z) * z
    ny = np.linalg.norm(y)
    if ny < 0.2:
        raise ValueError(f"panel normal {np.round(z, 3)} is too close to vertical to define 'up'")
    y = y / ny
    x = np.cross(y, z)
    return np.asarray(origin, dtype=float), np.stack([x, y, z], axis=1)


def to_panel(p_base, origin, R) -> np.ndarray:
    """Base-frame point -> panel-frame coordinates."""
    return R.T @ (np.asarray(p_base, dtype=float) - origin)


def from_panel(p_panel, origin, R) -> np.ndarray:
    """Panel-frame coordinates -> base-frame point."""
    return origin + R @ np.asarray(p_panel, dtype=float)


def rot_to_panel(R_base, R) -> np.ndarray:
    """Base-frame orientation (3x3) -> the same orientation expressed in the panel frame."""
    return R.T @ np.asarray(R_base, dtype=float)


def rot_from_panel(R_local, R) -> np.ndarray:
    """Panel-frame orientation (3x3) -> base frame."""
    return R @ np.asarray(R_local, dtype=float)


def pose_path(p0, R0, p1, R1, max_step_m, max_step_deg):
    """Straight-line tool poses from (p0, R0) to (p1, R1): position interpolated, rotation slerped.

    Uses enough points that no step moves more than max_step_m or turns more than max_step_deg.
    Returns [(p, R), ...], excluding the start and ending exactly at (p1, R1).
    """
    from scipy.spatial.transform import Rotation, Slerp  # noqa: PLC0415

    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    rots = Rotation.from_matrix(np.stack([np.asarray(R0, dtype=float), np.asarray(R1, dtype=float)]))
    ang = float(np.degrees((rots[1] * rots[0].inv()).magnitude()))
    n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / max_step_m)), int(np.ceil(ang / max_step_deg)))
    slerp = Slerp([0.0, 1.0], rots)
    return [(p0 + (p1 - p0) * (k / n), slerp(k / n).as_matrix()) for k in range(1, n + 1)]


def rot_angle_deg(R0, R1) -> float:
    """Angle of the rotation taking R0 to R1, degrees."""
    c = (np.trace(np.asarray(R1, dtype=float) @ np.asarray(R0, dtype=float).T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def describe(p_panel) -> str:
    """Human-readable panel-frame offset (x right, y up, z out of the panel)."""
    x, y, z = (float(v) * 100 for v in p_panel)
    return (
        f"{abs(z):.1f} cm {'out from' if z >= 0 else 'INTO'} the panel, "
        f"{abs(x):.1f} cm {'right' if x >= 0 else 'left'}, {abs(y):.1f} cm {'above' if y >= 0 else 'below'} the button"
    )
