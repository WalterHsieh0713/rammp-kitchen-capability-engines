"""grasp.ring_in_frame and grasp.detect_ring on a brim rendered from the wrist camera's view pose: the
fitted outer edge lands on the true one, and the checks reject a wrong-sized ring, a ring cut off by
the image's edge, too few frames and frames that disagree. grasp.grab hands `motion` to every move."""

import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grab_container import grasp

W, H = 640, 400
K = (398.4, 398.3, 338.0, 203.2)
CENTRE, HEIGHT = np.array([0.98, -0.01]), 0.266


def view(c=CENTRE, back=0.15, up=0.25):
    """The camera (R, t) in the base frame VIEW_BACK_M back and VIEW_UP_M above the brim, looking at it."""
    t = np.array([c[0] - back, c[1], HEIGHT + up])
    z = np.array([*c, HEIGHT]) - t
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z]), t


def render(R, t, c=CENTRE, r_out=grasp.RING_RADIUS_M, r_in=0.07, gap_deg=(100, 115), noise=0.003, seed=0):
    """The brim's mask (a flat annulus, broken over `gap_deg`, as YOLO's masks often are) and a depth
    image of it in metres (plane depth + noise; 0 off the brim)."""
    fx, fy, cx, cy = K
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], axis=-1) @ R.T
    s = (HEIGHT - t[2]) / rays[..., 2]
    p = t + s[..., None] * rays
    d = p[..., :2] - c
    rad, ang = np.linalg.norm(d, axis=-1), np.degrees(np.arctan2(d[..., 1], d[..., 0])) % 360
    m = (s > 0) & (rad >= r_in) & (rad <= r_out) & ~((ang >= gap_deg[0]) & (ang <= gap_deg[1]))
    dm = np.where(m, s + np.random.default_rng(seed).normal(0, noise, s.shape), 0.0).astype(np.float32)
    return m, dm


def test_fit_finds_the_outer_edge():
    R, t = view()
    m, dm = render(R, t)
    c, r, h, rms = grasp.ring_in_frame(m, dm, K, R, t)
    assert np.linalg.norm(c - CENTRE) < 1e-3
    assert abs(r - grasp.RING_RADIUS_M) < 1e-3
    assert abs(h - HEIGHT) < 1e-3
    assert rms < 1e-3


def test_wrong_size_is_rejected():
    R, t = view()
    m, dm = render(R, t, r_out=grasp.RING_RADIUS_M - 0.01, r_in=0.06)
    assert grasp.ring_in_frame(m, dm, K, R, t).startswith("radius")


def test_ring_cut_by_the_image_edge_is_rejected():
    R, t = view(back=0.02, up=0.12)  # so close that the brim runs off the image
    m, dm = render(R, t)
    assert m[:, 0].any() or m[:, -1].any() or m[0].any() or m[-1].any()
    assert isinstance(grasp.ring_in_frame(m, dm, K, R, t), str)


class FakeCamera:
    """Camera's interface for detect_ring, serving one rendered frame per new stamp."""

    def __init__(self, frames):
        self.frames, self.i, self.stamp = frames, 0, None

    def frame(self, timeout, after=None):
        if self.i >= len(self.frames):
            return None
        self.R, self.t, m, dm = self.frames[self.i]
        self.i += 1
        self.stamp, self.mask = self.i, m
        return np.zeros((H, W, 3), np.uint8), dm

    def intrinsics(self):
        return K

    def pose_in_base(self, stamp=None):
        return self.R, self.t


class FakeModel:
    """YOLO's predict() result shape: one grabber mask (the camera's current one) at conf 0.9."""

    def __init__(self, camera):
        self.camera = camera

    def predict(self, bgr, **kw):
        m = self.camera.mask.astype(np.float32)
        cpu = types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: m))
        boxes = types.SimpleNamespace(cls=types.SimpleNamespace(tolist=lambda: [0.0]), conf=[0.9])
        return [types.SimpleNamespace(masks=types.SimpleNamespace(data=[cpu]), boxes=boxes, names={0: "grabber"})]


def frames(centres):
    out = []
    for i, c in enumerate(centres):
        R, t = view()
        out.append((R, t, *render(R, t, c=np.asarray(c), seed=i)))
    return out


def test_detect_ring_returns_the_edge_facing_the_approach():
    cam = FakeCamera(frames([CENTRE] * grasp.N_FRAMES))
    edge, ring = grasp.detect_ring(cam, FakeModel(cam))
    a = grasp.grasp_orientation()[:2, 2]
    a /= np.linalg.norm(a)
    assert np.linalg.norm(edge - [*(CENTRE - grasp.RING_RADIUS_M * a), HEIGHT]) < 1e-3
    assert ring["spread_m"] < 1e-3 and ring["conf"] == 0.9


def test_detect_ring_refuses_too_few_frames():
    cam = FakeCamera(frames([CENTRE] * (grasp.N_FRAMES - 1)))
    with pytest.raises(grasp.Abort, match="not seen stably"):
        grasp.detect_ring(cam, FakeModel(cam))


def test_detect_ring_refuses_frames_that_disagree():
    centres = [CENTRE] * (grasp.N_FRAMES - 1) + [CENTRE + [0.0, 3 * grasp.AGREE_M]]
    cam = FakeCamera(frames(centres))
    with pytest.raises(grasp.Abort, match="disagree"):
        grasp.detect_ring(cam, FakeModel(cam))


class FakeArm:
    """self.arm for grasp.grab: records each move's label, waypoints and keyword arguments."""

    last_error = ""

    def __init__(self):
        self.moves = []

    def not_ready(self):
        return None

    def gripper(self, *a, **kw):
        return True

    def go_through(self, waypoints, label, speed_scale=None, **kw):
        self.moves.append((label, list(waypoints), speed_scale, kw))
        return True

    def go_to_ee_pose(self, pos, quat, label, **kw):
        self.moves.append((label, quat, kw))
        return True


def test_grab_approaches_in_one_motion_and_raises_level():
    cam, arm = FakeCamera(frames([CENTRE] * grasp.N_FRAMES)), FakeArm()
    motion = {"control_mode": 1, "gains": "medium"}
    out = grasp.grab(arm, cam, FakeModel(cam), True, lambda s: None, lambda *a: None, 1, **motion)
    assert out["raised"]
    (label, (pre, at), speed, kw), (raise_label, quat, raise_kw) = arm.moves
    assert (label, speed, kw) == ("approach", grasp.APPROACH_SPEED_SCALE, motion)
    assert (pre.blend_radius, at.blend_radius) == (grasp.BLEND_M, 0.0)  # cut the pre-grasp corner
    assert (pre.hold, at.hold) == (0, 1)  # the hold on the pre-grasp -> rim leg only
    assert (raise_label, raise_kw) == ("raise", motion)
    approach = Rotation.from_quat(quat).as_matrix()[:, 2]
    assert approach[2] == pytest.approx(0.0, abs=1e-9)  # level with the world
