"""press_button's scene-camera far stage: combining the per-frame fits, the first-look order
(scene -> wrist domes) and the staging decision. No ROS spinning, no arm."""

import argparse
import time

import numpy as np
import pytest

pb = pytest.importorskip("rammp_button_press.press_button")  # needs rclpy + rammp interfaces
from rammp_button_press import Abort  # noqa: E402
from rammp_button_press.scene_finder import MicrowaveFit  # noqa: E402


def fit(button, n=(-1.0, 0.0, 0.0)):
    n = np.asarray(n, float) / np.linalg.norm(n)
    z = np.zeros(3)
    return MicrowaveFit(
        n=n,
        centre=np.asarray(button, float),
        ex=z,
        ey=z,
        width_m=0.445,
        height_m=0.26,
        corner_br=z,
        button=np.asarray(button, float),
        window_dark=0.7,
        strip_dark=0.1,
        lab=(164, 125, 135),
        px_box=(0, 0, 1, 1),
        n_points=1000,
    )


B = np.array([0.79, -0.30, 0.06])


# ---- combine_scene_fits ----------------------------------------------------------------------


def test_agreeing_fits_give_their_median_button_and_mean_normal():
    fits = [fit(B + [0, 0.002 * i, 0]) for i in range(5)]
    button, n, kept = pb.combine_scene_fits(fits, seen=5, reasons=[])
    assert np.allclose(button, B + [0, 0.004, 0])
    assert np.allclose(n, [-1, 0, 0])
    assert kept == 5


def test_one_odd_frame_is_dropped():
    fits = [fit(B) for _ in range(4)] + [fit(B + [0, 0.10, 0])]
    button, _, kept = pb.combine_scene_fits(fits, seen=5, reasons=[])
    assert kept == 4 and np.allclose(button, B)


def test_too_few_fits_is_refused_with_the_finders_reason():
    with pytest.raises(Abort, match="2/5 .*no microwave front: size"):
        pb.combine_scene_fits([fit(B), fit(B)], seen=5, reasons=["no microwave front: size 20 x 25 cm"] * 3)


def test_three_fits_are_enough_when_yolo_narrowed_the_search():
    fits = [fit(B) for _ in range(3)]
    with pytest.raises(Abort, match="3/5 .*need 4"):
        pb.combine_scene_fits(fits, seen=5, reasons=["x"] * 2)
    button, _, kept = pb.combine_scene_fits(fits, seen=5, reasons=["x"] * 2, min_fits=pb.SCENE_MIN_FITS_YOLO)
    assert kept == 3 and np.allclose(button, B)


def test_fits_that_disagree_are_refused_as_unstable():
    fits = [fit(B + [0, 0.03 * i, 0]) for i in range(5)]
    with pytest.raises(Abort, match="unstable"):
        pb.combine_scene_fits(fits, seen=5, reasons=[])


def test_normals_that_disagree_are_refused_as_unstable():
    fits = [fit(B, n=(-1.0, np.tan(np.radians(4.0 * i)), 0.0)) for i in range(5)]  # 0..16 deg
    with pytest.raises(Abort, match="unstable"):
        pb.combine_scene_fits(fits, seen=5, reasons=[])


def test_no_frames_at_all_says_the_camera_is_silent():
    with pytest.raises(Abort, match="no frames"):
        pb.combine_scene_fits([], seen=0, reasons=[])


# ---- first_look ------------------------------------------------------------------------------


def args(far="scene", target="start_30s"):
    return argparse.Namespace(far=far, target=target)


class FakePer:
    def __init__(self):
        self.scene_stopped = 0

    def stop_scene(self):
        self.scene_stopped += 1


class FakeRun:
    def __init__(self):
        self.logged = []
        self.per = FakePer()

    def log(self, rec):
        self.logged.append(rec)


def looks(monkeypatch, scene=None, domes=None):
    """Patch the two detectors: each is a frame (origin, R) or an exception to raise."""
    calls = []

    def make(name, result):
        def f(run, target, *rest, **kw):
            calls.append(name)
            if isinstance(result, BaseException):  # Abort is a SystemExit
                raise result
            return result

        return f

    frame = (B, np.eye(3))
    monkeypatch.setattr(pb, "measure_panel_frame_scene", make("scene", frame if scene is None else scene))
    monkeypatch.setattr(pb, "measure_panel_frame_domes", make("domes", frame if domes is None else domes))
    return calls


def test_scene_camera_looks_first(monkeypatch):
    calls = looks(monkeypatch)
    _, _, source = pb.first_look(FakeRun(), args())
    assert calls == ["scene"] and source == "scene"


def test_stage_out_option_parses():
    assert pb.build_parser().parse_args([]).stage_out is None
    assert pb.build_parser().parse_args(["--stage-out", "0.22"]).stage_out == 0.22


def test_a_low_scene_estimate_is_raised_to_the_staging_floor(monkeypatch):
    low = np.array([0.84, -0.273, 0.05])  # 2026-10-15: staging from this was IK_FAIL
    looks(monkeypatch, scene=(low, np.eye(3)))
    run = FakeRun()
    origin, _, _ = pb.first_look(run, args())
    assert np.allclose(origin, [0.84, -0.273, pb.SCENE_MIN_BUTTON_Z_M])
    assert any(r.get("stage") == "scene_min_z" for r in run.logged)


def test_a_scene_estimate_above_the_floor_is_kept(monkeypatch):
    high = np.array([0.80, -0.25, 0.104])
    looks(monkeypatch, scene=(high, np.eye(3)))
    origin, _, _ = pb.first_look(FakeRun(), args())
    assert np.allclose(origin, high)


def test_the_floor_does_not_touch_a_wrist_estimate(monkeypatch):
    low = np.array([0.72, -0.27, 0.05])
    looks(monkeypatch, scene=Abort("scene camera: no frames in 8 s"), domes=(low, np.eye(3)))
    origin, _, source = pb.first_look(FakeRun(), args())
    assert source == "wrist" and np.allclose(origin, low)


def test_scene_failure_falls_back_to_the_wrist_and_is_logged(monkeypatch):
    calls = looks(monkeypatch, scene=Abort("scene camera: no frames in 8 s"))
    run = FakeRun()
    _, _, source = pb.first_look(run, args())
    assert calls == ["scene", "domes"] and source == "wrist"
    assert any(r.get("stage") == "scene_fallback" and "no frames" in r.get("why", "") for r in run.logged)


def test_scene_and_domes_failing_aborts(monkeypatch):
    calls = looks(monkeypatch, scene=Abort("no microwave front"), domes=Abort("dome detector: nothing in 7 frames"))
    with pytest.raises(Abort, match="dome detector"):
        pb.first_look(FakeRun(), args())
    assert calls == ["scene", "domes"]


def test_the_scene_subscriptions_close_after_the_first_look(monkeypatch):
    """Left open, the scene camera's 1280x720 colour + depth at 30 Hz starved the wrist re-detect:
    4 frames in 8 s instead of 7 in under 1 s, and the press aborted at staging (2026-10-14)."""
    looks(monkeypatch)
    run = FakeRun()
    pb.first_look(run, args())
    assert run.per.scene_stopped == 1


def test_the_scene_subscriptions_close_when_the_scene_look_fails(monkeypatch):
    looks(monkeypatch, scene=Abort("no microwave front"))
    run = FakeRun()
    pb.first_look(run, args())
    assert run.per.scene_stopped == 1


def test_far_wrist_is_todays_behaviour(monkeypatch):
    calls = looks(monkeypatch)
    _, _, source = pb.first_look(FakeRun(), args(far="wrist"))
    assert calls == ["domes"] and source == "wrist"


def test_domes_too_close_aborts_with_the_reason(monkeypatch):
    looks(monkeypatch, scene=Abort("no microwave front"), domes=Abort("dome detector: nothing (TOO CLOSE: 18 cm)"))
    with pytest.raises(Abort, match="TOO CLOSE"):
        pb.first_look(FakeRun(), args())


# ---- staging_plan ----------------------------------------------------------------------------


def test_a_scene_estimate_always_stages_without_the_turn_limit():
    out, limit_turn = pb.staging_plan("scene", cam_d=0.20)
    assert out == pb.SCENE_STAGE_OUT_M and limit_turn is False


def test_wrist_estimates_stage_only_from_afar_and_keep_the_turn_limit():
    assert pb.staging_plan("wrist", cam_d=0.40) == (pb.STAGE_OUT_M, True)
    assert pb.staging_plan("wrist", cam_d=0.20) == (None, True)


# ---- command line and the run log ------------------------------------------------------------


def test_push_and_stop_options_default_and_are_bounded(capsys):
    a = pb.build_parser().parse_args([])
    assert a.push_mm == pb.CONTACT_PUSH_M * 1000 and a.stop_nm == pb.CONTACT_STOP_NM
    assert pb.main(["--stop-nm", "2"]) == 2 and pb.main(["--push-mm", "30"]) == 2


def test_the_default_camera_is_the_wrist_d405_not_the_kinova_vision_module():
    """2026-10-15: the arm driver began publishing camera_color_optical_frame (the Kinova vision module,
    90 deg sideways); the old default picked it up and put the button 57 cm off."""
    assert pb.build_parser().parse_args([]).camera_frame == "wrist_d405_color_optical"


def test_the_yolo_box_is_used_unless_turned_off():
    assert pb.build_parser().parse_args([]).yolo is True
    assert pb.build_parser().parse_args(["--no-yolo"]).yolo is False


def test_far_defaults_to_the_scene_camera():
    a = pb.build_parser().parse_args([])
    assert a.far == "scene" and a.scene_ns == "/scene_camera"
    assert pb.build_parser().parse_args(["--far", "wrist"]).far == "wrist"


def test_an_abort_reason_reaches_the_run_log(monkeypatch):
    run = FakeRun()

    class Arm:
        def close(self):
            pass

    run.arm = Arm()
    run.per = type("Per", (), {"create_publisher": lambda self, *a: None})()
    monkeypatch.setattr(pb, "Run", lambda a: run)

    def boom(run_, a, pub, decide=None):
        raise Abort("scene camera: unstable -- only 2/5 frames agree")

    monkeypatch.setattr(pb, "press_button", boom)
    assert pb.main([]) == 2
    assert run.logged[-1] == {"stage": "abort", "why": "scene camera: unstable -- only 2/5 frames agree"}


# ---- measure_panel_frame_scene: the frames' searches --------------------------------------------


class SceneFeed:
    """A scene camera with a new colour and depth frame on every get(); no YOLO box, no ROS."""

    bridge = None

    def get(self, name, max_age=None):
        return None if name == "yolo" else object()  # a new object is a new frame

    def transform(self, frame):
        return np.eye(4)


def scene_run():
    arm = argparse.Namespace(ee_pos=lambda: np.zeros(3), kin=argparse.Namespace(base_pos=np.zeros(3)))
    return argparse.Namespace(per=SceneFeed(), arm=arm, log=lambda rec: None)


def test_the_scene_frames_are_searched_side_by_side(monkeypatch):
    def find(**kw):
        time.sleep(0.3)  # 5 frames one after another: >= 1.5 s
        return fit(B), ""

    monkeypatch.setattr(pb.scene_finder, "find_microwave", find)
    monkeypatch.setattr(pb, "scene_frame_from_msgs", lambda *a: {})
    t0 = time.monotonic()
    origin, _ = pb.measure_panel_frame_scene(scene_run(), "start_30s", use_yolo=False)
    assert time.monotonic() - t0 < 0.9
    assert np.allclose(origin, B)


def test_the_scene_finders_reasons_still_reach_the_abort(monkeypatch):
    found = iter([True, False, False, False, True])
    monkeypatch.setattr(
        pb.scene_finder,
        "find_microwave",
        lambda **kw: (fit(B), "") if next(found) else (None, "no microwave front: size 20 x 25 cm"),
    )
    monkeypatch.setattr(pb, "scene_frame_from_msgs", lambda *a: {})
    with pytest.raises(Abort, match="2/5 .*no microwave front: size"):
        pb.measure_panel_frame_scene(scene_run(), "start_30s", use_yolo=False)
