"""Refining FACE_CORNER_TO_START_MM: the wrist-detected button expressed from the scene front's
corner (scene_finder.start_offset_mm) and its collection from press_button run logs
(tools/scene_offset_from_logs.py)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from rammp_button_press import scene_finder as sf

TOOL = Path(__file__).resolve().parents[1] / "tools" / "scene_offset_from_logs.py"
CORNER = np.array([0.79, -0.317, 0.0])
EX, EY = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])


def button_at(offset_mm, target="start_30s"):
    rel = np.asarray(sf.BUTTON_FROM_DISPLAY_MM[target]) - np.asarray(sf.BUTTON_FROM_DISPLAY_MM["start_30s"])
    x, y = (np.asarray(offset_mm) + rel) / 1000.0
    return CORNER + x * EX + y * EY


def test_start_offset_inverts_the_finders_placement():
    assert np.allclose(sf.start_offset_mm(CORNER, EX, EY, button_at((-30.0, 70.0)), "start_30s"), (-30.0, 70.0))


def test_another_target_gives_the_start_buttons_offset():
    assert np.allclose(
        sf.start_offset_mm(CORNER, EX, EY, button_at((-30.0, 70.0), "timer_clock"), "timer_clock"), (-30.0, 70.0)
    )


def load_tool():
    spec = importlib.util.spec_from_file_location("scene_offset_from_logs", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_log(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def scene_rec(target):
    return {
        "panel_frame_scene": {
            "target": target,
            "front": {"corner_br": CORNER.tolist(), "ex": EX.tolist(), "ey": EY.tolist()},
        }
    }


def test_offsets_come_from_scene_staged_runs_only(tmp_path):
    tool = load_tool()
    write_log(
        tmp_path / "press_1.jsonl",
        [
            scene_rec("start_30s"),
            {"stage": "staging_redetect", "source": "scene", "button_xyz": button_at((-30.0, 70.0)).tolist()},
        ],
    )
    write_log(
        tmp_path / "press_2.jsonl",
        [
            scene_rec("timer_clock"),
            {
                "stage": "staging_redetect",
                "source": "scene",
                "button_xyz": button_at((-20.0, 60.0), "timer_clock").tolist(),
            },
        ],
    )
    write_log(tmp_path / "press_3.jsonl", [{"stage": "staging_redetect", "source": "wrist", "button_xyz": [0, 0, 0]}])
    write_log(tmp_path / "press_4.jsonl", [scene_rec("start_30s"), {"stage": "abort", "why": "no domes"}])
    got = tool.offsets(sorted(tmp_path.glob("*.jsonl")))
    assert [p.name for p, _ in got] == ["press_1.jsonl", "press_2.jsonl"]
    assert np.allclose([o for _, o in got], [(-30.0, 70.0), (-20.0, 60.0)])
