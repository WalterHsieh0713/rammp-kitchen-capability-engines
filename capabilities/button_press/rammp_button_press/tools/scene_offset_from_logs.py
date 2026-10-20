"""Refine scene_finder.FACE_CORNER_TO_START_MM from press_button run logs. READ-ONLY: reads logs.

Every run that staged from the scene camera logs the front it found (`panel_frame_scene`, median
over its frames) and then the wrist camera's dome fix at the staging pose (`staging_redetect`,
source "scene"). The fix, expressed from that front's bottom-right corner, is what the constant
should have been for that run. This prints it per run and the median over all of them.

    python3 scene_offset_from_logs.py                      # every ~/press_logs/press_*.jsonl
    python3 scene_offset_from_logs.py ~/press_logs/press_20261015_*.jsonl

Paste the printed line into scene_finder.py once several runs agree (spread of a few mm). The
offset is from the corner AS THE FINDER MEASURES IT, so re-run this after changing the finder.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package source next to this tools/ dir
from rammp_button_press import scene_finder as sf  # noqa: E402


def offsets(paths):
    """[(log path, (x mm, y mm))] for every run that staged from the scene camera and re-detected."""
    out = []
    for p in paths:
        scene = redetect = None
        for line in Path(p).read_text().splitlines():
            rec = json.loads(line)
            if "panel_frame_scene" in rec:
                scene = rec["panel_frame_scene"]
            elif rec.get("stage") == "staging_redetect" and rec.get("source") == "scene":
                redetect = rec
        if scene is None or redetect is None or "front" not in scene:
            continue
        f = scene["front"]
        out.append(
            (
                Path(p),
                sf.start_offset_mm(
                    f["corner_br"], f["ex"], f["ey"], redetect["button_xyz"], scene.get("target", "start_30s")
                ),
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*")
    a = ap.parse_args()
    paths = [Path(x) for x in a.logs] or sorted((Path.home() / "press_logs").glob("press_*.jsonl"))
    got = offsets(paths)
    if not got:
        print(f"no scene-staged run with a staging re-detect in {len(paths)} logs")
        return 1
    for p, (x, y) in got:
        print(f"  {p.name}: x {x:+.1f} mm, y {y:+.1f} mm")
    xy = np.array([o for _, o in got])
    med = np.median(xy, axis=0)
    print(
        f"{len(got)} runs, spread x {np.ptp(xy[:, 0]):.1f} mm, y {np.ptp(xy[:, 1]):.1f} mm "
        f"(now {sf.FACE_CORNER_TO_START_MM})"
    )
    print(f"FACE_CORNER_TO_START_MM = ({med[0]:.1f}, {med[1]:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
