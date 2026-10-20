"""Per-rig calibration of the pre-press spot: prints two constants for press_button.py. Never moves the arm.

press_button aims the tool at a stored pose relative to the detected button: where the tool
frame is when the LEFT fingertip is STANDOFF_M (2 cm) in front of the button. That depends on
the fingers, camera mount and hand-eye calibration, so re-measure on every rig and after any
of them changes.

  1. sheppy profile `arm-camera` up; camera ~25 cm from the panel (press_button's staging
     distance: from further away the same button reads ~1.5 cm lower), all 5 domes in view.
  2. Run this. It detects the panel frame from that view.
  3. When prompted: switch sheppy to `camera-only` (stops the driver; never hand-guide or
     teleop with it up, it crashes), hand-guide the left fingertip 2 cm straight out from the
     button centre, wrist roughly square to the panel, switch back to `arm-camera`, press
     Enter. Do NOT move the microwave.
     With --touch: put the fingertip lightly ON the button centre instead; the script backs off
     STANDOFF_M itself and also prints the detection's error at the button.
  4. Paste the printed PREPRESS_EE_OFFSET_M / PREPRESS_EE_QUAT_PANEL into press_button.py.

    ros2 run rammp_button_press measure_prepress_offset --target timer_clock
"""

import argparse
import sys
import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from rammp_button_press import Abort, say
from rammp_button_press.panel_frame import describe, from_panel, rot_to_panel, to_panel
from rammp_button_press.press_button import (
    FINGERTIP_PAST_TOOL_M,
    MIN_PREPRESS_FINGERTIP_CLEAR_M,
    STANDOFF_M,
    Run,
    build_parser,
    fingertip_out_of_panel,
    measure_panel_frame_domes,
)

DRIVER_BACK_TIMEOUT_S = 120.0


def wait_for_driver(run: Run) -> None:
    """Block until /joint_states and /ee_state are fresh again (sheppy back on `arm-camera`)."""
    t0 = time.time()
    while time.time() - t0 < DRIVER_BACK_TIMEOUT_S:
        try:
            run.arm.state()
            time.sleep(1.0)  # let a few ~100 Hz readings in after the restart
            run.arm.state()
            return
        except Abort:
            time.sleep(0.5)
    raise Abort(f"arm driver not back within {DRIVER_BACK_TIMEOUT_S:.0f} s -- is the `arm-camera` profile up?")


def main() -> int:
    """Detect the panel frame, wait for the hand-placed fingertip, then print the tool pose in
    the panel frame as the two press_button constants."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True)
    ap.add_argument(
        "--touch",
        action="store_true",
        help=f"fingertip touching the button (not {STANDOFF_M*100:.0f} cm in front); backed off here",
    )
    a, rest = ap.parse_known_args()
    run_args = build_parser().parse_args(["--target", a.target, *rest])  # never --execute
    if run_args.execute:
        say("measure_prepress_offset never moves the arm; drop --execute")
        return 2

    rclpy.init()
    run = None
    try:
        try:
            run = Run(run_args)
            run.preflight(require_ready=False)
            say("\n== 1/2: panel frame from this view ==")
            origin, R = measure_panel_frame_domes(run, a.target)
            where = "ON the centre of" if a.touch else f"{STANDOFF_M*100:.0f} cm in front of"
            input(
                f"\n== 2/2: sheppy -> camera-only, hand-guide the LEFT fingertip {where} "
                f"{a.target}, let go, sheppy -> arm-camera, then press Enter =="
            )
            wait_for_driver(run)
            problems, _ = run.arm.status()
            tool = [pr for pr in problems if pr.startswith("tool frame")]
            if tool:
                raise Abort(tool[0])
        except Abort as e:
            say(f"\nABORT: {e}")
            return 2
        p, Rm = run.arm.ee_pose()
        local = to_panel(p, origin, R)
        if a.touch:
            tip = to_panel(p + Rm[:, 2] * FINGERTIP_PAST_TOOL_M, origin, R)
            say(
                f"\n  touch: fingertip {describe(tip)} from the detected button "
                f"(the detection's error at this button, if the fingertip is really on its centre)"
            )
            local = local + np.array([0.0, 0.0, STANDOFF_M])  # back off along the panel normal
            p = from_panel(local, origin, R)
        q_local = Rotation.from_matrix(rot_to_panel(Rm, R)).as_quat()
        tool_z = rot_to_panel(Rm, R)[:, 2]
        say(f"\n  tool frame: {describe(local)}")
        say(
            f"  fingertip {fingertip_out_of_panel(p, Rm, origin, R)*100:+.1f} cm out of the fitted panel plane "
            f"(aim: {STANDOFF_M*100:.0f}; press_button refuses below {MIN_PREPRESS_FINGERTIP_CLEAR_M*100:.0f})"
        )
        say(
            f"  tool z axis in the panel frame {np.round(tool_z, 3)} "
            f"({np.degrees(np.arccos(np.clip(-tool_z[2], -1, 1))):.0f} deg off straight-in)"
        )
        say("\nPaste into rammp_button_press/press_button.py:\n")
        say(f"PREPRESS_EE_OFFSET_M = np.array([{local[0]:.4f}, {local[1]:.4f}, {local[2]:.4f}])")
        say(
            f"PREPRESS_EE_QUAT_PANEL = np.array([{q_local[0]:.5f}, {q_local[1]:.5f}, "
            f"{q_local[2]:.5f}, {q_local[3]:.5f}])   # xyzw"
        )
        run.log(
            {
                "stage": "measure_prepress_offset",
                "target": a.target,
                "ee_local": local.tolist(),
                "touch": a.touch,
                "quat_panel": q_local.tolist(),
                "panel_origin": origin.tolist(),
                "panel_R": R.tolist(),
            }
        )
        return 0
    finally:
        if run is not None:
            run.arm.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
