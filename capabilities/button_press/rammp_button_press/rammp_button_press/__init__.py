"""Autonomous microwave button press for the Kinova Gen3 on RAMMP's arm driver (ROS 2).

Port of ``feeding_deployment.button_press`` (branch ``button-task``) to RAMMP's
``kinova_gen3_node``, shipped as the ``button_press`` capability image. Interface:
``capabilities/button_press/docs/interface.md``; design and history: ``README.md``.

Modules:
  press_server             the capability's node: the /press_button action server
  press_client             test client for press_server
  press_button             one press run: detect -> plan and gate every leg -> go -> press -> return
  decide                   --interactive's decisions (space / q from the terminal)
  dome_pattern             main button detector: the 5 chrome domes, wrist camera
  scene_finder             first rough look from the fixed scene camera, staging only
  panel_frame, geometry    pure maths: the button-centred panel frame, plane fit, pixel rays
  kinematics               PyBullet IK and every motion safety gate (no ROS)
  rammp_arm                the real arm: state, trajectories, gripper, arbitration
  perception               ROS node holding the latest camera frames and tf
  measure_prepress_offset  one-off, per-rig measurement of the pre-press spot
  view_detection           live detection viewer (read-only)

``dome_pattern``, ``panel_frame`` and ``geometry`` were copied from ``feeding_deployment``.
"""

import sys

_output = None


class Abort(SystemExit):
    """Stop the run and HOLD the arm where it is.

    Raised by every safety gate; ``press_button.main`` catches it and prints the reason.
    """


def say(*parts, sep=" ", end="\n", flush=False):
    """A run's progress line, with print()'s signature. Goes to stdout, or wherever set_output()
    points it (press_server: the goal's action feedback and the node's log)."""
    out = _output or sys.stdout
    out.write(sep.join(map(str, parts)) + end)
    if flush:
        out.flush()


def set_output(stream):
    """Send say() to `stream` (anything with write/flush) instead of stdout; None restores stdout."""
    global _output
    _output = stream
