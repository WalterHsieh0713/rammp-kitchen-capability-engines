# `rammp_button_press` — the microwave button press on RAMMP's arm driver

The autonomous button press from `feeding_deployment/button_press/` (branch `button-task`; read
its `README.md` for how detection and the panel frame work), ported from the Cornell arm RPC
(`ArmInterfaceClient` / `arm_server.py`) to RAMMP's `kinova_gen3_node`, running natively next
to sheppy (RAMMP's hybrid workflow). An `ament_python` package, self-contained: it needs
neither `feeding_deployment` nor its RPC stack installed.

**New rig (different Jetson / arm)? Start with [`NEW_RIG.md`](NEW_RIG.md).**

**Status (2026-10-12):** builds; 35 offline tests pass; full dry runs against a mock driver +
synthetic camera (detect → panel frame → every leg gated). On hardware: session checks, live
detection and a dry run pass; `PREPRESS_EE_OFFSET_M` re-measured on this rig (hand-placed
pre-press spot from a 33.5 cm view), and an attended 1.0 cm push along the normal from that spot
clicked start_30s (offset measured against timer_clock, shifted by the dome layout). **The full `press_button --execute` sequence has not run yet.**

**Scene camera far stage (2026-10-14):** the first look now comes from the fixed scene camera
(see [below](#scene-camera-far-stage)), so the arm can start anywhere. Offline tests pass and the
finder works on live frames; **no press has been staged from it yet**.

## Setup (once per box)

```bash
# the package lives in the feeding-deployment repo; ~/ros2_ws/src holds a symlink
ln -s ~/feeding-deployment/rammp_button_press ~/ros2_ws/src/rammp_button_press
cd ~/ros2_ws && source /opt/ros/humble/setup.bash
colcon build --packages-select rammp_button_press    # NOT --symlink-install (see below); rebuild after edits (~2 s)
```

- `rammp-interfaces-ros2` must be built in the same workspace at the tag the arm image uses
  (v1.1.0 for `kinova-gen3-ros2:1.1.0`). Don't modify it.
- Python deps come from a venv created with `--system-site-packages` (so `rclpy`/`cv_bridge`
  come from `/opt/ros`): numpy \< 2, scipy, opencv, pybullet. On the Jetson this is
  `~/feeding-deployment/.venv`. No system-wide pip.
- Scripts get `#!/usr/bin/env python3` (`setup.cfg`), so `ros2 run` uses the activated venv.
  `--symlink-install` ignores that and writes `/usr/bin/python3`, which has no pybullet.
- The hand-eye calibration (`end_effector_link → camera_color_optical_frame`, easy_handeye2
  `.calib`) for **this** camera mount: default
  `~/.ros2/easy_handeye2/calibrations/wrist_camera_calib.calib`, override with `BUTTON_CALIB`.

## A session

Every terminal: `source ~/feeding-deployment/rammp_button_press/sheppy/env.sh`

```bash
# 1. bring-up (no motion: the driver holds the arm where it is)
cd ~/feeding-deployment/rammp_button_press/sheppy
sheppy up --manifest sheppy-manifest.yaml arm-camera
sheppy status                                         # arm, arm_tf, camera_calib_tf, wrist_camera running

# 2. session-start checks: gripper + arm state, camera, tf
ros2 topic echo --once /control_status                # estopped false; arbitration / owner
ros2 topic echo --once --qos-reliability best_effort /gripper_state
ros2 topic hz /wrist_camera/aligned_depth_to_color/image_raw   # ~15 Hz
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame

# 3. panel in view, camera >= 22 cm from it (watch the domes)
ros2 run rammp_button_press view_detection --target timer_clock

# 3b. the scene camera's view of the microwave (front outlined, button ringed)
ros2 run rammp_button_press view_detection --scene --target timer_clock

# 4. dry run (default): detect, plan and gate every leg, move nothing
ros2 run rammp_button_press press_button --target timer_clock

# 5. ONLY with the go-ahead for this specific move, someone at the e-stop:
ros2 run rammp_button_press press_button --target timer_clock --execute

sheppy down
```

Releasing the arm for hand-guiding or teleop is `sheppy up ... camera-only`, which stops the
driver. **Never hand-guide or teleop with the driver up**: `WRONG_SERVOING_MODE` crash (2026-10-07).
`sheppy up ... arm-camera` brings it back, and it starts holding wherever the arm is.

### Re-measuring the pre-press spot (required on this rig)

`PREPRESS_EE_OFFSET_M` / `PREPRESS_EE_QUAT_PANEL` in `press_button.py` were measured on
rchi-cpu-5, in that rig's *detected* panel frame, so they absorb its hand-eye bias. With this
URDF's fingertip (tool frame + 5.96 cm) they put the fingertip **7 cm into the panel**. The
10-07 microwave port saw the same rig's grasp constants land 9.8 cm too deep on this Jetson.
`check_fingertip` refuses any pre-press spot whose fingertip is less than 1 cm in front of the
fitted plane. To re-measure:

```bash
ros2 run rammp_button_press measure_prepress_offset --target timer_clock
# detects from ~25 cm, then asks you to: sheppy -> camera-only, hand-guide the LEFT fingertip
# 2 cm in front of the button, sheppy -> arm-camera, Enter. Prints the two constants to paste.
```

## Interactive run: a person confirms each step

```bash
ros2 run rammp_button_press press_button --target start_30s --execute --interactive
```

In a terminal of its own (it reads single keys; under sheppy's TUI there is no terminal and it
refuses to start), with Foxglove showing the wrist camera: connect to `ws://<jetson>:8780` (the
read-only bridge from box-opening's `press_demo_launch`, which the `button-press` profile runs, see
`RAMMP-box-opening/docs/FOXGLOVE.md`) and import `foxglove/button_press.json`.

1. The scene camera finds the microwave; the arm goes to the staging pose and **stops**.
   **space**: find the buttons and go to the button. **q**: go back to the start.
1. At the button (the pre-press spot) it **stops** again. **space**: one press, then it waits
   again (space again = press again). **q**: back out 15 cm, then back to the start.

The start is wherever the arm was when the run began. Going back always backs straight out from
the panel before cuRobo takes over (as every return does). Keys typed while the arm moves are
thrown away. If something fails at a stop (no domes at staging, a gate, a press), the arm holds and
says why; then only **q** works, and the run ends back at the start. Ctrl-C ends the run where it is.
Each answer is a `decision` record in the run log. `--presses` is ignored (one press per space),
`--no-return` is refused, and without `--execute` it is the ordinary dry run (no stops).
Design: `docs/superpowers/specs/2026-10-16-interactive-press-design.md`.

## Scene camera far stage

The first look (`--far scene`, the default) uses the fixed **scene camera** (Orbbec Gemini 336L,
`/scene_camera`), not the wrist camera, so the arm may start anywhere: facing away, at HOME, wherever
the last task left it. `scene_finder.py` finds the cream Comfee's **front** by geometry (a vertical
plane of the front's size, standing on a surface, facing the arm, with the dark door window on the
left and the control strip on the right; cream as a confirmation) and places the button by a stored
offset from the front's bottom-right corner. From ~1.2 m it is good to a few cm: **staging only**.
cuRobo takes the arm straight to the pre-press pose + `SCENE_STAGE_OUT_M` (15 cm), the wrist-turn
limit lifted for that one move, and the wrist camera's dome detection there plans the press, exactly
as before.

- **Needs** the `scene_camera` node and its tf (`base_link -> scene_camera_*`), which box-opening's
  launch publishes from its `config/camera_scene.yaml`. This package reads that tf as published and
  changes nothing of box-opening's. This package's own sheppy manifest has no scene camera: run the
  bench profile that has it.
- **Falls back** to the wrist camera (domes, then the display) whenever the scene camera finds
  nothing usable: no frames, no tf, fewer than 4 of 5 frames agreeing within 2 cm / 5 deg, two
  microwave-like fronts, or no front passing the gates (a partly hidden one fails the size gate;
  "arm in the way?" when the tool stands in front of it). Every reason is printed and logged.
  `--far wrist` is the behaviour before 2026-10-14.
- **Constants** (`scene_finder.py`): `FRONT_W_M` / `FRONT_H_M` are what the finder measured on the
  HERL microwave (tape-measure to confirm). `FACE_CORNER_TO_START_MM` started from the button's
  pixel in the scene image; after a few staged runs, `python3 tools/scene_offset_from_logs.py`
  prints the median of what the wrist camera actually found.
- **Logs**: `panel_frame_scene` (the front, the frames used, rejected frames' reasons),
  `scene_fallback`, `staging_redetect` (now with `source` and `moved_cm`: the scene-vs-wrist distance,
  which is how calibration drift shows up), and `abort` (every abort's reason, for all runs).
- If the wrist camera finds no domes at the staging pose, the run holds there and names the scene
  calibration or the corner offset as the likely cause.
- **Not covered**: a front turned away from the scene camera, a front hidden by the fridge door
  (recorded as `tools/frames/scene/neg_fridge_door_open_*`), other microwave models, the microwave
  in cuRobo's world (it is not there: watch the staging move).
- **Tests**: `test/test_scene_finder.py` ray-casts a box model of the microwave (moved, turned,
  wrong size, no window, grey, two of them, hidden, floating, facing away) and runs the recorded
  scene frames in `tools/frames/scene/` (`record_scene_frames.py LABEL`; `neg_*` = must find nothing).

## As a RAMMP module: the `/press_button` action server

`press_server` serves `/press_button` (`rammp_button_press_interfaces/action/PressButton`, its
own small interface package next to this one -- `rammp-interfaces-ros2` stays untouched):

```
goal     target, presses, press_in_m (must be 0: the depth is fixed at PRESS_IN_M), execute, no_return
result   success, moved, message ("done" / why it stopped; "HOLDING HERE" = holding), log_file
feedback stage -- one line per step, exactly what press_button prints
```

Same code path as the script (`press_button.press_button` per goal), with the camera, arm
interface and kinematics kept loaded (~1 s per dry-run goal). **The arm moves only if the server
was started with `allow_execute:=true` and the goal asks to execute**; otherwise the goal runs as a
dry run and the result says so. Cancel = stop the trajectory in flight and hold. One goal at a
time. Sheppy runs it as node `button_press`:

```bash
sheppy up --manifest sheppy-manifest.yaml arm-camera           # server: dry runs only
sheppy up --manifest sheppy-manifest.yaml arm-camera-execute   # server may move the arm -- e-stop manned
ros2 run rammp_button_press press_client --target start_30s [--presses 3] [--execute]   # Ctrl-C = cancel
```

Don't run the `press_button` script while the server is executing a goal: they would both
command the arm. `rammp-alternative.yaml` (package root) is the fragment for adding the server to
a RAMMP manifest, with what it needs from the rest of the stack (driver, cuRobo planner, wrist
camera topics, a calibrated camera tf -- per rig, with a re-measured pre-press offset).

Build: `rammp_button_press_interfaces` must be built in the same workspace:
`colcon build --packages-select rammp_button_press_interfaces rammp_button_press`.

## What changed from `button-task`

| Old (`ArmInterfaceClient`)                       | Now (`RammpArm`, `rammp_arm.py`)                                                                                                                    |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_state()["position"]`                        | `/joint_states` (best effort; `joint_1..7`, wrapped to ±π)                                                                                          |
| `get_state()["ee_pos"]` (Kortex tool pose)       | `/ee_state` + 0.120 m along the flange z (`/ee_state` is the flange, `end_effector_link`)                                                           |
| `get_state()["gripper_pos"]`                     | `/gripper_state.position` (0 open … 1 closed)                                                                                                       |
| `get_arm_state()` (`SERVOING_READY`)             | `/control_status` (e-stop, arbitration, owner) + `/diagnostics` (`Arm` task)                                                                        |
| `get_speed()` low/medium                         | none. Each leg is timed here: min-jerk, peak 6 cm/s tool, 0.35 rad/s per joint                                                                      |
| `CartesianTrajectoryCommand` (Kortex's own IK)   | `execute_joint_trajectory`: **the exact joint solutions the sim gates checked**, 20 ms samples, driver-side `path_tolerance` 0.15 rad               |
| full `FeedingDeploymentPyBulletSimulator` for IK | `urdf/gen3_tool_frame_kinematic.urdf` alone (same chain, same `tool_frame`; FK/IK identical to the full sim, tested); loads in 10 ms instead of 2 s |
| camera `/camera/...`, frame `arm_base_link`      | `/wrist_camera/...` (`--camera-ns`), frame `base_link`                                                                                              |
| `feeding_deployment.ros2.calibration_tf`         | `calibration_tf` in this package (RAMMP publishes no wrist-camera tf)                                                                               |

Unchanged: `dome_pattern.py`, `panel_frame.py`, `geometry.py` are copies (only the import path
differs; `test/test_sync_with_feeding_deployment.py` fails if they drift). The detection, panel
frame, staging and planning in `press_button.py`, and every existing gate: IK error ≤ 5 mm,
orientation ≤ 1°, joint step ≤ 10°, reach ≤ 0.91 m, z band 0.12–0.75 m, wrist turn ≤ 45° (the 40 cm move cap was removed 2026-10-12),
press ≤ 3 cm, final position within 1 cm (approach and press; staging, bail and back-to-start within 5 cm since 2026-10-12).

**New refusals** (none of them loosen anything):

- **Joint limits:** J2/J4/J6 must stay 2° inside the driver URDF's limits. Kortex used to do
  the IK; now our solutions go to the driver as they are. Raw PyBullet IK does return
  out-of-range J4.
- **Fingertip clearance:** see above.
- **Tool-frame check at preflight:** sim FK at the live joints must match `/ee_state` + 0.12 m
  to within 5 mm. If the driver ever reports a different frame, the pre-press offset would be
  silently wrong, so the run refuses.
- **Start drift:** a leg is refused if the arm moved more than 0.5° since it was planned.
- **Arbitration:** if `arbitration_mode` is `enforced`, `--acquire-control` acquires a token
  only when nobody owns the arm. `/acquire_control` *seizes*, so it never takes the arm from
  another owner. The token is released at exit.

**Long moves (staging, back to start, bail)** always go through the driver's `go_to_ee_pose`, but never starting at the panel: back to start and a bail from the approach first back straight out `RETREAT_OUT_M` (15 cm) along the panel normal, and an aborted press stroke goes straight back to the pre-press spot (2026-10-14: cuRobo's return from 2 cm in front of the button swept the gripper into the microwave). Then
cuRobo plans from the measured joints and the driver
executes. The target is `tool_frame` in `base_link`, the same frame as `ee_pose()`. What that gives up:
the path can't be previewed or gated before it runs (only the goal's reach and z band are checked,
and a dry run checks only the goal). These moves run at cuRobo's **full planned speed**
(`GO_TO_SPEED_SCALE` = 1.0 in `rammp_arm.py`, the goal's `speed_scale`; lower it to slow them down): on
2026-10-14 a 51 cm staging move from HOME took 1.2 s at full speed and hit the microwave. cuRobo's world
has no microwave, counter, wheelchair or user. The approach and the press
never use `go_to_ee_pose` (its path is not a straight line). The only gripper command is preflight's close on an `--execute` run (never in a dry run, never opened).

## Open questions (for Swapnil) and what we found

1. **Does RAMMP publish a wrist-camera tf?** No calibrated one. The December manifest runs the
   camera and the driver only, so we publish `arm_tf` (robot_state_publisher on the driver
   URDF) and `camera_calib_tf` ourselves.
1. **Which tool frame does `/ee_state` report?** The flange (`end_effector_link`), checked
   2026-10-07: equal to `tf2_echo base_link end_effector_link`. The driver doc calls it "tool
   pose". The preflight check above keeps verifying it.
1. **Is arbitration enforced?** `disabled` in RAMMP's own fragment and in our manifest.
   Enforced at HERL? `--acquire-control` covers it, but only when no one owns the arm.
1. **Own rammp-org repo/module later?** Open. The package is self-contained for that. A
   `rammp-alternative.yaml` fragment would follow once the press runs as a node, not a
   one-shot script.

## Still to check on hardware

- The camera on the wrist: the local manifest says D435i (serial 140122079014, 2026-10-07),
  RAMMP's December manifest says D405. The detector's ~22 cm minimum range and depth behaviour
  were tuned on rchi-cpu-5's camera.
- The calibration on this box is Capybara's (copied 2026-10-07). Touch-checked to ~2 cm for
  the door handle, never for buttons.
- The first motion: `--presses 0 --no-return` from 23 cm, attended, before any press.
