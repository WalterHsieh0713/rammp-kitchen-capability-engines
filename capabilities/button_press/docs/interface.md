# Button Press Capability — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @EvuhLi
**Tier:** experimental

## What it does

Presses one button on the microwave's control panel with the Kinova arm. On a `/press_button`
goal it finds the microwave (scene camera first, then the wrist camera's dome detector), stages in
front of the panel, re-detects the button from about 25 cm, plans and gates every leg in a
kinematic sim, then goes to a pre-press spot, presses, and returns to where the arm started. Any
failed check stops the run with the arm holding where it is. Without it, no kitchen task can start
the microwave.

**The arm moves only if the node runs with `allow_execute: true` and the goal sets `execute`.**
Otherwise every goal is a dry run: detect, plan and gate, nothing commanded.

## Action

| Name            | Type                                               | Meaning                     |
| --------------- | -------------------------------------------------- | --------------------------- |
| `/press_button` | `rammp_button_press_interfaces/action/PressButton` | Press one button, see below |

- **Goal:** `target` (`start_30s`, `timer_clock`, `stop_eco`, `power_level`,
  `wgt_time_defrost`), `presses` (0–10; 0 = go to the spot and back), `press_in_m` (must be 0),
  `execute`, `no_return`.
- **Result:** `success`, `moved`, `message` ("done", or why it stopped; "HOLDING HERE" = the arm
  holds where it is), `log_file` (the run's JSONL log, in the container's `log_dir`).
- **Feedback:** `stage`, one progress line per step.
- **Cancel:** stops the trajectory in flight; the arm holds where it is.

One goal at a time; others are rejected, as are all goals until the node has reached the arm
driver (it retries every 2 s and never exits for want of it).

```bash
docker exec -it <container> /ros_entrypoint.sh ros2 run rammp_button_press press_client --target start_30s
```

## Publishes

| Topic                       | Type                                   | Rate             | Meaning                                      |
| --------------------------- | -------------------------------------- | ---------------- | -------------------------------------------- |
| `/button_press/target_pose` | `geometry_msgs/PoseStamped`            | once per goal    | The planned pre-press tool pose, for viewing |
| `/setpoint/twist`           | `rammp_arm_interfaces/TwistSetpoint`   | 100 Hz, in press | The press (driver's `ee_twist` stream)       |
| `/setpoint/gripper`         | `rammp_arm_interfaces/GripperSetpoint` | at preflight     | Closes the gripper on an execute run         |

It also calls the driver's `/execute_joint_trajectory` and `/go_to_ee_pose` actions,
`/open_stream`, `/close_stream`, and (with `acquire_control`) `/acquire_control` /
`/release_control`.

## Subscribes

| Topic                                                                                                                 | Type                              | Required | Meaning                                             |
| --------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------- | --------------------------------------------------- |
| `/joint_states`, `/ee_state`                                                                                          | `JointState`, `EeState`           | yes      | Arm state (driver)                                  |
| `/gripper_state`, `/control_status`                                                                                   | `GripperState`, `ControlStatus`   | yes      | Gripper closed? e-stop, arbitration                 |
| `/diagnostics`                                                                                                        | `diagnostic_msgs/DiagnosticArray` | no       | Arm health at preflight                             |
| `/tf`, `/tf_static`                                                                                                   | `tf2_msgs/TFMessage`              | yes      | Arm and camera mounts into `arm_frame`              |
| `/wrist_camera/color/{image_raw,camera_info}`                                                                         | `Image`, `CameraInfo`             | yes      | Dome detection                                      |
| `/wrist_camera/<depth_topic>`                                                                                         | `sensor_msgs/Image`               | yes      | Depth **aligned to colour**                         |
| `/scene_camera/color/{image_raw,camera_info}`, `/scene_camera/<scene_depth_topic>`, `/scene_camera/depth/camera_info` | `Image`, `CameraInfo`             | no       | First look; without it the wrist camera looks first |
| `/yolo/detections`                                                                                                    | `vision_msgs/Detection2DArray`    | no       | Narrows the scene search to the microwave           |

## Parameters

ROS parameters, declared with defaults under `params:` in both fragments (flatbot's values). A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy passes them as a
params file.

| Name                | Default (fragment)     | Meaning                                                                                |
| ------------------- | ---------------------- | -------------------------------------------------------------------------------------- |
| `allow_execute`     | `false`                | `true` lets execute goals move the arm. Attended runs only                             |
| `acquire_control`   | `false`                | Take an arbitration token (only if nobody holds the arm) where arbitration is enforced |
| `arm_frame`         | `arm_mount`            | The frame the run works in: `/ee_state`'s and `go_to_ee_pose`'s. Its z must point up   |
| `joint_prefix`      | `arm_`                 | Driver joint names are `<prefix>joint_1..7`                                            |
| `mount_rpy_deg`     | `[0.0, 30.0, 0.0]`     | The arm base's orientation in `arm_frame` (the robot's `mount.yaml`)                   |
| `camera_frame`      | `""`                   | Wrist camera optical frame; `""` = the colour image's own `frame_id`                   |
| `camera_ns`         | `/wrist_camera`        | Wrist camera namespace                                                                 |
| `depth_topic`       | `depth/image_rect_raw` | Wrist depth under `camera_ns`, aligned to colour                                       |
| `far`               | `scene`                | First look: `scene` (scene camera, falls back to wrist) or `wrist`                     |
| `scene_ns`          | `/scene_camera`        | Scene camera namespace                                                                 |
| `scene_depth_topic` | `depth/image_rect_raw` | Scene depth under `scene_ns`                                                           |
| `log_dir`           | `/logs`                | Run logs (mounted from the host's `/tmp/button_press_logs` in the real fragment)       |

Per-rig constants that are not parameters yet: `PREPRESS_EE_OFFSET_M` / `PREPRESS_EE_QUAT_PANEL`
in `press_button.py` (measure with `measure_prepress_offset`, then rebuild), and the scene finder's
`FACE_CORNER_TO_START_MM` (refine with `tools/scene_offset_from_logs.py`).

## Hardware

No devices of its own: it reaches the arm, the gripper and both cameras through their drivers'
topics. It needs the arm with a closed 2F-85 gripper, a wrist RGB-D camera with depth aligned to
colour (from ~22 cm), and optionally a fixed scene camera seeing the microwave's front.

**Per robot:** the camera mounts in tf (rammp-deployments `transforms/`), and the pre-press offset
measured on that robot against those mounts.

## Alternatives

| id                  | What it is                                                                           |
| ------------------- | ------------------------------------------------------------------------------------ |
| `button_press_real` | The module proper                                                                    |
| `button_press_mock` | Same interface, no hardware — every goal succeeds at once, nothing detected or moved |
