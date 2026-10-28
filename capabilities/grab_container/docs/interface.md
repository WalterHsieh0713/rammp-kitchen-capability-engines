# Grab Container Capability — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @seshk944
**Tier:** experimental

## What it does

Grabs the food container (the tupperware) by its protruding sleeve ring, a side
pinch of the flat brim, and holds it a little above where it stood. `grab`
finds the ring in the wrist camera (YOLO segmentation, 5 frames that agree to
1 cm, the grasp point on the ring's near rim), then opens the gripper, moves to
a pre-grasp 10 cm back along the approach, moves in, closes and raises 6 cm.
Every move is a cuRobo move through `self.arm.go_to_ee_pose`. `park` sends the
arm to the scan pose by a direct joint move, which works when cuRobo refuses the
start state; it is not collision-checked. Any failed check stops the action
with the arm holding where it is. It is activated when the user selects a
`container`. Without it, nothing picks the container up.

Ported from the tupperware grasp in `dhyi-svg/feeding-deployment@8f5a3352`
(`rammp_grasp_tupperware/`).

The arm moves only when the node runs with `allow_execute: true` and the goal
sets `execute`; otherwise every goal is a dry run (`grab` detects and plans
every leg with cuRobo; `park` reports the move) and nothing moves.

## Publishes

| Topic                    | Type              | Rate | Meaning                                                                                                   |
| ------------------------ | ----------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/grab_container/status` | `std_msgs/String` | 1 Hz | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |

## Subscribes

Only while `grab` runs; `self.arm` also subscribes to the arm driver's state then.

| Topic                                            | Type                     | Required | Meaning                        |
| ------------------------------------------------ | ------------------------ | -------- | ------------------------------ |
| `/wrist_camera/color/image_raw`                  | `sensor_msgs/Image`      | yes      | Colour, for the ring detector  |
| `/wrist_camera/color/camera_info`                | `sensor_msgs/CameraInfo` | yes      | Intrinsics                     |
| `/wrist_camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image`      | yes      | Depth aligned to colour        |
| `/tf`, `/tf_static`                              | `tf2_msgs/TFMessage`     | yes      | `arm.frame` \<- `camera_frame` |

## Orchestrator interface

Served under `/grab_container/`, types from `rammp_orchestrator_interfaces`:

| Name                             | Type                   | Meaning                                       |
| -------------------------------- | ---------------------- | --------------------------------------------- |
| `/grab_container/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/grab_container/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/grab_container/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/grab_container/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

| Action | Params                            | Result payload                                                    |
| ------ | --------------------------------- | ----------------------------------------------------------------- |
| `grab` | `execute` (bool, default `false`) | `{"executed", "grasp_point", "approach", "raised", "note"}`, in m |
| `park` | `execute` (bool, default `false`) | `{"executed", "note"}`                                            |

A failed check fails the action with `refused` (recoverable); the arm holds where it is.

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name               | Default                        | Meaning                                                   |
| ------------------ | ------------------------------ | --------------------------------------------------------- |
| `use_orchestrator` | `true`                         | `false` runs standalone: no registration, drive it by CLI |
| `mock`             | `false`                        | `true` skips hardware (the mock alternative sets it)      |
| `allow_execute`    | `false`                        | `false`: every goal is a dry run, whatever it asks        |
| `model_path`       | `"/models/best.pt"`            | YOLO segmentation weights with the `grabber` class        |
| `camera_ns`        | `"/wrist_camera"`              | Namespace of the wrist camera's topics                    |
| `camera_frame`     | `"camera_color_optical_frame"` | The camera's tf frame                                     |

`self.arm` reads these `arm.*` parameters once at startup and refuses changes
afterwards. They are set per robot like any other; `arm.mock_urdf` names the
URDF `make_sim()` loads in mock runs (the real driver supplies its own).
`arm.sender_id` is the one left out of the real fragment: its default is the
engine id, so each capability sends its goals under its own name.

| Name                           | Default                                            | Meaning                                                    |
| ------------------------------ | -------------------------------------------------- | ---------------------------------------------------------- |
| `arm.tool_z`                   | `0.12`                                             | Tool frame offset along the flange z (m)                   |
| `arm.dt`                       | `0.02`                                             | Trajectory sample spacing (s)                              |
| `arm.joint_speed`              | `0.85`                                             | Peak joint speed of joint moves (rad/s)                    |
| `arm.cart_speed`               | `0.16`                                             | Peak tool speed of Cartesian paths (m/s)                   |
| `arm.contact_joint_speed`      | `0.5`                                              | Peak joint speed of contact moves (rad/s)                  |
| `arm.contact_cart_speed`       | `0.09`                                             | Peak tool speed of contact moves (m/s)                     |
| `arm.min_duration_s`           | `1.0`                                              | Shortest trajectory duration (s)                           |
| `arm.cart_step_m`              | `0.02`                                             | Longest Cartesian step before IK (m)                       |
| `arm.cart_step_deg`            | `5.0`                                              | Largest Cartesian turn per step before IK (deg)            |
| `arm.rot_equiv_m`              | `0.15`                                             | Travel a 1 rad turn counts as when timing a path (m)       |
| `arm.max_knot_jump_deg`        | `20.0`                                             | Largest joint change between Cartesian knots (deg)         |
| `arm.path_tol_rad`             | `0.15`                                             | Per-joint tracking tolerance of trajectories (rad)         |
| `arm.gripper_force`            | `0.5`                                              | Default gripper force (0..1)                               |
| `arm.gripper_speed`            | `0.5`                                              | Default gripper speed (0..1)                               |
| `arm.frame`                    | `"arm_mount"`                                      | Frame of go_to_ee_pose goals and /ee_state                 |
| `arm.sender_id`                | the engine id (RammpArm standalone: `"microwave"`) | sender_id on goals (the engine id under the engine)        |
| `arm.go_to_speed_scale`        | `0.3`                                              | speed_scale of go_to goals (0.01..1)                       |
| `arm.go_to_timeout_s`          | `120.0`                                            | go_to planning plus execution limit (s)                    |
| `arm.go_to_track_m`            | `0.01`                                             | go_to_ee_pose: largest tool error at the end (m)           |
| `arm.go_to_track_deg`          | `2.0`                                              | go_to_joint_config: largest end joint error (deg)          |
| `arm.max_ik_err`               | `0.02`                                             | IK position error above which a pose is unreachable (m)    |
| `arm.max_step_jump_deg`        | `20.0`                                             | Largest joint change of a planned step (deg)               |
| `arm.j6_guard_deg`             | `115.0`                                            | Largest absolute J6 the sim accepts (deg)                  |
| `arm.j4_guard_deg`             | `144.0`                                            | Largest absolute J4 the sim accepts (deg)                  |
| `arm.max_continuous_delta_deg` | `170.0`                                            | Largest short-way step of a free joint (deg)               |
| `arm.keep_out_err`             | `9.0`                                              | IK error reported for a keep-out solution (m)              |
| `arm.ik_max_iterations`        | `200`                                              | PyBullet IK iterations per solve                           |
| `arm.clearance_query_m`        | `0.5`                                              | clearance() ignores bodies farther than this (m)           |
| `arm.fingertip_open_rad`       | `0.0`                                              | Sim finger joint angle with the gripper open (rad)         |
| `arm.finger_closed_rad`        | `0.7`                                              | Sim finger joint angle with the gripper closed (rad)       |
| `arm.state_wait_s`             | `10.0`                                             | First arm call: wait for the driver's state (s)            |
| `arm.state_stale_s`            | `0.5`                                              | Motion state older than this is an error (s)               |
| `arm.diagnostics_stale_s`      | `3.0`                                              | /diagnostics older than this is an error (s)               |
| `arm.goal_accept_s`            | `5.0`                                              | Wait for the driver to accept or reject a goal (s)         |
| `arm.result_margin_s`          | `30.0`                                             | A trajectory's result may come this long after its end (s) |
| `arm.poll_s`                   | `0.01`                                             | Poll period of a running motion (s)                        |
| `arm.cancel_settle_s`          | `0.4`                                              | Wait for a cancelled goal to settle (s)                    |
| `arm.halt_status_wait_s`       | `0.2`                                              | After a HALTED goal, wait for an e-stop status (s)         |
| `arm.planner_wait_s`           | `5.0`                                              | Wait for the cuRobo planner's action server (s)            |
| `arm.go_to_settle_s`           | `0.3`                                              | After a go_to goal, let the state catch up (s)             |
| `arm.gripper_period_s`         | `0.02`                                             | gripper() setpoint period (s)                              |
| `arm.gripper_moved`            | `0.002`                                            | gripper(): position change that counts as moving (0..1)    |
| `arm.gripper_judge_after_s`    | `0.5`                                              | gripper(): judge the fingers still only after (s)          |
| `arm.gripper_still_s`          | `0.3`                                              | gripper(): still this long means settled (s)               |
| `arm.plan_timeout_s`           | `30.0`                                             | Wait for a cuRobo plan (s)                                 |
| `arm.service_wait_s`           | `5.0`                                              | Wait for a service server to come up (s)                   |
| `arm.service_call_s`           | `10.0`                                             | Wait for a service's answer (s)                            |
| `arm.stream_timeout_s`         | `0.1`                                              | stream(): default setpoint timeout of a session (s)        |
| `arm.publisher_match_s`        | `1.0`                                              | Wait for a new publisher to reach the driver (s)           |
| `arm.close_stream_wait_s`      | `1.0`                                              | Wait for /close_stream's answer (s)                        |
| `arm.mock_joints`              | `[0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]`        | Mock mode: the joints it starts at (rad, J1..J7)           |
| `arm.mock_urdf`                | `""`                                               | Mock mode: the URDF file make_sim() loads (empty: none)    |

## Hardware

The arm and gripper (through the arm driver), the wrist camera, and the GPU for
the detector (`gpus: all`). The detector's weights are mounted from
`/home/jetson/oxo` at `/models` (`model_path`).

## Alternatives

| id                    | What it is                                                                         |
| --------------------- | ---------------------------------------------------------------------------------- |
| `grab_container_real` | The module proper                                                                  |
| `grab_container_mock` | Same interface, no hardware — for developing against this module without owning it |
