# Place Container — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @WalterHsieh0713
**Tier:** experimental

## What it does

Puts the food container, already gripped by its lip, down at a spot on the table and lets go.
It squares the hand (level, pointing 5° down), carries the container over to above the spot,
lowers it along a straight line in the driver's joint impedance mode until the table stops it,
holds a light press while the gripper opens, then backs the hand straight out. The spot is
given in `arm_mount` (x, y, table height), or marked by touching it with the fingertips (`mark`).
Finding a free spot from the cameras is not part of it (free-space segmentation belongs in
rammp-perception, rammp-deployments#96). Without it, the container stays in the hand after the
grasp. Ported from RAMMP-table, where `mark` → `place` put it on a marked sticky note on the
flatbot (2026-10-26).

## Publishes

| Topic                     | Type              | Rate | Meaning                                                                                                   |
| ------------------------- | ----------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/place_container/status` | `std_msgs/String` | 1 Hz | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |

## Subscribes

| Topic | Type | Required | Meaning                                                            |
| ----- | ---- | -------- | ------------------------------------------------------------------ |
| —     | —    | —        | Only the arm driver's state, through `self.arm` (during an action) |

## Orchestrator interface

Served under `/place_container/`, types from `rammp_orchestrator_interfaces`:

| Name                              | Type                   | Meaning                                       |
| --------------------------------- | ---------------------- | --------------------------------------------- |
| `/place_container/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/place_container/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/place_container/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/place_container/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

Every action is a dry run (plan and check, nothing moves) unless the node runs with
`allow_execute: true` and the goal sets `execute: true`.

| Action    | Params                                                                                              | Result payload                                        |
| --------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `place`   | `execute`; `x`, `y`, `table_z` (m, arm_mount; all or none = the marked spot); `allow_empty_gripper` | `{"executed", "plan"}` or `{"executed", "recording"}` |
| `mark`    | —                                                                                                   | `{"spot": [x, y], "table_z"}`                         |
| `retreat` | `execute`                                                                                           | `{"executed", "to"}`                                  |
| `reset`   | `execute`, `save` (save the current joints as the start)                                            | `{"executed", "plan"}` or `{"saved"}`                 |

Progress phases of `place`: `adjust`, `lift`, `transit`, `lower`, `open`, `retreat`. A failed
check ends the action with `refused` and the reason; after contact the arm is left holding in
impedance, before it in position control.

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file. The defaults are the flatbot's (2026-10-26).

| Name                     | Default                                      | Meaning                                                                             |
| ------------------------ | -------------------------------------------- | ----------------------------------------------------------------------------------- |
| `use_orchestrator`       | `true`                                       | `false` runs standalone: no registration, drive it by CLI                           |
| `mock`                   | `false`                                      | `true` skips hardware (the mock alternative sets it)                                |
| `allow_execute`          | `false`                                      | The arm moves only with this and `execute`; attended runs only                      |
| `flange_to_tip`          | `0.17955`                                    | Flange face to the closed fingertips (m); not ruler-measured on the flatbot         |
| `pitch_deg`              | `5.0`                                        | The hand points this many deg down while placing                                    |
| `vz_stop`                | `-0.01`                                      | Contact: the hand's z velocity back up to this (m/s, + up)                          |
| `hold_press`             | `0.003`                                      | The hold after contact, this far below the contact point (m)                        |
| `settle_s`               | `3.0`                                        | Held in impedance at the start this long before going down (s)                      |
| `early_stop`             | `0.03`                                       | Contact more than this above where the bottom meets the table: refused (m)          |
| `impedance_kq`           | `[80.0, 80.0, 80.0, 80.0, 30.0, 30.0, 30.0]` | Joint stiffness of every impedance move, J1..J7                                     |
| `impedance_zeta`         | `0.5`                                        | Damping ratio of every impedance move                                               |
| `impedance_torque_limit` | `[39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0]`    | Torque limit of every impedance move (Nm), J1..J7                                   |
| `lower_speed`            | `0.02`                                       | The hand's speed where the table should be (m/s): the contact test's speed          |
| `go_to_speed_scale`      | `0.4`                                        | speed_scale of the carrying moves (2x RAMMP-table's)                                |
| `adjust_speed_scale`     | `1.0`                                        | speed_scale of the squaring turn (2x)                                               |
| `retreat_speed`          | `0.06`                                       | The retreat's peak speed (m/s; 2x)                                                  |
| `reset_speed_scale`      | `1.0`                                        | speed_scale of reset's joint moves (2x)                                             |
| `reach`                  | `[0.7, 1.1]`                                 | Spot distance from the arm base axis with the full descent (m)                      |
| `allow_reconfigure`      | `false`                                      | Switch to another arm posture where the line down can't reach the table in this one |
| `max_adjust_deg`         | `15.0`                                       | Largest squaring turn (deg)                                                         |
| `max_spot_age_min`       | `30.0`                                       | A marked spot older than this is refused (min)                                      |
| `tool_frame_tol`         | `0.015`                                      | The arm's tool frame vs the model's (m)                                             |
| `state_dir`              | `"/config/place_container"`                  | Where the marked spot, start position and descent recordings are kept               |

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

The arm and gripper (through the arm driver). No devices or cameras of its own.

## Alternatives

| id                     | What it is                                                                         |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `place_container_real` | The module proper                                                                  |
| `place_container_mock` | Same interface, no hardware — for developing against this module without owning it |
