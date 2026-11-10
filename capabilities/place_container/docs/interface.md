# Place Container — Interface

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @WalterHsieh0713
**Tier:** experimental

## What it does

Puts the food container, already gripped by its lip, down at a spot on the table and lets go.
It squares the hand (level, pointing 5° down), carries the container over to above the spot,
takes it quickly straight down to just above the table (`lower_from`), lowers the last few cm along
a straight line in the driver's joint impedance mode until the table stops it, holds a light press
while the gripper opens, drops the open hand 1 cm (`release_drop`) and backs it out: straight
back, or, where the elbow runs out of room close to the arm, a short way back and then a 60° turn
of the hand about the wrist (seen from above) that swings the fingertips off the lip. The spot is
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
| `reset`   | `execute`, `save` (save the current joints as the start), `empty` (no tilt check), `big_turn`       | `{"executed", "plan"}` or `{"saved"}`                 |

Progress phases of `place`: `carry` (adjust, lift, transit and the approach down in one blended motion
through the planned joints; with `blend` false or `adjust_mode` joints, `adjust`, `lift`, `transit` on
their own), `lower`, `open`, `retreat`. `reset` with `big_turn` (empty hand only) lets a joint turn
more than 120° on the way back, to recover from a different arm posture. A failed
check ends the action with `refused` (recoverable) and the reason: what happened, where (m in
`arm_mount`), what to do next, then the driver's or cuRobo's own words. An error of the arm driver
itself ends it with the error's name (`StateTimeout`, `ControlTaken`, ...), also recoverable; after contact the arm is left holding in
impedance, before it in position control.

## On the user's UI

Every action puts a panel on the UI bridge (`ctx.update_user_panel`, PR #42): a title ("Placing the
container", ...), the phase in words (checking the plan, carrying it over, lowering it onto the
table, letting go, backing the hand away) and a progress bar. A failure is shown there with its full
reason (what happened, where, what to do next) and an OK button; the action waits for the OK (at
most `ui_error_wait_s`) before it ends with the error, because the panel closes when the action
ends. Standalone (no orchestrator, so no session token) the panel calls are logged no-ops.

In mock (`mock: true`), `place` walks through its phases on the panel, 1 s each, without hardware,
and `mock_fail` (a phase: `carry`, `lower`, `open` or `retreat`) ends it there with a refusal, to
try the error panel in a sim.

## Collision volumes

On flatbot the cuRobo planner composes the chair's platform, seat and occupant volumes
(rammp-deployments `config/robots/flatbot/curobo/`) into every world; this capability sets no
world of its own. Run it with the `flatbot` profile, not `flatbot_empty_world`. The table, the
container and anything on the table are NOT in that world.

On 2026-11-07 the arm moved 8 in (0.2032 m) forward on the chair (rammp-deployments #156). A spot
or start position saved before then is refused (`state_valid_after`): `mark` and `reset` with
`save` again.

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file. The defaults are the flatbot's (2026-10-26; speeds, settle and
contact arming 2026-11-10).

| Name                     | Default                                      | Meaning                                                                                                                                                                              |
| ------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `use_orchestrator`       | `true`                                       | `false` runs standalone: no registration, drive it by CLI                                                                                                                            |
| `mock`                   | `false`                                      | `true` skips hardware (the mock alternative sets it)                                                                                                                                 |
| `allow_execute`          | `false`                                      | The arm moves only with this and `execute`; attended runs only                                                                                                                       |
| `flange_to_tip`          | `0.17955`                                    | Flange face to the closed fingertips (m); not ruler-measured on the flatbot                                                                                                          |
| `pitch_deg`              | `5.0`                                        | The hand points this many deg down while placing                                                                                                                                     |
| `vz_stop`                | `-0.01`                                      | Contact: the hand's z velocity back up to this (m/s, + up)                                                                                                                           |
| `hold_press`             | `0.003`                                      | The hold after contact, this far below the contact point (m)                                                                                                                         |
| `release_drop`           | `0.01`                                       | After the gripper opens, the hand goes this far straight down before backing out, which then doesn't lift (m); 0 = off                                                               |
| `settle_s`               | `0.0`                                        | Held in impedance at the start this long before going down (s); 0 = no pause                                                                                                         |
| `contact_ignore_m`       | `0.08`                                       | Contact is ignored for the descent's first this-many m (the arm's sag and catch-up); at most up to 1 cm above where the table should be                                              |
| `early_stop`             | `0.03`                                       | Contact more than this above where the bottom meets the table: refused (m)                                                                                                           |
| `impedance_kq`           | `[80.0, 80.0, 80.0, 80.0, 30.0, 30.0, 30.0]` | Joint stiffness of every impedance move, J1..J7                                                                                                                                      |
| `impedance_zeta`         | `0.5`                                        | Damping ratio of every impedance move                                                                                                                                                |
| `impedance_torque_limit` | `[39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0]`    | Torque limit of every impedance move (Nm), J1..J7                                                                                                                                    |
| `lower_from`             | `0.06`                                       | The slow impedance lowering starts this far above where the container's bottom meets the table; the carry goes straight down to there (m)                                            |
| `lower_speed`            | `0.02`                                       | The hand's speed where the table should be (m/s): the contact test's speed. Up to `arm.cart_speed` / 1.5; the contact thresholds scale with it, `vz_stop` must stay above -0.75 x it |
| `go_to_speed_scale`      | `0.6`                                        | speed_scale of the carrying moves and the reset (3x RAMMP-table's)                                                                                                                   |
| `blend`                  | `true`                                       | adjust -> lift -> transit as one blended cuRobo motion; `false` = separate moves as on 2026-10-29                                                                                    |
| `adjust_mode`            | `"pose"`                                     | With `blend`: `pose` = the squaring turn is the motion's first target; `joints` = a joint move to our checked joints first                                                           |
| `carry_targets`          | `"joints"`                                   | With `blend`: `joints` = the carrying motion through our planned joints (keeps the start's posture); `pose` = hand poses, cuRobo picks the joints                                    |
| `adjust_speed_scale`     | `1.0`                                        | speed_scale of the squaring turn (2x)                                                                                                                                                |
| `retreat_speed`          | `0.10`                                       | The retreat's peak fingertip speed (m/s)                                                                                                                                             |
| `reset_speed_scale`      | `1.0`                                        | speed_scale of reset's joint moves with `reset_mode` legacy (2x)                                                                                                                     |
| `reset_mode`             | `"curobo"`                                   | `curobo`: reset is one cuRobo motion around the chair; `legacy`: the old impedance joint moves                                                                                       |
| `blend_radius_near`      | `0.03`                                       | blend_radius near the table or the container, which cuRobo doesn't know (m)                                                                                                          |
| `blend_radius_far`       | `0.05`                                       | blend_radius at the carrying height (m)                                                                                                                                              |
| `curobo_check`           | `true`                                       | cuRobo checks the ends of the line down and the retreat against the chair's volumes first                                                                                            |
| `reach`                  | `[0.7, 1.1]`                                 | Spot distance from the arm base axis with the full descent (m); kinematics only, re-measure with the chair volumes                                                                   |
| `allow_reconfigure`      | `false`                                      | Switch to another arm posture where the line down can't reach the table in this one                                                                                                  |
| `max_adjust_deg`         | `15.0`                                       | Largest squaring turn (deg)                                                                                                                                                          |
| `max_spot_age_min`       | `30.0`                                       | A marked spot older than this is refused (min)                                                                                                                                       |
| `tool_frame_tol`         | `0.015`                                      | The arm's tool frame vs the model's (m)                                                                                                                                              |
| `state_dir`              | `"/config/place_container"`                  | Where the marked spot, start position and descent recordings are kept                                                                                                                |
| `state_valid_after`      | `1794074180.0`                               | A spot or start position saved before this (unix s) is refused: when the arm moved on the chair                                                                                      |
| `ui_error_wait_s`        | `60.0`                                       | A failure waits this long for the OK on the UI panel before the action ends with it (s); 0 = don't wait                                                                              |
| `ui_bridge_node_name`    | `"ui_bridge_node"`                           | The UI bridge node the panel talks to (declared by the engine)                                                                                                                       |

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
