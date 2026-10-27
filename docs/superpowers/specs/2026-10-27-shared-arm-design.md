# `self.arm`: the shared arm interface for kitchen capabilities

Status: draft for review, 2026-10-27.

## Why

Every December demo capability moves the Kinova Gen3 through RAMMP's arm driver
([kinova-gen3-ros2](https://rammp-org.github.io/kinova-gen3-ros2/interface/)) and
the cuRobo planner ([RAMMP-CuRobo](https://github.com/rammp-org/RAMMP-CuRobo)).
Today the student code in [dhyi-svg/feeding-deployment](https://github.com/dhyi-svg/feeding-deployment)
does this through three different clients: the Cornell `ArmInterfaceClient`
(Kortex directly), `SheppyArm`, and `RammpArm`. Each holds its own node,
executor thread and subscriptions for its whole lifetime, and each re-implements
IK, timing and safety checks.

The students writing capabilities do not know ROS. Previous demos lost time to
subscriptions held open and wasting resources. The `CapabilityEngine` base
class (PR #2) gives every capability one arm, `self.arm`, that:

- reproduces the `SheppyArm` API **exactly**, so the team's working code keeps
  working;
- exposes **every** arm driver capability as a plain interface;
- is deliberate about ROS resources and about the driver's state.

The students' designs carry a lot of hard-won nuance (contact speed caps, the
straight-line paths their sim checks were run on, loose grasps for the door
handle). `self.arm` must not lose any of it.

## Goals

1. **`SheppyArm`, exactly.** Same names, signatures, return values, module
   constants and behaviour. **Same joint paths is the correctness test of the
   port**: with the students' model loaded, the shared sim reproduces their code's
   joint paths. On the real robots the paths will change anyway (the model
   differs between robots, and it now knows the mount); that is accepted. A
   capability migrates by passing `self.arm` wherever it constructed
   `SheppyArm()`; the rest of its code is unchanged.
2. **The whole driver, plainly.** One thin method per driver and planner
   interface, with the driver's names and fields, in numpy and plain values.
3. **Driver-state tracking.** `self.arm` tracks the driver's state (trajectory goal,
   streaming session, control token, e-stop) and enforces the driver's rules
   with clear messages.
4. **Intentional ROS resources.** Under the engine, the arm's subscriptions,
   clients and publishers exist only while an action runs.
5. **A mount-aware sim.** Kinematics and collision checks know the robot's
   `arm_mount` (30° on flatbot and gen_1_5, identity on dojo).
6. **Compliant streaming.** `ee_twist_impedance` (on the driver's nightly) is
   supported from the start.

## Non-goals

- Performance tuning of the stack; that is done separately.
- Changing any capability's motion. Conveniences are additive and opt-in.
- Changes to RAMMP-CuRobo or the driver. They are listed as follow-ups below,
  each its own issue in that repo.
- Perception, TF to the cameras, the utensil wrist, F/T sensing.
- Drinking and feeding: their code is not in the student branches.
- Migrating the capabilities themselves. This deliverable is `self.arm`; each
  capability moves onto it under its own issue (#4, #6–#16).

## Anti-goals (failure even if it works)

- The port changes a joint path, as measured against the students' own model.
- Capability code imports `rclpy`, creates subscriptions, or handles tokens.
- A streaming session or subscription outlives the action that needed it.
- The engine or the arm acquires control. Control comes only from the
  orchestrator.
- Client-side checks layered on top of cuRobo plans. cuRobo's limits are trusted.

## What exists

**The driver** (`kinova-gen3-ros2`). Target: the nightly channel the flatbot
profile runs (`arm_joint_*` names, `arm_mount` frame), whose interface is the
one documented on `dev`. Four tiers:

- *State:* `/joint_states` and `/ee_state` (~50 Hz, best-effort; `/ee_state` is
  the flange in `arm_mount`, with twist and measured wrench),
  `/gripper_state` (20 Hz), `/control_status` and `/stream_status` (latched,
  on change), `/diagnostics` (1 Hz), `/arm/robot_description` (latched URDF,
  rendered from the robot's `mount.yaml`).
- *Trajectory and planned moves:* `execute_joint_trajectory`,
  `go_to_ee_pose`, `go_to_joint_config`, `go_to_preset`. Each takes
  `control_mode` (position or impedance) and `ImpedanceGains` (a profile:
  session default, soft, medium, stiff, or custom `kq`/`zeta`/`torque_limit`),
  plus `speed_scale`, `sender_id` and `token`.
- *Arbitration:* `/acquire_control` (seizes: halts the current owner),
  `/release_control`, `/revoke_control`, `/set_gains`.
- *Streaming:* `/list_controllers`, `/open_stream`, `/close_stream`, then
  setpoints on `/setpoint/{joint_position,joint_velocity,joint_torque,pose,twist}`.
  Controllers: `joint_position`, `joint_impedance`, `ee_pose_position`,
  `ee_pose_impedance`, `joint_torque`, `joint_velocity`, `ee_twist`, and on
  nightly `ee_twist_impedance`. `/setpoint/gripper` and `/estop` are separate.

The driver's rules `self.arm` must respect (from its streaming and
arbitration guides):

- A stream and a trajectory goal never run at once. Each side **refuses** the
  other; nothing queues. A goal accepted just before a stream opens is settled
  `INVALID_GOAL`.
- A session's controller and gains are fixed for its lifetime; changing either
  means close and reopen.
- A session whose setpoints stop past `timeout_s` closes
  (`kDeadlineExpired`): it zeroes the command, freezes at the measured position
  and latches. A pose the IK cannot reach for `ik_fault_s` closes it with
  `kIkFault`; re-opening the same request fails again.
- Acquiring control seizes it and halts the previous owner's motion (settled
  `-9 HALTED`); `generation` in `/control_status` changes.
- An e-stop drops all grants, halts and latches.

**The planner** (`RAMMP-CuRobo`). Plan-only actions `/rammp_curobo/plan_to_pose`
(with orientation `hold`) and `/rammp_curobo/plan_to_joints` return a full timed
`JointTrajectory`. `/rammp_curobo/set_world` swaps a whole packaged world. The
core has `update_world(boxes)`, `check_state_valid(q)` and `fk(q)` in Python, not
over ROS. Known gap: cuRobo plans with an older model's joint limits than the
driver's (driver README).

**The engine** (`capability_engine.CapabilityEngine`, PR #2). One node per
capability on a `MultiThreadedExecutor` with a `ReentrantCallbackGroup`.
`execute(action_id, params, ctx)` runs inside an action callback, one at a time;
`ctx.checkpoint()` blocks while paused and raises `Cancelled` once cancelled.
In standalone mode (`use_orchestrator:=false`) there is no activate or
hibernate.

**`SheppyArm`**: `microwave/sheppy_arm.py`, reference copy at
`microwave-wip@7bc328ae` (481 lines). `microwave-integration@54a4a794` has an
older copy that adds `raw_joints()` and lacks `not_ready`, `go_to_*` and
`contact_speed_scale`. `self.arm` reproduces the union of the two.

## Design

The arm is part of the `CapabilityEngine` base class. Every capability gets
`self.arm`, an `ArmInterface` the base class sets up; students call
`self.arm.<method>` and never create, spin, configure or close it.

```python
class OpenMicrowave(CapabilityEngine):
    def execute(self, action_id, params, ctx):
        q = self.arm.get_state()["position"]
        self.arm.go_to_ee_pose(pre_p, g_q, "to line-up")
        self.arm.move_cartesian([(g_p, g_q)])
        self.arm.gripper(1.0, force=0.25)
        ...
```

`ArmInterface` is flat: one class whose methods are the `SheppyArm` API and the
driver API side by side. There are no layers for students to learn; the sim is
public and used exactly as today.

### 1. `self.arm` in the engine

- **Setup.** `CapabilityEngine.__init__` creates `self.arm` on the engine's own
  node and callback group (a `MultiThreadedExecutor` with a
  `ReentrantCallbackGroup`, so arm callbacks run while `execute()` blocks). No
  arm-owned node, executor or thread.
- **Resource lifetime.** The arm's subscriptions, action and service clients and
  publishers are created on the first arm call inside an `execute()` and
  destroyed when that `execute()` ends, after any goal or stream has settled. An
  action that never touches the arm costs nothing. *(Claude's call: per
  `execute()` rather than per activation, because standalone mode has no
  activate/hibernate.)* Any arm call outside `execute()` raises "the arm is only
  available inside an action". A test asserts the node holds no arm
  subscriptions, clients or publishers after `execute()` returns.
- **Control comes from the orchestrator, never from the engine.** The
  orchestrator acquires control and hands the engine the token with the action;
  `self.arm` attaches it to every goal, session and setpoint for that action and
  forgets it when `execute()` ends. It never calls `/acquire_control` or
  `/release_control`. **With no token** (standalone mode, `make run
  STANDALONE=1`, or an orchestrator that does not send one yet), `self.arm`
  sends commands without one: that works with arbitration disabled, and under
  enforced arbitration the driver refuses (`NOT_AUTHORIZED`), which `self.arm`
  reports as "no control token: arbitration is enforced and none was given".
  Handing over a token needs a field in the orchestrator's interface (follow-up
  below); `ExecuteAction` and `Activate` carry none today.
- **Pause and cancel.** The engine gives `self.arm` the running action's `ctx`.
  Every motion call runs `ctx.checkpoint()` before sending, so pause takes effect
  between motions *(Claude's call: a trajectory cannot be resumed mid-way with
  the same path)*. Cancel during a motion cancels the goal (or zeroes and closes
  the stream), waits for it to settle, and raises `Cancelled`.
- **Standalone (bench scripts).** `from sheppy_arm import SheppyArm;
  SheppyArm(node_name=...)` still works and is the same class, constructed with
  its own node as today, destroyed by `close()`, refusing to move under enforced
  arbitration.

### 2. The `ArmInterface` methods

Results carry the driver's `error_code` and `error_string`. Arguments and
returns are numpy arrays, floats, strings and small dataclasses, never ROS
messages.

**The `SheppyArm` API, exactly.** Every name keeps its signature and meaning;
internals change only as listed.

| Name | Kept exactly | What changes inside |
|---|---|---|
| `get_state()` | dict: `position` (wrapped), `velocity`, `effort`, `ee_pos` (tool_frame: flange + `TOOL_Z`), `gripper_pos` | Joint names `arm_joint_1..7`, poses in `arm_mount`. |
| `raw_joints()` | yes (from `microwave-integration`) | — |
| `not_ready()` | `None` or a reason | Reads the driver state (section 3). |
| `execute_command(cmd)` | Cornell `JointCommand`, `CartesianTrajectoryCommand`, `Open/CloseGripperCommand` | The command classes are vendored, so `feeding_deployment` is not a dependency. |
| `move_joints(knots, cart_lengths=None, speed_scale=1.0, impedance=None, monitor=None)` | min-jerk timing at `DT`, `JOINT_SPEED`/`CART_SPEED`, `MIN_DURATION_S`, one goal, `PATH_TOL_RAD`, `monitor` cancel = success, keep-out check | `impedance` dicts map to the `CUSTOM` gains profile with the same `kq`, `zeta`, `torque_limit`. |
| `move_cartesian(traj, impedance=None, speed_scale=1.0)` | `cartesian_knots` with the shared `solve_ik`, same steps and gates, then `move_joints` | — |
| `go_to_ee_pose(pos, quat, label=, speed_scale=None)`, `go_to_joint_config(q, label=, speed_scale=None)` | driver plans and executes, `GO_TO_SPEED_SCALE`, end checks `GO_TO_TRACK_M` / `GO_TO_TRACK_DEG` | Goal frame `arm_mount`. |
| `gripper(position, timeout_s=4.0, force=None)` | streams until settled, returns `True` | Token attached. |
| `last_path`, `monitor_stopped` | yes | — |
| `close()` | yes | A no-op under the engine. |
| Module `sheppy_arm`: `JOINT_SPEED`, `CART_SPEED`, `CONTACT_*_SPEED`, `GO_TO_SPEED_SCALE`, `IMPEDANCE`, `ROT_EQUIV_M`, `TOOL_Z`, `ERRORS`, `SENDER_ID`, `FRAME`, `contact_speed_scale()`, `cartesian_knots()`, `SheppyArm` | yes, including callers mutating the speed constants at runtime (`self.arm` reads them at call time) | `FRAME = "arm_mount"`. `SENDER_ID` comes from the capability. |

Prints become log lines through the node's logger, with the same text.

**The driver API, alongside.** One method per driver and planner interface,
named as in the driver, with every field:

- State: `joint_states()`, `ee_state()` (pose, twist, wrench), `gripper_state()`,
  `control_status()`, `stream_status()`, `diagnostics()`, `robot_description()`.
  Each returns the latest message and its age; a value older than a threshold
  is an error, not a silent old value.
- Actions: `execute_joint_trajectory(...)`, `go_to_ee_pose(...)`,
  `go_to_joint_config(...)`, `go_to_preset(...)`. Blocking; feedback available
  to a callback.
- Planner: `plan_to_pose(...)`, `plan_to_joints(...)` (plan only), `set_world(...)`.
- Services: `list_controllers()`, `set_gains(...)`. `acquire_control`,
  `release_control` and `revoke_control` exist for standalone bench tools and
  refuse under the engine with "control comes from the orchestrator".
- Streaming: `stream(controller, gains=None, timeout_s=0.1)`, a context manager
  that opens the session, yields an object with `send(setpoint)`, and always
  closes it, including on exceptions. It checks `/list_controllers` first and
  refuses a controller the running driver reports unavailable (for example
  `ee_twist_impedance` on a build without it).
- Setpoints outside sessions: `gripper(...)`, `estop(...)`.

**Where the two share a name** (`go_to_ee_pose`, `go_to_joint_config`,
`gripper`), there is one method: `SheppyArm`'s positional signature and
behaviour, with the driver's remaining fields (`orientation_hold`,
`control_mode`, `gains`, `speed`, ...) as extra keyword arguments that default to
`SheppyArm`'s behaviour. *(Claude's call.)*

**No magic numbers.** Every tunable in `self.arm` (speeds, tolerances, gains,
timeouts, staleness thresholds, the sim's IK guards) is a named, documented
value, declared as a ROS parameter of the capability with today's value as
its default, listed in the capability's fragments and `docs/interface.md` as
CONTRIBUTING requires. The `sheppy_arm` module constants keep their names (and
stay writable, for existing code) and are initialised from those parameters.
New code in this work introduces no unexplained literals.

### 3. How `self.arm` tracks the driver's state

It holds one mode: `idle`, `goal` (with its handle), or `stream` (with
controller and gains), plus the token and the last `generation`.

| Situation | Behaviour |
|---|---|
| A goal requested while a stream is open, or a stream while a goal runs | Refuse with an error naming what is running. *(Claude's call: refuse rather than hand over. Sequential, blocking code never collides; a collision is almost always a bug.)* |
| A different controller or gains than the open session | Not possible: sessions only exist inside `stream(...)`. Nested `stream` blocks are refused. |
| `/stream_status` shows the session closed unexpectedly | Raise inside the `with` block with the cause: "your loop stopped sending for more than `timeout_s`" (deadline) or "the driver could not solve that pose" (IK fault). |
| `generation` changes, or a goal settles `-9 HALTED`, while holding the token | Stop, drop the token, raise "control taken by `<owner_id>`". Never re-acquire. |
| `/control_status` shows e-stopped | Refuse every command with "e-stopped". |

Whether `/setpoint/gripper` is excluded from the stream/goal rule is to be
confirmed in the driver source; the gripper has its own tier.

### 4. The sim

Students keep using the sim exactly as they do now: `make_sim()`, `solve_ik`,
`clearance`, `cartesian_knots(sim, ...)`, the guards, their own door model and
keep-out boxes, called directly from their code. Nothing about it is hidden
behind `self.arm`.

What changes is where it lives and what model it loads. `SheppyArm`'s
`move_cartesian` executes the joint path the callers' own sim checks ran on
(`door_push._chain`, `demo_replay`, the swing check), because both chain the
same `microwave_common.solve_ik`. So there is one copy, shared by students and
`self.arm`:

- Moved into `capability_engine` from `microwave_common`: `solve_ik`,
  `make_sim`, `wrap_joints`, `short_delta`, `continuous_ok`, `MAX_IK_ERR`,
  `J4_GUARD_DEG`, `J6_GUARD_DEG`, `MAX_STEP_JUMP_DEG`, the `KEEP_OUT` hook and
  `keep_out_message`, and `clearance`, with the same names and signatures.
  `microwave_common` re-exports the functions, so their callers do not change,
  but no settable value (see the contract below).
- `make_sim()` no longer builds Cornell's `FeedingDeploymentPyBulletSimulator`
  from `vention.yaml`. It loads a URDF into a bare PyBullet client and returns
  the same `(scene, robot)` shape the callers use (`robot_id`,
  `end_effector_id`, `physics_client_id`, `scene.robot_base_pose`), so callers'
  bodies (door model, keep-out box) still attach to it.
- **One copy per engine.** Today a single microwave run builds half a dozen
  copies (each script and several `microwave_common` and `door_push` helpers
  call `make_sim()`, and `SheppyArm` builds its own), each a full Cornell scene.
  `make_sim()` instead returns the process's single copy, built on first use,
  and `self.arm`'s internal sim is that same copy. This is safe for how they use
  it: nothing steps physics; `solve_ik` and `clearance` set the joints (and
  fingers) they need before every query, so a leftover pose changes no answer;
  `clearance` and `keep_out` query only the body ids they are given, so door
  boxes that callers add and never remove (only `door_push.py:128` removes
  them) accumulate harmlessly; and the engine runs one action at a time, with
  no arm thread touching the sim. Because there is only one model in the
  process, "same model" cannot be broken. Each capability is its own process,
  so this is one copy per engine.
- **One way in.** Students keep calling the module-level `make_sim()` as today;
  `self.arm` calls the same function and gets the same copy. No engine method.
- **Model source.** Production: the driver's `/arm/robot_description`, rooted at
  `arm_mount`, so the 30° mount is in the model and IK targets, door boxes and
  keep-out boxes are all in `arm_mount`, the frame of `/ee_state` and every
  goal. Verification: the Cornell URDF and base pose that `make_sim` uses today.
  Callers also run `make_sim()` offline (sim-check tools, CI, a laptop), so it
  takes an optional URDF path: with none it reads the live topic; offline it
  loads a file. The Cornell URDF and per-robot URDFs rendered from the driver's
  `kinova_gen3_description` (dojo, flatbot, gen_1_5) are committed as test
  fixtures. *(Claude's call.)*

**The contract between `self.arm` and the sim.** We own the sim's code and its
model; students own how they call it and the task bodies they add. These rules
keep the students' side from changing `self.arm`'s behaviour without an error:

| Way it could break quietly | Rule |
|---|---|
| A sim built from a different model, so their checks validate a different robot than the one `self.arm` solves IK on | There is one copy per process (above), built from `/arm/robot_description` under a live driver. An explicit URDF path is refused while the driver is up; offline it is allowed. |
| A settable value written through a re-export changes only the copy: `fridge_push.py:68` turns the wheelchair keep-out on with `microwave_common.KEEP_OUT = keep_out`, which would stop reaching `solve_ik` and `move_joints` and switch the check off silently | `microwave_common` re-exports no settable value. The hook is set with `set_keep_out(fn)` (the one line in `fridge_push` that changes); a test asserts it is set after `enable_keep_out()`. Speed writes such as `sheppy_arm.CART_SPEED *= s` are safe: `sheppy_arm` is our module. |
| Poses checked with one IK, sent to `move_cartesian`, re-solved by another (the close script has its own `_solve`, 1000 iterations against `solve_ik`'s 200; today it sends the joints it solved, which is fine) | Documented for students: if you checked joints, send joints (`move_joints`); if you send poses (`move_cartesian`), check them with `cartesian_knots` on the shared sim. |
| A pose looked up in `arm_base_link` instead of `arm_mount`: identical on dojo, 30° off on flatbot and gen_1_5 | Documented for students: look poses up in `sheppy_arm.FRAME`. |
| The sim's end-effector link is the flange, not the tool frame `ee_pos` reports (0.120 m apart) | Ours: checked in verification step 2. |

**The students' rig is tilted, and their code compensates by hand.** Their model
(Cornell's sim, and `FRAME = "base_link"` on the 1.1.0 driver) has no mount.
`microwave/NOTES.md` (10-21) records the user's hand-placed grasp coming out
13.6° nose-down "because THIS RIG'S ARM BASE IS TILTED", so `GRASP_PITCHES_DEG`
now tries 14° first, marked TEMPORARY: "Real deployment: base will be level ->
restore (0, 10, 17, 25, 30, 35)." Other geometry assumes the arm base frame is
gravity-aligned: the door arc turns about the frame's z (a vertical hinge), and
"level" checks (`CLOSE_MAX_TILT_DEG`, the close's roll/pitch limits) are in it.
With a mount-aware model these assumptions become true on every robot, and the
hand compensation must come out. Production joint paths change on the new
robots regardless (accepted); each capability gets its hardware pass under
rammp-deployments step 4. "Same joint path" is enforced only by verification
step 1, against their model.

Verification, in order:

1. **Port check (Cornell URDF).** Record golden outputs offline from the
   students' code at `microwave-wip@7bc328ae`: `solve_ik` and `cartesian_knots`
   over a set of poses and seeds, including a door arc and a straight approach.
   The shared sim with the Cornell URDF must reproduce the joints to ~1e-6 rad.
   The same for `move_joints`'s sampled trajectory (`t`, `q`, `qd`). All cases
   run in sequence through the one shared copy, with door boxes added between
   them as the callers do, so any state left behind that changed an answer
   would fail the test.
2. **Model check (driver URDF).** Same inputs on dojo's identity mount: report
   differences rather than assert. A gap is a model difference (link
   geometry, the tool frame, the wrist camera mount missing from the generated
   model), not a porting bug. Then on flatbot's mount: the same targets in
   `arm_mount` give the same arm motion through the tilt.

The tool frame needs explicit checking in step 2: `SheppyArm` reports
`tool_frame` as the flange plus 0.120 m along its z, and the Cornell robot's
`end_effector_id` must be that same frame.

### 5. Migrating `RammpArm` users

Button press (PR #5, which closes issue #4; `feature/4-button-press`) and
container grab (`rammp_grasp_tupperware/` on `sk-demo-testing` and
`microwave-integration` in feeding-deployment) move onto `self.arm`:

- `state()`, `joints()`, `ee_pose()`, `raw_joints()`, `gripper_present()` →
  `get_state()` / `raw_joints()`.
- `status()`, `take_control()`, and the private `_token` and
  `_grasp_grip_pub` → handled by `self.arm`; `gripper()`.
- `cancel_motion()` → engine cancel.
- `precheck_poses()` + `smooth_poses()` → `move_cartesian`.
- `plan_curobo()` / `gate_joint_path()` / `curobo_move()` → `go_to_ee_pose` /
  `go_to_joint_config` with `speed_scale`. The client-side gates are dropped:
  cuRobo is trusted.
- `press_impedance()` → `self.arm.stream("ee_pose_impedance", ...)`; the press
  monitor (sag, lateral, overshoot aborts) stays in button press.

Button press's joint paths change: it had its own IK (`kinematics.py`) with
different thresholds. That is accepted; its full sequence has not run end to end
on the new stack yet (rammp-deployments#94), so it needs a hardware pass anyway.

### 6. Conveniences (later, opt-in, each its own PR)

Each is one more method on `ArmInterface`:

- `rotate_about(point, axis, angle, mode=, gains=, stop_when=)` on
  `ee_twist` / `ee_twist_impedance`: a 100 Hz loop commanding
  `ω × (tool − hinge)` from the measured pose, for the door and fridge swings.
  Stops on angle reached, wrench thresholds from `/ee_state`, slip, stall, or
  `stop_when()`.
- `plan_cartesian(traj)` / `execute(plan)` on a cuRobo straight-line action,
  once it exists (follow-up below). A capability adopts it by moving its own
  checks onto `plan.joints`, so checking and execution switch together.
- Smooth multi-leg motion (`plan_a + plan_b` executed as one trajectory), once
  the blending retimer exists.

## Risks to handle in the plan

1. **Meshes behind `package://` URIs.** The driver's URDF includes the arm and
   gripper from `kortex_description` and `robotiq_description`, so its meshes
   are `package://` references. Capability images do not install those
   packages, and PyBullet does not resolve `package://`; a mesh that fails to
   load is dropped with a warning, and `clearance` would measure against an arm
   with pieces missing. Install both packages in the image, resolve the URIs,
   and have `make_sim()` fail if any collision mesh does not load.
2. **The two `SheppyArm` copies disagree on speed.** `microwave-wip` has
   `JOINT_SPEED` 0.85 and `CART_SPEED` 0.16; `microwave-integration` (placement
   track B) still has 0.5 and 0.09. Accepted: the newer values are the
   defaults, and as parameters (see "No magic numbers") a capability can set
   its own.
3. **PyBullet has no arm64 wheel** (PyPI 3.2.7 ships x86_64 wheels only), so
   the Jetson image builds it from source. Do it in its own cached Docker
   layer and check the build time early.
4. **Ruff forbids `print()` in capability code** (CONTRIBUTING), and the
   students' code prints throughout. Decided: the rule stays, and each
   migration converts prints to the node's logger. Ruff's own fix for this rule
   deletes the `print` rather than converting it, so the conversion is a
   mechanical edit done in the migration, not an autofix.
5. **PR #2 lands first.** `self.arm` lives in `CapabilityEngine`, which is
   unmerged and conflicts with `dev`.
6. **Recording goldens needs the students' environment once**: their code at
   `microwave-wip@7bc328ae` with Cornell's sim, `feeding_deployment` and
   `pybullet_helpers`, run offline.

7. **Arbitration will be enabled for the demo, once ready.** It stays off
   (`arbitration_mode: disabled`, as in the December manifest today) until the
   orchestrator hands out tokens; `self.arm` works both ways, sending the token
   when it has one and none otherwise. The token handover must land before
   arbitration is turned on.

For context: on flatbot nothing publishes `base_link`, and cuRobo's world is
empty until one is authored in `arm_mount` (flatbot profile). Students moving
their frames to `arm_mount` is an agreed contract.

## Follow-ups outside this repo (one issue each)

- **Prerequisite.** rammp-interfaces-ros2 and rammp-orchestrator: the
  orchestrator acquires control and passes the token to the engine with each
  action (a `token` field on `ExecuteAction`, or on `Activate`). Must land
  before arbitration is turned on for the demo.

- RAMMP-CuRobo: plan with the driver's model (`/arm/robot_description`), which
  fixes the joint-limit mismatch.
- RAMMP-CuRobo: runtime obstacles over ROS (expose `update_world`), so task
  geometry (door, fridge, wheelchair) is in the planner's world.
- RAMMP-CuRobo: a straight-line `plan_cartesian_path` action (dense samples, IK
  seeded along the path, `check_state_valid`, timed `JointTrajectory`).
- RAMMP-CuRobo: revive `approach_via` (kinova-gen3-ros2#40) for one smooth
  "line up and insert" trajectory.
- RAMMP-CuRobo: a corner-blending retimer for chaining plans without stops.
- kinova-gen3-ros2: the pose stream switching IK solutions, which contradicts the
  documented bounded IK step.
- kinova-gen3-ros2: `/set_gains` is missing from `docs/interface.md`.
- kinova-gen3-driver: confirm the stale-stream hold ("FREEZES") on hardware.

## Left to the builder

- Module and file names inside `capability_engine`, beyond `import sheppy_arm`
  and the sim functions' names working.
- Test infrastructure for driver-state tracking (the driver's mock alternative, or
  fakes).
- Staleness thresholds for state reads.
- How golden outputs are recorded and stored.
- Phasing within the work, provided `SheppyArm` exact reproduction (sections 1–5,
  verification step 1) lands before any convenience.
