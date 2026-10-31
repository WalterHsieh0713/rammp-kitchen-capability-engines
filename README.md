# rammp-kitchen-capability-engines

Shared codebase for all kitchen tasks implemented on RAMMP, for the December
2026 demos.

Each capability (button press, door opening, ...) is its own RAMMP module: one
directory under `capabilities/`, one image at
`ghcr.io/rammp-org/rammp-kitchen-capability-engines/<capability>`, one real and
one mock sheppy alternative. This repo enforces the
[RAMMP module standards](https://github.com/rammp-org/rammp-module-template)
for every capability, automatically, on every PR.

## Quickstart

Prerequisites: [`uv`](https://docs.astral.sh/uv/), `pre-commit`, Docker. Images
are arm64 (`rammp-base` is Jetson-only), so build on the Jetson or let CI do it.

```bash
pre-commit install            # once per clone; see CONTRIBUTING.md
make new CAP=button_press     # start a capability from the example
make check                    # repo rules + sheppy's own loader, all capabilities
make build CAP=button_press   # docker build (arm64 host)
make smoke CAP=button_press   # publishes what it declares, exits on SIGTERM
make lint                     # pre-commit run --all-files
```

## Standalone or with the orchestrator

Every capability is a `capability_engine.CapabilityEngine`: it implements the
[RAMMP orchestrator](https://github.com/rammp-org/rammp-orchestrator)'s engine
contract (register, activate, execute_action, control, hibernate) and only
writes its own `execute()`. Every capability, and `capability_engine` itself,
is an `ament_python` package, so it builds natively with colcon or in its image,
and both run the same launch file with the same two switches:
`use_orchestrator:=false` (standalone) and `mock:=true` (no hardware).

### Natively with colcon (the RAMMP hybrid workflow)

Clone into the workspace from [developer setup](https://rammp-org.github.io/setup/),
next to `rammp-interfaces-ros2` (at a version that carries
`rammp_orchestrator_interfaces`, currently its `dev` branch):

```bash
cd ~/ros2_ws/src
git clone git@github.com:rammp-org/rammp-kitchen-capability-engines.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --packages-up-to example
source install/setup.bash

ros2 launch example example.launch.py                          # registers with the orchestrator; waits for it if absent
ros2 launch example example.launch.py use_orchestrator:=false  # standalone; add mock:=true to skip hardware
```

Edit, `colcon build --packages-select example`, relaunch. No image involved.
To have sheppy start your build, add a local `kind: launch_file` alternative as in
[hybrid workflow](https://rammp-org.github.io/development/hybrid/) (an uncommitted
edit to your manifest).

### In the image

```bash
make build CAP=example                # arm64 host
make run CAP=example                  # with the orchestrator
make run CAP=example STANDALONE=1     # standalone; MOCK=1 skips hardware
```

### Sending it an action

Either way, drive a running capability as the orchestrator would:

```bash
make action CAP=example ACTION=count PARAMS='{"to": 5}'
# or natively:
ros2 action send_goal --feedback /example/execute_action rammp_orchestrator_interfaces/action/ExecuteAction \
  '{engine_id: example, action_id: count, params_json: "{\"to\": 5}"}'
ros2 service call /example/control rammp_orchestrator_interfaces/srv/Control '{command: 0}'  # 0 pause, 1 resume, 2 cancel
ros2 topic echo /example/status
```

Standalone, the orchestrator calls a capability makes (`ctx.request_intent`,
`ctx.write_world_state`, `ctx.update_object`) are logged and skipped. Arm control
arbitration (`AcquireControl`) is not wired in yet: the orchestrator does not
call it today.

To run it on the robot, transcribe the capability's two fragments into
[`rammp-deployments`](https://github.com/rammp-org/rammp-deployments)'s
`december_2026/sheppy-manifest.yaml` as a node with both alternatives.

## API

A capability is a Python class. It subclasses `CapabilityEngine`, lists its
actions, and implements `execute()`, which the engine calls each time the
orchestrator starts one of those actions. Inside `execute()`, `self.arm`
controls the arm and `ctx` talks to the orchestrator:

```python
from capability_engine import Action, CapabilityEngine, EngineError, run


class Capability(CapabilityEngine):
    display_name = "Open microwave"
    perception_classes = ["microwave"]
    required_resources = ["arm", "gripper"]
    actions = [Action("open", "Open the door")]

    def __init__(self):
        super().__init__("open_microwave")

    def execute(self, action_id, params, ctx):
        if not self.arm.gripper(0.0):  # open the fingers
            raise EngineError("gripper_failed", self.arm.last_error)
        ctx.checkpoint()  # stop here if the action was paused or cancelled
        if not self.arm.move_joints([[0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57]]):
            raise EngineError("move_failed", self.arm.last_error)
        return {"opened": True}


def main():
    run(Capability)
```

`capabilities/example` is a complete one you can copy. Everything below imports
from `capability_engine` unless another module is named. Before you write a
capability, read the rules in [Using `self.arm`](CONTRIBUTING.md#using-selfarm):
how to handle a cancel, when to subscribe to topics, how to use the sim, and
the `arm.*` parameters. Terms such as *knot*, *tool frame* and *cuRobo* are
explained in [Terms](#terms) at the end.

### `CapabilityEngine`

Set these class attributes. The engine sends them to the orchestrator when it
registers:

| Attribute            | What it is                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `display_name`       | The name shown to the user                                                                  |
| `perception_classes` | Object classes this engine handles. Selecting an object of one of them activates the engine |
| `required_resources` | What the orchestrator locks while an action runs: `"arm"`, `"base"`, `"gripper"`            |
| `state_reads`        | World-state keys the engine reads                                                           |
| `state_writes`       | World-state keys the engine writes                                                          |
| `intents_handled`    | Intents (such as `MOVE_BASE`) that other engines may hand to this one                       |
| `actions`            | The actions it offers, as a list of `Action`                                                |

`Action(action_id, label, params_schema={}, sticky_followup=False, perception_classes=[])`
describes one action. `params_schema` is a JSON Schema for the `params` dict
that `execute()` receives.

Methods you override:

| Method                                    | What it does                                                                               |
| ----------------------------------------- | ------------------------------------------------------------------------------------------ |
| `execute(action_id, params, ctx) -> dict` | Required. Runs one action and returns its result. To fail, raise `EngineError`             |
| `availability(instance_id, world_state)`  | Optional. Says which actions can run on this object right now. By default, all of them can |
| `hibernate()`                             | Optional. Clears anything you stored for the selected object. By default, it does nothing  |

What the engine gives you:

| Name                                                         | What it is                                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `__init__(engine_id, **node_kwargs)`                         | Call it as `super().__init__("<engine_id>")`. The ROS node's name is the engine id                                  |
| `self.arm`                                                   | The arm (see [below](#selfarm-arminterface))                                                                        |
| `self.mock`                                                  | True when the `mock` parameter is set, meaning there is no hardware                                                 |
| `self.instance_id`                                           | The object the engine was activated on                                                                              |
| `create_action_subscription(msg_type, topic, callback, qos)` | Like `Node.create_subscription`, but the subscription exists only until the action ends. Call it inside `execute()` |
| `create_action_publisher(msg_type, topic, qos)`              | The same, for a publisher                                                                                           |
| `create_action_tf_buffer()`                                  | A `tf2_ros.Buffer` that stays up to date until the action ends. Use it instead of a `TransformListener`             |
| `stop()`                                                     | Stops accepting new actions and cancels the running one. `run()` calls it on SIGTERM or SIGINT                      |

`run(EngineClass)` starts a capability. It runs until SIGTERM or Ctrl-C.
`EngineError(error_code, message="", recoverable=False)` fails the action with
`error_code`.

### `ctx` (`Context`)

`execute()` receives a `ctx`. Use it to follow pause and cancel and to talk to
the orchestrator:

| Method                                               | What it does                                                                                              |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `checkpoint()`                                       | Call it between steps. It waits while the action is paused and raises `Cancelled` once it is cancelled    |
| `cancelled() -> bool`                                | Whether the action has been cancelled. Returns at once, even while paused                                 |
| `progress(phase, pct, message="", pausable=True)`    | Reports progress to the orchestrator. `pct` runs from 0 to 100                                            |
| `request_intent(intent_type, params=None) -> Result` | Asks the orchestrator to hand an intent (such as `MOVE_BASE`) to another engine, and waits for its result |
| `write_world_state(**updates) -> bool`               | Updates the world state. The keys are `gripper`, `doors`, `objects`, `arm` and `base_pose`                |
| `update_object(instance_id, **fields) -> bool`       | Updates one object's fields in the world state                                                            |

### `self.arm` (`ArmInterface`)

Use `self.arm` only inside `execute()`. Anywhere else, a call raises
`NotInAction`.

It has two sets of methods. Both talk to the same arm driver, but they report
failure differently.

**The `RammpArm` methods** are the ones the microwave students wrote against.
A motion returns `True` when it succeeds. It returns `False` when the arm refuses
it or it fails, an e-stop included, and puts the reason in `self.arm.last_error`.

```python
get_state() -> dict
raw_joints()
move_joints(knots, cart_lengths=None, speed_scale=1.0, impedance=None, monitor=None)
move_cartesian(traj, impedance=None, speed_scale=1.0)
go_to_ee_pose(pos, quat, label="go_to_ee_pose", speed_scale=None, *,
              orientation_hold=HOLD_NONE, control_mode=POSITION, gains=None, feedback=None)
go_to_joint_config(q, label="go_to_joint_config", speed_scale=None, *,
                   control_mode=POSITION, gains=None, feedback=None)
go_to_preset(name, speed_scale=None, *, control_mode=POSITION, gains=None, feedback=None)
gripper(position, timeout_s=4.0, force=None, *, speed=None)
execute_command(cmd)
```

| Name                                         | What it does                                                                                                                              |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `get_state()`                                | A dict: `position` (joints, wrapped to ±π), `velocity`, `effort`, `ee_pos` (the tool frame as `[x, y, z, qx, qy, qz, qw]`), `gripper_pos` |
| `raw_joints()`                               | The measured joints, not wrapped. A cuRobo plan must start from these                                                                     |
| `move_joints`                                | Moves through the joint `knots`. If you pass `monitor`, the arm stops as soon as `monitor()` returns True, and `monitor_stopped` is set   |
| `move_cartesian`                             | Moves the tool in straight lines through `traj`, a list of `(pos, quat_xyzw)` in the arm's base frame                                     |
| `go_to_ee_pose`                              | cuRobo plans a path for the tool to the pose, then the driver runs it                                                                     |
| `go_to_joint_config`                         | The same, to the joints `q`                                                                                                               |
| `go_to_preset`                               | The same, to one of the driver's named poses                                                                                              |
| `gripper`                                    | `position` goes from 0 (open) to 1 (closed). Returns when the fingers stop moving                                                         |
| `execute_command`                            | Runs a `cornell_commands` command: `JointCommand`, `CartesianTrajectoryCommand`, `OpenGripperCommand` or `CloseGripperCommand`            |
| `last_error`, `last_path`, `monitor_stopped` | The last refusal or failure; the tool path the last move measured, as `[(t, x, y, z)]`; whether its monitor stopped it                    |

**The driver methods** expose the arm driver directly. Instead of returning
`False`, they raise one of the `ArmError` exceptions.

```python
joint_states(), ee_state(), gripper_state()
control_status(), stream_status(), diagnostics()
not_ready() -> str | None
robot_description() -> str
execute_joint_trajectory(names, points, *, path_tolerance=None, goal_tolerance=None,
                         goal_time_tolerance=0.0, control_mode=POSITION, preemption=QUEUE,
                         gains=None, speed_scale=1.0, feedback=None) -> TrajectoryResult
stream(controller, gains=None, timeout_s=None)  # use as: with self.arm.stream(...) as s:
list_controllers() -> list[Controller]
set_gains(gains) -> (accepted, message)
plan_to_pose(pos, quat, start_joints, hold=HOLD_NONE) -> Plan
plan_to_joints(q_goal, start_joints) -> Plan
set_world(world) -> (success, message)
```

| Name                                                   | What it does                                                                                                                                          |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `joint_states()`, `ee_state()`, `gripper_state()`      | The latest message from that topic, as a dataclass. `age_s` says how old it is. Raises `StateTimeout` if none arrives or it is too old                |
| `control_status()`, `stream_status()`, `diagnostics()` | The same, for `/control_status`, `/stream_status` and `/diagnostics`                                                                                  |
| `not_ready()`                                          | Why the arm must not be commanded right now (for example, it is e-stopped), or None if it can be                                                      |
| `robot_description()`                                  | The driver's URDF                                                                                                                                     |
| `execute_joint_trajectory`                             | The driver's trajectory action, with all its options. `points` is a list of `(t, positions, velocities)`, sent exactly as you give it                 |
| `stream`                                               | Opens a streaming session. Call `s.send(setpoint)` at least every `timeout_s`. When the `with` block ends, the arm holds still and the session closes |
| `list_controllers()`                                   | The driver's streaming controllers, and whether each can be opened now                                                                                |
| `set_gains`                                            | Sets the session's default impedance gains                                                                                                            |
| `plan_to_pose`, `plan_to_joints`                       | Asks cuRobo for a plan without moving the arm                                                                                                         |
| `set_world`                                            | Replaces cuRobo's collision world with one of the packaged worlds                                                                                     |

The constants and dataclasses are in `capability_engine.arm`:

- `HOLD_NONE`, `HOLD_LEVEL`, `HOLD_FIXED`: how much of the tool's orientation a
  planned path keeps.
- `POSITION`, `IMPEDANCE`: stiff or compliant control.
- `QUEUE`, `LATEST_WINS`: whether a new trajectory waits for the running one or
  replaces it.

Other modules:

- `rammp_arm` has `RammpArm`'s speeds and other settings, and its path helpers
  (`cartesian_knots`, `joint_trajectory`, `FRAME`, ...).
- `capability_engine.sim` has the shared PyBullet sim (`make_sim`, `solve_ik`,
  `tool_pose`, `set_keep_out`).

The arm's tunable values (speeds, tolerances, timeouts) are `arm.*` ROS
parameters.

### Exceptions

| Exception      | Raised when                                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| `Cancelled`    | The orchestrator cancelled the action. `ctx.checkpoint()` and every arm motion raise it                        |
| `ArmError`     | The parent class of the arm errors below. Also raised when the driver refuses a driver-method call or it fails |
| `NotInAction`  | You called the arm outside `execute()`, or called `send()` outside its `with` block                            |
| `ControlTaken` | Another program took control of the arm in the middle of your action                                           |
| `EStopped`     | You called a driver method while the arm is e-stopped                                                          |
| `StreamClosed` | The driver closed your stream                                                                                  |
| `ModeConflict` | You started a stream or a trajectory while another one was running                                             |
| `StateTimeout` | A state read got no recent message                                                                             |

### Terms

- **Orchestrator**: the RAMMP program that engines register with. It starts and
  cancels their actions and holds the world state.
- **Knot**: one set of 7 joint angles that a path passes through.
- **Wrapped to ±π**: each joint angle given between -π and π.
- **Tool frame**: a point 0.12 m past the arm's flange, along the flange's z
  axis.
- **cuRobo**: the motion planner. It finds a path that avoids the obstacles in
  its collision world.
- **Impedance control**: the arm gives way when pushed, instead of holding its
  position stiffly.
- **Stream**: sending setpoints one after another, instead of a whole trajectory
  at once.
- **URDF**: the file that describes the robot's links and joints.

## A capability

```
capability_engine/               shared ament_python package: the orchestrator engine contract
capabilities/<name>/             one ament_python package per capability (`make new CAP=<name>`)
  package.xml, setup.py, setup.cfg
  <name>/node.py                the capability: a CapabilityEngine subclass
  launch/<name>.launch.py       what both the native build and the image run
  Dockerfile                    FROM a pinned rammp-base / rammp-cuda tag, built from the repo root
  rammp-alternative.yaml        the real alternative: sheppy's schema plus `tier`
  rammp-alternative.mock.yaml   same topics, no hardware — CI smoke-tests this one
  docs/interface.md             the page rammp-docs pulls; keep its headings
```

plus an owner line in `.github/CODEOWNERS`.

## What CI enforces

| Check                      | Rule                                                                                                                                          |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `pre-commit`               | Ruff (lint + format, no `print`), clang-format, hadolint, mdformat, actionlint, file hygiene, no large files or keys                          |
| `checks`                   | the files above exist; fragments are schema-valid; mock topics equal real topics; pinned base; owner in CODEOWNERS; **sheppy loads the node** |
| `smoke (<capability>)`     | on arm64, the mock stays up, every declared `publishes` topic appears, and it exits on SIGTERM within 5 s                                     |
| `tier: demo-ready` (extra) | image pinned to a release tag or digest (never `:dev`/`:main`/`:latest`), and a `/…/status` health topic                                      |

Not machine-checkable, so it is on the PR author and the owner: a
hardware-in-the-loop pass (attended, human on the e-stop) before a capability
is marked `demo-ready`, and an on-call owner for the demo.

Merges to `dev` and `main` publish images tagged with the branch name and commit
SHA. A `v*` tag publishes semver tags for demo profiles to pin.
