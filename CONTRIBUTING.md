# Contributing

This repo follows the RAMMP module workflow
([git workflow](https://rammp-org.github.io/git-workflow/),
[pre-PR checklist](https://rammp-org.github.io/publishing/checklist/)).

## Pre-commit hooks

Style is enforced before each commit, and again in CI, where a failure blocks
the merge. Run this once after cloning:

```bash
uv tool install pre-commit
pre-commit install
```

Ruff, clang-format and mdformat fix files in place: stage the changes and
commit again. Hook revisions are pinned; updating them is a deliberate PR
(`pre-commit autoupdate`). The hooks also refuse commits made directly on `dev`
or `main`.

Ruff forbids `print()` in capability code: log through the node's
`get_logger()`.

## Branches

- `main` — demo-ready, stable, deployable. Updated only by a PR from `dev`,
  **merged with a merge commit** so promotions keep history.
- `dev` — the default branch and staging ground. PRs land here, **squash-merged**.
- `feature/<issue>-<description>` or `bug/<issue>-<description>` — your working
  branch, forked from the latest `dev`.

Both `dev` and `main` require a PR, one approval from the capability's code
owner, resolved review threads, and green `pre-commit` and `ci-ok` checks.
Force-pushes and deletion are blocked.

## Workflow

1. Open (or claim) a GitHub issue and assign yourself.
1. `git checkout dev && git pull && git checkout -b feature/42-button-press-mock`
1. Develop. `make check` and `make smoke CAP=<name>` pass locally (smoke needs
   an arm64 host; CI runs it either way).
1. Open a PR into `dev` using the template: what changed, how it was tested
   (mock, sim or hardware — say which), what was observed, `Closes #42`.

## What belongs in a capability

Code that runs on the robot. Not scratch scripts, versioned copies
(`_v2`, `_final`), session logs, run outputs, or hardcoded IPs and home-dir
paths: take hosts and paths as ROS parameters, and keep experiments on your
branch or in your own repo. Runtime behaviour follows the
[RAMMP patterns](https://rammp-org.github.io/patterns/): never exit because a
peer is absent, publish status from startup, exit on SIGTERM.

## Parameters

Declare each ROS parameter with its default under `params:` in both fragments,
and list it in `docs/interface.md`. Deployments set per-robot values by
overriding those keys in a profile, so anything not listed there cannot be
changed without editing the manifest. sheppy appends
`--ros-args --params-file` to the container's command, so a fragment with
`params:` needs an explicit `command:` that runs the node's executable, not
`ros2 launch`; `validate_fragment.py` enforces both. Large configuration ships in
the image, chosen by a parameter. See
[Configuring a deployment](https://rammp-org.github.io/configuring-a-deployment).

## Using `self.arm`

Every `CapabilityEngine` has `self.arm`, an `ArmInterface` with the same methods
as the students' `SheppyArm` (`get_state`, `move_joints`, `move_cartesian`,
`gripper`, `go_to_ee_pose`, ...), plus a sim of the arm for checking poses before
they are sent. Use it inside `execute()`: the arm's ROS resources exist from the
first arm call to the end of that `execute()`.

**Several targets in one motion.** Each `go_to_*` call plans from rest and
stops dead at its goal. `go_to_ee_poses` (a list of `PoseTarget`) and
`go_to_joint_configs` (a list of `JointTarget`) send all their targets as ONE
goal to the driver. It plans every leg and blends them, so the arm passes a
target without stopping wherever its `blend_radius` (m) lets the tool cut the
corner:

```python
from capability_engine.arm import HOLD_FIXED, PoseTarget

self.arm.go_to_ee_poses(
    [
        PoseTarget(view_pos, grasp_quat, blend_radius=0.05),
        PoseTarget(grasp_pos, grasp_quat, hold=HOLD_FIXED, speed_scale=0.2),
    ],
    "to the grasp",
)
```

Each target has its own `hold` and `speed_scale` (default `GO_TO_SPEED_SCALE`);
`control_mode` and `gains` are one for the whole motion. A hold is checked
against where the arm is when its leg starts. So a held first target must keep
the tool's current orientation (its tilt under `HOLD_LEVEL`), as for
`go_to_ee_pose`, and a later held target must keep the previous target's. Give
both the same quat, as above, or the leg can fail with `PLANNING_FAILED leg 2/2`.
A radius of 0, and the last target, stop. A radius above 0 lets the planner cut
the corner, but it may still stop there. With a radius above 0 a `JointTarget`'s joints are only passed
near, not reached. The call returns `True` or `False` and raises like
`go_to_ee_pose`, and its end check is against the last target. cuRobo checks
every cut corner for collisions in its world. Near something that isn't in that
world, keep the radius small.

**What a motion can raise.** `RammpArm`'s methods return `False` for a refused
or failed motion, an e-stop included, as they always did. On top of that, any
motion can raise:

- `Cancelled`: the orchestrator cancelled the action (the goal or stream is
  stopped first);
- `ControlTaken`: another sender took control of the arm mid-action;
- `StreamClosed`: the driver closed your `stream(...)` session;
- `NotInAction`: an arm call outside `execute()`, or `send()` outside its
  `with` block;
- `EStopped`: the driver API (`execute_joint_trajectory`, `stream`, `send`)
  while the arm is e-stopped.

Import them from the package, as in
`from capability_engine import Cancelled, ControlTaken`. All but `Cancelled` are
`ArmError`s.

**Rules for the sim.** One sim per process, shared with `self.arm`, built from
the driver's own model. These rules keep your checks and the arm in agreement:

- Get the sim from `make_sim()` (`capability_engine.sim`). Never build your own
  PyBullet world or pass a different URDF while the driver is up.
- Call `make_sim()` for the first time inside `execute()`: the model comes from
  the driver through `self.arm`, which is only available in an action. Later
  calls, anywhere, return the same copy.
- If you checked joints, send joints (`move_joints`). If you send poses
  (`move_cartesian`), check them with `cartesian_knots` on the shared sim, not
  with your own IK: a different solver may land on different joints.
- Look poses up in `rammp_arm.FRAME`, not a hard-coded frame. `arm_mount` and
  `arm_base_link` agree on dojo and are 30 degrees apart on flatbot and gen_1_5.
- Settable values are set through functions, not by assigning to a re-export:
  turn the keep-out on with `set_keep_out(fn)`. Speeds such as
  `rammp_arm.CART_SPEED *= s` are fine.

**Control comes from the orchestrator.** It grants the arm when it starts your
action. Never call `/acquire_control`, `/release_control` or `/revoke_control`, and
never publish `/estop`: the orchestrator and other nodes own them, and `self.arm`
has no way to. It only reads them: it refuses commands while the arm is e-stopped
and raises `ControlTaken` when another sender takes the arm.

**Your own topics: subscribe only during the action.** An idle subscription still
costs CPU for every message it receives (about 2 ms each on the Orin:
Demo-Software #263), so a camera or detections subscription should exist only
while an action needs it. Make it inside `execute()` with
`self.create_action_subscription(...)` (and `self.create_action_publisher(...)`,
`self.create_action_tf_buffer()` in place of a `TransformListener`): the same
arguments as rclpy's, and the engine destroys them when the action ends. Do not
destroy subscriptions yourself: on Humble, destroying one while the executor
spins can crash the node, and the engine does it in the safe order. Plain
`self.create_subscription()` still works for what the node needs all its life.

**Cancel ends the action promptly.** Call `ctx.checkpoint()` between steps and keep
each step short: on a cancel or a stop (`docker stop`), `self.arm` cancels its
motion at once, but your own code ends only at its next checkpoint, and a stop
waits at most a few seconds for it.

**No `print`.** Ruff forbids it. When you migrate a student script, turn each
`print(msg)` into `self.get_logger().info(msg)` (`warn`/`error` where it was a
warning or failure) with the same message text.

**Parameters.** Every tunable of `self.arm` is a ROS parameter `arm.<name>` whose
default is the students' value: speeds, tolerances, timeouts, IK guards. They are
read once at startup (a change needs a restart) and are listed with their meaning
in [`capabilities/example/docs/interface.md`](capabilities/example/docs/interface.md);
copy that table and the `arm.*` keys under `params:` in the real fragment into
your capability. A test checks the example's fragment and table against
`capability_engine/arm_params.py`. In mock runs (`mock:=true`) there is no
driver, so no model: set `arm.mock_urdf` to the URDF file `make_sim()` should
load (and `arm.mock_joints` for where the arm starts). Launch files take it as
`mock_urdf:=/path/robot.urdf`; native runs can also use `--ros-args -p arm.mock_urdf:=...`.
Two values are deliberately not in the fragment: `arm.sender_id` defaults to the
engine id, and `rammp_arm.IMPEDANCE` is the one `RammpArm` value that is not an
`arm.*` parameter at all, because it is only used when a caller passes it
(`move_joints(..., impedance=IMPEDANCE)`).
