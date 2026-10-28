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
action. Never call `/acquire_control` or `/release_control`; `self.arm` does not
either.

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
