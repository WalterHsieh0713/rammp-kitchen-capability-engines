# AGENTS.md

Instructions for coding agents in this repo. `README.md` has the API
reference, the development flow and the perception table; `CONTRIBUTING.md`
has the rules for `self.arm`, parameters and branches. Read both before writing
a capability. This file adds what an agent needs on top of them.

## What this repo is

Kitchen capabilities (button press, microwave open and close, grab container,
...) for RAMMP's December 2026 demos on a Kinova Gen3 with a Robotiq 2F-85.
Each is an `ament_python` package under `capabilities/<name>/`, built into its
own image and launched by sheppy from `rammp-deployments`.
`capability_engine/` is the shared base package.

## Rules

- **Every capability subclasses `capability_engine.CapabilityEngine`.** Each
  complete behavior the user can ask for is one `Action`, run by `execute()`
  and following the base class's contract: class attributes, `EngineError`,
  `ctx.checkpoint()` between steps, `create_action_subscription` for anything
  you subscribe to. No behavior lives in a side script, a service of your own,
  or a sequence of actions the orchestrator has to stitch together.
- **Prefer the RAMMP arm driver API.** `self.arm` has two APIs:
  - Recommended: `go_to_ee_pose(s)`, `go_to_joint_config(s)`, `go_to_preset`
    (cuRobo plans, the driver executes), and the driver methods
    (`execute_joint_trajectory`, `stream`, `plan_to_*`, `joint_states()`,
    `ee_state()`, ...). Write new code against these.
  - Legacy, for migration only: the Cornell feeding-deployment API
    (`move_joints`, `move_cartesian`, `execute_command` with
    `capability_engine.cornell_commands`) and the PyBullet sim
    (`capability_engine.sim`, the `rammp_arm` path helpers). It exists so a
    ported script runs with few changes. Don't add new code on it. When you
    keep a legacy move, say why in the PR.
- **Use the perception already running** (`README.md`, "Perception you can
  use"). Don't add a detector or segmenter to a capability without asking.
- **Leave arm control and safety alone.** Never call `/acquire_control`,
  `/release_control`, `/revoke_control`, or publish `/estop`. A cancel,
  failure or exception must leave the arm still and the gripper safe.
- **Keep `capability_engine/` changes small and separate.** Every capability
  depends on it. Make additive changes in their own PR; propose a redesign in
  an issue, not inside a capability PR.
- **Hardware runs are attended**, with a human on the e-stop. Don't start
  anything that can move the arm unless the user has asked you to and is there.

## Development flow

Follow the steps in `README.md` ("Development flow"), in order:

1. Port the behavior into a capability and test it (mock, then hardware).
1. Switch to the driver API where it can do the same job.
1. Polish the motion: blend consecutive moves into one with
   `go_to_ee_poses` / `go_to_joint_configs` and `blend_radius`, cut needless
   stops, tune speeds.
1. Publish the image (merge to `dev`, then a `v*` tag) and mark the
   capability `tier: demo-ready` after an attended hardware pass.
1. Open a PR to `rammp-org/rammp-deployments` adding the node, real and mock
   alternatives pinned to the release, to `december_2026/sheppy-manifest.yaml`
   and the demo profiles.

Don't skip ahead: polish on top of a port that hasn't been tested on the arm
hides real failures.

## Commands

```bash
make new CAP=<name>      # scaffold from capabilities/example
make check               # repo rules, fragment schemas, sheppy's loader
make test-arm            # self.arm / capability_engine tests in an x86 Humble image
make lint                # pre-commit on everything (ruff, mdformat, hadolint, ...)
make build CAP=<name>    # arm64 only (rammp-base is Jetson-only); CI builds otherwise
make smoke CAP=<name>    # mock image publishes its topics and exits on SIGTERM
make run CAP=<name> STANDALONE=1 MOCK=1
make action CAP=<name> ACTION=<id> PARAMS='{...}'
```

Natively: `colcon build --packages-up-to <name>`, then
`ros2 launch <name> <name>.launch.py use_orchestrator:=false mock:=true`.

## Other repos

Read these when you need to know how a capability is used or deployed. The
local clones may be stale: fetch, or read the default branch on GitHub.

| Repo                                                     | Look here for                                                                                                                                                                          |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rammp-org/rammp-deployments`                            | How everything is deployed. `december_2026/sheppy-manifest.yaml` (every node, its topics and parameters, including the perception nodes), `profiles/`, robot config, `REVIEW_RULES.md` |
| `rammp-org/rammp-orchestrator`                           | The engine contract from the orchestrator's side. Code is on `feat/orchestrator`: `docs/ENGINE_DEVELOPER_GUIDE.md`, `docs/RUNTIME_FLOWS.md`                                            |
| `rammp-org/sheppy`                                       | Launch tooling: manifests, alternatives, profiles, `sheppy up <profile>`                                                                                                               |
| `rammp-org/rammp-org.github.io`                          | Project docs at https://rammp-org.github.io: patterns, publishing checklist, configuring a deployment, arm interfaces                                                                  |
| `rammp-org/rammp-perception`                             | The perception nodes and `docs/interface.md` (topics, parameters, classes)                                                                                                             |
| `rammp-org/rammp-interfaces-ros2`                        | Message, service and action types. Use these; never redefine one                                                                                                                       |
| `kinova-gen3-driver`, `kinova-gen3-ros2`, `rammp-curobo` | The arm driver, its ROS 2 wrapper and the planner. You shouldn't need them; `self.arm` covers them                                                                                     |

## Pull requests

- Branch `feature/<issue>-<description>` from the latest `dev`; PR into `dev`
  with the template's sections.
- The reviewer is an experienced roboticist who hasn't seen this code. Explain
  the implementation in robotics terms: the motion sequence and its frames
  (approach, grasp, retreat), which moves are planned, Cartesian or streamed,
  where control is compliant and why, what perception it uses and how a
  detection becomes a target pose, and how a failure or cancel leaves the arm.
  Don't narrate the code; point at files for detail.
- Report test results with evidence: mock, sim or hardware (say which), the
  number of runs and successes, the cause of each failure, and links to logs or
  video. Say what wasn't tested. Mock results are not robot validation.
- Keep the diff clean: no scratch scripts, debug output, run artifacts,
  versioned copies or unrelated changes. `make check` and `make lint` pass
  before you open it.
- Check it against `REVIEW_RULES.md` in `rammp-deployments` before asking for
  review.
