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
