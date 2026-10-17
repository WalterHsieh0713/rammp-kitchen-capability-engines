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
writes its own `execute()`. The same image runs either way:

```bash
make run CAP=example                  # registers with the orchestrator; waits for it if absent
make run CAP=example STANDALONE=1     # no orchestrator; MOCK=1 also skips hardware
make action CAP=example ACTION=count PARAMS='{"to": 5}'   # send a goal, as the orchestrator would
```

Standalone, the orchestrator calls a capability makes (`ctx.request_intent`,
`ctx.write_world_state`, `ctx.update_object`) are logged and skipped. Arm control
arbitration (`AcquireControl`) is not wired in yet: the orchestrator does not
call it today.

To run it on the robot, transcribe the capability's two fragments into
[`rammp-deployments`](https://github.com/rammp-org/rammp-deployments)'s
`december_2026/sheppy-manifest.yaml` as a node with both alternatives.

## A capability

```
capabilities/<name>/
  Dockerfile                    FROM a pinned rammp-base / rammp-cuda version, built from the repo root
  rammp-alternative.yaml        the real alternative: sheppy's schema plus `tier`
  rammp-alternative.mock.yaml   same topics, no hardware — CI smoke-tests this one
  docs/interface.md             the page rammp-docs pulls; keep its headings
  ...                           your code
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
