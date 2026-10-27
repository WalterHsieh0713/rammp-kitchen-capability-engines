# Example Capability — Interface

> Replace this page with your capability's real interface. `rammp-docs` pulls this
> file, so its section headings are load-bearing — keep them.

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @SwapnilPande
**Tier:** experimental

## What it does

The reference capability. It offers one action, `count`, which counts up to
`to` with progress feedback and honours pause and cancel. It is activated when
the user selects a `counter`. Replace this paragraph with what your capability
does and what would break without it.

## Publishes

| Topic             | Type              | Rate | Meaning                                                                                                   |
| ----------------- | ----------------- | ---- | --------------------------------------------------------------------------------------------------------- |
| `/example/status` | `std_msgs/String` | 1 Hz | Health topic: JSON `state` (idle/executing/paused), `action`, `instance_id`, `mode`, `registered`, `mock` |

## Subscribes

| Topic | Type | Required | Meaning                      |
| ----- | ---- | -------- | ---------------------------- |
| —     | —    | —        | This module consumes nothing |

## Orchestrator interface

Served under `/example/`, types from `rammp_orchestrator_interfaces`:

| Name                      | Type                   | Meaning                                       |
| ------------------------- | ---------------------- | --------------------------------------------- |
| `/example/activate`       | `srv/Activate`         | Which actions are available for the selection |
| `/example/execute_action` | `action/ExecuteAction` | Run an action; progress as feedback           |
| `/example/control`        | `srv/Control`          | Pause, resume, cancel the running action      |
| `/example/hibernate`      | `srv/Hibernate`        | Drop the selection                            |

| Action  | Params                                                | Result payload      |
| ------- | ----------------------------------------------------- | ------------------- |
| `count` | `to` (int ≥ 1, default 3), `interval_s` (default 1.0) | `{"counted_to": n}` |

Registers with `/rammp_orchestrator/register_engine` on startup, retrying until
the orchestrator is up.

## Parameters

ROS parameters, declared with defaults under `params:` in the real fragment. A
deployment sets one per robot with an `overrides:` entry in its profile; sheppy
passes them as a params file.

| Name               | Default | Meaning                                                   |
| ------------------ | ------- | --------------------------------------------------------- |
| `publish_rate_hz`  | `1.0`   | Status publication rate                                   |
| `use_orchestrator` | `true`  | `false` runs standalone: no registration, drive it by CLI |
| `mock`             | `false` | `true` skips hardware (the mock alternative sets it)      |

## Hardware

None. Devices and GPUs belong in `rammp-alternative.yaml`, never in a setup
step — if this module needed `/dev/ttyUSB0`, the fragment would declare it.

## Alternatives

| id             | What it is                                                                         |
| -------------- | ---------------------------------------------------------------------------------- |
| `example_real` | The module proper                                                                  |
| `example_mock` | Same interface, no hardware — for developing against this module without owning it |
