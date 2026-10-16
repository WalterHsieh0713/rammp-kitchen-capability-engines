# Example Capability — Interface

> Replace this page with your capability's real interface. `rammp-docs` pulls this
> file, so its section headings are load-bearing — keep them.

**Rung:** ad-hoc (declared by this module, not standardized)
**Owner:** @SwapnilPande
**Tier:** experimental

## What it does

One paragraph. What job does this module do in the system, and what would break
if it were missing?

## Publishes

| Topic             | Type              | Rate | Meaning                                                         |
| ----------------- | ----------------- | ---- | --------------------------------------------------------------- |
| `/example/status` | `std_msgs/String` | 1 Hz | Health topic; the payload names the alternative that is running |

## Subscribes

| Topic | Type | Required | Meaning                      |
| ----- | ---- | -------- | ---------------------------- |
| —     | —    | —        | This module consumes nothing |

## Parameters

| Name              | Default | Meaning                 |
| ----------------- | ------- | ----------------------- |
| `publish_rate_hz` | `1.0`   | Status publication rate |

## Hardware

None. Devices and GPUs belong in `rammp-alternative.yaml`, never in a setup
step — if this module needed `/dev/ttyUSB0`, the fragment would declare it.

## Alternatives

| id             | What it is                                                                         |
| -------------- | ---------------------------------------------------------------------------------- |
| `example_real` | The module proper                                                                  |
| `example_mock` | Same interface, no hardware — for developing against this module without owning it |
