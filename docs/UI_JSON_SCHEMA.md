# UI bridge JSON schema

`ctx.update_user_panel()` / `ctx.poll_ui_event()` / `ctx.publish_overlay_image()`
(see the README's [`ctx` UI section](../README.md#ui-ctx)) talk to
`rammp_ui_bridge`'s `RequestUserInput` action and `UpdateUserInputPanel`
service, both of which carry a JSON string in a `*_json` field rather than
typed ROS fields. This page documents the shape of that JSON — what a
capability sends, and what it gets back.

None of this is validated against a formal JSON Schema file today; it's a
plain-dict convention enforced only by the engine and bridge code that reads
it. Malformed/non-object JSON is handled fail-soft (treated as empty/absent)
rather than raising, on both ends.

## Request: `update_user_panel(panel_json, objects, selected_instance_id)`

All three arguments are optional and independent — pass only the ones you want
to change; a `None` argument means "leave this aspect of the panel unchanged".
At least one of `panel_json`/`objects` is required the first time
`update_user_panel()` is called in a run, to have something to open the goal
with.

### `panel_json`

A dict, JSON-encoded by the engine before it's sent. Keys:

| Key        | Type                 | Meaning                                 |
| ---------- | -------------------- | --------------------------------------- |
| `title`    | `str`                | Panel heading                           |
| `message`  | `str`                | Body text                               |
| `progress` | `object` (see below) | A progress bar; omit the key for no bar |
| `buttons`  | `array` of `object`s | Tappable buttons; omit/empty for none   |

`progress` object:

| Key        | Type     | Meaning                                                      |
| ---------- | -------- | ------------------------------------------------------------ |
| `phase`    | `str`    | Short label for the current step                             |
| `pct`      | `number` | 0-100                                                        |
| `pausable` | `bool`   | Whether the bridge should offer a pause control alongside it |

Each entry in `buttons`:

| Key         | Type  | Meaning                                                                 |
| ----------- | ----- | ----------------------------------------------------------------------- |
| `button_id` | `str` | Opaque id this capability matches against in `poll_ui_event()`'s result |
| `label`     | `str` | Button text                                                             |

Example (from `capabilities/example/example/node.py`'s `count_with_panel`):

```json
{
  "title": "Counting",
  "message": "3/10",
  "progress": {"phase": "counting", "pct": 30.0, "pausable": false},
  "buttons": [{"button_id": "stop", "label": "Stop"}]
}
```

A `button_press` for any `button_id` in the currently-open panel always ends
the `RequestUserInput` goal (see **Result**, below) — there's no "button that
keeps the panel open" concept. `update_user_panel()`'s next call transparently
opens a fresh goal.

### `objects`

A list of dicts, one per tappable object to overlay on the UI's camera view
(e.g. candidate items for the user to pick between). Each dict:

| Key                | Type                           | Meaning                                                                                                                                     |
| ------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `instance_id`      | `str`                          | Opaque id this capability matches against `poll_ui_event()`'s `object_tapped` events                                                        |
| `perception_class` | `str`                          | The object's class, shown as a label                                                                                                        |
| `box`              | `[x_min, y_min, x_max, y_max]` | Pixel-space bounding box in the video frame named by `source`                                                                               |
| `centroid`         | `[x, y]` (optional)            | Pixel-space center; derived from `box` if omitted                                                                                           |
| `source`           | `"scene"` or `"overlay"`       | Which video channel `box`'s pixel coordinates are relative to — this capability's scene camera, or its own `publish_overlay_image()` stream |

Example:

```json
[
  {
    "instance_id": "fridge_door_1",
    "perception_class": "fridge_door",
    "box": [120.0, 80.0, 340.0, 410.0],
    "source": "scene"
  }
]
```

### `selected_instance_id`

A single `str`: which `instance_id` (from `objects`, current or previous) the
bridge should render as highlighted/selected. Pass `""` to clear the
highlight.

## Response: `poll_ui_event()`

Returns `None` if nothing new has happened since the last call, otherwise one
of:

```json
{"type": "button", "button_id": "stop"}
```

The panel's `RequestUserInput` goal has ended (a button press, an engine
cancel, or a bridge-forced abort). The next `update_user_panel()` call
transparently opens a fresh goal.

```json
{"type": "object_tapped", "instance_id": "fridge_door_1"}
```

The user tapped one of the objects most recently pushed via `objects`. This
does **not** end the goal — the user may tap several different objects in a
row while deciding — it's just feedback; call `update_user_panel()` again with
a new `selected_instance_id` if you want the highlight to follow the tap.

Only one event is held at a time; `poll_ui_event()` consumes it. A later event
overwrites an unread earlier one rather than queuing.

## `publish_overlay_image(image)`

Takes a `sensor_msgs/Image`, not JSON — published as-is to the bridge's shared
overlay topic, stamped with this run's session token. Its pixel frame is what
an `objects` entry's `"source": "overlay"` refers to.
