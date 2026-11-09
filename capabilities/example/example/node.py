"""Reference capability: `count` counts up with progress; `count_with_panel`
does the same while also driving a live UI-bridge panel.

Copy this directory with `make new CAP=<name>` and replace these with your
capability's actions. Everything orchestrator-facing (registration, the
activate/execute/control/hibernate servers, /example/status, SIGTERM) comes
from capability_engine.CapabilityEngine.

    ros2 launch example example.launch.py                          # with the orchestrator
    ros2 launch example example.launch.py use_orchestrator:=false  # standalone
    make action CAP=example ACTION=count PARAMS='{"to": 5}'        # send it a goal
    make action CAP=example ACTION=count_with_panel PARAMS='{"to": 5}'
"""

import time

from capability_engine import Action, CapabilityEngine, EngineError, run


class Capability(CapabilityEngine):
    display_name = "Example"
    # Activated when the user selects one of these perception classes.
    perception_classes = ["counter"]
    # Resources the orchestrator locks while an action runs: "arm", "base",
    # "gripper". This example touches no hardware.
    required_resources = []
    _COUNT_PARAMS_SCHEMA = {
        "type": "object",
        "properties": {
            "to": {"type": "integer", "minimum": 1, "default": 3},
            "interval_s": {"type": "number", "minimum": 0, "default": 1.0},
        },
    }
    actions = [
        Action("count", "Count up", params_schema=_COUNT_PARAMS_SCHEMA),
        Action("count_with_panel", "Count up (live panel)", params_schema=_COUNT_PARAMS_SCHEMA),
    ]

    def __init__(self):
        super().__init__("example")

    def execute(self, action_id, params, ctx):
        to = int(params.get("to", 3))
        interval = float(params.get("interval_s", 1.0))
        if to < 1:
            raise EngineError("bad_params", "'to' must be at least 1", recoverable=True)

        if not self.mock:
            # self.arm talks to the arm driver; the first call opens it for this action.
            state = self.arm.get_state()
            self.get_logger().info(f"arm joints {state['position']}, tool at {state['ee_pos'][:3]}")

        if action_id == "count_with_panel":
            return self._count_with_panel(to, interval, ctx)
        return self._count(to, interval, ctx)

    def _count(self, to: int, interval: float, ctx) -> dict:
        for n in range(1, to + 1):
            ctx.checkpoint()  # honours pause and cancel between steps
            # A real capability would command hardware here, unless self.mock.
            time.sleep(interval)
            ctx.progress("counting", 100.0 * n / to, f"{n}/{to}")

        # A capability that changes the world reports it, e.g.
        # ctx.update_object(self.instance_id, opened=True) or
        # ctx.write_world_state(gripper={"holding": "cup_1"}).

        return {"counted_to": to}

    def _count_with_panel(self, to: int, interval: float, ctx) -> dict:
        """Same as `count`, but also drives a live, engine-owned panel on the
        UI bridge directly (independent of ctx.progress(), which only reports
        to the orchestrator and is never shown to the user): a progress bar
        plus a Stop button, refreshed every step, and polled for a tap.
        ctx.update_user_panel()/ctx.poll_ui_event() are both non-blocking --
        this loop keeps counting in between calls rather than waiting on the
        user. No session token (e.g. running standalone) makes every bridge
        call a harmless no-op, so this still behaves like plain `count`.

        A capability with its own camera (e.g. a wrist cam) would also call
        ctx.publish_overlay_image(image) here, once per tick, to show the
        user a live view alongside the panel -- omitted since this example
        has no camera to publish.
        """
        for n in range(1, to + 1):
            ctx.checkpoint()
            pct = 100.0 * n / to
            ctx.update_user_panel(
                panel_json={
                    "title": "Counting",
                    "message": f"{n}/{to}",
                    "progress": {"phase": "counting", "pct": pct, "pausable": False},
                    "buttons": [{"button_id": "stop", "label": "Stop"}],
                }
            )
            event = ctx.poll_ui_event()
            if event is not None and event.get("type") == "button" and event["button_id"] == "stop":
                self.get_logger().info(f"count_with_panel: user pressed Stop at {n}/{to}")
                return {"counted_to": n, "stopped_by_user": True}
            time.sleep(interval)
            ctx.progress("counting", pct, f"{n}/{to}")

        return {"counted_to": to}


def main():
    run(Capability)


if __name__ == "__main__":
    main()
