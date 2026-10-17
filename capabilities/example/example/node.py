"""Reference capability: one action, `count`, that counts up with progress.

Copy this directory with `make new CAP=<name>` and replace `count` with your
capability's actions. Everything orchestrator-facing (registration, the
activate/execute/control/hibernate servers, /example/status, SIGTERM) comes
from capability_engine.CapabilityEngine.

    ros2 launch example example.launch.py                          # with the orchestrator
    ros2 launch example example.launch.py use_orchestrator:=false  # standalone
    make action CAP=example ACTION=count PARAMS='{"to": 5}'        # send it a goal
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
    actions = [
        Action(
            "count",
            "Count up",
            params_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "integer", "minimum": 1, "default": 3},
                    "interval_s": {"type": "number", "minimum": 0, "default": 1.0},
                },
            },
        )
    ]

    def __init__(self):
        super().__init__("example")

    def execute(self, action_id, params, ctx):
        to = int(params.get("to", 3))
        interval = float(params.get("interval_s", 1.0))
        if to < 1:
            raise EngineError("bad_params", "'to' must be at least 1", recoverable=True)

        for n in range(1, to + 1):
            ctx.checkpoint()  # honours pause and cancel between steps
            # A real capability would command hardware here, unless self.mock.
            time.sleep(interval)
            ctx.progress("counting", 100.0 * n / to, f"{n}/{to}")

        # A capability that changes the world reports it, e.g.
        # ctx.update_object(self.instance_id, opened=True) or
        # ctx.write_world_state(gripper={"holding": "cup_1"}).

        return {"counted_to": to}


def main():
    run(Capability)


if __name__ == "__main__":
    main()
