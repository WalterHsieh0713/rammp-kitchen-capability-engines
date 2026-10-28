import importlib

_ENGINE = ["Action", "Cancelled", "CapabilityEngine", "Context", "EngineError", "run"]
# self.arm's exceptions (driver_state is pure Python, like sim).
_ARM_ERRORS = [
    "ArmError",
    "ControlTaken",
    "EStopped",
    "ModeConflict",
    "NotInAction",
    "StateTimeout",
    "StreamClosed",
]
__all__ = _ENGINE + _ARM_ERRORS


def __getattr__(name):
    # Lazy, so `import capability_engine.sim` works without rclpy (engine imports it).
    if name in _ENGINE:
        return getattr(importlib.import_module("capability_engine.engine"), name)
    if name in _ARM_ERRORS:
        return getattr(importlib.import_module("capability_engine.driver_state"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
