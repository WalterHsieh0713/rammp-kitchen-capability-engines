import importlib

__all__ = ["Action", "Cancelled", "CapabilityEngine", "Context", "EngineError", "run"]


def __getattr__(name):
    # Lazy, so `import capability_engine.sim` works without rclpy (engine imports it).
    if name in __all__:
        return getattr(importlib.import_module("capability_engine.engine"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
