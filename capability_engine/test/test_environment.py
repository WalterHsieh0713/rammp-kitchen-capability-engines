from pathlib import Path


def test_interfaces_and_libs_present():
    import numpy  # noqa: F401
    import pybullet  # noqa: F401
    import scipy  # noqa: F401
    from rammp_arm_interfaces.action import (  # noqa: F401
        ExecuteJointTrajectory,
        GoToEEPose,
        GoToJointConfig,
        GoToPreset,
    )
    from rammp_arm_interfaces.msg import EeState, ImpedanceGains
    from rammp_arm_interfaces.srv import (  # noqa: F401
        CloseStream,
        ListControllers,
        OpenStream,
        SetGains,
    )
    from rammp_common_interfaces.srv import AcquireControl  # noqa: F401
    from rammp_curobo_interfaces.action import PlanToJoints, PlanToPose  # noqa: F401
    from rammp_orchestrator_interfaces.action import ExecuteAction  # noqa: F401

    assert hasattr(EeState(), "wrench")
    assert ImpedanceGains.PROFILE_CUSTOM == 4


def test_fixtures_present():
    assert Path("/fixtures/cornell/robot.urdf").is_file()
    for robot in ("dojo", "flatbot", "gen_1_5"):
        assert Path(f"/fixtures/urdf/{robot}.urdf").is_file()
