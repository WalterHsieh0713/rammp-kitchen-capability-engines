"""The Cornell arm commands that `ArmInterface.execute_command()` takes.

Vendored from feeding-deployment microwave-wip@7bc328ae
src/feeding_deployment/control/robot_controller/command_interface.py, so `feeding_deployment` is
not a dependency. Only the four commands SheppyArm executes.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class KinovaCommand:
    """Establish an interface for commands that can be sent to the robot."""


@dataclass(frozen=True)
class CartesianTrajectoryCommand(KinovaCommand):
    """Command to follow a cartesian trajectory."""

    traj: list[tuple[NDArray, NDArray]]

    def __init__(self, traj):
        object.__setattr__(self, "traj", [(np.array(pos), np.array(quat)) for pos, quat in traj])
        assert all(pos.shape == (3,) and quat.shape == (4,) for pos, quat in self.traj)


@dataclass(frozen=True)
class JointCommand(KinovaCommand):
    """Command to set the joint positions."""

    pos: NDArray

    def __init__(self, pos):
        object.__setattr__(self, "pos", np.array(pos))  # convert list to numpy array
        num_dof = 7
        assert self.pos.shape == (num_dof,)


class OpenGripperCommand(KinovaCommand):
    """Command to open the gripper."""


class CloseGripperCommand(KinovaCommand):
    """Command to close the gripper."""
