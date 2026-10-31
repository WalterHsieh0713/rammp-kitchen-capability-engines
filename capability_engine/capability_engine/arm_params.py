"""Every tunable of self.arm, as the capability's ROS parameters arm.<name>.

`CapabilityEngine.__init__` declares each one with today's value as its default and writes the
value it gets to `target`, the module constant the code reads at call time (callers may still
mutate those constants at runtime, as RammpArm's did). The values are therefore process-wide, and
the engine refuses arm.* changes after startup.
"""

from dataclasses import dataclass

import rammp_arm

from capability_engine import arm, sim


@dataclass(frozen=True)
class Param:
    name: str  # the ROS parameter is arm.<name>
    default: object  # today's value
    description: str
    target: tuple  # (module, attribute) the value is written to


def _param(module, attr, description):
    default = getattr(module, attr)
    if isinstance(default, list):
        default = list(default)  # a copy: the default stays today's value
    return Param(attr.lower(), default, description, (module, attr))


PARAMS = [
    # rammp_arm: RammpArm's speeds, path and go_to settings
    _param(rammp_arm, "TOOL_Z", "Tool frame offset along the flange z (m)"),
    _param(rammp_arm, "DT", "Trajectory sample spacing (s)"),
    _param(rammp_arm, "JOINT_SPEED", "Peak joint speed of joint moves (rad/s)"),
    _param(rammp_arm, "CART_SPEED", "Peak tool speed of Cartesian paths (m/s)"),
    _param(rammp_arm, "CONTACT_JOINT_SPEED", "Peak joint speed of contact moves (rad/s)"),
    _param(rammp_arm, "CONTACT_CART_SPEED", "Peak tool speed of contact moves (m/s)"),
    _param(rammp_arm, "MIN_DURATION_S", "Shortest trajectory duration (s)"),
    _param(rammp_arm, "CART_STEP_M", "Longest Cartesian step before IK (m)"),
    _param(rammp_arm, "CART_STEP_DEG", "Largest Cartesian turn per step before IK (deg)"),
    _param(rammp_arm, "ROT_EQUIV_M", "Travel a 1 rad turn counts as when timing a path (m)"),
    _param(rammp_arm, "MAX_KNOT_JUMP_DEG", "Largest joint change between Cartesian knots (deg)"),
    _param(rammp_arm, "PATH_TOL_RAD", "Per-joint tracking tolerance of trajectories (rad)"),
    _param(rammp_arm, "GRIPPER_FORCE", "Default gripper force (0..1)"),
    _param(rammp_arm, "GRIPPER_SPEED", "Default gripper speed (0..1)"),
    _param(rammp_arm, "FRAME", "Frame of go_to_ee_pose goals and /ee_state"),
    _param(rammp_arm, "SENDER_ID", "sender_id on goals (the engine id under the engine)"),
    _param(rammp_arm, "GO_TO_SPEED_SCALE", "speed_scale of go_to goals (0.01..1)"),
    _param(rammp_arm, "GO_TO_TIMEOUT_S", "go_to planning plus execution limit (s)"),
    _param(rammp_arm, "GO_TO_TRACK_M", "go_to_ee_pose: largest tool error at the end (m)"),
    _param(rammp_arm, "GO_TO_TRACK_DEG", "go_to_joint_config: largest end joint error (deg)"),
    # sim: the IK and planning guards
    _param(sim, "MAX_IK_ERR", "IK position error above which a pose is unreachable (m)"),
    _param(sim, "MAX_STEP_JUMP_DEG", "Largest joint change of a planned step (deg)"),
    _param(sim, "J6_GUARD_DEG", "Largest absolute J6 the sim accepts (deg)"),
    _param(sim, "J4_GUARD_DEG", "Largest absolute J4 the sim accepts (deg)"),
    _param(sim, "MAX_CONTINUOUS_DELTA_DEG", "Largest short-way step of a free joint (deg)"),
    _param(sim, "KEEP_OUT_ERR", "IK error reported for a keep-out solution (m)"),
    _param(sim, "IK_MAX_ITERATIONS", "PyBullet IK iterations per solve"),
    _param(sim, "CLEARANCE_QUERY_M", "clearance() ignores bodies farther than this (m)"),
    _param(sim, "FINGERTIP_OPEN_RAD", "Sim finger joint angle with the gripper open (rad)"),
    _param(sim, "FINGER_CLOSED_RAD", "Sim finger joint angle with the gripper closed (rad)"),
    # arm: waits, timeouts and staleness limits of self.arm
    _param(arm, "STATE_WAIT_S", "First arm call: wait for the driver's state (s)"),
    _param(arm, "STATE_STALE_S", "Motion state older than this is an error (s)"),
    _param(arm, "DIAGNOSTICS_STALE_S", "/diagnostics older than this is an error (s)"),
    _param(arm, "GOAL_ACCEPT_S", "Wait for the driver to accept or reject a goal (s)"),
    _param(arm, "RESULT_MARGIN_S", "A trajectory's result may come this long after its end (s)"),
    _param(arm, "POLL_S", "Poll period of a running motion (s)"),
    _param(arm, "CANCEL_SETTLE_S", "Wait for a cancelled goal to settle (s)"),
    _param(arm, "HALT_STATUS_WAIT_S", "After a HALTED goal, wait for an e-stop status (s)"),
    _param(arm, "PLANNER_WAIT_S", "Wait for the cuRobo planner's action server (s)"),
    _param(arm, "GO_TO_SETTLE_S", "After a go_to goal, let the state catch up (s)"),
    _param(arm, "GRIPPER_PERIOD_S", "gripper() setpoint period (s)"),
    _param(arm, "GRIPPER_MOVED", "gripper(): position change that counts as moving (0..1)"),
    _param(arm, "GRIPPER_JUDGE_AFTER_S", "gripper(): judge the fingers still only after (s)"),
    _param(arm, "GRIPPER_STILL_S", "gripper(): still this long means settled (s)"),
    _param(arm, "PLAN_TIMEOUT_S", "Wait for a cuRobo plan (s)"),
    _param(arm, "SERVICE_WAIT_S", "Wait for a service server to come up (s)"),
    _param(arm, "SERVICE_CALL_S", "Wait for a service's answer (s)"),
    _param(arm, "STREAM_TIMEOUT_S", "stream(): default setpoint timeout of a session (s)"),
    _param(arm, "PUBLISHER_MATCH_S", "Wait for a new publisher to reach the driver (s)"),
    _param(arm, "CLOSE_STREAM_WAIT_S", "Wait for /close_stream's answer (s)"),
    _param(arm, "MOCK_JOINTS", "Mock mode: the joints it starts at (rad, J1..J7)"),
    _param(arm, "MOCK_URDF", "Mock mode: the URDF file make_sim() loads (empty: none)"),
]
