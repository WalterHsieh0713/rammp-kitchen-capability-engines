"""self.arm's dataclasses, and their conversions from and to the driver's messages: pure
functions, no node. capability_engine.arm re-exports the public ones."""

from dataclasses import dataclass, field

import numpy as np
import rammp_arm
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointTolerance
from rammp_arm import JOINTS
from rammp_arm_interfaces.msg import (
    ImpedanceGains,
    JointImpedanceGainValues,
    JointSetpoint,
    PoseSetpoint,
    TwistSetpoint,
)
from scipy.spatial.transform import Rotation as R

from capability_engine.driver_state import ArmError

# `gains` by name: the driver's impedance profiles ("default": the session's default)
GAIN_PROFILES = {
    "default": ImpedanceGains.PROFILE_SESSION_DEFAULT,
    "soft": ImpedanceGains.PROFILE_SOFT,
    "medium": ImpedanceGains.PROFILE_MEDIUM,
    "stiff": ImpedanceGains.PROFILE_STIFF,
}
GAIN_KEYS = ("kq", "zeta", "torque_limit")  # `gains` as a dict: custom joint impedance

HOLD_NONE = 0  # go_to_ee_pose orientation_hold: the planner reorients as it likes
HOLD_LEVEL = 1  # keep the tool's tilt; spin about arm_mount z is free
HOLD_FIXED = 2  # keep the orientation entirely

# What stream() sends on each /setpoint/<channel> a controller reads, and what send() takes.
SETPOINTS = {
    "joint_position": (JointSetpoint, "7 joint positions (rad)"),
    "joint_velocity": (JointSetpoint, "7 joint velocities (rad/s)"),
    "joint_torque": (JointSetpoint, "7 joint torques (N m)"),
    "pose": (PoseSetpoint, "(position xyz, quaternion xyzw) of /ee_state's frame in arm_mount"),
    "twist": (TwistSetpoint, "6 floats: linear (m/s) then angular (rad/s), in arm_mount axes"),
}


@dataclass
class JointStates:
    """/joint_states for arm_joint_1..7, as the driver reports them (not wrapped)."""

    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray
    age_s: float = 0.0


@dataclass
class EeState:
    """/ee_state: the flange (not the tool frame) in `frame_id`, its twist and measured wrench."""

    frame_id: str
    position: np.ndarray  # x, y, z
    orientation: np.ndarray  # x, y, z, w
    linear: np.ndarray
    angular: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    age_s: float = 0.0


@dataclass
class GripperState:
    position: float  # 0 (open) .. 1 (closed)
    effort: float
    current: float
    present: bool
    age_s: float = 0.0


@dataclass
class ControlStatus:
    arbitration_enabled: bool
    estopped: bool
    owned: bool
    owner_id: str
    generation: int
    rejected_count: int
    age_s: float = 0.0


@dataclass
class StreamStatus:
    open: bool
    controller: str
    channels: list[str]
    timeout_s: float
    rejected_count: int
    age_s: float = 0.0


@dataclass
class DiagnosticStatus:
    level: int
    name: str
    message: str
    hardware_id: str
    values: dict[str, str] = field(default_factory=dict)


@dataclass
class Diagnostics:
    status: list[DiagnosticStatus]
    age_s: float = 0.0


@dataclass
class TrajectoryResult:
    """execute_joint_trajectory's result, as the driver settled the goal."""

    error_code: int  # 0 SUCCESSFUL, or one of rammp_arm.ERRORS
    error_string: str
    final_error: np.ndarray  # joint position error (reference - actual) at the end, rad


@dataclass
class GoalFeedback:
    """A running goal's feedback."""

    fraction_complete: float  # execution progress, 0 .. 1 (0 while planning)
    actual: np.ndarray  # measured joint positions, rad
    phase: str = ""  # go_to goals: "planning" or "executing"
    planner_state: str = ""  # go_to goals: cuRobo's state while planning


@dataclass
class Plan:
    """A cuRobo plan (nothing moved): sample times (s), joints and joint velocities, (N, 7) in
    JOINTS order. A failed plan has success False, the planner's message and no samples."""

    success: bool
    message: str
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, len(JOINTS))))
    qd: np.ndarray = field(default_factory=lambda: np.zeros((0, len(JOINTS))))
    planning_time: float = 0.0  # s


@dataclass
class PoseTarget:
    """One target of go_to_ee_poses: the tool (tool_frame) at (pos, quat xyzw) in FRAME, reached
    with orientation hold `hold` (HOLD_*) at `speed_scale` (default GO_TO_SPEED_SCALE).
    `blend_radius` (m): how close the tool must pass this pose, cutting the corner into the next
    leg instead of stopping on it; 0 stops here, and the last target always stops. The planner
    cuts only as far as the joint limits, collision and the holds allow, which may be not at
    all. A hold is checked against where the arm is when its leg starts: a held first target
    must keep the tool's current orientation (its tilt under HOLD_LEVEL), as for go_to_ee_pose,
    and a later one the previous target's (give it the same quat), or the leg may fail to plan
    (PLANNING_FAILED leg k/N). `quat` is sent normalized."""

    pos: tuple
    quat: tuple
    hold: int = HOLD_NONE
    blend_radius: float = 0.0
    speed_scale: float | None = None


@dataclass
class JointTarget:
    """One target of go_to_joint_configs: joints `joints` (rad, J1..J7) at `speed_scale` (default
    GO_TO_SPEED_SCALE). `blend_radius` as PoseTarget's, around where the tool is at these joints:
    with a radius above 0 the joints themselves are not reached, only passed near."""

    joints: tuple
    blend_radius: float = 0.0
    speed_scale: float | None = None


@dataclass
class Controller:
    """A streaming controller of the driver (/list_controllers)."""

    name: str
    channels: list[str]  # the /setpoint/<channel> topics it reads
    available: bool  # whether the running driver can open it


def _goal_feedback(m):
    return GoalFeedback(
        float(m.fraction_complete),
        np.array(m.actual.positions),
        getattr(m, "phase", ""),
        getattr(m, "planner_state", ""),
    )


def _speed_scale(speed_scale):
    """A go_to speed scale: GO_TO_SPEED_SCALE when None."""
    return float(rammp_arm.GO_TO_SPEED_SCALE if speed_scale is None else speed_scale)


def _impedance_gains(gains) -> ImpedanceGains:
    """`gains` as the driver's ImpedanceGains: None (the message default), a profile name in
    GAIN_PROFILES, or a dict {"kq": 7 floats, "zeta": float, "torque_limit": 7 floats} (custom
    joint impedance, RammpArm's IMPEDANCE without path_tol). ArmError for anything else."""
    if gains is None:
        return ImpedanceGains()
    if isinstance(gains, str) and gains in GAIN_PROFILES:
        return ImpedanceGains(profile=GAIN_PROFILES[gains])
    if isinstance(gains, dict) and set(gains) == set(GAIN_KEYS):
        try:
            kq = np.asarray(gains["kq"], float)
            zeta = float(gains["zeta"])
            torque_limit = np.asarray(gains["torque_limit"], float)
        except (TypeError, ValueError):
            kq = None
        if kq is not None and kq.shape == torque_limit.shape == (len(JOINTS),):
            return ImpedanceGains(
                profile=ImpedanceGains.PROFILE_CUSTOM,
                custom=JointImpedanceGainValues(
                    kq=kq.tolist(), zeta=zeta, torque_limit=torque_limit.tolist()
                ),
            )
    raise ArmError(
        f"gains must be one of {', '.join(GAIN_PROFILES)} or a dict "
        f'{{"kq": 7 floats, "zeta": float, "torque_limit": 7 floats}}, not {gains!r}'
    )


def _duration(t):
    sec = int(t)
    return Duration(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def _seconds(d):
    return d.sec + d.nanosec * 1e-9


def _joint_index(names):
    """Where arm_joint_1..7 are in `names`, in JOINTS order, when `names` are exactly those or
    joint_1..7 (the sim's rule: cuRobo's config names them joint_<i>); None otherwise."""
    names = list(names)
    for fmt in ("arm_joint_{}", "joint_{}"):
        wanted = [fmt.format(i) for i in range(1, len(JOINTS) + 1)]
        if sorted(names) == sorted(wanted):
            return [names.index(n) for n in wanted]
    return None


def _plan(res):
    """A planner's result as a Plan, its joints reordered to JOINTS; None when its joints are
    not arm_joint_1..7 or joint_1..7 (_joint_index) or a point lacks positions or velocities
    for them."""
    names = list(res.trajectory.joint_names)
    idx = _joint_index(names)
    if idx is None:
        return None
    points = res.trajectory.points
    if any(len(p.positions) != len(names) or len(p.velocities) != len(names) for p in points):
        return None
    return Plan(
        True,
        res.message,
        np.array([_seconds(p.time_from_start) for p in points]),
        np.array([[p.positions[i] for i in idx] for p in points]).reshape(-1, len(JOINTS)),
        np.array([[p.velocities[i] for i in idx] for p in points]).reshape(-1, len(JOINTS)),
        float(res.planning_time),
    )


def _setpoint(channel, value):
    """`value` as the message on /setpoint/<channel> (see SETPOINTS), or None when it is not
    of that channel's shape."""
    msg_type = SETPOINTS[channel][0]
    try:
        if channel == "pose":
            pos, quat = (np.asarray(v, float) for v in value)
            if pos.shape != (3,) or quat.shape != (4,):
                return None
            m = msg_type()
            p, o = m.pose.position, m.pose.orientation
            p.x, p.y, p.z = pos.tolist()
            o.x, o.y, o.z, o.w = quat.tolist()
            return m
        v = np.asarray(value, float)
    except (TypeError, ValueError):
        return None
    if channel == "twist":
        if v.shape != (6,):
            return None
        m = msg_type()
        lin, ang = m.twist.linear, m.twist.angular
        lin.x, lin.y, lin.z, ang.x, ang.y, ang.z = v.tolist()
        return m
    if v.shape != (len(JOINTS),):
        return None
    return msg_type(values=v.tolist())


def _tolerances(names, position):
    """A position tolerance (rad) per joint: `position` is one for all or one per joint; None
    gives none (the driver's default)."""
    if position is None:
        return []
    per_joint = np.broadcast_to(np.asarray(position, float), (len(names),))
    return [
        JointTolerance(name=n, position=float(p)) for n, p in zip(names, per_joint, strict=True)
    ]


def _vec(v):
    return np.array([v.x, v.y, v.z])


def _joint_states(m):
    if not set(JOINTS) <= set(m.name):
        return None
    idx = [list(m.name).index(n) for n in JOINTS]
    return JointStates(
        np.array([m.position[i] for i in idx]),
        np.array([m.velocity[i] for i in idx]),
        np.array([m.effort[i] for i in idx]),
    )


def _ee_state(m):
    o = m.pose.orientation
    return EeState(
        m.header.frame_id,
        _vec(m.pose.position),
        np.array([o.x, o.y, o.z, o.w]),
        _vec(m.twist.linear),
        _vec(m.twist.angular),
        _vec(m.wrench.force),
        _vec(m.wrench.torque),
    )


def _tool_position(ee):
    """The tool frame's position: the flange + TOOL_Z along the flange z."""
    return ee.position + R.from_quat(ee.orientation).apply([0.0, 0.0, rammp_arm.TOOL_Z])


def _gripper_state(m):
    return GripperState(float(m.position), float(m.effort), float(m.current), bool(m.present))


def _control_status(m):
    return ControlStatus(
        m.arbitration_enabled, m.estopped, m.owned, m.owner_id, m.generation, m.rejected_count
    )


def _stream_status(m):
    return StreamStatus(m.open, m.controller, list(m.channels), m.timeout_s, m.rejected_count)


def _diagnostics(m):
    return Diagnostics(
        [
            DiagnosticStatus(
                ord(s.level),  # a ROS byte field: one-byte bytes
                s.name,
                s.message,
                s.hardware_id,
                {kv.key: kv.value for kv in s.values},
            )
            for s in m.status
        ]
    )
