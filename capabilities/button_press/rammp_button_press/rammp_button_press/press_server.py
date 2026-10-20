"""`/press_button` action server: the button_press capability's node.

Same detection, planning and gates as `press_button` (it runs `press_button.press_button` per
goal), but the camera, arm interface and kinematics stay loaded between goals, and progress
streams back as feedback (one line per stage, as the script prints them) and into the node's log.

SAFETY: the arm moves only if the node was started with `allow_execute:=true` AND the goal asks
for `execute`. Otherwise the goal runs as a dry run (detect + plan + gate every leg; nothing
moves) and the result says so. Someone stands at the e-stop whenever allow_execute is on.
Cancelling a goal stops the trajectory in flight; the driver holds the arm where it is, and the
goal ends there (no automatic return). One goal at a time: a second one is rejected.

Never exits because the arm or a camera is absent: until the arm driver answers it retries every
CONNECT_RETRY_S, and goals are rejected meanwhile. `--mock` serves the same interface with no
hardware (every goal succeeds at once, nothing is detected or moved): CI's smoke test and the
all-mock profile run it.

    press_server --ros-args -p allow_execute:=false
    press_client --target start_30s            # dry run
"""

from __future__ import annotations

import signal
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rammp_arm_interfaces.msg import GripperSetpoint, TwistSetpoint
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args

import rammp_button_press
from rammp_button_press import Abort, dome_pattern
from rammp_button_press.press_button import Run, build_parser, press_button
from rammp_button_press_interfaces.action import PressButton

MAX_PRESSES = 10
CONNECT_RETRY_S = 2.0
MOCK_SETPOINTS = (
    (TwistSetpoint, "/setpoint/twist"),
    (GripperSetpoint, "/setpoint/gripper"),
)
# ROS parameter -> press_button flag. Defaults are press_button's (the 1.1.0 driver, RealSense wrist
# camera); the fragments set flatbot's.
STRING_PARAMS = {
    "arm_frame": "--arm-frame",
    "joint_prefix": "--joint-prefix",
    "camera_frame": "--camera-frame",
    "camera_ns": "--camera-ns",
    "depth_topic": "--depth-topic",
    "far": "--far",
    "scene_ns": "--scene-ns",
    "scene_depth_topic": "--scene-depth-topic",
    "log_dir": "--log-dir",
}


class _FeedbackLines:
    """say() sink for one goal: every complete line goes to the node's log and out as action feedback."""

    def __init__(self, goal_handle, logger):
        self.gh, self.logger, self.buf = goal_handle, logger, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.logger.info(line.rstrip())
                fb = PressButton.Feedback()
                fb.stage = line.rstrip()
                self.gh.publish_feedback(fb)
        return len(s)

    def flush(self):
        pass


class PressServer(Node):
    def __init__(self, mock: bool):
        super().__init__("button_press")
        self.mock = mock
        defaults = build_parser().parse_args([])
        self.allow_execute = bool(self.declare_parameter("allow_execute", False).value)
        argv = []
        for name, flag in STRING_PARAMS.items():
            argv += [flag, str(self.declare_parameter(name, getattr(defaults, flag[2:].replace("-", "_"))).value)]
        rpy = self.declare_parameter("mount_rpy_deg", list(defaults.mount_rpy_deg)).value
        argv += ["--mount-rpy-deg", *map(str, rpy)]
        if self.declare_parameter("acquire_control", False).value:
            argv.append("--acquire-control")
        self.base_argv = argv
        build_parser().parse_args(argv)  # a bad parameter value fails here, at startup
        self.run = None  # connect(): camera, arm interface and kinematics, kept for the node's life
        self.pub = self.create_publisher(PoseStamped, "/button_press/target_pose", 1)
        if mock:
            # the setpoint topics RammpArm publishes (contact / impedance press, gripper close):
            # the mock declares the real node's interface, so it advertises them too
            for msg, topic in MOCK_SETPOINTS:
                self.create_publisher(msg, topic, qos_profile_sensor_data)
        self.busy = threading.Lock()
        group = ReentrantCallbackGroup()  # cancel requests must get through while a goal runs
        self.server = ActionServer(
            self,
            PressButton,
            "press_button",
            execute_callback=self.execute_mock if mock else self.execute,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=group,
        )
        mode = "MOCK: no hardware" if mock else f"allow_execute={self.allow_execute}"
        self.get_logger().info(
            f"/press_button up ({mode}; "
            f"{'the arm WILL move on execute goals' if self.allow_execute and not mock else 'nothing moves'})"
        )

    def connect(self):
        """Build the Run (arm, cameras, sim). Raises Abort, having cleaned up, if the driver is not up."""
        self.run = Run(build_parser().parse_args([*self.base_argv, *(["--execute"] if self.allow_execute else [])]))
        self.get_logger().info("connected to the arm driver")

    def on_goal(self, goal):
        if self.run is None and not self.mock:
            self.get_logger().warn("rejected: not connected to the arm driver yet")
            return GoalResponse.REJECT
        if self.busy.locked():
            self.get_logger().warn("rejected: a press is already running")
            return GoalResponse.REJECT
        if goal.target not in dome_pattern.NAMES:
            self.get_logger().warn(f"rejected: unknown target {goal.target!r} (known: {dome_pattern.NAMES})")
            return GoalResponse.REJECT
        if goal.presses > MAX_PRESSES or goal.press_in_m != 0.0:
            self.get_logger().warn(
                f"rejected: presses {goal.presses} (max {MAX_PRESSES}) / press_in_m "
                f"{goal.press_in_m} (must be 0: the contact press finds the button by touch)"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def on_cancel(self, _goal_handle):
        self.get_logger().warn("cancel requested: stopping the arm where it is")
        if self.run is not None:
            self.run.arm.cancel_motion()
        return CancelResponse.ACCEPT

    def execute_mock(self, gh):
        g = gh.request
        res = PressButton.Result()
        with self.busy:
            gh.publish_feedback(PressButton.Feedback(stage=f"mock: {g.target} -- nothing detected or moved"))
            res.success, res.moved, res.message = True, False, "mock: no hardware, nothing detected or moved"
            gh.succeed()
            return res

    def execute(self, gh):
        g = gh.request
        res = PressButton.Result()
        with self.busy:
            execute = bool(g.execute and self.allow_execute)
            argv = [*self.base_argv, "--target", g.target, "--presses", str(g.presses)]
            if execute:
                argv.append("--execute")
            if g.no_return:
                argv.append("--no-return")
            a = build_parser().parse_args(argv)
            run = self.run
            run.args, run.arm.execute = a, execute
            run.arm.cancel_requested = False
            run.start_log()
            res.log_file = run.log_path
            sent0 = run.arm.sent_goals
            note = "" if execute or not g.execute else "allow_execute is false: ran as a DRY RUN -- "
            self.get_logger().info(f"goal: {g.target} x{g.presses} {'EXECUTE' if execute else 'dry run'}")
            rammp_button_press.set_output(_FeedbackLines(gh, self.get_logger()))
            try:
                code = press_button(run, a, self.pub)
                res.success = code == 0
                res.message = note + ("done" if execute else "dry run passed every gate; nothing moved")
            except Abort as e:
                res.success, res.message = False, note + str(e)
                run.log({"stage": "abort", "why": str(e)})
            except Exception as e:  # noqa: BLE001 -- report, never kill the server mid-goal
                res.success, res.message = False, f"{type(e).__name__}: {e}"
                self.get_logger().error(res.message)
            finally:
                rammp_button_press.set_output(None)
            res.moved = run.arm.sent_goals > sent0
            if gh.is_cancel_requested:
                gh.canceled()
                res.message = "cancelled -- " + res.message
            elif res.success:
                gh.succeed()
            else:
                gh.abort()
            self.get_logger().info(f"result: success={res.success} moved={res.moved} {res.message}")
            return res

    def close(self):
        if self.run is not None:
            self.run.arm.cancel_motion()  # SIGTERM mid-goal: stop the arm where it is
            self.run.arm.close()


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    rclpy.init(args=argv)
    # A container's PID 1 gets no default action for SIGTERM: without this, `docker stop` (every
    # `sheppy woof`) waits out the grace period and SIGKILLs.
    signal.signal(signal.SIGTERM, lambda *_: rclpy.try_shutdown())
    node = PressServer(mock="--mock" in remove_ros_args(argv))
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    # The executor spins on its own thread; the main thread only waits, in short sleeps, so the
    # SIGTERM handler (Python runs it on the main thread) gets to run at once.
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        while rclpy.ok() and node.run is None and not node.mock:
            try:
                node.connect()
            except Abort as e:
                node.get_logger().warn(f"{e} -- retrying in {CONNECT_RETRY_S:.0f} s")
                time.sleep(CONNECT_RETRY_S)
        while rclpy.ok():
            time.sleep(0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
