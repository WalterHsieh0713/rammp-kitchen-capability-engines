"""Send one goal to /press_button and print its progress (test client for press_server).

    ros2 run rammp_button_press press_client --target start_30s              # dry run
    ros2 run rammp_button_press press_client --target start_30s --execute    # moves only if the server
                                                                              # runs with allow_execute:=true
Ctrl-C cancels the goal: the arm stops and holds where it is.
"""

import argparse
import sys

import rclpy
from rclpy.action import ActionClient

from rammp_button_press import say
from rammp_button_press_interfaces.action import PressButton


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="start_30s")
    ap.add_argument("--presses", type=int, default=1)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--no-return", action="store_true")
    a, _ = ap.parse_known_args(argv)
    rclpy.init()
    node = rclpy.create_node("press_client")
    cl = ActionClient(node, PressButton, "press_button")
    try:
        if not cl.wait_for_server(timeout_sec=10.0):
            say("no /press_button server -- is the sheppy `button_press` node up?")
            return 2
        g = PressButton.Goal(target=a.target, presses=a.presses, execute=a.execute, no_return=a.no_return)
        fut = cl.send_goal_async(g, feedback_callback=lambda fb: say(fb.feedback.stage, flush=True))
        rclpy.spin_until_future_complete(node, fut)
        gh = fut.result()
        if not gh.accepted:
            say("goal REJECTED (busy, unknown target or out-of-range presses; see the server log)")
            return 2
        rf = gh.get_result_async()
        try:
            rclpy.spin_until_future_complete(node, rf)
        except KeyboardInterrupt:
            say("\ncancelling -- the arm stops where it is")
            cf = gh.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cf, timeout_sec=3.0)
            rclpy.spin_until_future_complete(node, rf, timeout_sec=30.0)
        r = rf.result().result
        say(f"\nresult: success={r.success} moved={r.moved}\n  {r.message}\n  log: {r.log_file}")
        return 0 if r.success else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
