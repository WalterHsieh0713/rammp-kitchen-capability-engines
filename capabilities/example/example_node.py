#!/usr/bin/env python3
"""Reference capability: publishes a 1 Hz status topic and exits on SIGTERM.

Copy this directory with `make new CAP=<name>` and replace this file with your
capability. Keep the two things it demonstrates: a status topic published from
the moment the node starts, and the SIGTERM handler.
"""

import signal
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


class Example(Node):
    def __init__(self, mock: bool) -> None:
        super().__init__("example")
        self.declare_parameter("publish_rate_hz", 1.0)
        rate = self.get_parameter("publish_rate_hz").value
        self.label = "mock" if mock else "real"
        self.publisher = self.create_publisher(String, "/example/status", 10)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"example capability up ({self.label})")

    def tick(self) -> None:
        # State goes out on a timer; the log does not, or it floods sheppy's view.
        message = String()
        message.data = f"ok ({self.label})"
        self.publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = Example(mock="--mock" in sys.argv)

    # A container's PID 1 receives no default action for SIGTERM, so without
    # this handler `docker stop` — and every `sheppy woof` — would wait out the
    # full grace period and then SIGKILL. `make smoke` fails without it.
    signal.signal(signal.SIGTERM, lambda *_: rclpy.try_shutdown())

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
