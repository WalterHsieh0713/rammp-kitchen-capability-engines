"""Fixtures for the ArmInterface tests: an arm on a fresh node beside a FakeDriver, spun together.

FakeCtx and counts() live in fake_driver.py: with tests/arm/conftest.py in the same session,
`from conftest import ...` would reach whichever conftest pytest loaded last."""

import threading

import pytest
import rclpy
from fake_driver import FakeDriver
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from capability_engine.arm import ArmInterface


@pytest.fixture(scope="session")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def arm_env(ros):
    """(arm, node, fake): an ArmInterface on a fresh engine-like node, and a FakeDriver."""
    fake = FakeDriver()
    node = Node("arm_test_engine")
    arm = ArmInterface(node, ReentrantCallbackGroup())
    executor = MultiThreadedExecutor()
    executor.add_node(fake)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    yield arm, node, fake
    arm.end_action()
    executor.shutdown()
    spinner.join()
    node.destroy_node()
    fake.destroy_node()
