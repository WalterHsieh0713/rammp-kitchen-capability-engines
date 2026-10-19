#!/usr/bin/env python3
"""Runtime smoke test for one capability, run against its mock alternative.

Runs the image the way sheppyd will (no --init, host networking, the mock's
command and its `params:` as a params file) and checks the three things every
other container relies on:

1. it stays up rather than exiting because a peer or device is absent
2. every topic its fragment declares in `publishes` actually appears
3. it exits on SIGTERM within the stop grace period

The mock is used because CI has no arm, camera or GPU. That is also why the
mock must speak the real alternative's interface (check_capabilities.py).

Usage: smoke.py <capability> <image>
"""

import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DISCOVERY_TIMEOUT_S = 30
GRACE_S = 5


def docker(*args, check=True):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check).stdout


def fail(name, message):
    print(f"FAIL: {message}")
    print(docker("logs", "--tail", "20", name, check=False))
    return 1


def main(capability, image):
    mock = yaml.safe_load(
        (ROOT / "capabilities" / capability / "rammp-alternative.mock.yaml").read_text()
    )
    command = mock["container"].get("command", [])
    if isinstance(command, str):
        command = command.split()
    expected = set(mock.get("publishes", []))
    # A random domain keeps parallel smoke runs on one host from seeing each other.
    domain = str(random.randint(1, 100))
    name = f"rammp-smoke-{capability}-{domain}"

    # sheppy's params delivery: a file keyed by ros_node_name (or /**), and
    # --ros-args --params-file appended to the command.
    mount = []
    if mock.get("params"):
        key = mock.get("ros_node_name") or "/**"
        params = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({key: {"ros__parameters": mock["params"]}}, params)
        params.close()
        mount = ["-v", f"{params.name}:/sheppy/params.yaml:ro"]
        command += ["--ros-args", "--params-file", "/sheppy/params.yaml"]

    docker(
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "--network",
        "host",
        "-e",
        f"ROS_DOMAIN_ID={domain}",
        *mount,
        image,
        *command,
    )
    try:
        deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
        seen = set()
        while time.monotonic() < deadline:
            if docker("inspect", "-f", "{{.State.Running}}", name, check=False).strip() != "true":
                return fail(
                    name, "container exited on its own; never exit because a peer is absent"
                )
            listed = docker(
                "exec",
                "-e",
                f"ROS_DOMAIN_ID={domain}",
                name,
                "/ros_entrypoint.sh",
                "ros2",
                "topic",
                "list",
                "--no-daemon",
                check=False,
            )
            seen = set(listed.split())
            if expected <= seen:
                break
            time.sleep(2)
        if not expected <= seen:
            missing = ", ".join(sorted(expected - seen))
            return fail(
                name, f"declared but never published within {DISCOVERY_TIMEOUT_S}s: {missing}"
            )

        start = time.monotonic()
        docker("stop", "-t", str(GRACE_S), name)
        elapsed = time.monotonic() - start
        if elapsed >= GRACE_S:
            print(f"FAIL: did not exit on SIGTERM within {GRACE_S}s ({elapsed:.1f}s)")
            print("      A container that ignores SIGTERM makes every 'sheppy woof' a stall.")
            return 1
    finally:
        docker("rm", "-f", name, check=False)

    print(
        f"PASS: {capability} published {len(expected)} declared topic(s) "
        f"and exited on SIGTERM in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__.strip().splitlines()[-1])
    sys.exit(main(sys.argv[1], sys.argv[2]))
