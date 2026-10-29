# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=example`). The image colcon-builds
# the same packages you build natively in ~/ros2_ws — capability_engine and
# this capability — and runs the same launch file.
#
# Pin an explicit base tag (a release or a dated nightly), never :humble or
# :latest — the tag records which rammp-interfaces-ros2 version this image
# speaks. Use rammp-cuda only if the node needs the GPU, and then drop
# torch/torchvision from your requirements (the base ships the Jetson wheels).
# arm64 only.
#
# nightly-20261027 is the base the nightly arm driver is built on (interfaces
# 672a329f9036), so self.arm speaks the driver's exact messages. It carries the
# arm, common and orchestrator interfaces; rammp_curobo_interfaces it does not.
FROM ghcr.io/rammp-org/rammp-base:nightly-20261027

WORKDIR /module_ws

# self.arm plans and checks poses on a PyBullet model of the arm. PyPI ships no
# arm64 wheel, so this compiles from source (the base has gcc, g++ and Python.h).
# 3.2.7 is the version the SheppyArm goldens were recorded with. numpy is in the
# base; scipy comes from rosdep (capability_engine's package.xml). Its own layer,
# before the COPYs, so editing code never recompiles it.
RUN pip install --no-cache-dir pybullet==3.2.7

# rammp_curobo_interfaces (the cuRobo planner's messages) is not in the base:
# build it from a pinned RAMMP-CuRobo commit alongside the packages below.
ADD https://github.com/rammp-org/RAMMP-CuRobo/archive/525b7527df7ac75b5446f0a9367e651a0ecdb303.tar.gz /tmp/curobo.tar.gz
RUN mkdir -p src && tar -xzf /tmp/curobo.tar.gz -C src --strip-components=1 --wildcards '*/rammp_curobo_interfaces' && rm /tmp/curobo.tar.gz

COPY capability_engine/ src/capability_engine/
COPY capabilities/example/ src/example/

# /module_ws is an overlay on the base's /ros2_ws (the RAMMP interfaces); the
# base's entrypoint sources its install/ automatically.
RUN . /opt/ros/humble/setup.sh \
    && . /ros2_ws/install/setup.sh \
    && apt-get update \
    && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/* \
    && colcon build \
    && rm -rf build log

# `ros2 launch` on Humble has no SIGTERM handler. As PID 1 it ignores the
# SIGTERM `docker stop` (and `sheppy woof`) sends and is SIGKILLed after the
# grace period, leaving the node no chance to stop cleanly. On SIGINT it shuts
# its nodes down properly, so make that the stop signal.
STOPSIGNAL SIGINT
CMD ["ros2", "launch", "example", "example.launch.py"]
