# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=example`). The image colcon-builds
# the same packages you build natively in ~/ros2_ws — capability_engine and
# this capability — and runs the same launch file.
#
# Pin an explicit base VERSION, never :humble or :latest — the tag records
# which rammp-interfaces-ros2 version this image speaks. Use rammp-cuda only
# if the node needs the GPU, and then drop torch/torchvision from your
# requirements (the base ships the Jetson wheels). arm64 only.
FROM ghcr.io/rammp-org/rammp-base:1.1.0-jp6

WORKDIR /module_ws

# TEMPORARY: rammp_orchestrator_interfaces is on rammp-interfaces-ros2 dev
# (1.2.0) but not yet in a rammp-base release, so build it from a pinned
# commit alongside the packages below. Delete these two lines when the FROM
# above moves to a base that carries 1.2.0.
ADD https://github.com/rammp-org/rammp-interfaces-ros2/archive/2bf67d4099842fd3da96bdc465c0ae8323d8df4b.tar.gz /tmp/interfaces.tar.gz
RUN mkdir -p src && tar -xzf /tmp/interfaces.tar.gz -C src --strip-components=1 --wildcards '*/rammp_orchestrator_interfaces' && rm /tmp/interfaces.tar.gz

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
