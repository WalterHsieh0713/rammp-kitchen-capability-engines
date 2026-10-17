# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=example`), so a capability can
# COPY the shared capability_engine package as well as its own directory.
#
# Pin an explicit base VERSION, never :humble or :latest — the tag records
# which rammp-interfaces-ros2 version this image speaks. Use rammp-cuda only
# if the node needs the GPU, and then drop torch/torchvision from your
# requirements (the base ships the Jetson wheels). arm64 only.
FROM ghcr.io/rammp-org/rammp-base:1.1.0-jp6

# TEMPORARY: rammp_orchestrator_interfaces is on rammp-interfaces-ros2 dev
# (1.2.0) but not yet in a rammp-base release. Built here from a pinned commit
# into the /module_ws overlay, which the base's entrypoint sources. Delete
# this block when the FROM above moves to a base that carries 1.2.0.
ADD https://github.com/rammp-org/rammp-interfaces-ros2/archive/2bf67d4099842fd3da96bdc465c0ae8323d8df4b.tar.gz /tmp/interfaces.tar.gz
WORKDIR /module_ws
RUN mkdir -p src \
    && tar -xzf /tmp/interfaces.tar.gz -C src --strip-components=1 \
        --wildcards '*/rammp_orchestrator_interfaces' \
    && . /opt/ros/humble/setup.sh \
    && colcon build \
    && rm -rf build log src /tmp/interfaces.tar.gz

WORKDIR /app
COPY capability_engine/ /app/capability_engine/
COPY capabilities/example/example_node.py /app/
ENV PYTHONPATH=/app
CMD ["python3", "/app/example_node.py"]
