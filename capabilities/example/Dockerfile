# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=example`), so a capability can
# COPY shared code from elsewhere in the repo as well as its own directory.
#
# Pin an explicit base VERSION, never :humble or :latest — the tag records
# which rammp-interfaces-ros2 version this image speaks. Use rammp-cuda only
# if the node needs the GPU, and then drop torch/torchvision from your
# requirements (the base ships the Jetson wheels). arm64 only.
FROM ghcr.io/rammp-org/rammp-base:1.1.0-jp6

# PATH B — wrap plain Python. For ROS packages, colcon-build into /module_ws
# instead (RAMMP-docker/templates/module.Dockerfile); the base's entrypoint
# sources /module_ws/install automatically.
WORKDIR /app
COPY capabilities/example/example_node.py /app/
CMD ["python3", "/app/example_node.py"]
