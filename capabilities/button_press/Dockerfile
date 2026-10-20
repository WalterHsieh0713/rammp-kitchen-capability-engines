# syntax=docker/dockerfile:1

# Built from the repo root (`make build CAP=button_press`). arm64 only, like its base.
#
# rammp-base, not rammp-cuda: nothing here uses the GPU (cuRobo runs in the planner's own image).
FROM ghcr.io/rammp-org/rammp-base:1.1.0-jp6

# PATH A: two ROS packages, colcon-built into /module_ws, which the base's entrypoint sources.
# rammp_button_press_interfaces is this capability's own PressButton action; the RAMMP
# interfaces come from the base's /ros2_ws and are never copied in.
WORKDIR /module_ws
COPY capabilities/button_press/rammp_button_press_interfaces src/rammp_button_press_interfaces
COPY capabilities/button_press/rammp_button_press src/rammp_button_press

# rosdep installs package.xml's apt dependencies (cv_bridge, vision_msgs, numpy, scipy, opencv,
# pytest); the base ships rosdep's sources but not its cache. pybullet has no rosdep key, so
# pip, pinned; it builds from source on arm64.
RUN . /opt/ros/humble/setup.sh \
    && . /ros2_ws/install/setup.sh \
    && rosdep update --rosdistro humble \
    && apt-get update \
    && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --no-cache-dir pybullet==3.2.6 \
    && colcon build \
    && rm -rf /module_ws/build /module_ws/log

# The node's executable, not `ros2 run`: that would be PID 1 and swallow SIGTERM.
CMD ["/module_ws/install/rammp_button_press/lib/rammp_button_press/press_server"]
