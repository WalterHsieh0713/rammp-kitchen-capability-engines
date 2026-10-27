# Test image for self.arm: x86 ROS 2 Humble with the arm/common/curobo/orchestrator
# interfaces, the pure-Python deps, and the robot fixtures. It is NOT a production
# image (those are arm64, FROM rammp-base); it only has to speak the same messages.
# The repo is mounted at /repo at run time, so editing Python never needs a rebuild:
#   make test-arm [ARGS=...]
FROM ros:humble-ros-base

# Pins. Interfaces and curobo are full SHAs; kinova_gen3_description is dev.
ARG INTERFACES_REF=672a329f9036c640ce7b872ae02c77116780fb21
ARG CUROBO_REF=fd8b89cb18fd854f3add36ab9b1b72933dbabf76
ARG KINOVA_REF=2546edbab9cda031cdfe82b33289f05e01341f88
ARG DEPLOYMENTS_REF=9e0cd960fd7e1876a8a592b8fdb33a27da82a734
# feeding-deployment microwave-wip: the students' Cornell arm model.
ARG CORNELL_REF=7bc328aef5433b1617b61b30758d7d626b90af8b

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git python3-dev python3-pip python3-colcon-common-extensions \
        ros-humble-kortex-description ros-humble-robotiq-description ros-humble-xacro \
    && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3013
RUN pip install --no-cache-dir numpy scipy pybullet pytest

# Fetch commit $2 of GitHub repo $1 into dir $3 (full SHA required for a shallow fetch).
# hadolint ignore=SC2016
RUN printf '%s\n' '#!/bin/sh' 'set -e' \
        'git init -q "$3" && cd "$3"' \
        'git fetch -q --depth 1 "https://github.com/$1.git" "$2" && git checkout -q FETCH_HEAD' \
        > /usr/local/bin/fetch-commit \
    && chmod +x /usr/local/bin/fetch-commit

WORKDIR /ws
RUN mkdir src \
    && fetch-commit rammp-org/rammp-interfaces-ros2 ${INTERFACES_REF} /tmp/interfaces \
    && cp -r /tmp/interfaces/rammp_arm_interfaces /tmp/interfaces/rammp_common_interfaces \
        /tmp/interfaces/rammp_orchestrator_interfaces src/ \
    && fetch-commit rammp-org/RAMMP-CuRobo ${CUROBO_REF} /tmp/curobo \
    && cp -r /tmp/curobo/rammp_curobo_interfaces src/ \
    && fetch-commit rammp-org/kinova-gen3-ros2 ${KINOVA_REF} /tmp/kinova \
    && cp -r /tmp/kinova/kinova_gen3_description src/ \
    && rm -rf /tmp/interfaces /tmp/curobo /tmp/kinova

RUN . /opt/ros/humble/setup.sh && colcon build --event-handlers console_direct-

# Cornell arm model (URDF + meshes), straight from the students' repo.
RUN git clone -q --filter=blob:none --no-checkout https://github.com/dhyi-svg/feeding-deployment /tmp/fd \
    && git -C /tmp/fd sparse-checkout set src/feeding_deployment/assets/robot \
    && git -C /tmp/fd checkout -q ${CORNELL_REF} \
    && mkdir -p /fixtures \
    && cp -r /tmp/fd/src/feeding_deployment/assets/robot /fixtures/cornell \
    && rm -rf /tmp/fd

# One URDF per robot, rendered from its mount.yaml exactly as description.launch.py does.
RUN . /ws/install/setup.sh && fetch-commit rammp-org/rammp-deployments ${DEPLOYMENTS_REF} /tmp/dep \
    && mkdir -p /fixtures/urdf \
    && for r in dojo flatbot gen_1_5; do \
        python3 /ws/src/kinova_gen3_description/scripts/kinova_gen3_render.py \
            --mount /tmp/dep/december_2026/config/robots/$r/mount.yaml \
            --out /fixtures/urdf/$r.urdf; \
    done \
    && rm -rf /tmp/dep

# Sources come from the mounted repo, so there is nothing to rebuild when they change.
ENV PYTHONPATH=/repo/capability_engine
RUN printf '%s\n' '#!/bin/bash' \
        '. /opt/ros/humble/setup.bash' '. /ws/install/setup.bash' 'exec "$@"' \
        > /entrypoint.sh \
    && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
WORKDIR /repo
