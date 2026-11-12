"""One rosbag per action run, with a run.json beside it: started, then succeeded / failed / cancelled
with the error and its traceback. Adapted from microwave_place's RunBag (PR #40, which follows the
fridge's recording pattern), so every capability's runs look the same.

Recorded: /joint_states, /ee_state, /gripper_state (state_hz), /tf, /tf_static, /arm/robot_description,
/diagnostics, /rosout (all of this node's log; other nodes' warnings and errors), the run's own
events (each phase, with its time: EVENT_TOPIC), and optionally the scene camera's colour, its
camera_info and depth together at frame_hz. Only action-owned subscriptions (the engine destroys
them when the action ends); no arm commands, no subprocess. A recording that can't start refuses
the action before anything moves.
"""

import json
import math
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import message_filters
import rosbag2_py
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from rammp_arm_interfaces.msg import EeState, GripperState
from rcl_interfaces.msg import Log
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

LOG_WARN = 30
LOG_ERROR = 40
SYNC_SLOP_S = 0.05
EVENT_TOPIC = "/place_container/run_event"  # bag-only JSON events, not a live publisher
STATIC_QOS = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
LOG_QOS = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)


def _level(value):
    # Humble may expose uint8 constants as bytes even though Log.level is an int.
    return value[0] if isinstance(value, bytes) else int(value)


def jsonable(value):
    """`value` as plain JSON types (numpy numbers and arrays, Paths: as their str)."""
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))


class RecordingError(Exception):
    """Recording could not start: refuse the action before any motion."""


class _Input(message_filters.SimpleFilter):
    def __init__(self, node, msg_type, topic):
        super().__init__()
        node.create_action_subscription(msg_type, topic, self.signalMessage, qos_profile_sensor_data)


def _qos_yaml(qos):
    def duration(value):
        sec, nsec = divmod(value.nanoseconds, 1_000_000_000)
        return dict(sec=sec, nsec=nsec)

    return yaml.safe_dump(
        [
            dict(
                history=int(qos.history),
                depth=qos.depth,
                reliability=int(qos.reliability),
                durability=int(qos.durability),
                deadline=duration(qos.deadline),
                lifespan=duration(qos.lifespan),
                liveliness=int(qos.liveliness),
                liveliness_lease_duration=duration(qos.liveliness_lease_duration),
                avoid_ros_namespace_conventions=qos.avoid_ros_namespace_conventions,
            )
        ]
    )


class RunBag:
    """One action's bag (<state_dir>/runs/<action>_<UTC time>_<id>/bag) and run.json; closes on
    success, failure, cancellation or exception (a context manager). `camera`: (colour,
    colour camera_info, depth) topics, or () for none."""

    def __init__(self, node, state_dir, action, params, config, camera=(), frame_hz=1.0, state_hz=25.0):
        for name, value in (("bag_frame_hz", frame_hz), ("bag_state_hz", state_hz)):
            if not math.isfinite(value) or value <= 0:
                raise RecordingError(f"{name} must be finite and positive")
        if camera and len(camera) != 3:
            raise RecordingError(f"bag_camera_topics: colour, camera_info, depth -- got {list(camera)}")
        self.node, self.camera = node, tuple(camera)
        self.frame_period, self.state_period = 1 / frame_hz, 1 / state_hz
        self._lock, self._last, self.counts = threading.Lock(), {}, {}
        self._writer, self._closed, self.result = None, False, None
        self.errors = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.run_dir = Path(state_dir) / "runs" / f"{action}_{stamp}_{uuid.uuid4().hex[:8]}"
        self.path = self.run_dir / "bag"
        self.metadata = dict(
            action=action,
            params=jsonable(params),
            config=jsonable(config),
            camera=list(self.camera),
            started_ns=self._now(),
            frame_hz=frame_hz,
            state_hz=state_hz,
        )
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            self._writer = rosbag2_py.SequentialWriter()
            self._writer.open(
                rosbag2_py.StorageOptions(uri=str(self.path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"),
            )
            self._subscribe()
            self.event("started", self.metadata)
            if self.errors:
                raise RecordingError("; ".join(self.errors))
            self._save_metadata(dict(self.metadata, status="running"))
        except Exception as exc:
            self.close(exc)
            raise RecordingError(f"cannot start the run bag at {self.path}: {exc}") from exc

    def _now(self):
        return self.node.get_clock().now().nanoseconds

    def _topic(self, topic, msg_type, qos):
        self._writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic, type=msg_type, serialization_format="cdr", offered_qos_profiles=_qos_yaml(qos)
            )
        )
        self.counts[topic] = 0

    def _subscribe(self):
        if self.camera:
            for topic, name in zip(self.camera, ("Image", "CameraInfo", "Image"), strict=True):
                self._topic(topic, f"sensor_msgs/msg/{name}", qos_profile_sensor_data)
            inputs = [
                _Input(self.node, cls, topic)
                for cls, topic in zip((Image, CameraInfo, Image), self.camera, strict=True)
            ]
            self._sync = message_filters.ApproximateTimeSynchronizer(inputs, 10, SYNC_SLOP_S)
            self._sync.registerCallback(self._on_frame)
        sp = self.state_period
        streams = [
            ("/joint_states", "sensor_msgs/msg/JointState", JointState, qos_profile_sensor_data, sp),
            ("/ee_state", "rammp_arm_interfaces/msg/EeState", EeState, qos_profile_sensor_data, sp),
            ("/gripper_state", "rammp_arm_interfaces/msg/GripperState", GripperState, qos_profile_sensor_data, sp),
            ("/tf", "tf2_msgs/msg/TFMessage", TFMessage, QoSProfile(depth=100), 0.1),
            ("/tf_static", "tf2_msgs/msg/TFMessage", TFMessage, STATIC_QOS, 0),
            ("/arm/robot_description", "std_msgs/msg/String", String, STATIC_QOS, 0),
            ("/diagnostics", "diagnostic_msgs/msg/DiagnosticArray", DiagnosticArray, LOG_QOS, 1.0),
            ("/rosout", "rcl_interfaces/msg/Log", Log, LOG_QOS, 0),
        ]
        for topic, type_name, cls, qos, period in streams:
            self._topic(topic, type_name, qos)
            self.node.create_action_subscription(
                cls, topic, lambda msg, topic=topic, period=period: self._on_message(topic, msg, period), qos
            )
        self._topic(EVENT_TOPIC, "std_msgs/msg/String", QoSProfile(depth=1))

    def _due(self, key, period):
        now = time.monotonic()
        if period and key in self._last and now - self._last[key] < period:
            return False
        self._last[key] = now
        return True

    def _write(self, topic, message, stamp):
        # Caller holds the lock. A late callback after close must never touch the writer.
        if self._closed or self._writer is None or self.errors:
            return
        try:
            self._writer.write(topic, serialize_message(message), stamp)
            self.counts[topic] += 1
        except Exception as exc:
            self.errors.append(f"write {topic}: {exc}")

    def _on_frame(self, color, info, depth):
        with self._lock:
            if self._closed or not self._due("camera", self.frame_period):
                return
            stamp = self._now()
            for topic, msg in zip(self.camera, (color, info, depth), strict=True):
                self._write(topic, msg, stamp)

    def _on_message(self, topic, msg, period):
        # This node's whole log, and every node's warnings and errors.
        if topic == "/rosout" and _level(msg.level) < LOG_WARN and msg.name != self.node.get_logger().name:
            return
        key = topic
        if topic == "/tf":
            key = (topic, tuple(sorted((t.header.frame_id, t.child_frame_id) for t in msg.transforms)))
        if topic == "/diagnostics" and any(_level(s.level) >= 2 for s in msg.status):
            period = 0  # never thin away an error diagnostic
        with self._lock:
            if self._closed or not self._due(key, period):
                return
            self._write(topic, msg, self._now())

    def event(self, status, detail):
        """A JSON event in the bag (EVENT_TOPIC): "started", "phase" {phase, pct, message}, ..."""
        with self._lock:
            self._write(EVENT_TOPIC, String(data=json.dumps(dict(status=status, detail=jsonable(detail)))), self._now())

    def _save_metadata(self, metadata):
        temporary = self.run_dir / "run.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(self.run_dir / "run.json")

    def close(self, error=None):
        with self._lock:
            if self._closed:
                return
            cancelled = type(error).__name__ == "Cancelled"
            status = "succeeded" if error is None else ("cancelled" if cancelled else "failed")
            outcome = dict(status=status, result=self.result, finished_ns=self._now())
            if error is not None:
                outcome.update(
                    error_type=type(error).__name__,
                    error=str(error),
                    traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)),
                )
                # Its own copy: the engine may log the failure only after execute() returns.
                self._write(
                    "/rosout",
                    Log(
                        stamp=self.node.get_clock().now().to_msg(),
                        level=LOG_ERROR,
                        name=self.node.get_logger().name,
                        msg=str(error),
                    ),
                    self._now(),
                )
            self._write(EVENT_TOPIC, String(data=json.dumps(jsonable(outcome))), self._now())
            self._closed = True
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception as exc:
                    self.errors.append(f"close: {exc}")
                finally:
                    self._writer = None
            metadata = dict(self.metadata, **outcome, counts=dict(self.counts), recording_errors=list(self.errors))
            try:
                self._save_metadata(jsonable(metadata))
            except Exception as exc:
                self.errors.append(f"run.json: {exc}")
        if self.errors:
            self.node.get_logger().error(f"run recording incomplete at {self.path}: {'; '.join(self.errors)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(exc)
        return False
