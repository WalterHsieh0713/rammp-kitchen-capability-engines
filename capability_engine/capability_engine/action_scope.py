"""The ROS entities of one action: made during it, destroyed safely when it ends.

Idle subscriptions cost CPU (rclpy's executor spends ~2 ms per message on the Orin: Demo-Software
#263), so whatever only an action needs lives only as long as the action. Destroying an entity
while rclpy's executor spins needs care (_Gate); ActionScope does it once, for self.arm and for
the capability's own subscriptions and publishers alike.
"""

import threading

import tf2_ros
from rclpy.action import ActionClient
from rclpy.callback_groups import CallbackGroup
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage

TF_QOS = QoSProfile(depth=100)
TF_STATIC_QOS = QoSProfile(
    depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
)
ENDED_MESSAGE = "this action has ended: per-action entities are made only while it runs"


class ScopeClosed(RuntimeError):
    """An entity was asked of a scope whose action has ended."""


class _Gate(CallbackGroup):
    """The engine's callback group, closable. rclpy (Humble) raises InvalidHandle out of the
    executor's spin() when an entity is destroyed after the executor found it ready but before it
    took the message, so close() lets no new callback start and waits for the running ones; the
    entities in the gate can then be destroyed while the engine's executor spins. close() waits
    for running callbacks with no timeout, so a callback on a gate must never block or end its
    action. (Demo-Software #263 needed the same order.)"""

    def __init__(self, group):
        super().__init__()
        self._group = group
        self._cond = threading.Condition()
        self._running = 0
        self._closed = False

    def can_execute(self, entity):
        return not self._closed and self._group.can_execute(entity)

    def beginning_execution(self, entity):
        with self._cond:
            if self._closed or not self._group.beginning_execution(entity):
                return False
            self._running += 1
            return True

    def ending_execution(self, entity):
        self._group.ending_execution(entity)
        with self._cond:
            self._running -= 1
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.wait_for(lambda: self._running == 0)


class ActionScope:
    """Entities on `node` for one action, their callbacks in a _Gate over `group`. close()
    destroys them all, in the safe order, and refuses any made after it (ScopeClosed)."""

    def __init__(self, node, group):
        self._node = node
        self._gate = _Gate(group)
        self._lock = threading.Lock()
        self._entities = []  # (destroy, entity), in creation order
        self._closed = False

    def _add(self, make, destroy):
        with self._lock:
            if self._closed:
                raise ScopeClosed(ENDED_MESSAGE)
            entity = make()
            self._entities.append((destroy, entity))
            return entity

    def create_subscription(self, msg_type, topic, callback, qos_profile):
        node = self._node
        return self._add(
            lambda: node.create_subscription(
                msg_type, topic, callback, qos_profile, callback_group=self._gate
            ),
            node.destroy_subscription,
        )

    def create_publisher(self, msg_type, topic, qos_profile):
        node = self._node
        return self._add(
            lambda: node.create_publisher(msg_type, topic, qos_profile), node.destroy_publisher
        )

    def create_client(self, srv_type, name):
        node = self._node
        return self._add(
            lambda: node.create_client(srv_type, name, callback_group=self._gate),
            node.destroy_client,
        )

    def create_action_client(self, action_type, name):
        return self._add(
            lambda: ActionClient(self._node, action_type, name, callback_group=self._gate),
            ActionClient.destroy,
        )

    def create_tf_buffer(self) -> tf2_ros.Buffer:
        """A tf2 Buffer fed by this scope's own /tf and /tf_static subscriptions
        (TransformListener's unregister() is not safe to call while the executor spins)."""
        buffer = tf2_ros.Buffer()

        def store(msg, static):
            for t in msg.transforms:
                if static:
                    buffer.set_transform_static(t, "action_scope")
                else:
                    buffer.set_transform(t, "action_scope")

        self.create_subscription(TFMessage, "/tf", lambda m: store(m, False), TF_QOS)
        self.create_subscription(TFMessage, "/tf_static", lambda m: store(m, True), TF_STATIC_QOS)
        return buffer

    def close(self):
        """Stop new callbacks, wait for running ones, then destroy every entity, newest first."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            entities, self._entities = self._entities, []
        self._gate.close()
        for destroy, entity in reversed(entities):
            destroy(entity)
