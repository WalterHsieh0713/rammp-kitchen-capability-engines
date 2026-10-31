"""What the arm driver is doing and who controls it; pure Python, no rclpy.

ArmInterface feeds in /control_status and /stream_status and brackets its goals and
streams; check_command() says whether a command may go out now.
"""

import threading

STREAM_CLOSED_MESSAGE = (
    "the driver closed the stream: your loop stopped sending for more than timeout_s, "
    "or the driver could not solve that pose"
)
ESTOPPED_REASON = "arm is E-STOPPED -- clear it first"  # not_ready()'s reason for an e-stop


class ArmError(Exception):
    """Base class of every error self.arm raises."""


class NotInAction(ArmError):
    """An arm call was made outside an action, or a streaming call outside a stream(...) block."""


class ModeConflict(ArmError):
    """A goal or stream was requested while another is running."""


class ControlTaken(ArmError):
    """Another sender took control of the arm."""


class EStopped(ArmError):
    """The arm is e-stopped."""


class StreamClosed(ArmError):
    """The driver closed the stream this code had open."""


class NoToken(ArmError):
    """A command needed a control token and none was given."""


class StateTimeout(ArmError, RuntimeError):
    """The driver's state did not arrive in time (a RuntimeError, as RammpArm raised)."""


class DriverState:
    def __init__(self, *, standalone: bool = False):
        self._standalone = standalone  # True: RammpArm's wording when no token is held
        self._lock = threading.Lock()
        self._mode = "idle"  # "idle", "goal" or "stream"
        self._running = ""  # name of the goal or controller of the stream
        self._token = None
        self._status = None  # (estopped, arbitration_enabled, owned, owner_id)
        self._generation = None
        self._control_taken_by = None  # set when the generation changed under our token
        self._stream_closed = False
        self._stream_seen_open = False  # /stream_status showed this stream's session open

    @property
    def mode(self):
        with self._lock:
            return self._mode

    @property
    def token(self):
        with self._lock:
            return self._token

    def set_token(self, token):
        """Start an action with this token (or None); forgets any earlier takeover."""
        with self._lock:
            self._token = token
            self._control_taken_by = None

    def begin_goal(self, name):
        self._begin("goal", name)

    @property
    def stream_closed(self) -> bool:
        """The driver has closed the stream this code had open (its /stream_status said so)."""
        return self._stream_closed

    def begin_stream(self, controller):
        self._begin("stream", controller)

    def end_goal(self):
        self._end()

    def end_stream(self):
        self._end()

    def _begin(self, mode, name):
        with self._lock:
            if self._mode != "idle":
                raise ModeConflict(
                    f"cannot start {mode} {name!r}: {self._mode} {self._running!r} is running"
                )
            self._mode, self._running = mode, name
            self._stream_closed = self._stream_seen_open = False

    def _end(self):
        with self._lock:
            self._mode, self._running = "idle", ""
            self._stream_closed = self._stream_seen_open = False

    def on_control_status(self, estopped, arbitration_enabled, owned, owner_id, generation):
        with self._lock:
            self._status = (estopped, arbitration_enabled, owned, owner_id)
            changed = self._generation is not None and generation != self._generation
            self._generation = generation
            if changed and self._token is not None:
                self._token = None
                self._control_taken_by = owner_id

    def on_goal_halted(self):
        """A goal settled -9 HALTED: while holding the token, that means control was taken."""
        with self._lock:
            if self._token is not None:
                self._token = None
                self._control_taken_by = self._status[3] if self._status else ""

    def on_stream_status(self, open):  # noqa: A002 (the driver's field name)
        """A "closed" counts only once this stream's session has shown open: one arriving before
        that is a late status of the session before it."""
        with self._lock:
            if self._mode != "stream":
                return
            if open:
                self._stream_seen_open = True
            elif self._stream_seen_open:
                self._stream_closed = True

    def not_ready(self, sender_id):
        """Why the arm must not be commanded now, or None when it may be."""
        with self._lock:
            st, token = self._status, self._token
        if st is None:
            return "no /control_status (latched) -- arm driver not up?"
        estopped, enforced, owned, owner_id = st
        if estopped:
            return ESTOPPED_REASON
        # With a token, the driver judges it: the orchestrator acquires control under its own
        # owner_id, not this sender_id.
        if enforced and token is None:
            if owned and owner_id != sender_id:
                return f"the arm is owned by {owner_id!r}"
            if self._standalone:
                return (
                    "arbitration is ENFORCED and this code holds no token "
                    "(/acquire_control not implemented here)"
                )
            return "no control token: arbitration is enforced and none was given"
        return None

    def check_command(self):
        """Raise the ArmError that forbids sending a command now; return if it is fine."""
        with self._lock:
            if self._status is not None and self._status[0]:
                raise EStopped("e-stopped")
            if self._control_taken_by is not None:
                raise ControlTaken(f"control taken by {self._control_taken_by!r}")
            if self._stream_closed:
                raise StreamClosed(STREAM_CLOSED_MESSAGE)
