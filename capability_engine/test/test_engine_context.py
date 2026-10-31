import threading
import time
from types import SimpleNamespace

import pytest

from capability_engine.engine import Cancelled, Context, _Run

CANCEL_AFTER_S = 0.2  # when the action's goal is cancelled while paused
LIMIT_S = 1.0  # checkpoint() must raise well before the pause would end


def test_goal_cancel_during_pause_raises_without_resume():
    run = _Run("count", SimpleNamespace())
    goal = SimpleNamespace(is_cancel_requested=False)
    ctx = Context(SimpleNamespace(), goal, run)
    run.paused.set()
    threading.Timer(CANCEL_AFTER_S, setattr, args=(goal, "is_cancel_requested", True)).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        ctx.checkpoint()
    assert time.monotonic() - t0 < LIMIT_S
    assert run.paused.is_set()  # never resumed
    assert ctx.cancelled()
