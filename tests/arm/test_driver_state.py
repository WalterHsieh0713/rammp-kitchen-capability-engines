"""DriverState: one row of the spec's section 3 table per test."""

import pytest

from capability_engine.driver_state import (
    ControlTaken,
    DriverState,
    EStopped,
    ModeConflict,
    StreamClosed,
)

TOKEN = b"t" * 16


def status(ds, *, estopped=False, enforced=False, owned=False, owner="", generation=1):
    ds.on_control_status(estopped, enforced, owned, owner, generation)


def test_goal_while_stream_open_refused():
    ds = DriverState()
    ds.begin_stream("ee_twist")
    with pytest.raises(ModeConflict, match="ee_twist"):
        ds.begin_goal("x")


def test_stream_while_goal_refused():
    ds = DriverState()
    ds.begin_goal("move_joints")
    with pytest.raises(ModeConflict, match="move_joints"):
        ds.begin_stream("ee_twist")


def test_nested_stream_refused():
    ds = DriverState()
    ds.begin_stream("ee_twist")
    with pytest.raises(ModeConflict, match="ee_twist"):
        ds.begin_stream("ee_twist")


def test_mode_follows_begin_and_end():
    ds = DriverState()
    assert ds.mode == "idle"
    ds.begin_goal("x")
    assert ds.mode == "goal"
    ds.end_goal()
    ds.begin_stream("ee_twist")
    assert ds.mode == "stream"
    ds.end_stream()
    assert ds.mode == "idle"


def test_stream_closed_unexpectedly():
    ds = DriverState()
    ds.begin_stream("ee_twist")
    ds.on_stream_status(False)
    with pytest.raises(StreamClosed) as e:
        ds.check_command()
    assert "stopped sending for more than" in str(e.value)
    assert "could not solve" in str(e.value)


def test_stream_status_closed_without_stream_is_ignored():
    ds = DriverState()
    ds.on_stream_status(False)
    ds.check_command()


def test_generation_change_is_control_taken():
    ds = DriverState()
    ds.set_token(TOKEN)
    status(ds, generation=1)
    ds.check_command()
    status(ds, generation=2, owner="teleop")
    with pytest.raises(ControlTaken, match="teleop"):
        ds.check_command()
    assert ds.token is None


def test_generation_change_without_token_is_not_control_taken():
    ds = DriverState()
    status(ds, generation=1)
    status(ds, generation=2, owner="teleop")
    ds.check_command()


def test_estop_refuses():
    ds = DriverState()
    status(ds, estopped=True)
    with pytest.raises(EStopped):
        ds.check_command()


def test_not_ready_matches_sheppy():
    ds = DriverState(standalone=True)
    assert ds.not_ready("me") == "no /control_status (latched) -- arm driver not up?"
    status(ds, estopped=True)
    assert ds.not_ready("me") == "arm is E-STOPPED -- clear it first"
    status(ds, enforced=True, owned=True, owner="x")
    assert ds.not_ready("me") == "the arm is owned by 'x'"
    status(ds)
    assert ds.not_ready("me") is None


def test_not_ready_enforced_without_token_standalone():
    ds = DriverState(standalone=True)
    status(ds, enforced=True)
    assert ds.not_ready("me") == (
        "arbitration is ENFORCED and this code holds no token "
        "(/acquire_control not implemented here)"
    )


def test_not_ready_enforced_without_token_under_engine():
    ds = DriverState()
    status(ds, enforced=True)
    assert ds.not_ready("me") == "no control token: arbitration is enforced and none was given"


def test_not_ready_enforced_with_token():
    ds = DriverState()
    ds.set_token(TOKEN)
    status(ds, enforced=True, owned=True, owner="me")
    assert ds.not_ready("me") is None
    status(ds, enforced=True, owned=True, owner="x")
    assert ds.not_ready("me") == "the arm is owned by 'x'"


def test_set_token_clears_control_taken():
    ds = DriverState()
    ds.set_token(TOKEN)
    status(ds, generation=1)
    status(ds, generation=2, owner="teleop")
    ds.set_token(b"n" * 16)
    ds.check_command()


def test_stream_closed_does_not_outlive_stream():
    ds = DriverState()
    ds.begin_stream("ee_twist")
    ds.on_stream_status(False)
    ds.end_stream()
    ds.check_command()


def test_goal_halted_with_token_is_control_taken():
    ds = DriverState()
    ds.set_token(TOKEN)
    status(ds, owner="teleop")
    ds.on_goal_halted()
    with pytest.raises(ControlTaken, match="teleop"):
        ds.check_command()
    assert ds.token is None


def test_goal_halted_without_token_does_nothing():
    ds = DriverState()
    status(ds, owner="teleop")
    ds.on_goal_halted()
    ds.check_command()
