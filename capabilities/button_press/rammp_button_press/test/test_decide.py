"""The interactive run's decision source: space goes on, q goes back, nothing else counts, and a key
typed while the arm was moving never answers the next stop. No terminal needed: keys are fed in."""

import io

import pytest

from rammp_button_press import Abort
from rammp_button_press.decide import BACK, GO, TerminalKeys


def keys(*typed):
    """TerminalKeys reading `typed` one key at a time, recording every read and flush in order."""
    seen = []
    it = iter(typed)

    def read_key():
        k = next(it)
        seen.append(("read", k))
        return k

    tk = TerminalKeys(stdin=io.StringIO(), read_key=read_key, flush=lambda: seen.append(("flush", None)))
    return tk, seen


def test_space_goes_on_and_q_goes_back():
    assert keys(" ")[0]("staging", "go?") == GO
    assert keys("q")[0]("staging", "go?") == BACK
    assert keys("Q")[0]("button", "go?") == BACK


def test_other_keys_are_ignored():
    tk, seen = keys("x", "\n", " ")
    assert tk("staging", "go?") == GO
    assert [k for what, k in seen if what == "read"] == ["x", "\n", " "]


def test_a_key_not_allowed_is_ignored():
    tk, _ = keys(" ", "q")
    assert tk("failed", "q only", allowed=(BACK,)) == BACK


def test_stdin_is_flushed_before_each_prompt():
    tk, seen = keys(" ", "q")
    tk("staging", "go?")
    tk("button", "go?")
    assert seen == [("flush", None), ("read", " "), ("flush", None), ("read", "q")]


def test_the_prompt_names_the_stop(capsys):
    keys(" ")[0]("staging", "space: go on   q: go back")
    out = capsys.readouterr().out
    assert "staging" in out and "space: go on   q: go back" in out


def test_not_a_terminal_is_refused():
    with pytest.raises(Abort, match="terminal"):
        TerminalKeys(stdin=io.StringIO()).check()
