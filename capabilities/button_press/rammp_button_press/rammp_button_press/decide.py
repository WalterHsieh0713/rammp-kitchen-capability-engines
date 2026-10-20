"""Where an interactive run's decisions come from (press_button --interactive).

The run asks at each stop (the staging pose, the button) and gets back GO or BACK. For now the
operator answers from the terminal; later the wheelchair user's own input replaces TerminalKeys,
anything callable as ``decide(stop, prompt, allowed) -> GO | BACK``.

    space   GO    go on
    q       BACK  go back to the start
"""

from __future__ import annotations

import os
import sys
import termios
import tty

from rammp_button_press import Abort, say

GO = "go"
BACK = "back"
KEYS = {" ": GO, "q": BACK, "Q": BACK}


class TerminalKeys:
    """Single keypresses from the terminal (no Enter). Keys typed while the arm moved are thrown
    away before each prompt, so they never answer the next stop. Ctrl-C still interrupts."""

    def __init__(self, stdin=None, read_key=None, flush=None):
        self.stdin = sys.stdin if stdin is None else stdin
        self._read_key = read_key or self._tty_key
        self._flush = flush or self._tty_flush

    def check(self):
        """Abort unless stdin is a terminal (sheppy's TUI runs have none): before anything moves."""
        if not self.stdin.isatty():
            raise Abort(
                "--interactive needs a terminal to read the keys from (stdin is not one) -- "
                "run press_button by hand in a terminal"
            )

    def __call__(self, stop: str, prompt: str, allowed: tuple[str, ...] = (GO, BACK)) -> str:
        self._flush()
        say(f"\n  [{stop}] {prompt}", flush=True)
        while True:
            choice = KEYS.get(self._read_key())
            if choice in allowed:
                say(f"  [{stop}] {'space' if choice == GO else 'q'}", flush=True)
                return choice

    def _tty_key(self) -> str:
        fd = self.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # one key at a time, no echo; Ctrl-C still raises KeyboardInterrupt
            return os.read(fd, 1).decode(errors="ignore")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _tty_flush(self):
        termios.tcflush(self.stdin.fileno(), termios.TCIFLUSH)
