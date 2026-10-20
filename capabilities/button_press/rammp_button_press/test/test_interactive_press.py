"""press_button --interactive: the two stops (staging, the button), space / q at each, the re-detect
and plan running while the staging stop waits, a failure at a stop waiting for q, and the one-shot run
left as it was. Runs press_button() against a stand-in run whose steps record what they would do (no
ROS, nothing moves)."""

import argparse
import time

import numpy as np
import pytest

pb = pytest.importorskip("rammp_button_press.press_button")  # needs rclpy + rammp interfaces
from rammp_button_press import Abort  # noqa: E402
from rammp_button_press.decide import BACK, GO  # noqa: E402

START = np.array([0.40, 0.00, 0.40])
STAGING = np.array([0.60, -0.30, 0.15])
SPOT = np.array([0.68, -0.31, 0.12])
N_FAR = np.array([-1.0, 0.0, 0.0])
N_CLOSE = np.array([-0.99, 0.10, 0.0]) / np.linalg.norm([-0.99, 0.10, 0.0])


def frame(n):
    """A panel frame whose z (out of the panel) is n."""
    z = np.asarray(n, float)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class FakeArm:
    def __init__(self):
        self.p, self.R = START.copy(), np.eye(3)

    def ee_pose(self):
        return self.p.copy(), self.R


class Clock:
    def now(self):
        return argparse.Namespace(nanoseconds=123)


class FakeRun:
    def __init__(self, execute=True, presses=1, no_return=False):
        self.args = argparse.Namespace(execute=execute, presses=presses, no_return=no_return)
        self.arm = FakeArm()
        self.per = argparse.Namespace(cam_position_in_base=lambda: START, get_clock=Clock)
        self.calls, self.records = [], []

    def preflight(self):
        self.calls.append("preflight")

    def log(self, rec):
        self.records.append(rec)

    def reanchor_seed(self, why):
        pass

    def move(self, path, name, **kw):
        self.calls.append(name)
        self.arm.p = SPOT.copy()

    def pose_path(self, p_from, R_from, p_to, R_to, **kw):
        return [p_from, p_to]

    def go_back(self, p0, R0, why, n_out=None):
        self.calls.append("go_back")

    def press(self, p_spot, R_spot, n_out, count=None, first=1):
        count = self.args.presses if count is None else count
        for i in range(first, first + count):
            if ("press", i) in self.fail:
                raise Abort(f"press {i}: touched nothing -- HOLDING HERE")
            self.calls.append(("press", i))

    fail = ()


def answers(*said):
    """A decide() that gives these answers in order, recording each question."""
    asked = []
    it = iter(said)

    def decide(stop, prompt, allowed=(GO, BACK)):
        asked.append((stop, allowed))
        choice = next(it)
        assert choice in allowed
        return choice

    decide.asked = asked
    return decide


@pytest.fixture
def steps(monkeypatch):
    """Stand-ins for press_button's steps; `fail` names the ones that raise Abort ("measure once": only
    the first re-detect), "slow" makes the re-detect take 0.2 s."""
    fail = set()

    def first_look(run, a):
        return STAGING + 0.1 * N_FAR, frame(N_FAR), "scene"

    def go_to_staging(run, p_st, R_st, n_out, lead):
        run.calls.append("to staging")
        run.arm.p = STAGING.copy()

    def measure(run, target, after=None):
        if "slow" in fail:
            time.sleep(0.2)
        run.calls.append("measure")
        if "measure" in fail or ("measure once" in fail and run.calls.count("measure") == 1):
            raise Abort("dome detector: nothing in 7 frames -- is the panel in view?")
        return SPOT + 0.05 * N_CLOSE, frame(N_CLOSE)

    def plan(run, a, pub, p0, origin, R, presses, back):
        run.calls.append(("plan", presses, back))
        if "plan" in fail:
            raise Abort("press 1 in: IK error 12.0 mm -- nothing moved")
        return "go-path", "in", "out", SPOT, np.eye(3)

    def back_to_start(run, p0, R0, n_out):
        run.calls.append(("back_to_start", tuple(np.round(n_out, 3))))
        if "back" in fail:
            raise Abort("back to start: cuRobo found no plan -- HOLDING HERE")
        run.arm.p = np.asarray(p0, float).copy()

    monkeypatch.setattr(pb, "first_look", first_look)
    monkeypatch.setattr(pb, "check_fingertip", lambda *a, **k: None)
    monkeypatch.setattr(pb, "pick_lead_in", lambda run, p_st, R_st, n, wanted: (0.1, p_st, None))
    monkeypatch.setattr(pb, "go_to_staging", go_to_staging)
    monkeypatch.setattr(pb, "measure_panel_frame_domes", measure)
    monkeypatch.setattr(pb, "plan_press", plan)
    monkeypatch.setattr(pb, "back_to_start", back_to_start)
    return fail


def args(**kw):
    return argparse.Namespace(**dict(dict(target="start_30s", stage_out=None, lead_in=0.1, execute=True), **kw))


def run_it(run, decide, **kw):
    return pb.press_button(
        run,
        args(execute=run.args.execute, presses=run.args.presses, no_return=run.args.no_return, **kw),
        pub=None,
        decide=decide,
    )


BACK_FAR = ("back_to_start", tuple(np.round(N_FAR, 3)))
BACK_CLOSE = ("back_to_start", tuple(np.round(N_CLOSE, 3)))


def answers_at_staging(run, choice, wait_for=None, settle_s=0.0):
    """A decide() that, at the staging stop, first waits (up to 2 s) for `wait_for` to show up in
    run.calls and settle_s more, notes run.calls as they are then (decide.seen), then answers
    `choice`; at the button it answers q."""

    def decide(stop, prompt, allowed=(GO, BACK)):
        if stop == "staging":
            t0 = time.monotonic()
            while wait_for is not None and wait_for not in run.calls and time.monotonic() - t0 < 2.0:
                time.sleep(0.01)
            time.sleep(settle_s)
            decide.seen = list(run.calls)
            return choice
        return BACK

    return decide


def test_the_redetect_and_plan_run_while_the_staging_stop_waits(steps):
    run = FakeRun()
    decide = answers_at_staging(run, GO, wait_for=("plan", 1, True))
    assert run_it(run, decide) == 0
    assert decide.seen == ["preflight", "to staging", "measure", ("plan", 1, True)]  # before space; nothing moved
    assert run.calls[len(decide.seen) :] == ["to pre-press", BACK_CLOSE]  # space: straight to the button


def test_q_at_staging_waits_for_the_look_ahead_then_goes_back(steps):
    steps.add("slow")  # still detecting when q comes: the way back must wait for it (the sim)
    run, decide = FakeRun(), answers(BACK)
    assert run_it(run, decide) == 0
    assert run.calls == [
        "preflight",
        "to staging",
        "measure",
        ("plan", 1, True),
        BACK_FAR,
    ]  # straight out along the far normal
    assert decide.asked == [("staging", (GO, BACK))]


def test_space_before_the_look_ahead_is_done_waits_for_it(steps):
    steps.add("slow")
    run, decide = FakeRun(), answers(GO, BACK)
    assert run_it(run, decide) == 0
    assert run.calls == ["preflight", "to staging", "measure", ("plan", 1, True), "to pre-press", BACK_CLOSE]
    assert [s for s, _ in decide.asked] == ["staging", "button"]


def test_a_failed_look_ahead_backs_out_and_retries_only_after_space(steps):
    steps.add("measure once")
    run = FakeRun()
    decide = answers_at_staging(run, GO, wait_for="measure", settle_s=0.2)
    assert run_it(run, decide) == 0
    assert decide.seen == ["preflight", "to staging", "measure"]  # no back-out before space
    assert run.calls[len(decide.seen) :] == [
        "back out to re-detect",
        "measure",
        ("plan", 1, True),
        "to pre-press",
        BACK_CLOSE,
    ]


def test_a_plan_refused_in_the_look_ahead_holds_after_space_then_goes_back(steps):
    steps.add("plan")
    run, decide = FakeRun(), answers(GO, BACK)
    with pytest.raises(Abort, match="IK error 12.0 mm.*went back to the start"):
        run_it(run, decide)
    assert decide.asked == [("staging", (GO, BACK)), ("failed", (BACK,))]
    assert "to pre-press" not in run.calls and run.calls[-1] == BACK_CLOSE


def test_each_space_at_the_button_is_one_press(steps):
    run, decide = FakeRun(presses=3), answers(GO, GO, GO, BACK)
    assert run_it(run, decide) == 0
    assert [c for c in run.calls if isinstance(c, tuple) and c[0] == "press"] == [("press", 1), ("press", 2)]
    assert run.calls[-1] == BACK_CLOSE
    assert ("plan", 1, True) in run.calls  # one stroke and the way back, gated


def test_every_decision_is_logged(steps):
    run = FakeRun()
    run_it(run, answers(GO, GO, BACK))
    assert [(r["stop"], r["choice"]) for r in run.records if r.get("stage") == "decision"] == [
        ("staging", GO),
        ("button", GO),
        ("button", BACK),
    ]


def test_a_failed_redetect_holds_waits_for_q_then_goes_back(steps):
    steps.add("measure")
    run, decide = FakeRun(), answers(GO, BACK)
    with pytest.raises(Abort, match="nothing in 7 frames.*went back to the start"):
        run_it(run, decide)
    assert decide.asked == [("staging", (GO, BACK)), ("failed", (BACK,))]  # only q works
    assert run.calls[-1] == BACK_FAR
    assert any(r.get("stage") == "abort_at_stop" for r in run.records)


def test_a_failed_press_holds_waits_for_q_then_goes_back(steps):
    run, decide = FakeRun(), answers(GO, GO, GO, BACK)
    run.fail = {("press", 2)}
    with pytest.raises(Abort, match="press 2: touched nothing.*went back to the start"):
        run_it(run, decide)
    assert decide.asked[-1] == ("failed", (BACK,))
    assert run.calls[-2:] == [("press", 1), BACK_CLOSE]


def test_a_failure_on_the_way_back_is_not_caught_again(steps):
    steps.add("back")
    run, decide = FakeRun(), answers(GO, BACK)
    with pytest.raises(Abort, match="cuRobo found no plan"):
        run_it(run, decide)
    assert [s for s, _ in decide.asked] == ["staging", "button"]  # no second prompt, no retry
    assert run.calls.count(BACK_CLOSE) == 1


def test_a_failure_still_at_the_start_ends_the_run_without_asking(steps, monkeypatch):
    def first_look(run, a):
        raise Abort("scene camera: no frames")

    monkeypatch.setattr(pb, "first_look", first_look)
    run, decide = FakeRun(), answers()
    with pytest.raises(Abort, match="no frames"):
        run_it(run, decide)
    assert decide.asked == [] and run.calls == ["preflight"]


def test_a_plan_refused_at_the_start_does_not_ask(steps):
    run, decide = FakeRun(), answers()
    with pytest.raises(Abort, match="at the start"):
        with pb.Stops(run, decide, START, np.eye(3)).holding(N_FAR):
            raise Abort("refused at the start")
    assert decide.asked == []


def test_without_decide_the_one_shot_run_is_unchanged(steps):
    run = FakeRun(presses=2)
    assert run_it(run, None) == 0
    assert run.calls == [
        "preflight",
        "to staging",
        "measure",
        ("plan", 2, True),
        "to pre-press",
        ("press", 1),
        ("press", 2),
        BACK_CLOSE,
    ]


def test_one_shot_no_return_stays_at_the_spot(steps):
    run = FakeRun(presses=1, no_return=True)
    assert run_it(run, None) == 0
    assert run.calls == ["preflight", "to staging", "measure", ("plan", 1, False), "to pre-press", ("press", 1)]


def test_dry_run_has_no_stops(steps, monkeypatch):
    run = FakeRun(execute=False)
    run.arm.go_to_ee_pose = lambda *a, **k: run.calls.append("staging goal checked")
    decide = answers()
    assert run_it(run, decide) == 0
    assert decide.asked == []
    assert run.calls == ["preflight", "staging goal checked"]


def _no_ros(monkeypatch):
    def boom():
        raise AssertionError("rclpy.init must not run")

    monkeypatch.setattr(pb.rclpy, "init", boom)


def test_interactive_refuses_no_return(monkeypatch, capsys):
    _no_ros(monkeypatch)
    assert pb.main(["--interactive", "--no-return", "--execute"]) == 2
    assert "--no-return" in capsys.readouterr().out


def test_interactive_execute_needs_a_terminal(monkeypatch, capsys):
    _no_ros(monkeypatch)
    monkeypatch.setattr(pb.sys, "stdin", argparse.Namespace(isatty=lambda: False))
    assert pb.main(["--interactive", "--execute"]) == 2
    assert "terminal" in capsys.readouterr().out
