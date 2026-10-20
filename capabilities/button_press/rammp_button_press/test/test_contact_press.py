"""The contact press (rammp_arm.ContactPress, box-opening's touch + push): in until a wrist torque
moves CONTACT_TOUCH_NM off its baseline, then CONTACT_PUSH_M further, position-bounded; the
microwave sliding or the panel missing ends it as a failure, never as a press."""

from rammp_button_press.rammp_arm import (
    CONTACT_MAX_TOUCH_M,
    CONTACT_MIN_TOUCH_M,
    CONTACT_PUSH_M,
    CONTACT_SPEED_MPS,
    CONTACT_START_S,
    CONTACT_STOP_NM,
    CONTACT_TIMEOUT_S,
    CONTACT_TOUCH_NM,
    ContactPress,
    reads_channel,
)

DT = 0.01
V = CONTACT_SPEED_MPS


def run(dev_at, lateral_at=lambda along: 0.0, t_end=10.0):
    """Feed a simulated push: the arm follows the commanded travel exactly, and the wrist torque
    deviation is a function of how far in it is. Returns (stop, t, along, press)."""
    press, t = ContactPress(), 0.0
    while t < t_end:
        along = V * t
        stop = press.update(t, V * t, along, lateral_at(along), dev_at(along))
        if stop:
            return stop, t, along, press
        t += DT
    return None, t, V * t, press


def button(at=0.015, k_nm_per_mm=1.0):
    """The wrist torque deviation of a button whose top is `at` m in: zero in free air, then
    rising k N.m per mm pushed past it (noise-free)."""
    return lambda along: max(0.0, (along - at) * 1000 * k_nm_per_mm)


def test_a_button_is_touched_then_pushed_the_bounded_distance():
    stop, t, along, press = run(button(at=0.015, k_nm_per_mm=0.3))
    assert stop[0] == "pressed"
    touch_dev_mm = CONTACT_TOUCH_NM / 0.3  # trips this far past the top
    assert abs(press.touch_m - (0.015 + touch_dev_mm / 1000)) < 2 * V * DT
    assert abs(along - (press.touch_m + CONTACT_PUSH_M)) < 2 * V * DT


def test_a_stiff_stop_during_the_push_ends_it_early_as_a_press():
    stop, _, along, press = run(button(at=0.015, k_nm_per_mm=2.0))
    assert stop[0] == "stop"
    assert along - press.touch_m < CONTACT_PUSH_M


def test_the_push_is_bounded_even_if_the_microwave_slides():
    # touch, then the torque never reaches the stop: it slides -- the push still ends at the bound
    def dev(along):
        return CONTACT_TOUCH_NM + 0.5 if along > 0.015 else 0.0

    stop, _, along, press = run(dev)
    assert stop[0] == "pressed"
    assert along <= press.touch_m + CONTACT_PUSH_M + 2 * V * DT


def test_nothing_touched_is_no_surface_not_a_press():
    stop, _, along, _ = run(lambda along: 0.0)
    assert stop[0] == "no_surface"
    assert along <= CONTACT_MAX_TOUCH_M + V * DT


def test_a_touch_before_the_panel_struck_something_else():
    # contact 2 mm in, judged as soon as the launch transient is over
    stop, *_ = run(lambda along: 10.0 if along > 0.002 else 0.0)
    assert stop[0] == "early"
    assert CONTACT_MIN_TOUCH_M > V * CONTACT_START_S


def test_the_launch_transient_is_not_judged():
    press = ContactPress()
    assert press.update(CONTACT_START_S / 2, 0.001, 0.001, 0.0, CONTACT_STOP_NM * 2) is None
    assert press.phase == "touch"


def test_sideways_drift_stops_the_push():
    stop, *_ = run(button(), lateral_at=lambda along: 0.02 if along > 0.005 else 0.0)
    assert stop[0] == "lateral"


def test_a_stuck_arm_times_out():
    press, t, stop = ContactPress(), 0.0, None
    while stop is None:  # commanded never advances, nothing touched
        stop = press.update(t, 0.0, 0.0, 0.0, 0.0)
        t += DT
    assert stop[0] == "timeout" and t > CONTACT_TIMEOUT_S


def test_the_driver_s_bare_channel_names_are_accepted():
    assert reads_channel(["twist"], "twist")  # what the driver actually answers (2026-10-15)
    assert reads_channel(["/setpoint/twist"], "twist")
    assert reads_channel(["pose"], "pose")
    assert not reads_channel(["pose"], "twist")
    assert not reads_channel([], "twist")


def test_the_lighter_defaults_stop_before_the_old_8_mm_and_7_nm():
    """2026-10-15: the stop came 1.5-2.5 mm past the touch at ~7 N.m; pushing on bent the finger off the dome."""
    assert CONTACT_PUSH_M <= 0.004 and CONTACT_STOP_NM <= 5.0 and CONTACT_STOP_NM > CONTACT_TOUCH_NM


def test_a_touch_at_7_6_mm_is_a_press_not_an_early_strike():
    """2026-10-15: a press that clicked once registered its touch at 7.6 mm and was refused as early."""
    stop, _, _, press = run(button(at=0.005, k_nm_per_mm=1.2))
    assert stop[0] in ("pressed", "stop") and press.touch_m < 0.008


def test_push_and_stop_can_be_set_per_press():
    stop, _, along, press = run_with(ContactPress(push=0.002, stop_nm=9.0), button(at=0.015, k_nm_per_mm=0.3))
    assert stop[0] == "pressed" and abs(along - (press.touch_m + 0.002)) < 2 * V * DT


def run_with(press, dev_at):
    t = 0.0
    while t < 10.0:
        along = V * t
        stop = press.update(t, V * t, along, 0.0, dev_at(along))
        if stop:
            return stop, t, along, press
        t += DT
    return None, t, V * t, press
