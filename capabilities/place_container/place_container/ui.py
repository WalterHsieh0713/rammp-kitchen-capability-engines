"""What place_container shows on the user's UI panel (ctx.update_user_panel; the shapes are in
docs/UI_JSON_SCHEMA.md). Pure: no ROS, so it is tested on its own.

Every phase the action reports with ctx.progress() is also put on the panel, in words, with a
progress bar; every failure is put on it with an OK button before the action ends with the error
(the panel closes when the action ends, so an error not shown before then is never seen).
"""

OK = "ok"  # the error panel's button
DONE = "done"  # the one-tap place's "placed" panel's button

TITLES = {
    "place_here": "Placing the container",
    "place": "Placing the container",
    "mark": "Marking the spot",
    "retreat": "Backing the hand away",
    "reset": "Going back to the start",
}
FAILED_TITLES = {
    "place_here": "Couldn't place the container",
    "place": "Couldn't place the container",
    "mark": "Couldn't mark the spot",
    "retreat": "Couldn't back the hand away",
    "reset": "Couldn't go back to the start",
}
# A phase (what execute() reports with ctx.progress) -> what the user reads, and how far along the
# action is then (%). The carry's own legs (adjust, lift, transit, ...) are reported under their
# names when they run as separate moves.
PHASES = {
    "plan": ("Checking the plan", 5),
    "carry": ("Carrying the container over to the spot", 15),
    "adjust": ("Squaring the hand", 10),
    "lift": ("Lifting the container", 12),
    "transit": ("Carrying the container over to the spot", 20),
    "transit-reconfigure": ("Carrying the container over to the spot", 20),
    "lower": ("Lowering it onto the table", 45),
    "open": ("Letting go", 70),
    "retreat": ("Backing the hand away", 85),
    "return": ("Backing the hand away and going back to the start", 85),
    "reset": ("Going back to the start", 50),
}
# mock place: the phases it walks through, and how long it shows each (s)
MOCK_PHASES = ("carry", "lower", "open", "return")
MOCK_PHASE_S = 1.0


def title(action_id):
    return TITLES.get(action_id, action_id)


def progress_panel(action_id, phase, message=""):
    """The panel while `action_id` is in `phase`: its title, the phase in words, a progress bar.
    A phase not in PHASES shows `message` (or its own name) and no bar."""
    text, pct = PHASES.get(phase, (message or phase, None))
    panel = {"title": title(action_id), "message": text}
    if pct is not None:
        # no pause control: a pause in the middle of the lowering or the retreat is not offered
        panel["progress"] = {"phase": phase, "pct": float(pct), "pausable": False}
    return panel


def done_panel(action_id, message):
    return {
        "title": title(action_id),
        "message": message,
        "progress": {"phase": "done", "pct": 100.0, "pausable": False},
    }


def error_panel(action_id, message):
    """The panel for a failed action: what went wrong (the refusal's own words: what happened,
    where, what to do next) and an OK button."""
    return {
        "title": FAILED_TITLES.get(action_id, f"{title(action_id)}: failed"),
        "message": message,
        "buttons": [{"button_id": OK, "label": "OK"}],
    }


def placed_panel():
    """The one-tap place's end: done, until the user taps Done."""
    return {
        "title": TITLES["place_here"],
        "message": "The container is on the table and the arm is back at its start.",
        "progress": {"phase": "done", "pct": 100.0, "pausable": False},
        "buttons": [{"button_id": DONE, "label": "Done"}],
    }


def is_button(event, button_id):
    """Whether a poll_ui_event() result is a tap on `button_id`."""
    return bool(event) and event.get("type") == "button" and event.get("button_id") == button_id


def is_ok(event):
    """Whether a poll_ui_event() result is the error panel's OK."""
    return is_button(event, OK)
