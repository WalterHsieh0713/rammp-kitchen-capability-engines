"""What place_container puts on the user's UI panel (place_container.ui): pure, no ROS."""

from place_container import ui


def test_a_phase_shows_in_words_with_a_progress_bar():
    panel = ui.progress_panel("place", "lower")
    assert panel["title"] == "Placing the container"
    assert panel["message"] == "Lowering it onto the table"
    assert panel["progress"] == {"phase": "lower", "pct": 45.0, "pausable": False}
    assert "buttons" not in panel  # nothing to press while the arm moves


def test_the_phases_go_forward():
    pct = [ui.PHASES[p][1] for p in ("plan", "carry", "lower", "open", "retreat")]
    assert pct == sorted(pct) and all(0 < p < 100 for p in pct)
    assert all(p in ui.PHASES for p in ui.MOCK_PHASES)


def test_an_unknown_phase_shows_its_message_without_a_bar():
    panel = ui.progress_panel("place", "something_new", "doing something new")
    assert panel["message"] == "doing something new" and "progress" not in panel


def test_a_failure_shows_the_whole_reason_and_an_ok_button():
    why = "the spot is 0.83 m from the base axis -- move the spot"
    panel = ui.error_panel("place", why)
    assert panel["title"] == "Couldn't place the container"
    assert panel["message"] == why
    assert panel["buttons"] == [{"button_id": ui.OK, "label": "OK"}]
    assert ui.error_panel("reset", why)["title"] == "Couldn't go back to the start"


def test_only_the_ok_button_counts_as_ok():
    assert ui.is_ok({"type": "button", "button_id": ui.OK})
    assert not ui.is_ok({"type": "button", "button_id": "stop"})
    assert not ui.is_ok({"type": "object_tapped", "instance_id": ui.OK})
    assert not ui.is_ok(None)


def test_the_panels_are_json_the_bridge_takes():
    import json

    for panel in (ui.progress_panel("place", "carry"), ui.error_panel("mark", "x"), ui.done_panel("place", "Done")):
        assert set(panel) <= {"title", "message", "progress", "buttons"}  # docs/UI_JSON_SCHEMA.md
        json.dumps(panel)


def test_the_one_tap_place_ends_with_a_done_button_and_mock_walks_the_real_phases():
    panel = ui.placed_panel()
    assert panel["buttons"] == [{"button_id": ui.DONE, "label": "Done"}]
    assert panel["progress"]["pct"] == 100.0
    assert ui.is_button({"type": "button", "button_id": ui.DONE}, ui.DONE)
    assert ui.progress_panel("place_here", "return")["message"].startswith("Backing the hand away")
    assert ui.MOCK_PHASES[-1] == "return" and ui.title("place_here") == ui.title("place")
