"""Unit tests for CapabilityEngine.availability()'s default perception_class
filtering -- no rclpy node is constructed (object.__new__ skips Node.__init__),
since availability() only reads self.actions."""

from capability_engine.engine import Action, CapabilityEngine


def _engine(actions):
    engine = object.__new__(CapabilityEngine)
    engine.actions = actions
    return engine


def test_action_with_no_perception_classes_is_always_offered():
    engine = _engine([Action(action_id="deselect", label="Deselect")])
    result = engine.availability("fridge_1", {})
    assert [a.action_id for a in result] == ["deselect"]
    assert result[0].available is True


def test_action_restricted_to_matching_class_is_offered():
    engine = _engine([Action(action_id="open", label="Open", perception_classes=["fridge"])])
    world_state = {"objects": {"fridge_1": {"perception_class": "fridge"}}}
    result = engine.availability("fridge_1", world_state)
    assert [a.action_id for a in result] == ["open"]


def test_action_restricted_to_other_class_is_omitted_entirely():
    engine = _engine([Action(action_id="open", label="Open", perception_classes=["fridge"])])
    world_state = {"objects": {"microwave_1": {"perception_class": "microwave"}}}
    result = engine.availability("microwave_1", world_state)
    assert result == []


def test_unknown_instance_class_falls_back_to_offering_every_action():
    engine = _engine([Action(action_id="open", label="Open", perception_classes=["fridge"])])
    result = engine.availability("unknown_1", {})
    assert [a.action_id for a in result] == ["open"]
