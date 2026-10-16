"""Tests for the fragment validator.

A fragment is a sheppy `alternative` block plus RAMMP's `implements` and
`tier`. Sheppy ignores keys it does not know, so nothing downstream catches a
malformed fragment — this validator is the only thing that does.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_fragment import validate  # noqa: E402


def minimal():
    return {
        "id": "example_real",
        "tier": "experimental",
        "kind": "docker",
        "container": {"image": "ghcr.io/rammp-org/rammp-module-template:dev"},
        "publishes": ["/example/heartbeat"],
        "subscribes": [],
    }


def test_minimal_fragment_is_valid():
    assert validate(minimal()) == []


def test_implements_is_optional_but_must_be_a_string():
    fragment = minimal()
    fragment["implements"] = "perception"
    assert validate(fragment) == []

    fragment["implements"] = 42
    assert any("implements" in error for error in validate(fragment))


def test_missing_id_is_reported():
    fragment = minimal()
    del fragment["id"]
    assert any("'id'" in error for error in validate(fragment))


def test_kind_must_be_docker():
    fragment = minimal()
    fragment["kind"] = "executable"
    assert any("'kind'" in error for error in validate(fragment))


def test_unknown_tier_is_reported():
    fragment = minimal()
    fragment["tier"] = "platinum"
    assert any("'tier'" in error for error in validate(fragment))


def test_container_image_is_required():
    fragment = minimal()
    fragment["container"] = {}
    assert any("container.image" in error for error in validate(fragment))


def test_relative_topic_names_are_reported():
    fragment = minimal()
    fragment["publishes"] = ["example/heartbeat"]
    errors = validate(fragment)
    assert any("absolute topic name" in error for error in errors)


def test_topics_must_be_a_list():
    fragment = minimal()
    fragment["subscribes"] = "/camera/color/image_raw"
    assert any("'subscribes' must be a list" in error for error in validate(fragment))


def test_non_mapping_fragment_is_reported():
    assert validate(["not", "a", "mapping"]) == ["fragment must be a YAML mapping"]


@pytest.mark.parametrize("tier", ["experimental", "integrated", "supported", "demo-ready"])
def test_every_documented_tier_is_accepted(tier):
    fragment = minimal()
    fragment["tier"] = tier
    assert validate(fragment) == []
