"""Tests for the repo-level capability checker.

Each test copies the example capability into a scratch repo, breaks one rule,
and asserts the checker names it.
"""

import shutil
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_capabilities  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo(tmp_path, monkeypatch):
    shutil.copytree(REPO / "capabilities" / "example", tmp_path / "capabilities" / "example")
    (tmp_path / ".github").mkdir()
    shutil.copy(REPO / ".github" / "CODEOWNERS", tmp_path / ".github" / "CODEOWNERS")
    monkeypatch.setattr(check_capabilities, "ROOT", tmp_path)
    return tmp_path / "capabilities" / "example"


def errors(cap):
    return "\n".join(check_capabilities.check(cap)[0])


def edit(path, change):
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data))


def test_example_passes(repo):
    assert errors(repo) == ""


def test_missing_mock_fails(repo):
    (repo / "rammp-alternative.mock.yaml").unlink()
    assert "missing rammp-alternative.mock.yaml" in errors(repo)


def test_mock_with_different_topics_fails(repo):
    edit(repo / "rammp-alternative.mock.yaml", lambda d: d["publishes"].append("/extra"))
    assert "mock publishes differs" in errors(repo)


def test_image_outside_capability_namespace_fails(repo):
    edit(
        repo / "rammp-alternative.yaml",
        lambda d: d["container"].update(image="ghcr.io/someone/else:1.0.0"),
    )
    assert "must be ghcr.io/rammp-org/rammp-kitchen-capability-engines/example" in errors(repo)


@pytest.mark.parametrize(
    "base",
    [
        "ros:humble",
        "ghcr.io/rammp-org/rammp-base:humble",
        "ghcr.io/rammp-org/rammp-base:latest",
        "ghcr.io/rammp-org/rammp-base:nightly",
    ],
)
def test_unpinned_base_fails(repo, base):
    dockerfile = repo / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text().replace("ghcr.io/rammp-org/rammp-base:nightly-20261027", base)
    )
    assert f"FROM {base}" in errors(repo)


def test_multistage_reference_to_earlier_stage_passes(repo):
    (repo / "Dockerfile").write_text(
        "FROM ghcr.io/rammp-org/rammp-cuda:1.1.0-jp6 AS build\nFROM build\n"
    )
    assert errors(repo) == ""


def test_missing_codeowner_fails(repo):
    (repo.parent.parent / ".github" / "CODEOWNERS").write_text("* @someone\n")
    assert "no owner line for /capabilities/example/" in errors(repo)


def set_tier(repo, tier, image=None):
    for name in ("rammp-alternative.yaml", "rammp-alternative.mock.yaml"):

        def change(d):
            d["tier"] = tier
            if image:
                d["container"]["image"] = image

        edit(repo / name, change)


def test_demo_ready_on_floating_tag_fails(repo):
    set_tier(repo, "demo-ready")
    assert "must be pinned to a release tag or digest" in errors(repo)


def test_demo_ready_pinned_with_status_passes(repo):
    set_tier(repo, "demo-ready", "ghcr.io/rammp-org/rammp-kitchen-capability-engines/example:1.0.0")
    assert errors(repo) == ""


def test_demo_ready_without_status_topic_fails(repo):
    set_tier(repo, "demo-ready", "ghcr.io/rammp-org/rammp-kitchen-capability-engines/example:1.0.0")
    for name in ("rammp-alternative.yaml", "rammp-alternative.mock.yaml"):
        edit(repo / name, lambda d: d.update(publishes=["/example/heartbeat"]))
    assert "publishes no health topic" in errors(repo)


def test_mismatched_tiers_fail(repo):
    edit(repo / "rammp-alternative.mock.yaml", lambda d: d.update(tier="integrated"))
    assert "same tier" in errors(repo)


def test_non_snake_case_name_fails(tmp_path):
    cap = tmp_path / "Button-Press"
    cap.mkdir()
    assert "snake_case" in errors(cap)
