#!/usr/bin/env python3
"""Validate a rammp-alternative.yaml fragment.

The fragment is sheppy's `alternative` schema plus RAMMP's `implements` and
`tier`. Sheppy's loader rejects only an unknown `kind` and an unknown
`machine` — it ignores everything else it does not recognise — so a typo in a
topic name or a missing image would otherwise travel all the way to the robot.
"""

import sys

import yaml

TIERS = ("experimental", "integrated", "supported", "demo-ready")


def validate(fragment):
    """Return a list of human-readable problems. Empty means valid."""
    if not isinstance(fragment, dict):
        return ["fragment must be a YAML mapping"]

    errors = []

    if not isinstance(fragment.get("id"), str) or not fragment.get("id"):
        errors.append("'id' is required and must be a non-empty string")

    if fragment.get("kind") != "docker":
        errors.append("'kind' must be 'docker'")

    if fragment.get("tier") not in TIERS:
        errors.append(f"'tier' must be one of: {', '.join(TIERS)}")

    implements = fragment.get("implements")
    if implements is not None and not isinstance(implements, str):
        errors.append("'implements' must be a role name string when present")

    container = fragment.get("container")
    if not isinstance(container, dict):
        errors.append("'container' is required and must be a mapping")
    elif not isinstance(container.get("image"), str) or not container.get("image"):
        errors.append("'container.image' is required and must be a non-empty string")

    for key in ("publishes", "subscribes"):
        topics = fragment.get(key, [])
        if not isinstance(topics, list):
            errors.append(f"'{key}' must be a list")
            continue
        for topic in topics:
            if not isinstance(topic, str) or not topic.startswith("/"):
                errors.append(f"'{key}' entry {topic!r} must be an absolute topic name")

    return errors


def main(paths):
    failed = False
    for path in paths:
        with open(path) as handle:
            errors = validate(yaml.safe_load(handle))
        for error in errors:
            print(f"{path}: {error}", file=sys.stderr)
        if errors:
            failed = True
        else:
            print(f"{path}: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
