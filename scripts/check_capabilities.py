#!/usr/bin/env python3
"""Check every capability under capabilities/ against this repo's rules.

Each capability is one RAMMP module: its own image, its own real and mock
fragments, its own docs page and owner. This is the gate that keeps them that
way, on top of validate_fragment.py (the template's per-fragment schema check):

- the required files exist, including a package.xml (colcon-buildable)
- both fragments are schema-valid and point at this capability's image
- the mock speaks exactly the real alternative's interface
- every FROM pins a versioned rammp-base / rammp-cuda image
- CODEOWNERS names an owner for the capability
- a `tier: demo-ready` capability publishes a status topic and pins its image
- sheppy itself loads the capability as a node, if sheppy is installed

Usage: check_capabilities.py [capability ...]   (default: all of them)
"""

import re
import sys
from pathlib import Path

import yaml
from validate_fragment import validate

ROOT = Path(__file__).resolve().parent.parent
CAPABILITIES = ROOT / "capabilities"
IMAGE_PREFIX = "ghcr.io/rammp-org/rammp-kitchen-capability-engines/"
REQUIRED = (
    "package.xml",
    "Dockerfile",
    "rammp-alternative.yaml",
    "rammp-alternative.mock.yaml",
    "docs/interface.md",
)
NAME = re.compile(r"^[a-z][a-z0-9_]*$")
PINNED_BASE = re.compile(r"^ghcr\.io/rammp-org/rammp-(base|cuda):\d+\.\d+\.\d+\S*$")
# Mutable tags a demo profile must never point at.
FLOATING_TAGS = ("dev", "main", "latest")


def check_dockerfile(path):
    errors = []
    stages = set()
    found = False
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].upper() != "FROM":
            continue
        found = True
        args = [p for p in parts[1:] if not p.startswith("--")]
        image = args[0]
        if image not in stages and not PINNED_BASE.match(image):
            errors.append(
                f"FROM {image}: build from a versioned ghcr.io/rammp-org/rammp-base or "
                "rammp-cuda tag (e.g. rammp-base:1.1.0-jp6), not a floating or upstream image"
            )
        if len(args) >= 3 and args[1].upper() == "AS":
            stages.add(args[2])
    if not found:
        errors.append("no FROM line")
    return errors


def check_codeowners(name):
    path = ROOT / ".github" / "CODEOWNERS"
    owned = f"/capabilities/{name}/"
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == owned:
                return []
    return [f".github/CODEOWNERS has no owner line for {owned}"]


def check_demo_ready(real):
    errors = []
    image = real["container"]["image"]
    tag = image.split("@")[0].rsplit(":", 1)[-1] if ":" in image.split("/")[-1] else ""
    if "@sha256:" not in image and (not tag or tag in FLOATING_TAGS):
        errors.append(f"tier demo-ready: image {image} must be pinned to a release tag or digest")
    if not any(t.endswith("/status") for t in real.get("publishes", [])):
        errors.append("tier demo-ready: publishes no health topic ending in /status")
    return errors


def check_sheppy(name, real, mock):
    """Load the capability as a one-node manifest through sheppy's own loader."""
    try:
        from sheppy.launch.docker.compose import service_to_docker_args
        from sheppy.manifest.loader import parse_manifest
    except ImportError:
        return None
    result = parse_manifest({"nodes": [{"name": name, "alternatives": [real, mock]}]})
    errors = [f"sheppy: {e.location}: {e.message}" for e in result.errors]
    # sheppy ignores depends_on, restart and healthcheck (it owns lifecycle),
    # so it only warns about them; a fragment relying on one is still wrong.
    for alt in (real, mock):
        *_, warns = service_to_docker_args(alt["container"])
        errors += [f"sheppy: {alt['id']}: {w}" for w in warns]
    return errors


def check(cap):
    name = cap.name
    if not NAME.match(name):
        return [f"directory name must be snake_case ({NAME.pattern})"], False
    missing = [f for f in REQUIRED if not (cap / f).exists()]
    if missing:
        return [f"missing {f}" for f in missing], False

    real = yaml.safe_load((cap / "rammp-alternative.yaml").read_text())
    mock = yaml.safe_load((cap / "rammp-alternative.mock.yaml").read_text())
    errors = [f"rammp-alternative.yaml: {e}" for e in validate(real)]
    errors += [f"rammp-alternative.mock.yaml: {e}" for e in validate(mock)]
    if errors:
        return errors, False

    for label, frag in (("real", real), ("mock", mock)):
        image = frag["container"]["image"]
        if not image.startswith(f"{IMAGE_PREFIX}{name}:") and not image.startswith(
            f"{IMAGE_PREFIX}{name}@"
        ):
            errors.append(f"{label} image {image} must be {IMAGE_PREFIX}{name}:<tag>")
    for key in ("publishes", "subscribes"):
        if sorted(real.get(key, [])) != sorted(mock.get(key, [])):
            errors.append(f"mock {key} differs from real {key}: same interface, no hardware")
    if real["tier"] != mock["tier"]:
        errors.append("real and mock fragments must declare the same tier")
    if real["tier"] == "demo-ready":
        errors += check_demo_ready(real)

    errors += check_dockerfile(cap / "Dockerfile")
    errors += check_codeowners(name)
    sheppy_errors = check_sheppy(name, real, mock)
    if sheppy_errors is not None:
        errors += sheppy_errors
    return errors, sheppy_errors is not None


def main(names):
    caps = [CAPABILITIES / n for n in names] or sorted(
        p for p in CAPABILITIES.iterdir() if p.is_dir()
    )
    if not caps:
        print("no capabilities found", file=sys.stderr)
        return 1
    failed = False
    for cap in caps:
        errors, sheppy_ran = check(cap)
        for error in errors:
            print(f"capabilities/{cap.name}: {error}", file=sys.stderr)
        if errors:
            failed = True
        else:
            note = "" if sheppy_ran else " (sheppy not installed; loader check skipped)"
            print(f"capabilities/{cap.name}: ok{note}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
