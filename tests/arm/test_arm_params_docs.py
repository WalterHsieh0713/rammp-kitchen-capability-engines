"""The example's real fragment and interface.md list exactly arm_params.PARAMS, with its defaults.

Lives in tests/arm because arm_params imports sheppy_arm (numpy, pybullet), which the plain-python
`checks` job lacks.
"""

import json
import re
from pathlib import Path

import yaml

from capability_engine.arm_params import PARAMS

EXAMPLE = Path(__file__).parents[2] / "capabilities" / "example"
EXPECTED = {f"arm.{prm.name}": prm.default for prm in PARAMS}


def test_real_fragment_lists_every_arm_param_with_its_default():
    params = yaml.safe_load((EXAMPLE / "rammp-alternative.yaml").read_text())["params"]
    assert {k: v for k, v in params.items() if k.startswith("arm.")} == EXPECTED


def test_mock_fragment_has_no_params():
    assert "params" not in yaml.safe_load((EXAMPLE / "rammp-alternative.mock.yaml").read_text())


def test_interface_md_documents_every_arm_param_with_its_default():
    text = (EXAMPLE / "docs" / "interface.md").read_text()
    rows = re.findall(r"^\|\s*`(arm\.\w+)`\s*\|\s*`([^`]*)`\s*\|", text, re.M)
    assert {name: json.loads(default) for name, default in rows} == EXPECTED
    assert len(rows) == len(EXPECTED)
