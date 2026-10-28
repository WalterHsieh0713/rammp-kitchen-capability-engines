"""The example's real fragment and interface.md list exactly arm_params.PARAMS, with its defaults.

Lives in tests/arm because arm_params imports rammp_arm (numpy, pybullet), which the plain-python
`checks` job lacks.
"""

import json
import re
from pathlib import Path

import yaml

from capability_engine.arm_params import PARAMS

EXAMPLE = Path(__file__).parents[2] / "capabilities" / "example"
EXPECTED = {f"arm.{prm.name}": prm.default for prm in PARAMS}
# arm.sender_id is the one exception: its default under the engine is the engine id, so the real
# fragment leaves it out (a copied fragment would otherwise send every capability's goals as
# "microwave") and interface.md names that default in words.
SENDER_ID = "arm.sender_id"
SENDER_ID_DEFAULT_DOC = 'the engine id (RammpArm standalone: `"microwave"`)'


def test_real_fragment_lists_every_arm_param_with_its_default():
    params = yaml.safe_load((EXAMPLE / "rammp-alternative.yaml").read_text())["params"]
    expected = {k: v for k, v in EXPECTED.items() if k != SENDER_ID}
    assert {k: v for k, v in params.items() if k.startswith("arm.")} == expected


def test_mock_fragment_has_no_params():
    assert "params" not in yaml.safe_load((EXAMPLE / "rammp-alternative.mock.yaml").read_text())


def test_interface_md_documents_every_arm_param_with_its_default_and_meaning():
    text = (EXAMPLE / "docs" / "interface.md").read_text()
    lines = [ln for ln in text.splitlines() if re.match(r"\|\s*`arm\.", ln)]
    rows = {}
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        assert len(cells) == 3, f"expected 3 cells (an unescaped | in the text?): {line}"
        name = cells[0].strip("`")
        default = cells[1] if name == SENDER_ID else json.loads(cells[1].strip("`"))
        rows[name] = (default, cells[2])
    expected = {f"arm.{prm.name}": (prm.default, prm.description) for prm in PARAMS}
    expected[SENDER_ID] = (SENDER_ID_DEFAULT_DOC, expected[SENDER_ID][1])
    assert rows == expected
    assert len(lines) == len(PARAMS)
