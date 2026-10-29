"""The place_container package on the path, so it tests without a colcon install, in the arm test
image (plan imports capability_engine.sim, which needs pybullet; the flatbot model is the image's
/fixtures/urdf/flatbot.urdf):

    make test-arm ARM_CMD='python3 -m pytest -q -p no:cacheprovider capabilities/place_container/test'
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
