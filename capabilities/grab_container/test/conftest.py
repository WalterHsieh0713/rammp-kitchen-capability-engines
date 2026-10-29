"""The grab_container package on the path, so it tests without a colcon install, in the arm test
image (grasp imports capability_engine.arm):

    make test-arm ARM_CMD='bash -c "python3 -m pip install -q --user opencv-python-headless; \
        python3 -m pytest -q -p no:cacheprovider capabilities/grab_container/test"'

(the test image has no cv2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
