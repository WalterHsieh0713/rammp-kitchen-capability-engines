"""The grab_container package on the path, so its pure-Python parts test without a colcon install or ROS:

    uv run --no-project --with pytest --with numpy --with scipy --with opencv-python-headless \
        python3 -m pytest -q capabilities/grab_container/test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
