import sys
from pathlib import Path

# Add project root to sys.path to allow imports from kotogram and scripts
# This file intends to be imported by other scripts in this directory
# to setup sys.path as a side-effect, avoiding E402 issues in those scripts.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
