#!/usr/bin/env python3
"""Standalone locations script for shell script usage.

This script provides CLI access to kotogram.locations without triggering
the full kotogram package import, avoiding RuntimeWarnings.
"""

import sys
import os

# Import locations module directly without triggering kotogram/__init__.py
# We add the parent directory to path and import the module file directly
_kotogram_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _kotogram_dir)

# Import all functions from the canonical locations module
from kotogram.locations import (  # noqa: E402
    get_train_root,
    get_cache_dir,
    get_data_dir,
    get_models_dir,
    get_shards_cache_dir,
    get_style_dataset_cache_dir,
    get_style_output_dir,
    get_style_support_dir,
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "cache":
            print(get_cache_dir())
        elif sys.argv[1] == "data":
            print(get_data_dir())
        elif sys.argv[1] == "models":
            print(get_models_dir())
        elif sys.argv[1] == "shards":
            print(get_shards_cache_dir())
        elif sys.argv[1] == "style-dataset":
            print(get_style_dataset_cache_dir())
        elif sys.argv[1] == "style-output":
            print(get_style_output_dir())
        elif sys.argv[1] == "style-support":
            print(get_style_support_dir())
        elif sys.argv[1] == "train-root":
            print(get_train_root())
