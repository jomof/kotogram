#!/usr/bin/env python3
"""Standalone locations script for shell script usage.

This script provides CLI access to kotogram.locations without triggering
the full kotogram package import, avoiding RuntimeWarnings.
"""

import sys

from kotogram.locations import (
    get_cache_dir,
    get_data_dir,
    get_models_dir,
    get_shards_cache_dir,
    get_style_dataset_cache_dir,
    get_style_output_dir,
    get_style_support_dir,
    get_train_root,
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "shell-env":
            # Output all paths in shell-evaluable format
            print(f"export DATA_DIR='{get_data_dir()}'")
            print(f"export CACHE_DIR='{get_cache_dir()}'")
            print(f"export MODELS_DIR='{get_models_dir()}'")
            print(f"export SUPPORT_DIR='{get_style_support_dir()}'")
        elif sys.argv[1] == "cache":
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
