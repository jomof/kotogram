"""Path utilities for training infrastructure."""

import os
import platform

from kotogram import locations


def get_style_history_dir() -> str:
    """Returns the directory for style training history and config (resides in .cache)."""
    return os.path.join(get_cache_dir(), "style-training")


def get_data_dir() -> str:
    """Returns the data directory (data inside TRAIN_ROOT)."""
    return os.path.join(locations.get_train_root(), "data")


def get_cache_dir() -> str:
    """Returns the base cache directory (.cache inside TRAIN_ROOT)."""
    return os.path.join(locations.get_train_root(), ".cache")


def get_shards_cache_dir() -> str:
    """Returns the directory for kotogram shards."""
    return os.path.join(get_cache_dir(), "kotogram_shards")


def get_style_dataset_cache_dir() -> str:
    """Returns the directory for style dataset metadata and vocabulary."""
    return os.path.join(get_cache_dir(), "style_dataset")


def get_profile_dir() -> str:
    """Returns the directory for profiling/instrumentation output."""
    root = os.environ.get("TRAIN_ROOT", ".")
    hostname = platform.node().split(".")[0]
    return os.path.join(root, f".profile-{hostname}")
