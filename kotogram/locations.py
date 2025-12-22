import os


def get_train_root() -> str:
    """Returns the training root directory from TRAIN_ROOT env var, defaulting to current dir."""
    return os.environ.get("TRAIN_ROOT", ".")


def get_cache_dir() -> str:
    """Returns the base cache directory (.cache inside TRAIN_ROOT)."""
    return os.path.join(get_train_root(), ".cache")


def get_data_dir() -> str:
    """Returns the data directory (data inside TRAIN_ROOT)."""
    return os.path.join(get_train_root(), "data")


def get_models_dir() -> str:
    """Returns the models directory (models inside TRAIN_ROOT)."""
    return os.path.join(get_train_root(), "models")


def get_shards_cache_dir() -> str:
    """Returns the directory for kotogram shards."""
    return os.path.join(get_cache_dir(), "kotogram_shards")


def get_style_dataset_cache_dir() -> str:
    """Returns the directory for style dataset metadata and vocabulary."""
    return os.path.join(get_cache_dir(), "style_dataset")


def get_style_output_dir() -> str:
    """Returns the directory for style model outputs and artifacts."""
    return os.path.join(get_models_dir(), "style")


def get_style_support_dir() -> str:
    """Returns the directory for style model training support artifacts (checkpoints, logs)."""
    return os.path.join(get_models_dir(), "style-support")


def get_profile_dir() -> str:
    """Returns the profile directory (.profile inside TRAIN_ROOT)."""
    return os.path.join(get_train_root(), ".profile")


def ensure_dir(path: str) -> None:
    """Ensures the directory exists."""
    os.makedirs(path, exist_ok=True)
