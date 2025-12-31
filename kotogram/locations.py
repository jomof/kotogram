import os


def get_train_root() -> str:
    """Returns the training root directory from TRAIN_ROOT env var, defaulting to current dir."""
    return os.environ.get("TRAIN_ROOT", ".")


def get_models_dir() -> str:
    """Returns the models directory (models inside TRAIN_ROOT)."""
    return os.path.join(get_train_root(), "models")


def get_style_output_dir() -> str:
    """Returns the directory for style model outputs and artifacts."""
    return os.path.join(get_models_dir(), "style")
