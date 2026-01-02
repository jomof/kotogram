"""I/O utilities for model checkpoints and weights."""

import json
import os
import random
from typing import Any, Dict, Optional, Union, cast

import torch
from torch import nn

from kotogram.constants import (
    FORMALITY_LABEL_TO_ID,
    GENDER_LABEL_TO_ID,
)
from kotogram.model import (
    ModelConfig,
    StyleClassifier,
)
from kotogram.tokenizer import Tokenizer
from train.types import TrainingHistory


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    """Save tokenizer vocabularies to JSON file atomically."""
    # pylint: disable=import-outside-toplevel
    data = tokenizer.to_dict()

    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Atomic write pattern
    import tempfile

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=dir_name, delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_path = tmp_file.name
            json.dump(data, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_model(
    model: StyleClassifier,
    path: str,
    config: ModelConfig,
) -> None:
    """Save trained model, tokenizer, and config."""
    # pylint: disable=too-many-positional-arguments
    os.makedirs(path, exist_ok=True)

    # Save model weights (Always use FP8 if available)
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("FP8 requires PyTorch 2.1+.")

    state_dict = {
        k: v.cpu().to(torch.float8_e4m3fn) if v.dtype == torch.float32 else v.cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("kc_decoders.")
    }
    torch.save(state_dict, os.path.join(path, "model.pt"))

    # Save config
    config = config or model.config
    with open(os.path.join(path, "model.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Save label mappings
    formality_label_map = {k.value: v for k, v in FORMALITY_LABEL_TO_ID.items()}
    gender_label_map = {k.value: v for k, v in GENDER_LABEL_TO_ID.items()}
    grammaticality_label_map = {"agrammatic": 0, "grammatic": 1}
    with open(os.path.join(path, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "formality": formality_label_map,
                "gender": gender_label_map,
                "grammaticality": grammaticality_label_map,
            },
            f,
            indent=2,
        )

    # Mark as feature-based multi-task model
    with open(os.path.join(path, "model_type.txt"), "w", encoding="utf-8") as f:
        f.write("style-multitask")


def get_rng_states() -> Dict[str, Any]:
    """Capture RNG states for all relevant libraries."""
    states = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()

    import numpy as np

    states["numpy"] = np.random.get_state()
    return states


def set_rng_states(states: Dict[str, Any]) -> None:
    """Restore RNG states."""
    if "python" in states:
        random.setstate(states["python"])
    if "torch" in states:
        torch.set_rng_state(states["torch"].cpu())
    if "cuda" in states and torch.cuda.is_available():
        # states["cuda"] is a list of tensors for CUDA
        torch.cuda.set_rng_state_all([s.cpu() for s in states["cuda"]])
    if "numpy" in states:
        import numpy as np

        np.random.set_state(states["numpy"])


def save_training_state(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Union[Dict[str, Any], TrainingHistory],
    config: Any,
    global_step: int = 0,
    scheduler: Optional[Any] = None,
    filename: str = "checkpoint.pt",
) -> None:
    """Generic training state save."""
    # pylint: disable=too-many-positional-arguments
    os.makedirs(path, exist_ok=True)
    history_dict = (
        history.to_dict()
        if hasattr(history, "to_dict")
        else (vars(history) if not isinstance(history, dict) else history)
    )

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history_dict,
        "rng_states": get_rng_states(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        # Save as 'args' for legacy compatibility with scripts/train_style.py
        checkpoint["args"] = (
            config.to_dict() if hasattr(config, "to_dict") else vars(config)
        )
        checkpoint["config"] = checkpoint["args"]

    torch.save(checkpoint, os.path.join(path, filename))


def load_training_state(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    filename: str = "checkpoint.pt",
) -> Dict[str, Any]:
    """Generic training state load."""
    # pylint: disable=too-many-positional-arguments
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No checkpoint at {full_path}")

    checkpoint = torch.load(full_path, map_location="cpu", weights_only=False)

    # Load model weights
    model_state = checkpoint["model_state_dict"]
    # Handle 'module.' prefix (legacy support for checkpoints)
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    # Use strict=False to get missing keys without exception, avoiding banned try-except RuntimeError
    missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

    # Filter out known safe missing keys (kc_head initialization)
    # If we are loading a non-KC checkpoint into a KC model, kc_head weights will be missing.
    real_missing = [
        k
        for k in missing_keys
        if not k.startswith("kc_head.") and not k.startswith("kc_decoders.")
    ]

    if real_missing or unexpected_keys:
        err_msgs = []
        if real_missing:
            err_msgs.append(
                f"Missing key(s) in state_dict: {', '.join(map(repr, real_missing))}."
            )
        if unexpected_keys:
            err_msgs.append(
                f"Unexpected key(s) in state_dict: {', '.join(map(repr, unexpected_keys))}."
            )
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t"
            + "\n\t".join(err_msgs)
        )

    if missing_keys:
        print(
            f"Warning: Missing KC head weights in checkpoint. Initializing from scratch. (Missing: {len(missing_keys)} keys)"
        )

    # Load optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scaler
    if "scaler_state_dict" in checkpoint:
        # We don't have scaler argument anymore, so we can't load it.
        # But if scaler is None (as per recorder), we shouldn't be here?
        # Or scaler logic is removed entirely?
        pass

    # Load scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG
    if "rng_states" in checkpoint:
        set_rng_states(checkpoint["rng_states"])

    return cast(Dict[str, Any], checkpoint)
