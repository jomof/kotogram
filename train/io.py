"""I/O utilities for model checkpoints and weights."""

import importlib.util
import json
import os
import random
from typing import Any, Dict, Optional, cast

import torch
from torch import nn

from kotogram.model import (
    FORMALITY_LABEL_TO_ID,
    GENDER_LABEL_TO_ID,
    ModelConfig,
    StyleClassifier,
)
from kotogram.tokenizer import Tokenizer


def save_model(
    model: StyleClassifier,
    tokenizer: Tokenizer,
    path: str,
    config: Optional[ModelConfig] = None,
    fp16: bool = False,
    fp8: bool = False,
) -> None:
    """Save trained model, tokenizer, and config."""
    # pylint: disable=too-many-positional-arguments
    os.makedirs(path, exist_ok=True)

    # Save model weights
    if fp8:
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("FP8 requires PyTorch 2.1+. Use --fp16 instead.")
        state_dict = {
            k: v.cpu().to(torch.float8_e4m3fn) if v.dtype == torch.float32 else v.cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("mlm_head.") and not k.startswith("kc_decoders.")
        }
        torch.save(state_dict, os.path.join(path, "model.pt"))
    elif fp16:
        state_dict = {
            k: v.cpu().half() if v.dtype == torch.float32 else v.cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("mlm_head.") and not k.startswith("kc_decoders.")
        }
        torch.save(state_dict, os.path.join(path, "model.pt"))
    else:
        state_dict = {
            k: v.cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("mlm_head.") and not k.startswith("kc_decoders.")
        }
        torch.save(state_dict, os.path.join(path, "model.pt"))

    # Save tokenizer
    tokenizer.save(os.path.join(path, "tokenizer.json"))

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
    if importlib.util.find_spec("numpy"):
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
    if "numpy" in states and importlib.util.find_spec("numpy"):
        import numpy as np

        np.random.set_state(states["numpy"])


def save_training_state(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, Any],
    global_step: int = 0,
    batch_idx: int = 0,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[Any] = None,
    config: Optional[Any] = None,
    filename: str = "checkpoint.pt",
) -> None:
    """Generic training state save."""
    # pylint: disable=too-many-positional-arguments
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "rng_states": get_rng_states(),
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
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
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler: Optional[Any] = None,
    filename: str = "checkpoint.pt",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Generic training state load."""
    # pylint: disable=too-many-positional-arguments
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No checkpoint at {full_path}")

    checkpoint = torch.load(full_path, map_location=device, weights_only=False)

    # Load model weights
    model_state = checkpoint["model_state_dict"]
    # Handle DDP 'module.' prefix
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state)

    # Load optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scaler
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Load scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG
    if "rng_states" in checkpoint:
        set_rng_states(checkpoint["rng_states"])

    return cast(Dict[str, Any], checkpoint)
