"""I/O utilities for model checkpoints and weights."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

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
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Save label mappings
    formality_label_map = {k.value: v for k, v in FORMALITY_LABEL_TO_ID.items()}
    gender_label_map = {k.value: v for k, v in GENDER_LABEL_TO_ID.items()}
    grammaticality_label_map = {"agrammatic": 0, "grammatic": 1}
    with open(os.path.join(path, "labels.json"), "w") as f:
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
    with open(os.path.join(path, "model_type.txt"), "w") as f:
        f.write("style-multitask")


def save_checkpoint(
    path: str,
    model: nn.Module,
    tokenizer: Tokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: Dict[str, List[float]],
    best_val_loss: float,
    patience_counter: int,
    best_state: Optional[Dict[str, torch.Tensor]],
    args: Any,
    model_config: ModelConfig,
    is_best: bool = False,
) -> None:
    """Save training checkpoint for resumption."""
    # Note: caller should check is_main_process()
    os.makedirs(path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "best_state": best_state,
        "args": vars(args) if hasattr(args, "__dict__") else args,
    }
    torch.save(checkpoint, os.path.join(path, "checkpoint.pt"))

    # Also save tokenizer and config
    tokenizer.save(os.path.join(path, "tokenizer.json"))
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(model_config.to_dict(), f, indent=2)


def load_checkpoint(
    path: str,
    device: Optional[str] = None,
) -> Tuple[StyleClassifier, Tokenizer, Dict[str, Any], bool]:
    """Load training checkpoint for resumption."""
    checkpoint_path = os.path.join(path, "checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Load config and tokenizer
    with open(os.path.join(path, "config.json"), "r") as f:
        config_dict = json.load(f)
    config = ModelConfig.from_dict(config_dict)
    tokenizer = Tokenizer.load(os.path.join(path, "tokenizer.json"))

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")

    # Reconstruct model
    model = StyleClassifier(config)

    # Filter out MLM/KC head weights
    model_state = checkpoint["model_state_dict"]
    # Strip 'module.' prefix if present
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    model_state = {
        k: v
        for k, v in model_state.items()
        if not k.startswith("mlm_head.") and not k.startswith("kc_decoders.")
    }

    try:
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            compatible_state = {}
            current_state = model.state_dict()
            for k, v in model_state.items():
                if k in current_state and v.shape == current_state[k].shape:
                    compatible_state[k] = v
                elif k not in current_state:
                    compatible_state[k] = v
            missing_keys, unexpected_keys = model.load_state_dict(
                compatible_state, strict=False
            )
        else:
            raise e

    if device:
        model.to(device)

    return model, tokenizer, checkpoint, bool(missing_keys or unexpected_keys)
