"""I/O utilities for model checkpoints and weights."""

import json
import os

import torch

from kotogram.constants import (
    FORMALITY_LABEL_TO_ID,
    GENDER_LABEL_TO_ID,
)
from kotogram.model import (
    InferenceClassifier,
    ModelConfig,
)
from kotogram.tokenizer import ENCODER_FEATURE_FIELDS, Tokenizer
from train.kc import KC_FAMILIES, KcMseFamily


def save_tokenizer(
    tokenizer: Tokenizer, path: str, inference_only: bool = False
) -> None:
    """Save tokenizer vocabularies to JSON file atomically."""
    # pylint: disable=import-outside-toplevel
    if inference_only:
        data = {
            "field_vocabs": {
                f: tokenizer.field_vocabs[f]
                for f in ENCODER_FEATURE_FIELDS
                if f in tokenizer.field_vocabs
            },
            "frozen": True,
        }
    else:
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
            json.dump(data, tmp_file, ensure_ascii=False, indent=2, sort_keys=True)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_model(
    model: InferenceClassifier,
    path: str,
    config: ModelConfig,
) -> None:
    """Save trained model, tokenizer, and config."""
    # pylint: disable=too-many-positional-arguments, too-many-locals

    def _has_needed_mse_families() -> bool:
        """Check if any MSE family decoder should be kept."""
        return any(
            isinstance(fam, KcMseFamily) and not fam.is_slim_decoder
            for fam in KC_FAMILIES.values()
        )

    def _has_needed_label_families() -> bool:
        """Check if any label (non-MSE) family decoder should be kept."""
        return any(
            not isinstance(fam, KcMseFamily) and not fam.is_slim_decoder
            for fam in KC_FAMILIES.values()
        )

    def _should_strip_key(key: str) -> bool:
        """Returns True if this key should be stripped from slim model."""
        if not key.startswith("kc_decoders."):
            return False

        parts = key.split(".")
        if len(parts) < 3:
            raise ValueError(f"Unexpected kc_decoders key pattern: {key}")

        sublayer = parts[1]

        # Check pathway layers (MSE and label)
        pathway_checks = {
            ("mse_hidden1", "mse_hidden2", "tanh"): _has_needed_mse_families,
            (
                "label_hidden1",
                "label_hidden2",
                "activation",
            ): _has_needed_label_families,
        }

        for layer_names, check_func in pathway_checks.items():
            if sublayer in layer_names:
                return not check_func()  # Strip if no families need this pathway

        # Recon pathway is training-only (always strip from exported models)
        if sublayer in (
            "recon_pos_embed",
            "recon_hidden1",
            "recon_hidden2",
            "recon_decoders",
        ):
            return True

        # Grammar point pathway (always keep for inference)
        if sublayer in ("gp_hidden", "gp_decoder"):
            return False

        # Handle per-family decoders
        if sublayer in ("decoders", "mse_decoders"):
            if len(parts) < 4:
                raise ValueError(
                    f"Unexpected kc_decoders.{sublayer} key pattern: {key}"
                )
            family_name = parts[2]
            for fid, fam in KC_FAMILIES.items():
                if fid.name.lower() == family_name:
                    return fam.is_slim_decoder
            raise ValueError(
                f"Unknown KC family in state_dict: {family_name} (key: {key})"
            )

        raise ValueError(f"Unexpected kc_decoders sublayer: {sublayer} (key: {key})")

    os.makedirs(path, exist_ok=True)

    # Save model weights (Always use FP8 if available)
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("FP8 requires PyTorch 2.1+.")

    state_dict = {
        k: v.cpu().to(torch.float8_e4m3fn) if v.dtype == torch.float32 else v.cpu()
        for k, v in model.state_dict().items()
        if not _should_strip_key(k)
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

    # Verify model size (Strict Architecture Verification)
    # pylint: disable=import-outside-toplevel
    from train.pytorch_utils import verify_model_size

    # Check actual file size
    model_pt_path = os.path.join(path, "model.pt")
    actual_size = os.path.getsize(model_pt_path)

    # Policy check (raises RuntimeError on failure)
    verify_model_size(model, actual_size)

    # Mark as feature-based multi-task model
    with open(os.path.join(path, "model_type.txt"), "w", encoding="utf-8") as f:
        f.write("style-multitask")


def get_checkpoint_path() -> str:
    """Returns the path to the checkpoint file (.cache/checkpoint.pt)."""
    # pylint: disable=import-outside-toplevel
    from train.paths import get_cache_dir

    return os.path.join(get_cache_dir(), "checkpoint.pt")


_CHECKPOINT_SENTINEL_KEYS = frozenset(
    {
        "embedding.embeddings.surface.weight",
        "encoder.layers.0.self_attn.in_proj_weight",
        "pooler.query",
    }
)


def save_checkpoint(model: InferenceClassifier) -> None:
    """Save full model state as checkpoint for training resumption.

    Unlike save_model which strips KC decoders and converts to FP8,
    this saves the complete model state in FP32 for seamless resumption.
    """
    # pylint: disable=import-outside-toplevel
    from train.paths import get_cache_dir

    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    checkpoint_path = get_checkpoint_path()

    # Save full state dict in FP32 (no stripping, no fp8 conversion)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Guard: verify the model has production-level structure before writing.
    # This prevents test stubs or dummy models from silently corrupting
    # the real checkpoint (which would break training resumption).
    missing = _CHECKPOINT_SENTINEL_KEYS - state_dict.keys()
    if missing:
        raise RuntimeError(
            f"Refusing to save checkpoint: model state_dict is missing "
            f"production keys {sorted(missing)}. "
            f"This looks like a test stub, not a real model."
        )

    torch.save(state_dict, checkpoint_path)
