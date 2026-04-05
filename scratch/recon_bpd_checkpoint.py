"""Checkpoint I/O and shared types for the BPD training pipeline.

This module owns serialization format and callback contracts.
Changes here do NOT invalidate the training script hash —
they are infrastructure, not training semantics.
"""

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

# ── Hardware detection (duplicated from recon_bpd for standalone use) ──
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


_ARCHITECTURE_KEYS = (
    "d_model",
    "ffn_dim",
    "kc_vocab_size",
    "num_heads",
    "num_layers",
    "recon_hidden_dim",
    "recon_pos_embed_dim",
)


def compute_architecture_hash(config: Any) -> str:
    """SHA256 of architecture-defining config keys for compatibility checks.

    Accepts a TrainConfig dataclass or a plain dict.
    """
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        d = dataclasses.asdict(config)
    else:
        d = dict(config)
    arch = {k: d[k] for k in _ARCHITECTURE_KEYS if k in d}
    return hashlib.sha256(json.dumps(arch, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TrainCheckpoint:
    """Opaque checkpoint for resuming a partially-completed training run.

    The caller should treat this as a black box — pass it back to ``train()``
    to resume from the last completed epoch.
    """

    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, object]
    scaler_state: Dict[str, object]
    scheduler_state: Dict[str, object]
    epoch: int  # last completed epoch (0-indexed)
    latest_metrics: Dict[str, float]
    epoch_history: list  # [(epoch, metrics_dict), ...]

    # Optional provenance metadata (None for legacy checkpoints).
    config_dict: Optional[Dict[str, Any]] = None
    dataset_id: Optional[str] = None
    chive_id: Optional[str] = None
    parent_checkpoint_id: Optional[str] = None


@dataclass
class EpochContext:
    """Read-only context passed to epoch-end callbacks for observability.

    Allows callbacks (e.g. reconstruction tests, MLflow artifact uploads)
    to inspect training state without the training loop importing
    observability code.

    Any code with access to the context can append file paths to
    ``artifact_paths``; the callback drains them for upload.
    """

    model: torch.nn.Module
    tokenizer: object  # Tokenizer — typed as object to avoid import
    device: torch.device
    temperature: float
    checkpoint_path: str
    config: object = None
    run_name: str = ""
    artifact_paths: List[str] = field(default_factory=list)


def save_checkpoint(checkpoint: TrainCheckpoint, path: str) -> None:
    """Persist a checkpoint to disk (atomic write)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    data: Dict[str, Any] = {
        "model_state": checkpoint.model_state,
        "optimizer_state": checkpoint.optimizer_state,
        "scaler_state": checkpoint.scaler_state,
        "scheduler_state": checkpoint.scheduler_state,
        "epoch": checkpoint.epoch,
        "latest_metrics": checkpoint.latest_metrics,
        "epoch_history": checkpoint.epoch_history,
    }
    if checkpoint.config_dict is not None:
        data["config_dict"] = checkpoint.config_dict
    if checkpoint.dataset_id is not None:
        data["dataset_id"] = checkpoint.dataset_id
    if checkpoint.chive_id is not None:
        data["chive_id"] = checkpoint.chive_id
    if checkpoint.parent_checkpoint_id is not None:
        data["parent_checkpoint_id"] = checkpoint.parent_checkpoint_id
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> Optional[TrainCheckpoint]:
    """Load a checkpoint from disk, or return None if not found.

    Backward-compatible: ignores unknown keys and supplies defaults
    for optional fields missing from legacy checkpoints.
    """
    if not os.path.exists(path):
        return None
    data = torch.load(path, weights_only=False, map_location=DEVICE)
    known = {f.name for f in dataclasses.fields(TrainCheckpoint)}
    filtered = {k: v for k, v in data.items() if k in known}
    return TrainCheckpoint(**filtered)
