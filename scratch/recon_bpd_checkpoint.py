"""Checkpoint I/O and shared types for the BPD training pipeline.

This module owns serialization format and callback contracts.
Changes here do NOT invalidate the training script hash —
they are infrastructure, not training semantics.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

# ── Hardware detection (duplicated from recon_bpd for standalone use) ──
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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
    artifact_paths: List[str] = field(default_factory=list)


def save_checkpoint(checkpoint: TrainCheckpoint, path: str) -> None:
    """Persist a checkpoint to disk (atomic write)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(
        {
            "model_state": checkpoint.model_state,
            "optimizer_state": checkpoint.optimizer_state,
            "scaler_state": checkpoint.scaler_state,
            "scheduler_state": checkpoint.scheduler_state,
            "epoch": checkpoint.epoch,
            "latest_metrics": checkpoint.latest_metrics,
            "epoch_history": checkpoint.epoch_history,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> Optional[TrainCheckpoint]:
    """Load a checkpoint from disk, or return None if not found."""
    if not os.path.exists(path):
        return None
    data = torch.load(path, weights_only=False, map_location=DEVICE)
    return TrainCheckpoint(**data)
