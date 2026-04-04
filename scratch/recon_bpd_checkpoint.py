"""Checkpoint I/O and shared types for the BPD training pipeline.

This module owns serialization format and callback contracts.
Changes here do NOT invalidate the training script hash —
they are infrastructure, not training semantics.
"""

import os
import time
import concurrent.futures
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
    config: object = None  # TrainConfig — typed as object to avoid circular import
    run_name: str = ""  # human-readable experiment identifier


# Shared executor for backgrounding disk writes to avoid blocking the training loop.
_SAVE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def wait_for_checkpoints() -> None:
    """Block until all background checkpoint saves complete."""
    _SAVE_EXECUTOR.shutdown(wait=True)


def _do_save_disk(data: dict, path: str) -> None:
    """Background task: perform atomic disk write."""
    tmp_path = path + ".tmp"
    try:
        torch.save(data, tmp_path)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"  [ERROR] Background checkpoint save failed: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_checkpoint(checkpoint: TrainCheckpoint, path: str) -> None:
    """Capture a snapshot of training state and background the disk write."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Sync capture of model state to CPU to ensure consistency before
    # backgrounding. This blocks for ~100-200ms depending on model size
    # but avoids "torn" checkpoints if the next epoch starts immediately.
    # We do this for all state dicts to be safe.
    def _to_cpu(d: dict) -> dict:
        return {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in d.items()
        }

    data = {
        "model_state": _to_cpu(checkpoint.model_state),
        "optimizer_state": _to_cpu(checkpoint.optimizer_state),
        "scaler_state": checkpoint.scaler_state,  # Scaler/Scheduler are small/CPU-native
        "scheduler_state": checkpoint.scheduler_state,
        "epoch": checkpoint.epoch,
        "latest_metrics": dict(checkpoint.latest_metrics),
        "epoch_history": list(checkpoint.epoch_history),
    }

    # Background the slow part (SSD I/O)
    _SAVE_EXECUTOR.submit(_do_save_disk, data, path)


def load_checkpoint(path: str) -> Optional[TrainCheckpoint]:
    """Load a checkpoint from disk, or return None if not found."""
    if not os.path.exists(path):
        return None
    data = torch.load(path, weights_only=False, map_location=DEVICE)
    return TrainCheckpoint(**data)
