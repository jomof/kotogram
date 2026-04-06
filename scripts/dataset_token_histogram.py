"""Dense token-count histogram for grammatical sentences (dataset build + training)."""

import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from scripts.gcs import gcs_download_file, gcs_exists

# Must match scripts.dataset.GCS_PREFIX (avoid importing dataset -> circular).
_GCS_PREFIX = "kotogram-datasets"
TOKEN_LEN_HIST_PREFIX = "toklen-hist-"
_LOCAL_CACHE = os.path.join(".cache", "datasets")


def token_length_histogram_path(dataset_id: str) -> str:
    """Local path to the token-length histogram .npy (next to ds-{id}.pt)."""
    return os.path.join(_LOCAL_CACHE, f"{TOKEN_LEN_HIST_PREFIX}{dataset_id}.npy")


def grammatical_token_length_counts(bundle: Dict[str, Any]) -> np.ndarray:
    """Histogram of token counts for grammatical (gram==1) sentences only."""
    offsets = bundle["offsets"]
    n = int(offsets.shape[0]) - 1
    diffs = (offsets[1 : n + 1] - offsets[:n]).to(torch.int64).cpu().numpy()
    gram = bundle["labels"].get("gram")
    if gram is not None:
        g = gram[:n].cpu().numpy().astype(bool)
        diffs = diffs[g]
    if diffs.size == 0:
        return np.zeros(1, dtype=np.uint64)
    max_t = int(diffs.max())
    return np.bincount(diffs, minlength=max_t + 1).astype(np.uint64)


def save_token_length_histogram(bundle: Dict[str, Any], dataset_id: str) -> str:
    """Write token histogram .npy next to the dataset bundle; returns path."""
    counts = grammatical_token_length_counts(bundle)
    path = token_length_histogram_path(dataset_id)
    os.makedirs(_LOCAL_CACHE, exist_ok=True)
    np.save(path, counts)
    return path


def ensure_token_length_histogram_local(dataset_id: str) -> Optional[str]:
    """Return local histogram path, downloading from GCS if absent."""
    path = token_length_histogram_path(dataset_id)
    if os.path.exists(path):
        return path
    key = f"{_GCS_PREFIX}/datasets/{TOKEN_LEN_HIST_PREFIX}{dataset_id}.npy"
    if gcs_exists(key):
        print(f"  Downloading token histogram {dataset_id}...")
        gcs_download_file(key, path)
        return path
    return None
