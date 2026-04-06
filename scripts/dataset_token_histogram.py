"""Token-count histogram derived from a dataset bundle (embedded in ds-*.pt)."""

from typing import Any, Dict

import numpy as np
import torch


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


def grammatical_token_gram_freq(bundle: Dict[str, Any]) -> np.ndarray:
    """Per-token position-frequency across grammatical sentences.

    Returns a int64 array of shape ``[V]`` where ``V`` is the surface vocab
    size.  Entry ``i`` counts how many times surface token ID ``i`` appears
    across all token positions in gram==1 sentences.  Used by the token
    percentile reduction to decide which tokens to keep.
    """
    surface_vocab_size = len(bundle.get("vocab", {}).get("surface", {}))
    surface_ids = bundle["features"]["surface"]
    offsets = bundle["offsets"]
    n = int(offsets.shape[0]) - 1
    gram = bundle["labels"].get("gram")

    counts = np.zeros(surface_vocab_size, dtype=np.int64)
    for i in range(n):
        if gram is not None and gram[i].item() != 1:
            continue
        start = int(offsets[i].item())
        end = int(offsets[i + 1].item())
        ids = surface_ids[start:end].numpy().astype(np.int64)
        for tid in ids:
            counts[tid] += 1
    return counts
