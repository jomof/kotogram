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
