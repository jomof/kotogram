"""KC target computation logic."""

from typing import Any, Dict, List, Union

import torch

# KC Configuration
KC_HASH_BUCKETS = 16384
KC_NGRAM_ORDER = 3
KC_POS_BIASED_WINDOW = 5


def _compute_bag_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[str, Any]
) -> None:
    for field in ["lemma", "pos", "conjugated_form"]:
        if field in feature_ids:
            targets[f"bag_{field}"] = list(set(feature_ids[field]))

    for field in ["surface", "lemma", "pos", "conjugated_form"]:
        if field in feature_ids:
            ids = feature_ids[field]
            tail_ids = ids[-KC_POS_BIASED_WINDOW:] if len(ids) > 0 else []
            targets[f"tail_{field}"] = list(set(tail_ids))


def _compute_ngram_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[str, Any]
) -> None:
    for field in ["pos", "conjugated_form"]:
        if field in feature_ids:
            ids = feature_ids[field]
            hashes = set()
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(ids) >= n_val:
                    for i in range(len(ids) - n_val + 1):
                        ngram = tuple(ids[i : i + n_val])
                        h = hash(ngram) % KC_HASH_BUCKETS
                        hashes.add(h)
            targets[f"ngram_{field}"] = list(hashes)


def _compute_pair_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[str, Any]
) -> None:
    if "pos" in feature_ids and "conjugated_form" in feature_ids:
        p_ids = feature_ids["pos"]
        c_ids = feature_ids["conjugated_form"]
        if len(p_ids) == len(c_ids):
            pair_hashes = set()
            for i, p_id in enumerate(p_ids):
                pair = (p_id, c_ids[i])
                h = hash(pair) % KC_HASH_BUCKETS
                pair_hashes.add(h)
            targets["pair_pos_conj"] = list(pair_hashes)


def compute_kc_targets(
    feature_ids: Dict[str, Union[List[int], "torch.Tensor"]],
) -> Dict[str, Any]:
    """Compute KC targets from feature IDs."""
    # Optimization: Ensure inputs are lists, not tensors

    # Check if values are tensors
    keys = list(feature_ids.keys())
    for k in keys:
        val = feature_ids[k]
        if isinstance(val, torch.Tensor):
            feature_ids[k] = val.tolist()

    # Create targets dict
    targets: Dict[str, Any] = {}

    # Delegate to helpers
    # Note: feature_ids is now guaranteed to have List[int] values,
    # but type hint above is Union. My helpers expect Dict[str, List[int]].
    # Runtime check passes, type checker might complain strictly but this is Pylint.

    clean_feature_ids: Dict[str, List[int]] = feature_ids  # type: ignore

    _compute_bag_targets(clean_feature_ids, targets)
    _compute_ngram_targets(clean_feature_ids, targets)
    _compute_pair_targets(clean_feature_ids, targets)

    return targets
