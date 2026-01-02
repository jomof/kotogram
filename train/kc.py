"""KC target computation logic."""

from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from kotogram.exceptions import MissingMappingError
from kotogram.japanese_parser import POS_MAP

# KC Configuration
KC_HASH_BUCKETS = 16384
KC_NGRAM_ORDER = 3
KC_POS_BIASED_WINDOW = 5

# Verification Checklist:
# - Verify label.py Phase 1 injects "<READING_MASK>" into reading vocabulary
# - Verify kc.py derived reading_gram logic is called during Phase 2 labeling
# - Verify StyleDataset._init_features mmaps "reading" and "pos", populating feature_ids for kc.py

# Constants for domain separation (SALT) to reduce accidental collisions between different feature families
SALT = {
    "ngram_pos": 101,
    "ngram_pos_detail_1": 102,
    "ngram_conjugated_form": 103,
    "ngram_conjugated_type": 104,
    "ngram_reading_gram": 105,
    "tail_ngram_pos": 201,
    "tail_ngram_pos_detail_1": 202,
    "tail_ngram_conjugated_form": 203,
    "tail_ngram_conjugated_type": 204,
    "tail_ngram_reading_gram": 205,
    "pair_pos_conj": 301,
    "pair_pos1_conjform": 302,
    "pair_pos1_conjtype": 303,
}


def _salt(key: str) -> int:
    """Robust lookup for SALT keys with descriptive error messages."""
    val = SALT.get(key)
    if val is None:
        raise MissingMappingError(
            map_name="SALT", key=key, context=f"Available keys={sorted(SALT.keys())}"
        )
    return val


def _mix64(x: int) -> int:
    """Deterministic 64-bit integer mix (splitmix64 style)."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x &= 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    x &= 0xFFFFFFFFFFFFFFFF
    x = x ^ (x >> 31)
    return x & 0xFFFFFFFFFFFFFFFF


def stable_hash_ints(ints: Sequence[int]) -> int:
    """Deterministic hash of a sequence of integers with fixed seed and mixing."""
    # 0x9E3779B97F4A7C15 is a common 64-bit magic constant (golden ratio fraction)
    h = 0x9E3779B97F4A7C15
    for i in ints:
        # Convert to unsigned 64-bit; handle potential negative IDs gracefully
        u = i & 0xFFFFFFFFFFFFFFFF
        # Add constant per element to avoid structured behavior on sequences like [0, 0, 0]
        h = (h + u + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        h = _mix64(h)
    return h


# Whitelist for grammar-heavy POS to keep reading.
# Content-heavy POS (noun, adj, etc.) will have their reading masked.
GRAMMAR_POS_WHITELIST = {
    "particle",
    "aux-verb",
    "prefix",
    "suffix",
    "adnom",
    "conj",
}


def derive_reading_gram_ids(
    feature_ids: Dict[str, List[int]],
    tokenizer: Any,
) -> List[int]:
    """Derive reading_gram IDs (kept for grammar POS, masked for content POS)."""
    if "reading" not in feature_ids or "pos" not in feature_ids or not tokenizer:
        return []

    r_ids = feature_ids["reading"]
    p_ids = feature_ids["pos"]

    if len(r_ids) != len(p_ids):
        return []

    # Get the ID for the special mask sentinel, falling back to unk_id.
    # Fail safe via explicit check instead of try-except (KeyError forbidden).
    mask_id = getattr(tokenizer, "unk_id", 0)
    if "reading" in tokenizer.field_vocabs:
        reading_vocab = tokenizer.field_vocabs["reading"]
        if "<READING_MASK>" in reading_vocab:
            mask_id = reading_vocab["<READING_MASK>"]

    # Efficient reverse lookup for POS strings with fingerprint-based cache invalidation.
    # Slightly stronger than (len,max): catches reshuffles that keep len/max stable.
    # We cache this on the tokenizer instance to avoid rebuilding it per sentence.
    pos_vocab = tokenizer.field_vocabs.get("pos", {})
    ids = pos_vocab.values()
    fp = (
        len(pos_vocab),
        max(ids, default=-1),
        sum(ids) if pos_vocab else 0,
    )
    if getattr(tokenizer, "_rev_pos_cache_fingerprint", None) != fp:
        rev_pos = {v: k for k, v in pos_vocab.items()}
        setattr(tokenizer, "_rev_pos_cache", rev_pos)
        setattr(tokenizer, "_rev_pos_cache_fingerprint", fp)
    else:
        rev_pos = getattr(tokenizer, "_rev_pos_cache", {})

    derived = []
    for r_id, p_id in zip(r_ids, p_ids):
        # Resolve POS ID back to string.
        pos_raw = rev_pos.get(p_id, "")
        # Normalize if it's a Japanese label, otherwise keep for checking whitelist.
        pos_norm = POS_MAP.get(pos_raw, pos_raw)

        if pos_norm in GRAMMAR_POS_WHITELIST:
            derived.append(r_id)
        else:
            derived.append(mask_id)
    return derived


def _compute_bag_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[str, Any]
) -> None:
    for field in [
        "reading_gram",
        "pos",
        "pos_detail_1",
        "conjugated_form",
        "conjugated_type",
    ]:
        if field in feature_ids:
            targets[f"bag_{field}"] = sorted(set(feature_ids[field]))

    for field in [
        "reading_gram",
        "pos",
        "pos_detail_1",
        "conjugated_form",
        "conjugated_type",
    ]:
        if field in feature_ids:
            ids = feature_ids[field]
            tail_ids = ids[-KC_POS_BIASED_WINDOW:] if len(ids) > 0 else []
            targets[f"tail_{field}"] = sorted(set(tail_ids))


def _compute_ngram_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[str, Any]
) -> None:
    for field in [
        "pos",
        "pos_detail_1",
        "conjugated_form",
        "conjugated_type",
        "reading_gram",
    ]:
        if field in feature_ids:
            ids = feature_ids[field]
            hashes = set()
            salt = _salt(f"ngram_{field}")
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(ids) >= n_val:
                    for i in range(len(ids) - n_val + 1):
                        ngram = ids[i : i + n_val]
                        # Prepend salt for domain separation
                        h = stable_hash_ints([salt, *ngram]) % KC_HASH_BUCKETS
                        hashes.add(h)
            targets[f"ngram_{field}"] = sorted(hashes)


def _compute_tail_ngram_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[str, Any]
) -> None:
    """Compute n-gram targets biased toward the end of the sentence."""
    for field in [
        "pos",
        "pos_detail_1",
        "conjugated_form",
        "conjugated_type",
        "reading_gram",
    ]:
        if field in feature_ids:
            ids = feature_ids[field]
            tail_ids = ids[-KC_POS_BIASED_WINDOW:] if len(ids) > 0 else []
            if not tail_ids:
                continue

            hashes = set()
            salt = _salt(f"tail_ngram_{field}")
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(tail_ids) >= n_val:
                    for i in range(len(tail_ids) - n_val + 1):
                        ngram = tail_ids[i : i + n_val]
                        h = stable_hash_ints([salt, *ngram]) % KC_HASH_BUCKETS
                        hashes.add(h)
            targets[f"tail_ngram_{field}"] = sorted(hashes)


def _compute_pair_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[str, Any]
) -> None:
    # pair_pos_conj (pos, conjugated_form)
    if "pos" in feature_ids and "conjugated_form" in feature_ids:
        p_ids = feature_ids["pos"]
        c_ids = feature_ids["conjugated_form"]
        if len(p_ids) == len(c_ids):
            pair_hashes = set()
            salt = _salt("pair_pos_conj")
            for i, p_id in enumerate(p_ids):
                h = stable_hash_ints([salt, p_id, c_ids[i]]) % KC_HASH_BUCKETS
                pair_hashes.add(h)
            targets["pair_pos_conj"] = sorted(pair_hashes)

    # pair_pos1_conjform (pos_detail_1, conjugated_form)
    if "pos_detail_1" in feature_ids and "conjugated_form" in feature_ids:
        p_ids = feature_ids["pos_detail_1"]
        c_ids = feature_ids["conjugated_form"]
        if len(p_ids) == len(c_ids):
            pair_hashes = set()
            salt = _salt("pair_pos1_conjform")
            for i, p_id in enumerate(p_ids):
                h = stable_hash_ints([salt, p_id, c_ids[i]]) % KC_HASH_BUCKETS
                pair_hashes.add(h)
            targets["pair_pos1_conjform"] = sorted(pair_hashes)

    # pair_pos1_conjtype (pos_detail_1, conjugated_type)
    if "pos_detail_1" in feature_ids and "conjugated_type" in feature_ids:
        p_ids = feature_ids["pos_detail_1"]
        c_ids = feature_ids["conjugated_type"]
        if len(p_ids) == len(c_ids):
            pair_hashes = set()
            salt = _salt("pair_pos1_conjtype")
            for i, p_id in enumerate(p_ids):
                h = stable_hash_ints([salt, p_id, c_ids[i]]) % KC_HASH_BUCKETS
                pair_hashes.add(h)
            targets["pair_pos1_conjtype"] = sorted(pair_hashes)


def compute_kc_targets(
    feature_ids: Dict[str, Union[List[int], "torch.Tensor"]],
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute KC targets from feature IDs."""
    # Ensure inputs are lists, not tensors
    feature_ids_list: Dict[str, List[int]] = {}
    for k, val in feature_ids.items():
        # Using isinstance(val, torch.Tensor) is safer than hasattr(val, "tolist")
        # to ensure we don't accidentally call it on non-tensor types.
        if isinstance(val, torch.Tensor):
            feature_ids_list[k] = val.tolist()
        else:
            feature_ids_list[k] = list(val)  # type: ignore

    # Derive reading_gram IDs downstream if tokenizer is provided.
    if tokenizer:
        rg_ids = derive_reading_gram_ids(feature_ids_list, tokenizer)
        if rg_ids:
            feature_ids_list["reading_gram"] = rg_ids

    # Create targets dict
    targets: Dict[str, Any] = {}

    _compute_bag_targets(feature_ids_list, targets)
    _compute_ngram_targets(feature_ids_list, targets)
    _compute_tail_ngram_targets(feature_ids_list, targets)
    _compute_pair_targets(feature_ids_list, targets)

    return targets
