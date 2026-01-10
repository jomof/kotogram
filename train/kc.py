"""KC target computation logic."""

from enum import Enum
from typing import Any, Dict, List, Sequence, Union

import torch

from kotogram.exceptions import MissingMappingError
from kotogram.tokenizer import CLS_ID, PAD_ID, UNK_ID

# KC Configuration
KC_NGRAM_ORDER = 3
KC_POS_BIASED_WINDOW = 5

# Special token IDs to exclude from KC targets
# CLS is non-discriminative as it appears in every sequence
# PAD and UNK are kept for analysis purposes
SPECIAL_TOKEN_IDS = {CLS_ID}


class KcFamilyId(str, Enum):
    """Canonical opaque IDs for all KC families."""

    # Bag Families
    BAG_READING_GRAM = "bag_reading_gram"
    BAG_POS = "bag_pos"
    BAG_POS_DETAIL_1 = "bag_pos_detail_1"
    BAG_CONJUGATED_TYPE = "bag_conjugated_type"

    # Tail Bag Families
    TAIL_READING_GRAM = "tail_reading_gram"
    TAIL_POS = "tail_pos"
    TAIL_POS_DETAIL_1 = "tail_pos_detail_1"
    TAIL_CONJUGATED_TYPE = "tail_conjugated_type"

    # Ngram Families
    NGRAM_POS = "ngram_pos"
    NGRAM_POS_DETAIL_1 = "ngram_pos_detail_1"
    NGRAM_CONJUGATED_TYPE = "ngram_conjugated_type"
    NGRAM_READING_GRAM = "ngram_reading_gram"

    # Tail Ngram Families
    TAIL_NGRAM_POS = "tail_ngram_pos"
    TAIL_NGRAM_POS_DETAIL_1 = "tail_ngram_pos_detail_1"
    TAIL_NGRAM_CONJUGATED_TYPE = "tail_ngram_conjugated_type"
    TAIL_NGRAM_READING_GRAM = "tail_ngram_reading_gram"


ALL_KC_FAMILIES = list(KcFamilyId)


# Per-family bucket sizes for sparse (ngram) families
# Based on collision analysis: reading_gram needs more buckets due to larger vocabulary
FAMILY_BUCKET_SIZES: Dict[KcFamilyId, int] = {
    KcFamilyId.NGRAM_POS: 2048,
    KcFamilyId.NGRAM_POS_DETAIL_1: 4096,
    KcFamilyId.NGRAM_CONJUGATED_TYPE: 8192,
    KcFamilyId.NGRAM_READING_GRAM: 262144,  # 2^18: 234K unique ngrams
    KcFamilyId.TAIL_NGRAM_POS: 2048,
    KcFamilyId.TAIL_NGRAM_POS_DETAIL_1: 4096,
    KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE: 8192,
    KcFamilyId.TAIL_NGRAM_READING_GRAM: 131072,  # 2^17: 115K unique ngrams
}


def get_family_bucket_size(family_id: KcFamilyId) -> int:
    """Get the hash bucket size for a sparse KC family.

    Raises:
        KeyError: If family_id is not a sparse ngram family.
    """
    return FAMILY_BUCKET_SIZES[family_id]


def is_family_sparse(family: KcFamilyId) -> bool:
    """Check if a KC family uses sparse (hash-based) features.

    Dense families (Bag, Tail) use actual tokenizer vocab IDs.
    Sparse families (Ngram, Tail Ngram) use hash-based n-gram features.

    Args:
        family: The KC family to check.

    Returns:
        True if the family uses sparse features, False for dense.

    Raises:
        ValueError: If the family is not recognized.
    """
    # Dense families (use actual tokenizer vocab)
    if family in (
        KcFamilyId.BAG_READING_GRAM,
        KcFamilyId.BAG_POS,
        KcFamilyId.BAG_POS_DETAIL_1,
        KcFamilyId.BAG_CONJUGATED_TYPE,
        KcFamilyId.TAIL_READING_GRAM,
        KcFamilyId.TAIL_POS,
        KcFamilyId.TAIL_POS_DETAIL_1,
        KcFamilyId.TAIL_CONJUGATED_TYPE,
    ):
        return False

    # Sparse families (use hash-based n-gram features)
    if family in (
        KcFamilyId.NGRAM_POS,
        KcFamilyId.NGRAM_POS_DETAIL_1,
        KcFamilyId.NGRAM_CONJUGATED_TYPE,
        KcFamilyId.NGRAM_READING_GRAM,
        KcFamilyId.TAIL_NGRAM_POS,
        KcFamilyId.TAIL_NGRAM_POS_DETAIL_1,
        KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE,
        KcFamilyId.TAIL_NGRAM_READING_GRAM,
    ):
        return True

    raise ValueError(f"Unknown KC family: {family}")


# Map from KcFamilyId to the input feature field name it consumes
FAMILY_FEATURES: Dict[KcFamilyId, str] = {
    # Bag
    KcFamilyId.BAG_READING_GRAM: "reading_gram",
    KcFamilyId.BAG_POS: "pos",
    KcFamilyId.BAG_POS_DETAIL_1: "pos_detail_1",
    KcFamilyId.BAG_CONJUGATED_TYPE: "conjugated_type",
    # Tail
    KcFamilyId.TAIL_READING_GRAM: "reading_gram",
    KcFamilyId.TAIL_POS: "pos",
    KcFamilyId.TAIL_POS_DETAIL_1: "pos_detail_1",
    KcFamilyId.TAIL_CONJUGATED_TYPE: "conjugated_type",
    # Ngram
    KcFamilyId.NGRAM_POS: "pos",
    KcFamilyId.NGRAM_POS_DETAIL_1: "pos_detail_1",
    KcFamilyId.NGRAM_CONJUGATED_TYPE: "conjugated_type",
    KcFamilyId.NGRAM_READING_GRAM: "reading_gram",
    # Tail Ngram
    KcFamilyId.TAIL_NGRAM_POS: "pos",
    KcFamilyId.TAIL_NGRAM_POS_DETAIL_1: "pos_detail_1",
    KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE: "conjugated_type",
    KcFamilyId.TAIL_NGRAM_READING_GRAM: "reading_gram",
}


# Constants for domain separation (SALT) to reduce accidental collisions between different feature families
SALT: Dict[KcFamilyId, int] = {
    KcFamilyId.NGRAM_POS: 101,
    KcFamilyId.NGRAM_POS_DETAIL_1: 102,
    KcFamilyId.NGRAM_CONJUGATED_TYPE: 104,
    KcFamilyId.NGRAM_READING_GRAM: 105,
    KcFamilyId.TAIL_NGRAM_POS: 201,
    KcFamilyId.TAIL_NGRAM_POS_DETAIL_1: 202,
    KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE: 204,
    KcFamilyId.TAIL_NGRAM_READING_GRAM: 205,
}


def _salt(key: KcFamilyId) -> int:
    """Robust lookup for SALT keys with descriptive error messages."""
    val = SALT.get(key)
    if val is None:
        raise MissingMappingError(
            map_name="SALT",
            key=str(key),
            context=f"Available keys={[k.name for k in SALT]}",
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


def _compute_bag_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[KcFamilyId, Any]
) -> None:
    # Bag families
    bag_families = {
        KcFamilyId.BAG_READING_GRAM,
        KcFamilyId.BAG_POS,
        KcFamilyId.BAG_POS_DETAIL_1,
        KcFamilyId.BAG_CONJUGATED_TYPE,
    }

    for family_id in bag_families:
        field = FAMILY_FEATURES[family_id]
        if field in feature_ids:
            # Exclude special tokens (CLS only - PAD/UNK kept for analysis)
            filtered = [v for v in feature_ids[field] if v not in SPECIAL_TOKEN_IDS]
            # For pos_detail_1 and conjugated_type, also filter out UNK
            # These fields can have UNK when morphology is ambiguous
            if family_id in (
                KcFamilyId.BAG_POS_DETAIL_1,
                KcFamilyId.BAG_CONJUGATED_TYPE,
            ):
                filtered = [v for v in filtered if v != UNK_ID]
            targets[family_id] = sorted(set(filtered))

    # Validate: BAG_READING_GRAM should never contain special tokens
    # All reading_gram values should be in vocabulary after masking logic
    if KcFamilyId.BAG_READING_GRAM in targets:
        if UNK_ID in targets[KcFamilyId.BAG_READING_GRAM]:
            raise RuntimeError(
                "BAG_READING_GRAM contains UNK token. "
                "This indicates a reading_gram value is not in the vocabulary. "
                "Check the GRAMMAR_POS_WHITELIST in masking.py or re-run labeling."
            )
        if PAD_ID in targets[KcFamilyId.BAG_READING_GRAM]:
            raise RuntimeError(
                "BAG_READING_GRAM contains PAD token. "
                "This indicates an empty reading_gram value. "
                "Check extract_token_features in kotogram.py."
            )
        if CLS_ID in targets[KcFamilyId.BAG_READING_GRAM]:
            raise RuntimeError(
                "BAG_READING_GRAM contains CLS token. "
                "CLS should be filtered by SPECIAL_TOKEN_IDS."
            )


def _compute_tail_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[KcFamilyId, Any]
) -> None:
    # Tail families
    tail_families = {
        KcFamilyId.TAIL_READING_GRAM,
        KcFamilyId.TAIL_POS,
        KcFamilyId.TAIL_POS_DETAIL_1,
        # KcFamilyId.TAIL_CONJUGATED_FORM,
        KcFamilyId.TAIL_CONJUGATED_TYPE,
    }

    for family_id in tail_families:
        field = FAMILY_FEATURES[family_id]
        if field in feature_ids:
            ids = feature_ids[field]
            tail_ids = ids[-KC_POS_BIASED_WINDOW:] if len(ids) > 0 else []
            # Exclude special tokens (CLS)
            filtered = [v for v in tail_ids if v not in SPECIAL_TOKEN_IDS]
            # For pos_detail_1 and conjugated_type, also filter out UNK
            # These fields can have UNK when morphology is ambiguous
            if family_id in (
                KcFamilyId.TAIL_POS_DETAIL_1,
                KcFamilyId.TAIL_CONJUGATED_TYPE,
            ):
                filtered = [v for v in filtered if v != UNK_ID]
            targets[family_id] = sorted(set(filtered))


def _compute_ngram_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[KcFamilyId, Any]
) -> None:
    # Ngram families
    ngram_families = {
        KcFamilyId.NGRAM_POS,
        KcFamilyId.NGRAM_POS_DETAIL_1,
        KcFamilyId.NGRAM_CONJUGATED_TYPE,
        KcFamilyId.NGRAM_READING_GRAM,
    }

    for family_id in ngram_families:
        field = FAMILY_FEATURES[family_id]
        if field in feature_ids:
            # Filter out special tokens before computing ngrams
            ids = [v for v in feature_ids[field] if v not in SPECIAL_TOKEN_IDS]
            hashes = set()
            salt = _salt(family_id)
            bucket_size = get_family_bucket_size(family_id)
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(ids) >= n_val:
                    for i in range(len(ids) - n_val + 1):
                        ngram = ids[i : i + n_val]
                        # Prepend salt for domain separation
                        h = stable_hash_ints([salt, *ngram]) % bucket_size
                        hashes.add(h)
            targets[family_id] = sorted(hashes)


def _compute_tail_ngram_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[KcFamilyId, Any]
) -> None:
    """Compute n-gram targets biased toward the end of the sentence."""
    tail_ngram_families = {
        KcFamilyId.TAIL_NGRAM_POS,
        KcFamilyId.TAIL_NGRAM_POS_DETAIL_1,
        KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE,
        KcFamilyId.TAIL_NGRAM_READING_GRAM,
    }

    for family_id in tail_ngram_families:
        field = FAMILY_FEATURES[family_id]
        if field in feature_ids:
            ids = feature_ids[field]
            tail_ids = ids[-KC_POS_BIASED_WINDOW:] if len(ids) > 0 else []
            # Filter out special tokens
            tail_ids = [v for v in tail_ids if v not in SPECIAL_TOKEN_IDS]
            if not tail_ids:
                continue

            hashes = set()
            salt = _salt(family_id)
            bucket_size = get_family_bucket_size(family_id)
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(tail_ids) >= n_val:
                    for i in range(len(tail_ids) - n_val + 1):
                        ngram = tail_ids[i : i + n_val]
                        h = stable_hash_ints([salt, *ngram]) % bucket_size
                        hashes.add(h)
            targets[family_id] = sorted(hashes)


def compute_kc_targets(
    feature_ids: Dict[str, Union[List[int], "torch.Tensor"]],
) -> Dict[KcFamilyId, Any]:
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

    # Create targets dict
    targets: Dict[KcFamilyId, Any] = {}

    _compute_bag_targets(feature_ids_list, targets)
    _compute_tail_targets(feature_ids_list, targets)
    _compute_ngram_targets(feature_ids_list, targets)
    _compute_tail_ngram_targets(feature_ids_list, targets)
    _compute_tail_ngram_targets(feature_ids_list, targets)

    return targets
