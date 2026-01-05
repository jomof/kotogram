"""KC target computation logic."""

from enum import Enum
from typing import Any, Dict, List, Sequence, Union

import torch

from kotogram.exceptions import MissingMappingError

# KC Configuration
KC_HASH_BUCKETS = 16384
KC_NGRAM_ORDER = 3
KC_POS_BIASED_WINDOW = 5

DEFAULT_HASH_BUCKET_SIZE = 16384


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

# Map from KcFamilyId to boolean indicating if it uses sparse features
FAMILY_IS_SPARSE: Dict[KcFamilyId, bool] = {
    # Bag (Dense)
    KcFamilyId.BAG_READING_GRAM: False,
    KcFamilyId.BAG_POS: False,
    KcFamilyId.BAG_POS_DETAIL_1: False,
    KcFamilyId.BAG_CONJUGATED_TYPE: False,
    # Tail (Dense)
    KcFamilyId.TAIL_READING_GRAM: False,
    KcFamilyId.TAIL_POS: False,
    KcFamilyId.TAIL_POS_DETAIL_1: False,
    KcFamilyId.TAIL_CONJUGATED_TYPE: False,
    # Ngram (Sparse)
    KcFamilyId.NGRAM_POS: True,
    KcFamilyId.NGRAM_POS_DETAIL_1: True,
    KcFamilyId.NGRAM_CONJUGATED_TYPE: True,
    KcFamilyId.NGRAM_READING_GRAM: True,
    # Tail Ngram (Sparse)
    KcFamilyId.TAIL_NGRAM_POS: True,
    KcFamilyId.TAIL_NGRAM_POS_DETAIL_1: True,
    KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE: True,
    KcFamilyId.TAIL_NGRAM_READING_GRAM: True,
}

SPARSE_FAMILY_NAMES = {
    f.name.lower() for f, is_sparse in FAMILY_IS_SPARSE.items() if is_sparse
}


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
            targets[family_id] = sorted(set(feature_ids[field]))


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
            targets[family_id] = sorted(set(tail_ids))


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
            ids = feature_ids[field]
            hashes = set()
            salt = _salt(family_id)
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(ids) >= n_val:
                    for i in range(len(ids) - n_val + 1):
                        ngram = ids[i : i + n_val]
                        # Prepend salt for domain separation
                        h = stable_hash_ints([salt, *ngram]) % KC_HASH_BUCKETS
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
            if not tail_ids:
                continue

            hashes = set()
            salt = _salt(family_id)
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(tail_ids) >= n_val:
                    for i in range(len(tail_ids) - n_val + 1):
                        ngram = tail_ids[i : i + n_val]
                        h = stable_hash_ints([salt, *ngram]) % KC_HASH_BUCKETS
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
