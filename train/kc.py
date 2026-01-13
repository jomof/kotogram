"""KC target computation logic."""

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Union

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

# compound_1 tokens to exclude from all tail families.
# These tokens are high-frequency but low-information for style discrimination.
# Format: composite token strings matching compound_1 vocabulary (e.g., "noun:common-noun")
TAIL_DISALLOW = frozenset(
    {
        "noun:common-noun",
        "noun:numeral",
        "aux-symbol:comma",
        "aux-symbol:period",
        "aux-symbol:general",
        "aux-symbol:close-bracket",
        "aux-symbol:open-bracket",
        "suffix:nominal",
        "noun:proper-noun",
    }
)


class KcFamilyId(str, Enum):
    """Canonical opaque IDs for all KC families."""

    # Bag Families
    BAG_READING_GRAM = "bag_reading_gram"
    BAG_POS = "bag_pos"
    BAG_COMPOUND_1 = "bag_compound_1"
    BAG_COMPOUND_2 = "bag_compound_2"
    BAG_CONJUGATED_TYPE = "bag_conjugated_type"

    # Tail Bag Families
    TAIL_READING_GRAM = "tail_reading_gram"
    TAIL_POS = "tail_pos"
    TAIL_COMPOUND_1 = "tail_compound_1"
    TAIL_COMPOUND_2 = "tail_compound_2"
    TAIL_CONJUGATED_TYPE = "tail_conjugated_type"

    # Ngram Families
    NGRAM_POS = "ngram_pos"
    NGRAM_COMPOUND_1 = "ngram_compound_1"
    NGRAM_COMPOUND_2 = "ngram_compound_2"
    NGRAM_CONJUGATED_TYPE = "ngram_conjugated_type"
    NGRAM_READING_GRAM = "ngram_reading_gram"

    # Tail Ngram Families
    TAIL_NGRAM_POS = "tail_ngram_pos"
    TAIL_NGRAM_COMPOUND_1 = "tail_ngram_compound_1"
    TAIL_NGRAM_COMPOUND_2 = "tail_ngram_compound_2"
    TAIL_NGRAM_CONJUGATED_TYPE = "tail_ngram_conjugated_type"
    TAIL_NGRAM_READING_GRAM = "tail_ngram_reading_gram"


ALL_KC_FAMILIES = list(KcFamilyId)


# Per-family bucket sizes for sparse (ngram) families
# Based on collision analysis: reading_gram needs more buckets due to larger vocabulary
FAMILY_BUCKET_SIZES: Dict[KcFamilyId, int] = {
    KcFamilyId.NGRAM_POS: 2048,
    KcFamilyId.NGRAM_COMPOUND_1: 8192 * 4,
    KcFamilyId.NGRAM_COMPOUND_2: 4096,
    KcFamilyId.NGRAM_CONJUGATED_TYPE: 8192,
    KcFamilyId.NGRAM_READING_GRAM: 262144,  # 2^18: 234K unique ngrams
    KcFamilyId.TAIL_NGRAM_POS: 1024,
    KcFamilyId.TAIL_NGRAM_COMPOUND_1: 8192 * 2,
    KcFamilyId.TAIL_NGRAM_COMPOUND_2: 4096,
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
        KcFamilyId.BAG_COMPOUND_1,
        KcFamilyId.BAG_COMPOUND_2,
        KcFamilyId.BAG_CONJUGATED_TYPE,
        KcFamilyId.TAIL_READING_GRAM,
        KcFamilyId.TAIL_POS,
        KcFamilyId.TAIL_COMPOUND_1,
        KcFamilyId.TAIL_COMPOUND_2,
        KcFamilyId.TAIL_CONJUGATED_TYPE,
    ):
        return False

    # Sparse families (use hash-based n-gram features)
    if family in (
        KcFamilyId.NGRAM_POS,
        KcFamilyId.NGRAM_COMPOUND_1,
        KcFamilyId.NGRAM_COMPOUND_2,
        KcFamilyId.NGRAM_CONJUGATED_TYPE,
        KcFamilyId.NGRAM_READING_GRAM,
        KcFamilyId.TAIL_NGRAM_POS,
        KcFamilyId.TAIL_NGRAM_COMPOUND_1,
        KcFamilyId.TAIL_NGRAM_COMPOUND_2,
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
    KcFamilyId.BAG_COMPOUND_1: "compound_1",
    KcFamilyId.BAG_COMPOUND_2: "compound_2",
    KcFamilyId.BAG_CONJUGATED_TYPE: "conjugated_type",
    # Tail
    KcFamilyId.TAIL_READING_GRAM: "reading_gram",
    KcFamilyId.TAIL_POS: "pos",
    KcFamilyId.TAIL_COMPOUND_1: "compound_1",
    KcFamilyId.TAIL_COMPOUND_2: "compound_2",
    KcFamilyId.TAIL_CONJUGATED_TYPE: "conjugated_type",
    # Ngram
    KcFamilyId.NGRAM_POS: "pos",
    KcFamilyId.NGRAM_COMPOUND_1: "compound_1",
    KcFamilyId.NGRAM_COMPOUND_2: "compound_2",
    KcFamilyId.NGRAM_CONJUGATED_TYPE: "conjugated_type",
    KcFamilyId.NGRAM_READING_GRAM: "reading_gram",
    # Tail Ngram
    KcFamilyId.TAIL_NGRAM_POS: "pos",
    KcFamilyId.TAIL_NGRAM_COMPOUND_1: "compound_1",
    KcFamilyId.TAIL_NGRAM_COMPOUND_2: "compound_2",
    KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE: "conjugated_type",
    KcFamilyId.TAIL_NGRAM_READING_GRAM: "reading_gram",
}


# Constants for domain separation (SALT) to reduce accidental collisions between different feature families
SALT: Dict[KcFamilyId, int] = {
    KcFamilyId.NGRAM_POS: 101,
    KcFamilyId.NGRAM_COMPOUND_1: 102,
    KcFamilyId.NGRAM_COMPOUND_2: 103,
    KcFamilyId.NGRAM_CONJUGATED_TYPE: 104,
    KcFamilyId.NGRAM_READING_GRAM: 105,
    KcFamilyId.TAIL_NGRAM_POS: 201,
    KcFamilyId.TAIL_NGRAM_COMPOUND_1: 202,
    KcFamilyId.TAIL_NGRAM_COMPOUND_2: 203,
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


def _get_hierarchical_ids(
    feature_ids: Dict[str, List[int]], family_id: KcFamilyId
) -> List[int]:
    """Get IDs for a family.

    For pos_detail families, the tokenizer now creates composite vocabulary tokens
    (e.g., "noun:proper-noun" for compound_1), so we just return the field IDs directly.
    """
    field = FAMILY_FEATURES[family_id]
    if field not in feature_ids:
        return []
    return list(feature_ids[field])


def _compute_bag_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[KcFamilyId, Any]
) -> None:
    # Bag families
    bag_families = {
        KcFamilyId.BAG_READING_GRAM,
        KcFamilyId.BAG_POS,
        KcFamilyId.BAG_COMPOUND_1,
        KcFamilyId.BAG_COMPOUND_2,
        KcFamilyId.BAG_CONJUGATED_TYPE,
    }

    for family_id in bag_families:
        field = FAMILY_FEATURES[family_id]
        if field in feature_ids:
            # Get hierarchical IDs (includes parent fields for pos_detail families)
            ids = _get_hierarchical_ids(feature_ids, family_id)
            # Exclude special tokens (CLS only - PAD/UNK kept for analysis)
            filtered = [v for v in ids if v not in SPECIAL_TOKEN_IDS]
            # For compound_1, compound_2, and conjugated_type, also filter out UNK
            # These fields can have UNK when morphology is ambiguous
            if family_id in (
                KcFamilyId.BAG_COMPOUND_1,
                KcFamilyId.BAG_COMPOUND_2,
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


def get_tail_ids(
    feature_ids: Dict[str, List[int]],
    field: str,
    window: int = KC_POS_BIASED_WINDOW,
    filter_unk: bool = False,
    disallowed_positions: Optional[Set[int]] = None,
) -> List[int]:
    """Extract the tail IDs for a given field.

    This is the core tail selection logic used by KC training.
    Returns the last `window` tokens' IDs, excluding CLS and optionally UNK.

    Args:
        feature_ids: Dictionary mapping field names to ID lists.
        field: The feature field to extract (e.g., "compound_1").
        window: Number of tokens from the end to include (default: KC_POS_BIASED_WINDOW).
        filter_unk: If True, also filter out UNK tokens.
        disallowed_positions: Optional set of position indices to exclude.

    Returns:
        List of unique, sorted tail IDs for the field.
    """
    if field not in feature_ids:
        return []
    ids = list(feature_ids[field])
    seq_len = len(ids)
    if seq_len == 0:
        return []

    # Get tail positions (last `window` tokens)
    tail_start = max(0, seq_len - window)

    # Filter by position, special tokens, and optionally UNK
    if disallowed_positions is None:
        disallowed_positions = set()

    filtered = []
    for i in range(tail_start, seq_len):
        if i in disallowed_positions:
            continue
        v = ids[i]
        if v in SPECIAL_TOKEN_IDS:
            continue
        if filter_unk and v == UNK_ID:
            continue
        filtered.append(v)

    return sorted(set(filtered))


# Module-level cache for resolved disallow IDs
_DISALLOW_IDS_CACHE: Optional[Set[int]] = None


def initialize_disallow_filter(compound_1_vocab: Dict[str, int]) -> None:
    """Initialize the module-level disallow filter from tokenizer vocab.

    Call this once during startup (e.g., after loading tokenizer) to resolve
    TAIL_DISALLOW tokens to IDs. All subsequent calls to
    compute_kc_targets will use the cached disallow IDs automatically.

    Args:
        compound_1_vocab: Dictionary mapping compound_1 tokens to IDs.
    """
    global _DISALLOW_IDS_CACHE  # pylint: disable=global-statement
    _DISALLOW_IDS_CACHE = set()
    for token in TAIL_DISALLOW:
        if token in compound_1_vocab:
            _DISALLOW_IDS_CACHE.add(compound_1_vocab[token])


def get_disallowed_positions(feature_ids: Dict[str, List[int]]) -> Set[int]:
    """Get position indices where compound_1 matches disallow list.

    Uses the module-level cached disallow IDs (set via initialize_disallow_filter).

    Returns:
        Set of position indices to exclude from all tail/ngram families.
    """
    if _DISALLOW_IDS_CACHE is None or "compound_1" not in feature_ids:
        return set()

    compound_1_ids = feature_ids["compound_1"]
    return {i for i, pid in enumerate(compound_1_ids) if pid in _DISALLOW_IDS_CACHE}


def _compute_tail_targets(
    feature_ids: Dict[str, List[int]],
    targets: Dict[KcFamilyId, Any],
    disallowed_positions: Optional[Set[int]] = None,
) -> None:
    # Tail families
    tail_families = {
        KcFamilyId.TAIL_READING_GRAM,
        KcFamilyId.TAIL_POS,
        KcFamilyId.TAIL_COMPOUND_1,
        KcFamilyId.TAIL_COMPOUND_2,
        KcFamilyId.TAIL_CONJUGATED_TYPE,
    }

    for family_id in tail_families:
        field = FAMILY_FEATURES[family_id]
        if field in feature_ids:
            # For compound_1, compound_2, and conjugated_type, also filter UNK
            filter_unk = family_id in (
                KcFamilyId.TAIL_COMPOUND_1,
                KcFamilyId.TAIL_COMPOUND_2,
                KcFamilyId.TAIL_CONJUGATED_TYPE,
            )
            targets[family_id] = get_tail_ids(
                feature_ids,
                field,
                filter_unk=filter_unk,
                disallowed_positions=disallowed_positions,
            )


def _compute_ngram_targets(
    feature_ids: Dict[str, List[int]],
    targets: Dict[KcFamilyId, Any],
    disallowed_positions: Optional[Set[int]] = None,
) -> None:
    # pylint: disable=too-many-locals
    # Ngram families
    ngram_families = {
        KcFamilyId.NGRAM_POS,
        KcFamilyId.NGRAM_COMPOUND_1,
        KcFamilyId.NGRAM_COMPOUND_2,
        KcFamilyId.NGRAM_CONJUGATED_TYPE,
        KcFamilyId.NGRAM_READING_GRAM,
    }

    if disallowed_positions is None:
        disallowed_positions = set()

    for family_id in ngram_families:
        field = FAMILY_FEATURES[family_id]
        if field in feature_ids:
            # Get hierarchical IDs with position info for filtering
            raw_ids = _get_hierarchical_ids(feature_ids, family_id)
            # Filter by position: exclude disallowed positions, special tokens, and UNK for some families
            ids = []
            for i, v in enumerate(raw_ids):
                if i in disallowed_positions:
                    continue
                if v in SPECIAL_TOKEN_IDS:
                    continue
                ids.append(v)
            # For compound_1, compound_2, and conjugated_type, also filter out UNK
            if family_id in (
                KcFamilyId.NGRAM_COMPOUND_1,
                KcFamilyId.NGRAM_COMPOUND_2,
                KcFamilyId.NGRAM_CONJUGATED_TYPE,
            ):
                ids = [v for v in ids if v != UNK_ID]
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
    feature_ids: Dict[str, List[int]],
    targets: Dict[KcFamilyId, Any],
    disallowed_positions: Optional[Set[int]] = None,
) -> None:
    # pylint: disable=too-many-locals
    """Compute n-gram targets biased toward the end of the sentence."""
    tail_ngram_families = {
        KcFamilyId.TAIL_NGRAM_POS,
        KcFamilyId.TAIL_NGRAM_COMPOUND_1,
        KcFamilyId.TAIL_NGRAM_COMPOUND_2,
        KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE,
        KcFamilyId.TAIL_NGRAM_READING_GRAM,
    }

    if disallowed_positions is None:
        disallowed_positions = set()

    for family_id in tail_ngram_families:
        field = FAMILY_FEATURES[family_id]
        if field in feature_ids:
            # Get hierarchical IDs
            raw_ids = _get_hierarchical_ids(feature_ids, family_id)
            seq_len = len(raw_ids)
            # Get tail positions (last KC_POS_BIASED_WINDOW)
            tail_start = max(0, seq_len - KC_POS_BIASED_WINDOW)
            # Filter: position in tail window AND not disallowed AND not special
            tail_ids = []
            for i in range(tail_start, seq_len):
                if i in disallowed_positions:
                    continue
                if raw_ids[i] in SPECIAL_TOKEN_IDS:
                    continue
                tail_ids.append(raw_ids[i])
            # For compound_1, compound_2, and conjugated_type, also filter out UNK
            if family_id in (
                KcFamilyId.TAIL_NGRAM_COMPOUND_1,
                KcFamilyId.TAIL_NGRAM_COMPOUND_2,
                KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE,
            ):
                tail_ids = [v for v in tail_ids if v != UNK_ID]
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
    """Compute KC targets from feature IDs.

    Args:
        feature_ids: Dictionary mapping field names to token ID lists/tensors.

    Note:
        Call initialize_disallow_filter() once at startup to enable automatic
        filtering of TAIL_DISALLOW tokens from tail/ngram families.
    """
    # Ensure inputs are lists, not tensors
    feature_ids_list: Dict[str, List[int]] = {}
    for k, val in feature_ids.items():
        # Using isinstance(val, torch.Tensor) is safer than hasattr(val, "tolist")
        # to ensure we don't accidentally call it on non-tensor types.
        if isinstance(val, torch.Tensor):
            feature_ids_list[k] = val.tolist()
        else:
            feature_ids_list[k] = list(val)  # type: ignore

    # Compute disallowed positions using module-level cached disallow IDs
    disallowed_positions = get_disallowed_positions(feature_ids_list)

    # Initialize all families with empty lists
    # This ensures all families are present in the returned dict
    targets: Dict[KcFamilyId, Any] = {family: [] for family in ALL_KC_FAMILIES}

    _compute_bag_targets(feature_ids_list, targets)
    _compute_tail_targets(feature_ids_list, targets, disallowed_positions)
    _compute_ngram_targets(feature_ids_list, targets, disallowed_positions)
    _compute_tail_ngram_targets(feature_ids_list, targets, disallowed_positions)

    return targets
