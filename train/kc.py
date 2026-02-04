"""KC target computation logic with ABC-based family hierarchy."""

# pylint: disable=too-many-lines

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import torch

from kotogram.exceptions import MissingMappingError
from kotogram.tokenizer import CLS_ID, PAD_ID, UNK_ID

# KC Configuration
KC_NGRAM_ORDER = 3
KC_POS_BIASED_WINDOW = 5

# Special token IDs to exclude from KC targets
SPECIAL_TOKEN_IDS = {CLS_ID}

# High-frequency low-information tokens to exclude from tail families.
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
    BAG_CONJUGATED_TYPE = "bag_conjugated_type"

    # Tail Bag Families
    TAIL_READING_GRAM = "tail_reading_gram"
    TAIL_POS = "tail_pos"
    TAIL_COMPOUND_1 = "tail_compound_1"
    TAIL_CONJUGATED_TYPE = "tail_conjugated_type"

    # Ngram Families
    NGRAM_POS = "ngram_pos"
    NGRAM_COMPOUND_1 = "ngram_compound_1"
    NGRAM_CONJUGATED_TYPE = "ngram_conjugated_type"
    NGRAM_READING_GRAM = "ngram_reading_gram"

    # Tail Ngram Families
    TAIL_NGRAM_POS = "tail_ngram_pos"
    TAIL_NGRAM_COMPOUND_1 = "tail_ngram_compound_1"
    TAIL_NGRAM_CONJUGATED_TYPE = "tail_ngram_conjugated_type"
    TAIL_NGRAM_READING_GRAM = "tail_ngram_reading_gram"

    # DB-Sourced Families (labels from corpus.db, not computed)
    GRAMMAR_POINT = "grammar_point"
    GENDER = "gender"
    FORMALITY = "formality"
    GENDER_CLASS = "gender_class"  # Classification version (3 classes: -1, 0, +1)
    FORMALITY_CLASS = "formality_class"  # Classification version (5 classes)
    REGISTER = (
        "register"  # Multi-label classification (14 registers, can have multiple)
    )


class KcLogitMode(str, Enum):
    """Defines which KC logits a family is trained against."""

    SPARSE_LOGITS = "sparse_logits"  # k-budget sparse activations (localized)
    ALL_LOGITS = "all_logits"  # Full KC probabilities (diffuse style)
    HOT_LOGITS = "hot_logits"  # Only logits with prob >= 0.5 (thresholded)


# =============================================================================
# KC Family ABC and Subclasses
# =============================================================================


@dataclass(frozen=True)
class KcFamily(ABC):
    """Abstract base class for all KC families.

    Attributes are immutable properties that define family behavior.
    Subclasses set these via dataclass fields.
    """

    family_id: KcFamilyId

    @property
    @abstractmethod
    def is_sparse(self) -> bool:
        """True if uses hash-bucket sparse features, False for dense tokenizer vocab."""

    @property
    @abstractmethod
    def is_db_sourced(self) -> bool:
        """True if targets come from DB labels, not computed from morphology."""

    @property
    @abstractmethod
    def feature_field(self) -> str:
        """The tokenizer feature field this family uses (empty for DB-sourced)."""

    @property
    def is_tail(self) -> bool:
        """True if this family uses only tail-window tokens."""
        return False

    @property
    def bucket_size(self) -> Optional[int]:
        """Hash bucket size for sparse families, None for dense."""
        return None

    @property
    def salt(self) -> Optional[int]:
        """Domain separation salt for ngram hashing, None for non-ngram."""
        return None

    @property
    def filter_unk(self) -> bool:
        """Whether to filter UNK tokens from this family's targets."""
        return False

    @property
    def is_slim_decoder(self) -> bool:
        """True if this decoder should be stripped from slim (exported) models."""
        return True

    @property
    def loss_weight(self) -> float:
        """Per-family loss multiplier (calibrated so families contribute equally)."""
        return 1.0

    @property
    @abstractmethod
    def logit_mode(self) -> KcLogitMode:
        """Which KC logits this family trains against (SPARSE_LOGITS or ALL_LOGITS)."""


@dataclass(frozen=True)
class KcBagFamily(KcFamily):
    """Bag-of-words dense families using full sequence."""

    _feature_field: str
    _logit_mode: KcLogitMode
    _filter_unk: bool = False
    _loss_weight: float = 1.0

    @property
    def is_sparse(self) -> bool:
        return False

    @property
    def is_db_sourced(self) -> bool:
        return False

    @property
    def feature_field(self) -> str:
        return self._feature_field

    @property
    def filter_unk(self) -> bool:
        return self._filter_unk

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    @property
    def logit_mode(self) -> KcLogitMode:
        return self._logit_mode


@dataclass(frozen=True)
class KcTailFamily(KcFamily):
    """Tail-window dense families."""

    _feature_field: str
    _logit_mode: KcLogitMode
    _filter_unk: bool = False
    _loss_weight: float = 1.0

    @property
    def is_sparse(self) -> bool:
        return False

    @property
    def is_db_sourced(self) -> bool:
        return False

    @property
    def feature_field(self) -> str:
        return self._feature_field

    @property
    def is_tail(self) -> bool:
        return True

    @property
    def filter_unk(self) -> bool:
        return self._filter_unk

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    @property
    def logit_mode(self) -> KcLogitMode:
        return self._logit_mode


@dataclass(frozen=True)
class KcNgramFamily(KcFamily):
    """Full-sequence n-gram sparse families."""

    _feature_field: str
    _bucket_size: int
    _salt: int
    _logit_mode: KcLogitMode
    _filter_unk: bool = False
    _loss_weight: float = 1.0

    @property
    def is_sparse(self) -> bool:
        return True

    @property
    def is_db_sourced(self) -> bool:
        return False

    @property
    def feature_field(self) -> str:
        return self._feature_field

    @property
    def bucket_size(self) -> Optional[int]:
        return self._bucket_size

    @property
    def salt(self) -> Optional[int]:
        return self._salt

    @property
    def filter_unk(self) -> bool:
        return self._filter_unk

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    @property
    def logit_mode(self) -> KcLogitMode:
        return self._logit_mode


@dataclass(frozen=True)
class KcTailNgramFamily(KcFamily):
    """Tail-window n-gram sparse families."""

    _feature_field: str
    _bucket_size: int
    _salt: int
    _logit_mode: KcLogitMode
    _filter_unk: bool = False
    _loss_weight: float = 1.0

    @property
    def is_sparse(self) -> bool:
        return True

    @property
    def is_db_sourced(self) -> bool:
        return False

    @property
    def feature_field(self) -> str:
        return self._feature_field

    @property
    def is_tail(self) -> bool:
        return True

    @property
    def bucket_size(self) -> Optional[int]:
        return self._bucket_size

    @property
    def salt(self) -> Optional[int]:
        return self._salt

    @property
    def filter_unk(self) -> bool:
        return self._filter_unk

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    @property
    def logit_mode(self) -> KcLogitMode:
        return self._logit_mode


@dataclass(frozen=True)
class KcPnuFamily(KcFamily):
    """DB-sourced PNU (Positive-Negative-Unlabeled) families like GRAMMAR_POINT.

    Multi-label semi-supervised learning with explicit positives, negatives, and
    unlabeled data. Uses sparsity assumption: unlabeled treated as weak negatives.
    """

    _logit_mode: KcLogitMode
    _loss_weight: float = 1.0

    @property
    def is_sparse(self) -> bool:
        return False  # Dense vocab (e.g., 1374 grammar points)

    @property
    def is_db_sourced(self) -> bool:
        return True

    @property
    def feature_field(self) -> str:
        return ""  # No tokenizer feature, targets from DB

    @property
    def is_slim_decoder(self) -> bool:
        return False  # Needed at inference time for grammar point prediction

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    @property
    def logit_mode(self) -> KcLogitMode:
        return self._logit_mode


@dataclass(frozen=True)
class KcMseFamily(KcFamily):
    """DB-sourced MSE loss families for continuous targets (GENDER, FORMALITY)."""

    _logit_mode: KcLogitMode
    _loss_weight: float = 1.0

    @property
    def is_sparse(self) -> bool:
        return False  # Single scalar output

    @property
    def is_db_sourced(self) -> bool:
        return True

    @property
    def feature_field(self) -> str:
        return ""  # No tokenizer feature, targets from batch

    @property
    def is_slim_decoder(self) -> bool:
        return False  # Needed at inference time

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    @property
    def logit_mode(self) -> KcLogitMode:
        return self._logit_mode


@dataclass(frozen=True)
class KcDbClassFamily(KcFamily):
    """DB-sourced multi-class classification families (GENDER_CLASS, FORMALITY_CLASS)."""

    _logit_mode: KcLogitMode
    _num_classes: int
    _loss_weight: float = 1.0

    @property
    def is_sparse(self) -> bool:
        return False  # Dense multi-class output

    @property
    def is_db_sourced(self) -> bool:
        return True

    @property
    def feature_field(self) -> str:
        return ""  # No tokenizer feature, targets from batch

    @property
    def is_slim_decoder(self) -> bool:
        return True  # Experimental, not needed at inference

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    @property
    def logit_mode(self) -> KcLogitMode:
        return self._logit_mode

    @property
    def num_classes(self) -> int:
        return self._num_classes


@dataclass(frozen=True)
class KcDbMultilabelFamily(KcFamily):
    """DB-sourced multi-label classification for REGISTER (multiple labels per sample)."""

    _logit_mode: KcLogitMode
    _num_classes: int
    _loss_weight: float = 1.0

    @property
    def is_sparse(self) -> bool:
        return False  # Dense multi-hot output

    @property
    def is_db_sourced(self) -> bool:
        return True

    @property
    def feature_field(self) -> str:
        return ""  # No tokenizer feature, targets from batch

    @property
    def is_slim_decoder(self) -> bool:
        return False  # Needed at inference time for register prediction

    @property
    def loss_weight(self) -> float:
        return self._loss_weight

    @property
    def logit_mode(self) -> KcLogitMode:
        return self._logit_mode

    @property
    def num_classes(self) -> int:
        return self._num_classes


# =============================================================================
# Family Registry
# =============================================================================
_DEFAULT_STRUCTURE_LOGITS = KcLogitMode.ALL_LOGITS
_DEFAULT_SEMANTIC_LOGITS = KcLogitMode.ALL_LOGITS


KC_FAMILIES: Dict[KcFamilyId, KcFamily] = {
    # Bag families (dense, full sequence)
    KcFamilyId.BAG_READING_GRAM: KcBagFamily(
        family_id=KcFamilyId.BAG_READING_GRAM,
        _feature_field="reading_gram",
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _loss_weight=0.030,  # Reduced from 0.149 (20%)
    ),
    KcFamilyId.BAG_POS: KcBagFamily(
        family_id=KcFamilyId.BAG_POS,
        _feature_field="pos",
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _loss_weight=0.137,  # Reduced from 0.683 (20%)
    ),
    KcFamilyId.BAG_COMPOUND_1: KcBagFamily(
        family_id=KcFamilyId.BAG_COMPOUND_1,
        _feature_field="compound_1",
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _filter_unk=True,
        _loss_weight=0.057,  # Reduced from 0.283 (20%)
    ),
    KcFamilyId.BAG_CONJUGATED_TYPE: KcBagFamily(
        family_id=KcFamilyId.BAG_CONJUGATED_TYPE,
        _feature_field="conjugated_type",
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _filter_unk=True,
        _loss_weight=0.062,  # Reduced from 0.308 (20%)
    ),
    # Tail families (dense, tail window)
    KcFamilyId.TAIL_READING_GRAM: KcTailFamily(
        family_id=KcFamilyId.TAIL_READING_GRAM,
        _feature_field="reading_gram",
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _loss_weight=0.028,  # Reduced from 0.139 (20%)
    ),
    KcFamilyId.TAIL_POS: KcTailFamily(
        family_id=KcFamilyId.TAIL_POS,
        _feature_field="pos",
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _loss_weight=0.107,  # Reduced from 0.537 (20%)
    ),
    KcFamilyId.TAIL_COMPOUND_1: KcTailFamily(
        family_id=KcFamilyId.TAIL_COMPOUND_1,
        _feature_field="compound_1",
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _filter_unk=True,
        _loss_weight=0.048,  # Reduced from 0.240 (20%)
    ),
    KcFamilyId.TAIL_CONJUGATED_TYPE: KcTailFamily(
        family_id=KcFamilyId.TAIL_CONJUGATED_TYPE,
        _feature_field="conjugated_type",
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _filter_unk=True,
        _loss_weight=0.057,  # Reduced from 0.284 (20%)
    ),
    # Ngram families (sparse, full sequence)
    KcFamilyId.NGRAM_POS: KcNgramFamily(
        family_id=KcFamilyId.NGRAM_POS,
        _feature_field="pos",
        _bucket_size=2048,
        _salt=101,
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _loss_weight=0.004,  # Reduced from 0.022 (20%)
    ),
    KcFamilyId.NGRAM_COMPOUND_1: KcNgramFamily(
        family_id=KcFamilyId.NGRAM_COMPOUND_1,
        _feature_field="compound_1",
        _bucket_size=8192 * 4,
        _salt=102,
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _filter_unk=True,
        _loss_weight=0.003,  # Reduced from 0.015 (20%)
    ),
    KcFamilyId.NGRAM_CONJUGATED_TYPE: KcNgramFamily(
        family_id=KcFamilyId.NGRAM_CONJUGATED_TYPE,
        _feature_field="conjugated_type",
        _bucket_size=8192,
        _salt=104,
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _filter_unk=True,
        _loss_weight=0.0005,  # Reduced from 0.0025 (20%)
    ),
    KcFamilyId.NGRAM_READING_GRAM: KcNgramFamily(
        family_id=KcFamilyId.NGRAM_READING_GRAM,
        _feature_field="reading_gram",
        _bucket_size=262144,  # 2^18: 234K unique ngrams
        _salt=105,
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _loss_weight=0.004,  # Reduced from 0.022 (20%)
    ),
    # Tail ngram families (sparse, tail window)
    KcFamilyId.TAIL_NGRAM_POS: KcTailNgramFamily(
        family_id=KcFamilyId.TAIL_NGRAM_POS,
        _feature_field="pos",
        _bucket_size=1024,
        _salt=201,
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _loss_weight=0.001,  # Reduced from 0.0046 (20%)
    ),
    KcFamilyId.TAIL_NGRAM_COMPOUND_1: KcTailNgramFamily(
        family_id=KcFamilyId.TAIL_NGRAM_COMPOUND_1,
        _feature_field="compound_1",
        _bucket_size=8192 * 2,
        _salt=202,
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _filter_unk=True,
        _loss_weight=0.0006,  # Reduced from 0.0032 (20%)
    ),
    KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE: KcTailNgramFamily(
        family_id=KcFamilyId.TAIL_NGRAM_CONJUGATED_TYPE,
        _feature_field="conjugated_type",
        _bucket_size=8192,
        _salt=204,
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _filter_unk=True,
        _loss_weight=0.0003,  # Reduced from 0.0017 (20%)
    ),
    KcFamilyId.TAIL_NGRAM_READING_GRAM: KcTailNgramFamily(
        family_id=KcFamilyId.TAIL_NGRAM_READING_GRAM,
        _feature_field="reading_gram",
        _bucket_size=131072,  # 2^17: 115K unique ngrams
        _salt=205,
        _logit_mode=_DEFAULT_STRUCTURE_LOGITS,
        _loss_weight=0.0007,  # Reduced from 0.0034 (20%)
    ),
    # DB-sourced families
    KcFamilyId.GRAMMAR_POINT: KcPnuFamily(
        family_id=KcFamilyId.GRAMMAR_POINT,
        _logit_mode=_DEFAULT_SEMANTIC_LOGITS,
        _loss_weight=1.0,  # epoch1_loss=6.00
    ),
    KcFamilyId.GENDER: KcMseFamily(
        family_id=KcFamilyId.GENDER,
        _logit_mode=_DEFAULT_SEMANTIC_LOGITS,
        _loss_weight=0.5,  # Original weight (not 100x)
    ),
    KcFamilyId.FORMALITY: KcMseFamily(
        family_id=KcFamilyId.FORMALITY,
        _logit_mode=_DEFAULT_SEMANTIC_LOGITS,
        _loss_weight=0.5,  # Original weight (not 100x)
    ),
    # Classification versions
    KcFamilyId.GENDER_CLASS: KcDbClassFamily(
        family_id=KcFamilyId.GENDER_CLASS,
        _num_classes=3,  # Masculine (-1), Neutral (0), Feminine (+1)
        _logit_mode=_DEFAULT_SEMANTIC_LOGITS,
        _loss_weight=1.0,
    ),
    KcFamilyId.FORMALITY_CLASS: KcDbClassFamily(
        family_id=KcFamilyId.FORMALITY_CLASS,
        _num_classes=5,  # Very Casual, Casual, Neutral, Formal, Very Formal
        _logit_mode=_DEFAULT_SEMANTIC_LOGITS,
        _loss_weight=1.0,
    ),
    # Multi-label families
    KcFamilyId.REGISTER: KcDbMultilabelFamily(
        family_id=KcFamilyId.REGISTER,
        _num_classes=14,  # 14 register types (sonkeigo, kenjogo, etc.)
        _logit_mode=_DEFAULT_SEMANTIC_LOGITS,
        _loss_weight=1.0,
    ),
}


def get_family(family_id: KcFamilyId) -> KcFamily:
    """Get the family instance for a given family ID."""
    return KC_FAMILIES[family_id]


ALL_KC_FAMILIES = list(KcFamilyId)


# =============================================================================
# Backward-Compatible Function Wrappers
# =============================================================================

# Legacy dictionary - kept for backward compatibility
FAMILY_FEATURES: Dict[KcFamilyId, str] = {
    fid: fam.feature_field for fid, fam in KC_FAMILIES.items()
}


def get_family_bucket_size(family_id: KcFamilyId) -> int:
    """Get the hash bucket size for a sparse KC family.

    Raises:
        KeyError: If family_id is not a sparse ngram family.
    """
    family = get_family(family_id)
    if family.bucket_size is None:
        raise KeyError(f"Family {family_id} is not a sparse family")
    return family.bucket_size


def is_family_sparse(family: KcFamilyId) -> bool:
    """Check if a KC family uses sparse (hash-based) features.

    Dense families (Bag, Tail) use actual tokenizer vocab IDs.
    Sparse families (Ngram, Tail Ngram) use hash-based n-gram features.

    Args:
        family: The KC family to check.

    Returns:
        True if the family uses sparse features, False for dense.
    """
    return get_family(family).is_sparse


def is_family_db_sourced(family: KcFamilyId) -> bool:
    """Check if a KC family's targets come from DB labels (not computed from morphology).

    DB-sourced families have targets stored in corpus.db columns (e.g., grammar, grammar_negative)
    rather than being computed from token features during labeling.

    Args:
        family: The KC family to check.

    Returns:
        True if the family uses DB-sourced labels, False for computed targets.
    """
    return get_family(family).is_db_sourced


# =============================================================================
# Hashing Utilities
# =============================================================================


def _salt(key: KcFamilyId) -> int:
    """Robust lookup for SALT keys with descriptive error messages."""
    family = get_family(key)
    if family.salt is None:
        raise MissingMappingError(
            map_name="SALT",
            key=str(key),
            context=f"Family {key} is not an ngram family",
        )
    return family.salt


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


# =============================================================================
# Target Computation Helpers
# =============================================================================


def _get_hierarchical_ids(
    feature_ids: Dict[str, List[int]], family_id: KcFamilyId
) -> List[int]:
    """Get IDs for a family.

    For pos_detail families, the tokenizer now creates composite vocabulary tokens
    (e.g., "noun:proper-noun" for compound_1), so we just return the field IDs directly.
    """
    family = get_family(family_id)
    field = family.feature_field
    if field not in feature_ids:
        return []
    return list(feature_ids[field])


def _compute_bag_targets(
    feature_ids: Dict[str, List[int]], targets: Dict[KcFamilyId, Any]
) -> None:
    # Bag families
    bag_families = [
        fid for fid, fam in KC_FAMILIES.items() if isinstance(fam, KcBagFamily)
    ]

    for family_id in bag_families:
        family = get_family(family_id)
        field = family.feature_field
        if field in feature_ids:
            # Get hierarchical IDs (includes parent fields for pos_detail families)
            ids = _get_hierarchical_ids(feature_ids, family_id)
            # Exclude special tokens (CLS only - PAD/UNK kept for analysis)
            filtered = [v for v in ids if v not in SPECIAL_TOKEN_IDS]
            # For families that filter UNK, remove UNK tokens
            if family.filter_unk:
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
    tail_families = [
        fid for fid, fam in KC_FAMILIES.items() if isinstance(fam, KcTailFamily)
    ]

    for family_id in tail_families:
        family = get_family(family_id)
        field = family.feature_field
        if field in feature_ids:
            targets[family_id] = get_tail_ids(
                feature_ids,
                field,
                filter_unk=family.filter_unk,
                disallowed_positions=disallowed_positions,
            )


def _compute_ngram_targets(
    feature_ids: Dict[str, List[int]],
    targets: Dict[KcFamilyId, Any],
    disallowed_positions: Optional[Set[int]] = None,
) -> None:
    # pylint: disable=too-many-locals
    # Ngram families
    ngram_families = [
        fid
        for fid, fam in KC_FAMILIES.items()
        if isinstance(fam, KcNgramFamily) and not fam.is_tail
    ]

    if disallowed_positions is None:
        disallowed_positions = set()

    for family_id in ngram_families:
        family = get_family(family_id)
        field = family.feature_field
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
            # For families that filter UNK, remove UNK tokens
            if family.filter_unk:
                ids = [v for v in ids if v != UNK_ID]
            hashes = set()
            salt_val = family.salt
            bucket_size = family.bucket_size
            assert salt_val is not None and bucket_size is not None
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(ids) >= n_val:
                    for i in range(len(ids) - n_val + 1):
                        ngram = ids[i : i + n_val]
                        # Prepend salt for domain separation
                        h = stable_hash_ints([salt_val, *ngram]) % bucket_size
                        hashes.add(h)
            targets[family_id] = sorted(hashes)


def _compute_tail_ngram_targets(
    feature_ids: Dict[str, List[int]],
    targets: Dict[KcFamilyId, Any],
    disallowed_positions: Optional[Set[int]] = None,
) -> None:
    # pylint: disable=too-many-locals
    """Compute n-gram targets biased toward the end of the sentence."""
    tail_ngram_families = [
        fid for fid, fam in KC_FAMILIES.items() if isinstance(fam, KcTailNgramFamily)
    ]

    if disallowed_positions is None:
        disallowed_positions = set()

    for family_id in tail_ngram_families:
        family = get_family(family_id)
        field = family.feature_field
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
            # For families that filter UNK, remove UNK tokens
            if family.filter_unk:
                tail_ids = [v for v in tail_ids if v != UNK_ID]
            if not tail_ids:
                continue

            hashes = set()
            salt_val = family.salt
            bucket_size = family.bucket_size
            assert salt_val is not None and bucket_size is not None
            for n_val in range(2, KC_NGRAM_ORDER + 1):
                if len(tail_ids) >= n_val:
                    for i in range(len(tail_ids) - n_val + 1):
                        ngram = tail_ids[i : i + n_val]
                        h = stable_hash_ints([salt_val, *ngram]) % bucket_size
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

    # Initialize all computed (non-DB-sourced) families with empty lists
    # DB-sourced families (like GRAMMAR_POINT) get targets from corpus.db, not from tokens
    targets: Dict[KcFamilyId, Any] = {
        family_id: [] for family_id, fam in KC_FAMILIES.items() if not fam.is_db_sourced
    }

    _compute_bag_targets(feature_ids_list, targets)
    _compute_tail_targets(feature_ids_list, targets, disallowed_positions)
    _compute_ngram_targets(feature_ids_list, targets, disallowed_positions)
    _compute_tail_ngram_targets(feature_ids_list, targets, disallowed_positions)

    return targets
