"""Tokenizer for Kotogram that extracts morphological features.

This module provides the Tokenizer class and related constants. It is designed
to be lightweight and free of heavy dependencies (like PyTorch) to allow
fast imports in multiprocessing workers.
"""

import json
from collections import Counter
from typing import Any, Dict, List

from kotogram.kotogram import TokenFeatures, extract_token_features, split_kotogram

# Special token values for vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"

# Special token IDs - single source of truth
# These must match the order tokens are added to vocabularies in Tokenizer.__init__
PAD_ID = 0
UNK_ID = 1
CLS_ID = 2

# Feature fields used for token embedding
# NOTE: 'surface' is critical for gender detection (pronouns like 僕, 俺, あたし)
ALL_FEATURE_FIELDS = [
    # "surface",
    "pos",
    "pos_detail_1",
    "pos_detail_2",
    "pos_detail_3",
    "conjugated_type",
    # "conjugated_form",
    # "base_orth",
    "reading_gram",
]
FEATURE_FIELDS = ALL_FEATURE_FIELDS  # Default: use all features


def build_pos_detail_1_composite(
    pos: str, pos_detail_1: str, conjugated_form: str, conjugated_type: str = ""
) -> str:
    """Build the composite token string for pos_detail_1 field.

    This is the single source of truth for how pos_detail_1 composites are built.
    Format: "pos:detail:conjugated_form" where detail is pos_detail_1 or
    conjugated_type (for aux-verbs).

    For aux-verbs (ます/です/だ), Sudachi provides conjugated_type (aux-masu)
    but no pos_detail_1, so we use conjugated_type as the detail component.

    Args:
        pos: Part of speech (e.g., "verb", "particle")
        pos_detail_1: Detail level 1 (e.g., "general", "case-particle")
        conjugated_form: Conjugated form (e.g., "imperative", "terminal") or ""
        conjugated_type: Conjugated type (e.g., "aux-masu") - used as fallback

    Returns:
        Composite token string, or "" if both pos_detail_1 and conjugated_type are empty.
    """
    # Use pos_detail_1 if present, otherwise fall back to conjugated_type
    detail = pos_detail_1 or conjugated_type
    if not detail:
        return ""
    if conjugated_form:
        return f"{pos}:{detail}:{conjugated_form}"
    return f"{pos}:{detail}"


def build_pos_detail_2_composite(pos: str, pos_detail_1: str, pos_detail_2: str) -> str:
    """Build the composite token string for pos_detail_2 field.

    This is the single source of truth for how pos_detail_2 composites are built.
    Format: "pos:pos_detail_1:pos_detail_2" (only if pos_detail_2 exists).

    Must be used by both vocabulary building (label.py) and encoding (tokenizer.py)
    to ensure consistency.

    Args:
        pos: Part of speech (e.g., "verb", "noun")
        pos_detail_1: Detail level 1 (e.g., "general", "proper-noun")
        pos_detail_2: Detail level 2 (e.g., "person-name", "place-name") or ""

    Returns:
        Composite token string, or "" if pos_detail_2 is empty.
    """
    if not pos_detail_2:
        return ""
    return f"{pos}:{pos_detail_1}:{pos_detail_2}"


def get_vocab_strings(features: "TokenFeatures") -> Dict[str, str]:
    """Transform TokenFeatures into vocab-ready strings for all feature fields.

    This is the single source of truth for how TokenFeatures are converted to
    vocabulary strings. Both vocabulary building (label.py) and encoding
    (tokenizer.py) must use this function to ensure consistency.

    Args:
        features: A TokenFeatures object with morphological information.

    Returns:
        Dict mapping field name to vocab-ready string for that field.
    """
    return {
        "pos": features.pos,
        "pos_detail_1": build_pos_detail_1_composite(
            features.pos,
            features.pos_detail_1,
            features.conjugated_form,
            features.conjugated_type,
        ),
        "pos_detail_2": build_pos_detail_2_composite(
            features.pos, features.pos_detail_1, features.pos_detail_2
        ),
        "pos_detail_3": features.pos_detail_3,
        "conjugated_type": features.conjugated_type,
        "reading_gram": features.reading_gram,
    }


class Tokenizer:
    """Tokenizer that extracts morphological features from Kotogram tokens.

    Instead of treating each token as a single vocabulary item, this tokenizer
    extracts categorical features (pos, pos_detail1, conjugated_type, conjugated_form,
    lemma) and maintains separate vocabularies for each field.

    Attributes:
        field_vocabs: Dict mapping field name to {value: id} mapping
        field_vocab_sizes: Dict mapping field name to vocabulary size
    """

    def __init__(self) -> None:
        """Initialize feature tokenizer."""
        # Initialize vocabularies for each field with special tokens
        # Vocabularies are stored as dicts for simple JSON serialization and compatibility.
        self.field_vocabs: Dict[str, Dict[str, int]] = {}
        self._field_counters: Dict[str, Counter[str]] = {}
        for f in FEATURE_FIELDS:
            self.field_vocabs[f] = {
                PAD_TOKEN: PAD_ID,
                UNK_TOKEN: UNK_ID,
                CLS_TOKEN: CLS_ID,
            }
            self._field_counters[f] = Counter()

        self._frozen = False

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def cls_id(self) -> int:
        return CLS_ID

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for all fields."""
        return {field: len(vocab) for field, vocab in self.field_vocabs.items()}

    def get_id(self, field: str, value: str) -> int:
        """Get ID for a value in a field."""
        vocab = self.field_vocabs[field]
        return vocab.get(value, self.unk_id)

    @staticmethod
    def extract_features(kotogram: str) -> List[TokenFeatures]:
        """Extract features from each token in a Kotogram string."""
        tokens = split_kotogram(kotogram)
        features_list = []

        for token in tokens:
            features = extract_token_features(token)
            features_list.append(features)
        return features_list

    def encode_features(
        self,
        features_list: List[TokenFeatures],
    ) -> Dict[str, List[int]]:
        """Encode a list of token feature objects into field ID sequences."""
        # Initialize result dictionary
        result: Dict[str, List[int]] = {f: [] for f in FEATURE_FIELDS}

        # Add CLS token
        for field in FEATURE_FIELDS:
            cls_id = self.get_id(field, CLS_TOKEN)
            result[field].append(cls_id)

        # Encode each token
        for features in features_list:
            # Get vocab-ready strings for all fields using centralized function
            vocab_strings = get_vocab_strings(features)
            for field in FEATURE_FIELDS:
                result[field].append(self.get_id(field, vocab_strings[field]))

        return result

    def encode(
        self,
        kotogram: str,
    ) -> Dict[str, List[int]]:
        """Encode a Kotogram string to feature ID sequences."""
        features_list = self.extract_features(kotogram)
        return self.encode_features(features_list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tokenizer state to a dictionary."""
        return {
            "field_vocabs": self.field_vocabs,
            "frozen": self._frozen,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load tokenizer state from dictionary."""
        if not isinstance(state, dict):
            raise TypeError(f"State must be a dictionary, got {type(state)}")

        # Migration logic for old pos_detail naming
        if "field_vocabs" in state:
            vocabs = state["field_vocabs"]
            for i in range(1, 4):
                old_key = f"pos_detail{i}"
                new_key = f"pos_detail_{i}"
                if old_key in vocabs and new_key not in vocabs:
                    vocabs[new_key] = vocabs.pop(old_key)

        self.field_vocabs.update(state.get("field_vocabs", {}))
        self._frozen = state.get("frozen", self._frozen)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()
        # Merge loaded vocabs, preserving defaults for any new fields not in the file
        tokenizer.load_state(data)
        return tokenizer
