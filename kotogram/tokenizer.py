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
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
                CLS_TOKEN: 2,
            }
            self._field_counters[f] = Counter()

        self._frozen = False

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def cls_id(self) -> int:
        return 2

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for all fields."""
        return {field: len(vocab) for field, vocab in self.field_vocabs.items()}

    def get_id(self, field: str, value: str) -> int:
        """Get ID for a value in a field."""
        vocab = self.field_vocabs[field]
        return vocab.get(value, self.unk_id)

    def extract_features(self, kotogram: str) -> List[TokenFeatures]:
        """Extract features from each token in a Kotogram string."""
        tokens = split_kotogram(kotogram)
        features_list = []

        for token in tokens:
            features = extract_token_features(token)
            # Filter? TokenFeatures matches strict schema anyway.
            # Using dataclass directly.
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
            # Unrolled for type safety and static analysis visibility
            # result["surface"].append(self.get_id("surface", features.surface))
            result["pos"].append(self.get_id("pos", features.pos))
            result["pos_detail_1"].append(
                self.get_id("pos_detail_1", features.pos_detail_1)
            )
            result["pos_detail_2"].append(
                self.get_id("pos_detail_2", features.pos_detail_2)
            )
            result["pos_detail_3"].append(
                self.get_id("pos_detail_3", features.pos_detail_3)
            )
            result["conjugated_type"].append(
                self.get_id("conjugated_type", features.conjugated_type)
            )
            # result["conjugated_form"].append(
            #     self.get_id("conjugated_form", features.conjugated_form)
            # )
            # base_orth stripped
            result["reading_gram"].append(
                self.get_id("reading_gram", features.reading_gram)
            )

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
