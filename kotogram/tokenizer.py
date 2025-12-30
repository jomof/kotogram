"""Tokenizer for Kotogram that extracts morphological features.

This module provides the Tokenizer class and related constants. It is designed
to be lightweight and free of heavy dependencies (like PyTorch) to allow
fast imports in multiprocessing workers.
"""

import json
import os
from collections import Counter
from typing import Any, Dict, List

from kotogram.kotogram import extract_token_features, split_kotogram

# Special token values for vocabulary
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"

# Feature fields used for token embedding
# NOTE: 'surface' is critical for gender detection (pronouns like 僕, 俺, あたし)
ALL_FEATURE_FIELDS = [
    "surface",
    "pos",
    "pos_detail1",
    "pos_detail2",
    "pos_detail3",
    "conjugated_type",
    "conjugated_form",
    "lemma",
    "base_orth",
    "reading",
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
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def cls_id(self) -> int:
        return 2

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Get vocabulary sizes for all fields."""
        return {field: len(vocab) for field, vocab in self.field_vocabs.items()}

    def _add_value(self, field: str, value: str) -> int:
        """Add a value to field vocabulary and return its ID."""
        if not value:
            value = UNK_TOKEN

        vocab = self.field_vocabs[field]
        if value in vocab:
            return vocab[value]

        if self._frozen:
            return self.unk_id

        new_id = len(vocab)
        vocab[value] = new_id
        return new_id

    def get_id(self, field: str, value: str) -> int:
        """Get ID for a value in a field."""
        vocab = self.field_vocabs[field]
        return vocab.get(value, self.unk_id)

    def extract_features(self, kotogram: str) -> List[Dict[str, str]]:
        """Extract features from each token in a Kotogram string."""
        tokens = split_kotogram(kotogram)
        features_list = []

        for token in tokens:
            features = extract_token_features(token)
            # Only keep the fields we use
            # Explicit access avoids vulture flagging fields as unused
            all_features = {
                "surface": features.surface,
                "pos": features.pos,
                "pos_detail1": features.pos_detail1,
                "pos_detail2": features.pos_detail2,
                "pos_detail3": features.pos_detail3,
                "conjugated_type": features.conjugated_type,
                "conjugated_form": features.conjugated_form,
                "lemma": features.lemma,
                "base_orth": features.base_orth,
                "reading": features.reading,
            }
            filtered = {field: all_features[field] for field in FEATURE_FIELDS}
            features_list.append(filtered)

        return features_list

    def encode_features(
        self,
        features_list: List[Dict[str, str]],
        add_cls: bool = True,
        add_to_vocab: bool = True,
    ) -> Dict[str, List[int]]:
        """Convert list of feature dicts to sequences of field IDs."""
        result: Dict[str, List[int]] = {f: [] for f in FEATURE_FIELDS}

        if add_cls:
            for field in FEATURE_FIELDS:
                result[field].append(self.cls_id)

        for features in features_list:
            for field in FEATURE_FIELDS:
                value = features.get(field, "")
                if add_to_vocab and not self._frozen:
                    self._field_counters[field][value] += 1
                    token_id = self._add_value(field, value)
                else:
                    vocab = self.field_vocabs[field]
                    token_id = vocab.get(value, self.unk_id)
                result[field].append(token_id)

        return result

    def encode(
        self,
        kotogram: str,
        add_cls: bool = True,
        add_to_vocab: bool = True,
    ) -> Dict[str, List[int]]:
        """Encode a Kotogram string to feature ID sequences."""
        features_list = self.extract_features(kotogram)
        return self.encode_features(features_list, add_cls, add_to_vocab)

    def save(self, path: str, **kwargs: Any) -> None:
        """Save tokenizer vocabularies to JSON file atomically."""
        data = {
            "field_vocabs": self.field_vocabs,
            "frozen": self._frozen,
        }
        data.update(kwargs)

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Atomic write: dump to temp file then rename
        # This prevents concurrent readers from seeing partial content
        import tempfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=dir_name, delete=False, encoding="utf-8"
            ) as tmp_file:
                # We save path to clean up in finally block if something goes wrong
                tmp_path = tmp_file.name
                json.dump(data, tmp_file, ensure_ascii=False, indent=2)
                tmp_file.flush()
                # fsync to ensure data is on disk before rename
                os.fsync(tmp_file.fileno())

            # Context manager closed the file. Now replace atomically.
            os.replace(tmp_path, path)
            # Sentinel to prevent deletion in finally
            tmp_path = None

        finally:
            # If tmp_path is still set, it means we didn't complete the replace/success path
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load tokenizer state from dictionary."""
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
