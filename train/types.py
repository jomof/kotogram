"""Data types for Kotogram training."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Sample:
    """A single training sample."""

    feature_ids: Dict[str, List[int]]
    formality_value: float = 0.5
    formality_pragmatic: int = 1
    gender_value: float = 0.5
    gender_pragmatic: int = 1
    grammaticality_label: int = 1
    register_labels: List[int] = field(default_factory=lambda: [0])
    original_sentence: str = ""
    kotogram: str = ""
    kc_targets: Dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        """Get sequence length."""
        if not self.feature_ids:
            return 0
        first = next(iter(self.feature_ids.values()))
        return len(first)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_ids": self.feature_ids,
            "formality_value": self.formality_value,
            "formality_pragmatic": self.formality_pragmatic,
            "gender_value": self.gender_value,
            "gender_pragmatic": self.gender_pragmatic,
            "grammaticality_label": self.grammaticality_label,
            "register_labels": self.register_labels,
            "original_sentence": self.original_sentence,
            "kotogram": self.kotogram,
        }


@dataclass
class ProcessedSample:
    """A sample with associated metadata before indexing."""

    sentence: str
    kotogram: str
    formality_id: int
    gender_value: float
    gender_pragmatic: int
    register_ids: List[int]
    gram_label: int
    success: int
    feature_ids: Optional[Dict[str, List[int]]] = None

    def to_cache_tuple(
        self, feature_ids_override: Optional[Dict[str, List[int]]] = None
    ) -> Any:
        """Convert to cache tuple format (used by dataset cache)."""
        return (
            self.sentence,
            self.kotogram,
            self.formality_id,
            self.gender_value,
            self.gender_pragmatic,
            self.register_ids,
            self.gram_label,
            feature_ids_override
            if feature_ids_override is not None
            else self.feature_ids,
        )
