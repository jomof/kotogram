from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Sample:
    """Single data sample with features and labels."""

    feature_ids: Dict[str, List[int]]
    formality_value: float
    formality_pragmatic: int
    gender_value: float
    gender_pragmatic: int
    register_labels: List[int] = field(default_factory=lambda: [0])
    grammaticality_label: int = 1
    original_sentence: str = ""
    kotogram: str = ""

    @property
    def seq_len(self) -> int:
        """Get sequence length."""
        first = next(iter(self.feature_ids.values()))
        return len(first)


@dataclass
class ProcessedSample:
    """Processed sample result from labeling stage."""

    sentence: str
    kotogram: str
    formality_id: int
    gender_value: float
    gender_pragmatic: int
    register_ids: List[int]
    gram_label: int
    success: int
