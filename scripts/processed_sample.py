from typing import NamedTuple, List, Optional

class ProcessedSample(NamedTuple):
    sentence: str
    sentence_id: str
    kotogram: str
    formality_id: int
    gender_value: float
    gender_pragmatic: int
    register_ids: List[int]
    gram_label: int
    success: int
