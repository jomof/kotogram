"""Worker logic for style training data encoding."""

from typing import Any, Dict, List, Optional

from kotogram.tokenizer import Tokenizer
from train.types import ProcessedSample, Sample

_tokenizer: Optional[Tokenizer] = None


def init_worker(tokenizer_state: Dict[str, Any]) -> None:
    """Initialize worker process with tokenizer state."""
    global _tokenizer
    _tokenizer = Tokenizer()
    _tokenizer.field_vocabs = tokenizer_state["field_vocabs"]
    _tokenizer._frozen = True


def _encode_samples_batch(
    items: List[ProcessedSample],
) -> List[Sample]:
    """Encode samples using the initialized global tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        raise RuntimeError("Worker not initialized. Call init_worker first.")

    samples = []

    for item in items:
        feature_ids = _tokenizer.encode(item.kotogram, add_cls=True, add_to_vocab=False)

        # Map formality_id to value/pragmatic
        f_id = item.formality_id
        if f_id == 5:  # UNPRAGMATIC_FORMALITY
            f_val = 0.0
            f_prag = 0
        else:
            f_val = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}.get(f_id, 0.0)
            f_prag = 1

        sample = Sample(
            feature_ids=feature_ids,
            formality_value=f_val,
            formality_pragmatic=f_prag,
            gender_value=item.gender_value,
            gender_pragmatic=item.gender_pragmatic,
            register_labels=item.register_ids,
            grammaticality_label=item.gram_label,
            original_sentence=item.sentence,
            kotogram=item.kotogram,
        )
        samples.append(sample)

    return samples
