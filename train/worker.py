"""Worker logic for style training data encoding."""

import array
from typing import Any, Dict, List, Optional

import torch

from kotogram.tokenizer import Tokenizer
from train.kc import compute_kc_targets
from train.types import ProcessedSample, Sample

_TOKENIZER: Optional[Tokenizer] = None


def init_worker(tokenizer_state: Dict[str, Any]) -> None:
    """Initialize worker process with tokenizer state."""
    # pylint: disable=global-statement
    global _TOKENIZER
    _TOKENIZER = Tokenizer()
    # pylint: disable=protected-access
    _TOKENIZER.field_vocabs = tokenizer_state["field_vocabs"]
    _TOKENIZER._frozen = True


def _encode_samples_batch(
    items: List[ProcessedSample],
) -> List[Sample]:
    """Encode samples using the initialized global tokenizer."""
    if _TOKENIZER is None:
        raise RuntimeError("Worker not initialized. Call init_worker first.")

    samples = []

    for item in items:
        feature_ids = _TOKENIZER.encode_fast(item.kotogram)
        kc_targets = compute_kc_targets(feature_ids)

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
            kc_targets=kc_targets,
        )
        samples.append(sample)

    return samples


def encode_batch_fast(
    items: List[ProcessedSample],
) -> Dict[str, Any]:
    """Encode samples returning column-oriented data for efficient tensor construction.

    Returns:
        Dict with keys:
        - features: Dict[str, List[List[int]]]
        - kc: List[Dict[str, Any]]
        - f_val: List[float]
        - f_prag: List[int]
        - g_val: List[float]
        - g_prag: List[int]
        - reg: List[List[int]]
    - gram: List[int]
    """
    # pylint: disable=too-many-locals
    if _TOKENIZER is None:
        raise RuntimeError("Worker not initialized. Call init_worker first.")

    # columnar storage (flattened) using array.array for IPC efficiency
    c_features_flat: Dict[str, array.array] = {}
    c_features_lens: Dict[str, array.array] = {}

    f_vals = array.array("f")
    f_prags = array.array("B")
    g_vals = array.array("f")
    g_prags = array.array("B")

    # Flattened register
    r_ids_flat = array.array("B")
    r_ids_lens = array.array("B")

    grams = array.array("B")

    # Dynamic KC columns
    c_kc_ids: Dict[str, array.array] = {}
    c_kc_counts: Dict[str, array.array] = {}

    for item in items:
        # Encode
        enc = _TOKENIZER.encode_fast(item.kotogram)

        # Init columns if needed
        if not c_features_flat:
            for k in enc.keys():
                c_features_flat[k] = array.array("I")
                c_features_lens[k] = array.array("I")

            # Helper to init KC dicts implicitly when needed below?

        for k, v in enc.items():
            c_features_flat[k].extend(v)
            c_features_lens[k].append(len(v))

        # KC Targets
        kc = compute_kc_targets(enc)

        # Flatten KC into columns
        for k, v in kc.items():
            if k not in c_kc_ids:
                c_kc_ids[k] = array.array("I")
                c_kc_counts[k] = array.array("I")

            # Append data
            c_kc_ids[k].extend(v)
            c_kc_counts[k].append(len(v))

        # Formality
        f_id = item.formality_id
        if f_id == 5:
            f_vals.append(0.0)
            f_prags.append(0)
        else:
            val = {0: 1.0, 1: 0.5, 2: 0.0, 3: -0.5, 4: -1.0}.get(f_id, 0.0)
            f_vals.append(val)
            f_prags.append(1)

        # Gender
        val_g = item.gender_value if item.gender_value is not None else 0.0
        prag_g = item.gender_pragmatic if item.gender_pragmatic is not None else 0
        g_vals.append(val_g)
        g_prags.append(prag_g)

        # Register (Flattened)
        reg_list = item.register_ids
        if reg_list:
            r_ids_flat.extend(reg_list)
            r_ids_lens.append(len(reg_list))
        else:
            r_ids_lens.append(0)

        # Gram
        grams.append(item.gram_label)

    return {
        "features_flat": c_features_flat,
        "features_lens": c_features_lens,
        "kc_ids": c_kc_ids,
        "kc_counts": c_kc_counts,
        "f_val": f_vals,
        "f_prag": f_prags,
        "g_val": g_vals,
        "g_prag": g_prags,
        "reg_flat": r_ids_flat,
        "reg_lens": r_ids_lens,
        "gram": grams,
    }


def _worker_init_fn(_: int) -> None:
    """Worker initialization function to limit per-worker threads."""
    torch.set_num_threads(1)
    try:
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
    except RuntimeError as e:  # worker-init=special-carveout
        # This can happen if the runtime is already initialized with >1 thread.
        # It's safe to ignore if we simply cannot change it now.
        if "cannot set number of interop threads" not in str(e):
            raise e
