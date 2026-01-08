"""Formality analysis for Japanese sentences in kotogram format.

This module provides tools to analyze the formality level of Japanese sentences
by examining linguistic features such as verb forms, particles, and auxiliary verbs.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from kotogram.constants import FormalityLevel, GenderLevel, RegisterLevel

from . import locations

# This is required for cross-language furigana support to work on typescript
# canary CI machine without installing pytorch.
if TYPE_CHECKING:
    from kotogram.model import StyleClassifier
    from kotogram.tokenizer import Tokenizer


# Type alias for Knowledge Component distribution (KC ID -> Probability)
KCDistribution = Dict[int, float]


class StyleAnalyzer:
    """Encapsulates style analysis model and tokenizer state."""

    def __init__(self) -> None:
        self._model: Optional["StyleClassifier"] = None
        self._tokenizer: Optional["Tokenizer"] = None
        self._custom_model_dir: Optional[str] = None

    def set_model_dir(self, path: str) -> None:
        """Set a custom directory to load the model from."""
        self._custom_model_dir = path
        # Reset cache if dir changes
        self._model = None
        self._tokenizer = None

    def load(self) -> Tuple["StyleClassifier", "Tokenizer"]:
        # pylint: disable=import-outside-toplevel
        """Load and cache the style classifier model."""
        if self._model is None or self._tokenizer is None:
            from kotogram.model import load_default_style_model, load_model

            # Priority 0: Custom model dir from CLI
            if self._custom_model_dir:
                if not os.path.exists(os.path.join(self._custom_model_dir, "model.pt")):
                    raise FileNotFoundError(
                        f"Custom model not found at {self._custom_model_dir}"
                    )
                self._model, self._tokenizer = load_model(self._custom_model_dir)

            # Priority 1: Check for local model in style-output dir
            elif os.path.exists(
                os.path.join(locations.get_style_output_dir(), "model.pt")
            ):
                model_dir = locations.get_style_output_dir()
                self._model, self._tokenizer = load_model(model_dir)
            else:
                # Priority 2: Fall back to package-default model
                self._model, self._tokenizer = load_default_style_model()

        return self._model, self._tokenizer

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    def is_available(self) -> bool:
        """Check if a model is available to be loaded."""
        if self.is_loaded():
            return True

        if self._custom_model_dir:
            return os.path.exists(os.path.join(self._custom_model_dir, "model.pt"))

        from kotogram.model import is_default_style_model_available

        return is_default_style_model_available()


# Global singleton instance
_ANALYZER = StyleAnalyzer()


def check_model_available() -> bool:
    """Check if the style model is available for loading."""
    return _ANALYZER.is_available()


@dataclass
class GrammarAnalysis:
    """Consolidated analysis result for a Japanese sentence."""

    # Input
    kotogram: str

    # Formality
    formality: FormalityLevel
    formality_score: float  # -1.0 to 1.0 (continuous prediction)
    formality_is_pragmatic: bool

    # Gender
    gender: GenderLevel
    gender_score: float  # -1.0 (Masculine) to 1.0 (Feminine)
    gender_is_pragmatic: bool

    # Register
    registers: Set[RegisterLevel]  # Set of detected registers
    register_scores: Dict[RegisterLevel, float]  # All registers and their scores

    # Grammaticality
    is_grammatic: bool
    grammaticality_score: float  # Probability of being grammatic
    kc_top: Optional[KCDistribution] = None  # Top-K KC {id: prob}

    def to_json(self) -> str:
        """Serialize analysis result to JSON string."""
        d = asdict(self)
        # Convert Enums to strings
        d["formality"] = self.formality.value
        d["gender"] = self.gender.value
        # Convert Sets to sorted lists of strings
        d["registers"] = sorted([r.value for r in self.registers])
        # Convert Dict keys from Enums to strings
        d["register_scores"] = {k.value: v for k, v in self.register_scores.items()}
        # kc_top: Dict[int, float] -> json.dump will convert int keys to strings automatically
        if self.kc_top is None:
            del d["kc_top"]
        return json.dumps(d, ensure_ascii=False)


def grammars(kotograms: List[str]) -> List[GrammarAnalysis]:
    # pylint: disable=too-many-locals
    """Analyze a list of Japanese sentences in batch and return results.

    This function is significantly more efficient than calling grammar()
    repeatedly for multiple sentences as it performs single model inference pass.

    Args:
        kotograms: List of kotogram compact sentence representations.

    Returns:
        List of GrammarAnalysis objects.
    """
    if not kotograms:
        return []

    from kotogram.validation import ensure_string

    for k in kotograms:
        ensure_string(k, "kotogram")

    # Use the trained neural model for prediction
    import torch

    from kotogram.model import REGISTER_ID_TO_LABEL
    from kotogram.tokenizer import FEATURE_FIELDS

    model, tokenizer = _ANALYZER.load()

    # Encode all kotograms
    encoded_list = [tokenizer.encode(k) for k in kotograms]

    # Padding logic to handle variable lengths in batch
    max_len = max(len(e[FEATURE_FIELDS[0]]) for e in encoded_list)
    batch_size = len(kotograms)

    field_inputs = {}
    for field in FEATURE_FIELDS:
        # 0 is the PAD_TOKEN id
        batch_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, encoded in enumerate(encoded_list):
            ids = encoded[field]
            batch_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        field_inputs[f"input_ids_{field}"] = batch_ids

    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, encoded in enumerate(encoded_list):
        attention_mask[i, : len(encoded[FEATURE_FIELDS[0]])] = 1

    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model.predict(field_inputs, attention_mask)
        # Get Interpretable KCs separately
        kc_top_results = None
        if model.config.kc_enabled:
            kc_top_results = model.predict_kcs_top(
                field_inputs, attention_mask, min_prob=0.5, topk=16
            )

    results = []
    for i in range(batch_size):
        # 1. Formality
        f_val = float(prediction.formality_value[i].item())
        f_is_pragmatic = prediction.formality_pragmatic_probs[i][1].item() > 0.5

        if not f_is_pragmatic:
            formality_res = FormalityLevel.UNPRAGMATIC_FORMALITY
        elif f_val >= 0.75:
            formality_res = FormalityLevel.VERY_FORMAL
        elif f_val >= 0.25:
            formality_res = FormalityLevel.FORMAL
        elif f_val >= -0.25:
            formality_res = FormalityLevel.NEUTRAL
        elif f_val >= -0.75:
            formality_res = FormalityLevel.CASUAL
        else:
            formality_res = FormalityLevel.VERY_CASUAL

        # 2. Gender
        g_val = float(prediction.gender_value[i].item())
        g_is_pragmatic = prediction.gender_pragmatic_probs[i][1].item() > 0.5

        if not g_is_pragmatic:
            gender_res = GenderLevel.UNPRAGMATIC_GENDER
        elif g_val <= -0.5:
            gender_res = GenderLevel.MASCULINE
        elif g_val >= 0.5:
            gender_res = GenderLevel.FEMININE
        else:
            gender_res = GenderLevel.NEUTRAL

        # 3. Register
        detected_register_scores = {}
        for reg_id, score in enumerate(prediction.register_probs[i]):
            label = REGISTER_ID_TO_LABEL.get(reg_id)
            score_val = float(score.item())
            if label and score_val > 0.9:
                detected_register_scores[label] = score_val

        detected_registers = set(detected_register_scores.keys())
        if not detected_registers:
            detected_registers.add(RegisterLevel.NEUTRAL)
            # We don't have a model score for NEUTRAL usually as it's the fallback,
            # but if we wanted to provide one we could, for now we just leave it as is
            # or maybe add it with score 1.0 if it's the only one?
            # The prompt says "only return register_scores for detected registers".

        # 4. Grammaticality
        gram_score = float(prediction.grammaticality_probs[i][1].item())
        is_grammatic = gram_score > 0.5

        kc_top_sample = None
        if kc_top_results is not None:
            # Convert list of (int, float) tuples to {int: float} dict
            # We strictly cast int(k_id) to ensure it's an int.
            kc_top_sample = {int(k_id): prob for k_id, prob in kc_top_results[i]}
        results.append(
            GrammarAnalysis(
                kotogram=kotograms[i],
                formality=formality_res,
                formality_score=f_val,
                formality_is_pragmatic=f_is_pragmatic,
                gender=gender_res,
                gender_score=g_val,
                gender_is_pragmatic=g_is_pragmatic,
                registers=detected_registers,
                register_scores=detected_register_scores,
                is_grammatic=is_grammatic,
                grammaticality_score=gram_score,
                kc_top=kc_top_sample,
            )
        )

    return results


def grammar(kotogram: str) -> GrammarAnalysis:
    """Analyze a Japanese sentence and return a consolidated GrammarAnalysis.

    This function runs a single inference pass through the neural model to
    determine formality, gender association, specific registers, and
    grammaticality.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.

    Returns:
        GrammarAnalysis object containing all linguistic analysis results.

    Examples:
        >>> # Formal sentence: 食べます (I eat - polite)
        >>> kotogram1 = "⌈ˢ食べᵖverb:lower-ichidan-ba:continuative⌉⌈ˢますᵖaux-verb-masu:terminal⌉"
        >>> res = grammar(kotogram1)  # doctest: +SKIP
        >>> res.formality
        <FormalityLevel.FORMAL: 'formal'>
        >>> res.is_grammatic
        True
    """
    return grammars([kotogram])[0]
