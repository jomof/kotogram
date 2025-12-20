"""Formality analysis for Japanese sentences in kotogram format.

This module provides tools to analyze the formality level of Japanese sentences
by examining linguistic features such as verb forms, particles, and auxiliary verbs.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Set, TYPE_CHECKING
from kotogram.constants import FormalityLevel, GenderLevel, RegisterLevel

# This is required for cross-language furigana support to work on typescript
# canary CI machine without installing pytorch.
if TYPE_CHECKING:
    from kotogram.model import StyleClassifier, Tokenizer

# Global cache for loaded model (lazy loading)
_style_model: Optional['StyleClassifier'] = None
_style_tokenizer: Optional['Tokenizer'] = None
_style_model_path: str = "models/style"


def _load_style_model() -> Tuple['StyleClassifier', 'Tokenizer']:
    """Load and cache the style classifier model.

    Returns:
        Tuple of (model, tokenizer) for style classification.

    Raises:
        FileNotFoundError: If model files are not found at the expected path.
    """
    global _style_model, _style_tokenizer

    if _style_model is None or _style_tokenizer is None:
        from kotogram.model import load_default_style_model
        _style_model, _style_tokenizer = load_default_style_model()

    return _style_model, _style_tokenizer


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

    def to_json(self) -> str:
        """Serialize analysis result to JSON string."""
        d = asdict(self)
        # Convert Enums to strings
        d['formality'] = self.formality.value
        d['gender'] = self.gender.value
        # Convert Sets to sorted lists of strings
        d['registers'] = sorted([r.value for r in self.registers])
        # Convert Dict keys from Enums to strings
        d['register_scores'] = {k.value: v for k, v in self.register_scores.items()}
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "GrammarAnalysis":
        """Deserialize analysis result from JSON string."""
        d = json.loads(json_str)

        # Map strings back to Enums
        d['formality'] = FormalityLevel(d['formality'])
        d['gender'] = GenderLevel(d['gender'])
        d['registers'] = {RegisterLevel(r) for r in d['registers']}
        d['register_scores'] = {RegisterLevel(k): v for k, v in d['register_scores'].items()}

        return cls(**d)


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
    from kotogram.validation import ensure_string
    ensure_string(kotogram, "kotogram")

    # Use the trained neural model for prediction
    import torch
    from kotogram.model import FEATURE_FIELDS, REGISTER_ID_TO_LABEL

    model, tokenizer = _load_style_model()

    # Encode the kotogram
    feature_ids = tokenizer.encode(kotogram, add_cls=True, add_to_vocab=False)

    # Create batch tensors
    field_inputs = {
        f'input_ids_{field}': torch.tensor([feature_ids[field]], dtype=torch.long)
        for field in FEATURE_FIELDS
    }
    attention_mask = torch.ones(1, len(feature_ids[FEATURE_FIELDS[0]]), dtype=torch.long)

    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model.predict(field_inputs, attention_mask)

        # 1. Formality
        f_val = float(prediction.formality_value[0].item())
        f_is_pragmatic = prediction.formality_pragmatic_probs[0][1].item() > 0.5

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
        g_val = float(prediction.gender_value[0].item())
        g_is_pragmatic = prediction.gender_pragmatic_probs[0][1].item() > 0.5

        if not g_is_pragmatic:
            gender_res = GenderLevel.UNPRAGMATIC_GENDER
        elif g_val <= -0.5:
            gender_res = GenderLevel.MASCULINE
        elif g_val >= 0.5:
            gender_res = GenderLevel.FEMININE
        else:
            gender_res = GenderLevel.NEUTRAL

        # 3. Register
        # All scores
        all_register_scores = {}
        for reg_id, score in enumerate(prediction.register_probs[0]):
            label = REGISTER_ID_TO_LABEL.get(reg_id)
            if label:
                all_register_scores[label] = float(score.item())

        # Detected registers (score > 0.5)
        detected_registers = {
            label for label, score in all_register_scores.items() if score > 0.5
        }
        if not detected_registers:
            detected_registers.add(RegisterLevel.NEUTRAL)

        # 4. Grammaticality
        gram_score = float(prediction.grammaticality_probs[0][1].item())
        is_grammatic = gram_score > 0.5

    return GrammarAnalysis(
        kotogram=kotogram,
        formality=formality_res,
        formality_score=f_val,
        formality_is_pragmatic=f_is_pragmatic,
        gender=gender_res,
        gender_score=g_val,
        gender_is_pragmatic=g_is_pragmatic,
        registers=detected_registers,
        register_scores=all_register_scores,
        is_grammatic=is_grammatic,
        grammaticality_score=gram_score,
    )
