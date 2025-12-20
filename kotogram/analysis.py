"""Formality analysis for Japanese sentences in kotogram format.

This module provides tools to analyze the formality level of Japanese sentences
by examining linguistic features such as verb forms, particles, and auxiliary verbs.
"""

from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING, Set



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


class FormalityLevel(Enum):
    """Formality levels for Japanese sentences."""

    VERY_FORMAL = "very_formal"           # Keigo, honorific language (敬語)
    FORMAL = "formal"                     # Polite/formal (-ます/-です forms)
    NEUTRAL = "neutral"                   # Plain/dictionary form, balanced
    CASUAL = "casual"                     # Colloquial, informal contractions
    VERY_CASUAL = "very_casual"          # Highly casual, slang
    UNPRAGMATIC_FORMALITY = "unpragmatic_formality"  # Mixed/awkward formality


class GenderLevel(Enum):
    """Gender-associated speech patterns for Japanese sentences."""

    MASCULINE = "masculine"               # Male-associated speech (俺, ぜ, ぞ, etc.)
    FEMININE = "feminine"                 # Female-associated speech (わ, の, あたし, etc.)
    NEUTRAL = "neutral"                   # Gender-neutral speech
    UNPRAGMATIC_GENDER = "unpragmatic_gender"  # Mixed/awkward gender markers


class RegisterLevel(Enum):
    """Specific register/dialect classifications."""

    SONKEIGO = "sonkeigo"                 # Honorific (respectful)
    KENJOGO = "kenjogo"                   # Humble
    KANSAIBEN = "kansaiben"               # Kansai dialect
    HAKATABEN = "hakataben"               # Hakata dialect
    KYOSHIGO = "kyoshigo"                 # Teacher style
    NETSLANG = "netslang"                 # Internet slang
    OJOUSAMA = "ojousama"                 # Refined lady style
    GUNTAI = "guntai"                     # Military style
    JOSEIGO = "joseigo"                   # Feminine register
    DANSEIGO = "danseigo"                 # Masculine register
    BURIKKO = "burikko"                   # Burikko (exaggerated cuteness)
    NEUTRAL = "neutral"                   # Standard/Neutral
    TOHOKU = "tohoku"                     # Tohoku dialect
    BUSHI = "bushi"                       # Samurai/Archaic register



def formality(kotogram: str) -> FormalityLevel:
    """Analyze a Japanese sentence and return its formality level.

    This function examines the linguistic features encoded in a kotogram
    representation to determine the overall formality level of the sentence.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.

    Returns:
        FormalityLevel indicating the sentence's formality level, including
        UNPRAGMATIC_FORMALITY if the sentence has an awkward combination
        of different formality levels.

    Examples:
        >>> # Formal sentence: 食べます (I eat - polite)
        >>> kotogram1 = "⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉"
        >>> formality(kotogram1)  # doctest: +SKIP
        <FormalityLevel.FORMAL: 'formal'>

        >>> # Casual sentence: 食べる (I eat - plain)
        >>> kotogram2 = "⌈ˢ食べるᵖv:e-ichidan-ba:terminal⌉"
        >>> formality(kotogram2)  # doctest: +SKIP
        <FormalityLevel.NEUTRAL: 'neutral'>
    """
    from kotogram.validation import ensure_string
    ensure_string(kotogram, "kotogram")

    # Use the trained neural model for prediction
    import torch
    from kotogram.model import FEATURE_FIELDS

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
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model.predict(field_inputs, attention_mask)
        # Check pragmatic first (class 1 is pragmatic)
        is_pragmatic = prediction.formality_pragmatic_probs[0][1].item() > 0.5
        f_val = float(prediction.formality_value[0].item())

    if not is_pragmatic:
        return FormalityLevel.UNPRAGMATIC_FORMALITY

    # Bucket continuous value to discrete level
    # 1.0=VF, 0.5=F, 0.0=N, -0.5=C, -1.0=VC
    if f_val >= 0.75:
        return FormalityLevel.VERY_FORMAL
    elif f_val >= 0.25:
        return FormalityLevel.FORMAL
    elif f_val >= -0.25:
        return FormalityLevel.NEUTRAL
    elif f_val >= -0.75:
        return FormalityLevel.CASUAL
    else:
        return FormalityLevel.VERY_CASUAL






def style(kotogram: str) -> Tuple[FormalityLevel, Optional[float], Set[RegisterLevel], bool]:
    """Analyze a Japanese sentence and return formality, gender, register, and grammaticality.

    This is more efficient than calling formality(), gender(), register(), and grammaticality()
    separately, as it only runs inference once.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.

    Returns:
        Tuple of (FormalityLevel, Optional[float], Set[RegisterLevel], is_grammatic) for the sentence.
        Gender is float [-1, 1] or None if unpragmatic.

    Examples:
        >>> # Formal, neutral sentence: 食べます (I eat - polite)
        >>> kotogram1 = "⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉"
        >>> style(kotogram1)  # doctest: +SKIP
        (<FormalityLevel.FORMAL: 'formal'>, <GenderLevel.NEUTRAL: 'neutral'>, <RegisterLevel.NEUTRAL: 'neutral'>, True)
    """
    from kotogram.validation import ensure_string
    ensure_string(kotogram, "kotogram")

    # Use the trained neural model for prediction (single inference for all)
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
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model.predict(field_inputs, attention_mask)
        
        # Formality Logic
        is_f_pragmatic = prediction.formality_pragmatic_probs[0][1].item() > 0.5
        f_val = float(prediction.formality_value[0].item())
        
        if not is_f_pragmatic:
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

        gender_value = float(prediction.gender_value[0].item())
        gender_prag_idx = int(prediction.gender_pragmatic_probs[0].argmax().item())
        grammaticality_idx = int(prediction.grammaticality_probs[0].argmax().item())

    is_grammatic = grammaticality_idx == 1  # 1 = grammatic, 0 = agrammatic
    
    # Process register probabilities (multi-label)
    detected_registers = set()
    register_probs_list = prediction.register_probs[0].tolist()
    threshold = 0.5
    
    for i, prob in enumerate(register_probs_list):
        if prob > threshold:
            label = REGISTER_ID_TO_LABEL.get(i)
            if label:
                detected_registers.add(label)
                
    if not detected_registers:
        detected_registers.add(RegisterLevel.NEUTRAL)

    # Determine gender result
    gender_res: Optional[float] = None
    if gender_prag_idx == 1: # Pragmatic
        gender_res = gender_value

    return (
        formality_res,
        gender_res,
        detected_registers,
        is_grammatic,
    )


def register(kotogram: str) -> Set[RegisterLevel]:
    """Analyze a Japanese sentence and return its specific register/dialect.

    Args:
        kotogram: Kotogram compact sentence representation.

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
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model.predict(field_inputs, attention_mask)
    
    # Process register probabilities (multi-label)
    detected_registers = set()
    register_probs_list = prediction.register_probs[0].tolist()
    threshold = 0.5
    
    for i, prob in enumerate(register_probs_list):
        if prob > threshold:
            label = REGISTER_ID_TO_LABEL.get(i)
            if label:
                detected_registers.add(label)

    if not detected_registers:
        detected_registers.add(RegisterLevel.NEUTRAL)

    return detected_registers


def gender(kotogram: str) -> Optional[float]:
    """Analyze a Japanese sentence and return its gender-associated speech level.

    This function examines the linguistic features encoded in a kotogram
    representation to determine the gender association of the speech style.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.

    Returns:
        Float value between -1.0 (Masculine) and 1.0 (Feminine), or None if
        the sentence is unpragmatic (awkward gender markers).
        0.0 represents Neutral.

    Examples:
        >>> # Masculine sentence: 俺が行くぜ (I'll go - masculine)
        >>> kotogram1 = "⌈ˢ俺ᵖpn⌉⌈ˢがᵖprt⌉⌈ˢ行くᵖv:u-godan-ka:terminal⌉⌈ˢぜᵖprt:sentence_final_particle⌉"
        >>> gender(kotogram1)  # doctest: +SKIP
        <GenderLevel.MASCULINE: 'masculine'>

        >>> # Feminine sentence: あたしが行くわ (I'll go - feminine)
        >>> kotogram2 = "⌈ˢあたしᵖpn⌉⌈ˢがᵖprt⌉⌈ˢ行くᵖv:u-godan-ka:terminal⌉⌈ˢわᵖprt:sentence_final_particle⌉"
        >>> gender(kotogram2)  # doctest: +SKIP
        <GenderLevel.FEMININE: 'feminine'>
    """
    from kotogram.validation import ensure_string
    ensure_string(kotogram, "kotogram")

    # Use the trained neural model for prediction
    import torch
    from kotogram.model import FEATURE_FIELDS

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
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model.predict(field_inputs, attention_mask)
        
        # Check pragmatic probability (index 1 is pragmatic, 0 is unpragmatic)
        is_pragmatic = prediction.gender_pragmatic_probs[0][1].item() > 0.5
        
        if not is_pragmatic:
            return None
            
        # Return value describing gender (-1=M, 0=N, 1=F)
        return float(prediction.gender_value[0].item())

def grammaticality(kotogram: str) -> bool:
    """Analyze a Japanese sentence and return whether it is grammatically correct.

    This function uses a trained neural model to predict whether a sentence is
    grammatically correct.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.

    Returns:
        True if the sentence is predicted to be grammatically correct,
        False if predicted to be agrammatic (has grammatical errors).

    Examples:
        >>> # A grammatically correct sentence
        >>> kotogram1 = "⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉"
        >>> grammaticality(kotogram1)  # doctest: +SKIP
        True

        >>> # An agrammatic sentence (detected by model)
        >>> kotogram2 = "⌈ˢ食べᵖv:e-ichidan-ba:terminal⌉⌈ˢますᵖauxv-masu:terminal⌉"  # invalid
        >>> grammaticality(kotogram2)  # doctest: +SKIP
        False
    """
    from kotogram.validation import ensure_string
    ensure_string(kotogram, "kotogram")

    # Use the trained neural model for prediction
    import torch
    from kotogram.model import FEATURE_FIELDS

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
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model.predict(field_inputs, attention_mask)
        grammaticality_idx = int(prediction.grammaticality_probs[0].argmax().item())

    # 1 = grammatic, 0 = agrammatic
    return grammaticality_idx == 1