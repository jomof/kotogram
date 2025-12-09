"""Formality analysis for Japanese sentences in kotogram format.

This module provides tools to analyze the formality level of Japanese sentences
by examining linguistic features such as verb forms, particles, and auxiliary verbs.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from kotogram.kotogram import split_kotogram, extract_token_features

if TYPE_CHECKING:
    from kotogram.style_classifier import StyleClassifier, Tokenizer

# Global cache for loaded model (lazy loading)
_style_model: Optional['StyleClassifier'] = None
_style_tokenizer: Optional['Tokenizer'] = None
_style_model_path: str = "models/style"


def _load_style_model() -> tuple['StyleClassifier', 'Tokenizer']:
    """Load and cache the style classifier model.

    Returns:
        Tuple of (model, tokenizer) for style classification.

    Raises:
        FileNotFoundError: If model files are not found at the expected path.
    """
    global _style_model, _style_tokenizer

    if _style_model is None or _style_tokenizer is None:
        from kotogram.style_classifier import load_model
        _style_model, _style_tokenizer = load_model(_style_model_path)

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


def formality(kotogram: str, use_model: bool = False) -> FormalityLevel:
    """Analyze a Japanese sentence and return its formality level.

    This function examines the linguistic features encoded in a kotogram
    representation to determine the overall formality level of the sentence.
    It looks for:
    - Polite/formal verb endings (ます, です)
    - Honorific and humble forms (keigo)
    - Plain/dictionary forms
    - Casual contractions and colloquialisms
    - Mixed formality patterns that sound unpragmatic

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.
        use_model: If True, use the trained neural model for prediction instead
                  of rule-based analysis. The model must be available at the
                  default model path (models/style). Default is False.

    Returns:
        FormalityLevel indicating the sentence's formality level, including
        UNPRAGMATIC_FORMALITY if the sentence has an awkward combination
        of different formality levels.

    Examples:
        >>> # Formal sentence: 食べます (I eat - polite)
        >>> kotogram1 = "⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉"
        >>> formality(kotogram1)
        <FormalityLevel.FORMAL: 'formal'>

        >>> # Casual sentence: 食べる (I eat - plain)
        >>> kotogram2 = "⌈ˢ食べるᵖv:e-ichidan-ba:terminal⌉"
        >>> formality(kotogram2)
        <FormalityLevel.NEUTRAL: 'neutral'>

        >>> # Unpragmatic: Mixed formality
        >>> kotogram3 = "⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉⌈ˢよᵖprt⌉⌈ˢ〜ᵖauxs⌉"
        >>> formality(kotogram3)
        <FormalityLevel.UNPRAGMATIC_FORMALITY: 'unpragmatic_formality'>

        >>> # Using the trained model
        >>> formality(kotogram1, use_model=True)  # doctest: +SKIP
        <FormalityLevel.FORMAL: 'formal'>
    """
    if use_model:
        # Use the trained neural model for prediction
        import torch
        from kotogram.style_classifier import FEATURE_FIELDS

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
            formality_probs, _ = model.predict(field_inputs, attention_mask)
            formality_idx = int(formality_probs[0].argmax().item())

        # Map model output index to FormalityLevel
        # Model uses: 0=very_formal, 1=formal, 2=neutral, 3=casual, 4=very_casual, 5=unpragmatic
        formality_map = {
            0: FormalityLevel.VERY_FORMAL,
            1: FormalityLevel.FORMAL,
            2: FormalityLevel.NEUTRAL,
            3: FormalityLevel.CASUAL,
            4: FormalityLevel.VERY_CASUAL,
            5: FormalityLevel.UNPRAGMATIC_FORMALITY,
        }
        return formality_map.get(formality_idx, FormalityLevel.NEUTRAL)

    # Rule-based analysis
    # Split into tokens and extract linguistic features
    tokens = split_kotogram(kotogram)

    if not tokens:
        return FormalityLevel.NEUTRAL

    # Extract features from each token
    features = []
    for token in tokens:
        feature = extract_token_features(token)
        if feature:
            features.append(feature)

    # Analyze formality based on features
    return _analyze_formality_features(features)


def _analyze_formality_features(features: List[Dict[str, str]]) -> FormalityLevel:
    """Analyze extracted features to determine formality level.

    Args:
        features: List of feature dictionaries from tokens

    Returns:
        FormalityLevel based on the combination of features
    """
    if not features:
        return FormalityLevel.NEUTRAL

    # Formality indicators
    has_formal = False           # ます/です forms
    has_very_formal = False      # Honorific/humble forms (keigo)
    has_casual = False           # Plain forms with casual markers
    has_very_casual = False      # Very casual particles/forms

    # Track sentence-final particles for context
    sentence_final_particles = []

    for i, feature in enumerate(features):
        pos = feature.get('pos', '')
        pos_detail1 = feature.get('pos_detail1', '')
        conjugated_type = feature.get('conjugated_type', '')
        surface = feature.get('surface', '')

        # Check for formal auxiliary verbs (ます/です)
        if conjugated_type in ['auxv-masu', 'auxv-desu']:
            has_formal = True

        # Check for ください and なさい - formal but not very formal when imperative
        lemma = feature.get('lemma', '')
        conjugated_form = feature.get('conjugated_form', '')

        if lemma in ['くださる', '下さる']:
            # ください (imperative of くださる) is standard formal/polite
            # Only mark as very formal if it's NOT the imperative form
            if conjugated_form == 'imperative':
                has_formal = True
            else:
                # くださる in other forms (e.g., くださった, くださいます) is keigo
                has_very_formal = True

        if lemma in ['なさる', '為さる']:
            # なさい (imperative of なさる) is polite imperative
            # Only mark as very formal if it's NOT the imperative form
            if conjugated_form == 'imperative':
                has_formal = True
            else:
                # なさる in other forms is honorific keigo
                has_very_formal = True

        # Check for other very formal/honorific forms
        # Honorific verbs often have specific patterns or use special verb forms
        # Common indicators: いらっしゃる, おっしゃる, etc.
        if lemma in ['いらっしゃる', 'おっしゃる', 'ご覧になる', 'お～になる']:
            has_very_formal = True
        # Humble verbs (謙譲語)
        # Note: Sudachi may use potential forms like いただける
        if lemma in ['いたす', '致す', 'まいる', '申す', '申し上げる', 'お～する', 'いただく', '頂く', 'いただける']:
            has_very_formal = True

        # Check for casual copula (だ)
        # Only mark as casual for specific forms:
        # - terminal: だ at sentence end (not in embedded clauses)
        # - conjunctive-geminate: だっ (becomes だった, だって)
        # - volitional-presumptive: だろう
        # Do NOT mark as casual:
        # - attributive: な (normal adjectival form)
        # - conjunctive-ni: に (normal adverbial form)
        # - conjunctive: で (normal connective)
        # - terminal だ in embedded clauses (mid-sentence)
        if conjugated_type == 'auxv-da':
            casual_forms = ['conjunctive-geminate', 'volitional-presumptive']
            if conjugated_form in casual_forms:
                has_casual = True
            elif conjugated_form == 'terminal':
                # Terminal だ is casual if followed only by punctuation/brackets
                # This handles quoted speech like 「好きだ。」
                is_at_clause_end = True
                for j in range(i + 1, len(features)):
                    next_pos = features[j].get('pos', '')
                    next_surface = features[j].get('surface', '')
                    # Skip punctuation and brackets
                    if next_pos == 'auxs' or next_surface in ['」', '』', ')', '）']:
                        continue
                    # If we hit another token, だ is mid-sentence
                    is_at_clause_end = False
                    break
                if is_at_clause_end:
                    has_casual = True

        # Check for very casual auxiliary verbs
        if conjugated_type in ['auxv-ja', 'auxv-nanda', 'auxv-hin', 'auxv-hen', 'auxv-nsu']:
            has_very_casual = True

        # Sudachi may parse じゃ as conj instead of auxv-ja
        if pos == 'conj' and surface == 'じゃ':
            has_very_casual = True

        # Check for sentence-final particles
        if pos == 'prt' and pos_detail1 == 'sentence_final_particle':
            sentence_final_particles.append(surface)

    # Analyze sentence-final particles for casual/very casual markers
    very_casual_particles = ['ぜ', 'ぞ', 'ぞい', 'さ']  # Masculine/rough particles
    # Casual particles include base forms and lengthened variants (なあ, ねえ, よー, etc.)
    casual_particles = [
        'よ', 'ね', 'の', 'わ', 'な',  # Base forms
        'なあ', 'なー', 'ねえ', 'ねー',  # Lengthened な/ね
        'よお', 'よー', 'わあ', 'わー',  # Lengthened よ/わ
        'かしら',  # Feminine wondering particle
        'かい',  # Casual question particle (masculine)
        'もの', 'もん',  # Explanatory particle (feminine casual)
    ]
    # Note: These particles are acceptable with formal forms, but make plain forms casual

    # Combine adjacent sentence-final particles (e.g., か+い -> かい)
    combined_particles = ''.join(sentence_final_particles)

    # Check combined particles first for multi-character sequences
    for particle in casual_particles:
        if len(particle) > 1 and particle in combined_particles:
            if not has_formal:
                has_casual = True
    for particle in very_casual_particles:
        if len(particle) > 1 and particle in combined_particles:
            if has_formal:
                has_very_casual = True
            else:
                has_casual = True

    for particle in sentence_final_particles:
        if particle in very_casual_particles:
            # Very casual particles - inappropriate with formal forms
            if has_formal:
                has_very_casual = True  # Unpragmatic mixing
            else:
                has_casual = True
        elif particle in casual_particles:
            # Casual particles - acceptable with formal, but make plain forms casual
            if not has_formal:
                # With plain forms, these particles create casual speech
                has_casual = True
            # If has_formal, these are acceptable and don't change the formality

    # Decision logic based on features

    # Very formal (keigo) takes precedence
    if has_very_formal:
        return FormalityLevel.VERY_FORMAL

    # Check for unpragmatic formality mixing
    # Formal forms mixed with very casual markers is unpragmatic
    if has_formal and has_very_casual:
        return FormalityLevel.UNPRAGMATIC_FORMALITY

    # Formal forms (ます/です) - even with acceptable particles
    if has_formal:
        return FormalityLevel.FORMAL

    # Very casual markers without formal forms
    if has_very_casual:
        return FormalityLevel.VERY_CASUAL

    # Casual forms (だ copula or casual markers)
    if has_casual:
        return FormalityLevel.CASUAL

    # Default to neutral for plain forms
    return FormalityLevel.NEUTRAL


def style(kotogram: str, use_model: bool = False) -> Tuple[FormalityLevel, GenderLevel]:
    """Analyze a Japanese sentence and return both formality and gender levels.

    This is more efficient than calling formality() and gender() separately
    when using the model, as it only runs inference once.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.
        use_model: If True, use the trained neural model for prediction instead
                  of rule-based analysis. Default is False.

    Returns:
        Tuple of (FormalityLevel, GenderLevel) for the sentence.

    Examples:
        >>> # Formal, neutral sentence: 食べます (I eat - polite)
        >>> kotogram1 = "⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉"
        >>> style(kotogram1)
        (<FormalityLevel.FORMAL: 'formal'>, <GenderLevel.NEUTRAL: 'neutral'>)

        >>> # Using the trained model
        >>> style(kotogram1, use_model=True)  # doctest: +SKIP
        (<FormalityLevel.FORMAL: 'formal'>, <GenderLevel.NEUTRAL: 'neutral'>)
    """
    if use_model:
        # Use the trained neural model for prediction (single inference for both)
        import torch
        from kotogram.style_classifier import FEATURE_FIELDS

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
            formality_probs, gender_probs = model.predict(field_inputs, attention_mask)
            formality_idx = int(formality_probs[0].argmax().item())
            gender_idx = int(gender_probs[0].argmax().item())

        # Map model output indices to enum values
        formality_map = {
            0: FormalityLevel.VERY_FORMAL,
            1: FormalityLevel.FORMAL,
            2: FormalityLevel.NEUTRAL,
            3: FormalityLevel.CASUAL,
            4: FormalityLevel.VERY_CASUAL,
            5: FormalityLevel.UNPRAGMATIC_FORMALITY,
        }
        gender_map = {
            0: GenderLevel.MASCULINE,
            1: GenderLevel.FEMININE,
            2: GenderLevel.NEUTRAL,
            3: GenderLevel.UNPRAGMATIC_GENDER,
        }
        return (
            formality_map.get(formality_idx, FormalityLevel.NEUTRAL),
            gender_map.get(gender_idx, GenderLevel.NEUTRAL),
        )

    # Rule-based analysis
    return formality(kotogram), gender(kotogram)


def gender(kotogram: str, use_model: bool = False) -> GenderLevel:
    """Analyze a Japanese sentence and return its gender-associated speech level.

    This function examines the linguistic features encoded in a kotogram
    representation to determine the gender association of the speech style.
    It looks for:
    - Masculine pronouns (俺, 僕) and particles (ぜ, ぞ, ぞい)
    - Feminine pronouns (あたし) and particles (わ, の with rising intonation)
    - Mixed patterns that sound unpragmatic

    Note: These are sociolinguistic associations, not prescriptive rules.
    Modern Japanese speakers may use various combinations regardless of gender.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.
        use_model: If True, use the trained neural model for prediction instead
                  of rule-based analysis. The model must be available at the
                  default model path (models/style). Default is False.

    Returns:
        GenderLevel indicating the sentence's gender-associated speech level,
        including UNPRAGMATIC_GENDER if the sentence has an awkward combination
        of different gender markers.

    Examples:
        >>> # Masculine sentence: 俺が行くぜ (I'll go - masculine)
        >>> kotogram1 = "⌈ˢ俺ᵖpn⌉⌈ˢがᵖprt⌉⌈ˢ行くᵖv:u-godan-ka:terminal⌉⌈ˢぜᵖprt:sentence_final_particle⌉"
        >>> gender(kotogram1)
        <GenderLevel.MASCULINE: 'masculine'>

        >>> # Feminine sentence: あたしが行くわ (I'll go - feminine)
        >>> kotogram2 = "⌈ˢあたしᵖpn⌉⌈ˢがᵖprt⌉⌈ˢ行くᵖv:u-godan-ka:terminal⌉⌈ˢわᵖprt:sentence_final_particle⌉"
        >>> gender(kotogram2)
        <GenderLevel.FEMININE: 'feminine'>

        >>> # Neutral sentence: 私が行きます (I'll go - neutral/polite)
        >>> kotogram3 = "⌈ˢ私ᵖpn⌉⌈ˢがᵖprt⌉⌈ˢ行きᵖv:u-godan-ka:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉"
        >>> gender(kotogram3)
        <GenderLevel.NEUTRAL: 'neutral'>

        >>> # Using the trained model
        >>> gender(kotogram1, use_model=True)  # doctest: +SKIP
        <GenderLevel.MASCULINE: 'masculine'>
    """
    if use_model:
        # Use the trained neural model for prediction
        import torch
        from kotogram.style_classifier import FEATURE_FIELDS

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
            _, gender_probs = model.predict(field_inputs, attention_mask)
            gender_idx = int(gender_probs[0].argmax().item())

        # Map model output index to GenderLevel
        # Model uses: 0=masculine, 1=feminine, 2=neutral, 3=unpragmatic
        gender_map = {
            0: GenderLevel.MASCULINE,
            1: GenderLevel.FEMININE,
            2: GenderLevel.NEUTRAL,
            3: GenderLevel.UNPRAGMATIC_GENDER,
        }
        return gender_map.get(gender_idx, GenderLevel.NEUTRAL)

    # Rule-based analysis
    # Split into tokens and extract linguistic features
    tokens = split_kotogram(kotogram)

    if not tokens:
        return GenderLevel.NEUTRAL

    # Extract features from each token
    features = []
    for token in tokens:
        feature = extract_token_features(token)
        if feature:
            features.append(feature)

    # Analyze gender based on features
    return _analyze_gender_features(features)


def _analyze_gender_features(features: List[Dict[str, str]]) -> GenderLevel:
    """Analyze extracted features to determine gender-associated speech level.

    Args:
        features: List of feature dictionaries from tokens

    Returns:
        GenderLevel based on the combination of features
    """
    if not features:
        return GenderLevel.NEUTRAL

    # Gender indicators
    has_masculine = False
    has_feminine = False

    # Track particles and their positions for pattern detection
    particle_sequence = []  # List of (index, surface, pos_detail1)

    for i, feature in enumerate(features):
        pos = feature.get('pos', '')
        pos_detail1 = feature.get('pos_detail1', '')
        surface = feature.get('surface', '')
        lemma = feature.get('lemma', '')
        conjugated_type = feature.get('conjugated_type', '')
        conjugated_form = feature.get('conjugated_form', '')

        # Check for masculine pronouns
        # 俺 (ore) - strongly masculine
        # 僕 (boku) - masculine (but used by some women too)
        # お前 (omae) - masculine second-person pronoun
        # Check both surface form and lemma since parsers vary
        if pos == 'pron':
            if surface in ['俺', 'おれ', 'オレ'] or lemma in ['俺', 'おれ', 'オレ']:
                has_masculine = True
            if surface in ['僕', 'ぼく', 'ボク'] or lemma in ['僕', 'ぼく', 'ボク', '僕-代名詞']:
                has_masculine = True
            # お前 (omae) - rough masculine second-person pronoun
            if surface in ['お前', 'おまえ', 'オマエ'] or lemma in ['御前', 'お前']:
                has_masculine = True

            # Check for feminine pronouns
            # あたし (atashi) - feminine variant of 私
            # あたくし (atakushi) - very formal feminine
            # Note: lemma might be 私 for these, so check surface
            if surface in ['あたし', 'アタシ', 'あたくし', 'アタクシ']:
                has_feminine = True

        # Check for rough masculine auxiliary verb forms
        # ねえ (nee) - rough masculine negation (variant of ない)
        if pos == 'auxv' and conjugated_type == 'auxv-nai':
            if surface in ['ねえ', 'ねー', 'ネエ', 'ネー']:
                has_masculine = True

        # Check for だろ (daro) - masculine sentence-final assertive
        # volitional-presumptive form of だ used assertively
        if pos == 'auxv' and conjugated_type == 'auxv-da':
            if conjugated_form == 'volitional-presumptive' and surface in ['だろ', 'ダロ']:
                has_masculine = True

        # Track particles for pattern detection
        if pos == 'prt':
            particle_sequence.append((i, surface, pos_detail1))

        # Check for かしら (kashira) - feminine wonder/question marker
        if surface in ['かしら', 'カシラ']:
            has_feminine = True

    # Analyze particle patterns
    masculine_particles = ['ぜ', 'ゼ', 'ぞ', 'ゾ', 'ぞい', 'ゾイ']
    feminine_particles = ['わ', 'ワ']

    # Check for のよ / のね patterns (feminine sentence endings)
    # Pattern: の (pre_noun_particle) followed by よ/ね (sentence_final_particle)
    for j in range(len(particle_sequence) - 1):
        idx1, surf1, detail1 = particle_sequence[j]
        idx2, surf2, detail2 = particle_sequence[j + 1]
        # Check if consecutive particles
        if idx2 == idx1 + 1:
            if surf1 == 'の' and detail1 == 'pre_noun_particle':
                if surf2 in ['よ', 'ヨ'] and detail2 == 'sentence_final_particle':
                    has_feminine = True
                if surf2 in ['ね', 'ネ'] and detail2 == 'sentence_final_particle':
                    has_feminine = True

    # Check individual sentence-final particles
    for _, surface, pos_detail1 in particle_sequence:
        if pos_detail1 in ['sentence_final_particle', 'adverbial_particle']:
            if surface in masculine_particles:
                has_masculine = True
            elif surface in feminine_particles:
                has_feminine = True

    # Decision logic based on features

    # Check for unpragmatic gender mixing
    # Strong masculine markers mixed with strong feminine markers is unusual
    if has_masculine and has_feminine:
        return GenderLevel.UNPRAGMATIC_GENDER

    # Masculine speech markers
    if has_masculine:
        return GenderLevel.MASCULINE

    # Feminine speech markers
    if has_feminine:
        return GenderLevel.FEMININE

    # Default to neutral
    return GenderLevel.NEUTRAL