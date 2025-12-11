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


def _load_style_model() -> Tuple['StyleClassifier', 'Tokenizer']:
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
            formality_probs, _, _ = model.predict(field_inputs, attention_mask)
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


def style(kotogram: str, use_model: bool = False) -> Tuple[FormalityLevel, GenderLevel, bool]:
    """Analyze a Japanese sentence and return formality, gender, and grammaticality.

    This is more efficient than calling formality(), gender(), and grammaticality()
    separately when using the model, as it only runs inference once.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.
        use_model: If True, use the trained neural model for prediction instead
                  of rule-based analysis. Default is False.

    Returns:
        Tuple of (FormalityLevel, GenderLevel, is_grammatic) for the sentence.
        When use_model=False, is_grammatic uses rule-based checks to detect
        common errors like adjectival predicates followed by だ.

    Examples:
        >>> # Formal, neutral sentence: 食べます (I eat - polite)
        >>> kotogram1 = "⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉"
        >>> style(kotogram1)
        (<FormalityLevel.FORMAL: 'formal'>, <GenderLevel.NEUTRAL: 'neutral'>, True)

        >>> # Using the trained model
        >>> style(kotogram1, use_model=True)  # doctest: +SKIP
        (<FormalityLevel.FORMAL: 'formal'>, <GenderLevel.NEUTRAL: 'neutral'>, True)
    """
    if use_model:
        # Use the trained neural model for prediction (single inference for all)
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
            formality_probs, gender_probs, grammaticality_probs = model.predict(field_inputs, attention_mask)
            formality_idx = int(formality_probs[0].argmax().item())
            gender_idx = int(gender_probs[0].argmax().item())
            grammaticality_idx = int(grammaticality_probs[0].argmax().item())

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
        is_grammatic = grammaticality_idx == 1  # 1 = grammatic, 0 = agrammatic
        return (
            formality_map.get(formality_idx, FormalityLevel.NEUTRAL),
            gender_map.get(gender_idx, GenderLevel.NEUTRAL),
            is_grammatic,
        )

    # Rule-based analysis
    return formality(kotogram), gender(kotogram), _rule_based_grammaticality(kotogram)


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
            _, gender_probs, _ = model.predict(field_inputs, attention_mask)
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
    # Also match lengthened variants like のねー, のよー
    for j in range(len(particle_sequence) - 1):
        idx1, surf1, detail1 = particle_sequence[j]
        idx2, surf2, detail2 = particle_sequence[j + 1]
        # Check if consecutive particles
        if idx2 == idx1 + 1:
            if surf1 == 'の' and detail1 == 'pre_noun_particle':
                if surf2 in ['よ', 'ヨ', 'よー', 'よお', 'ヨー'] and detail2 == 'sentence_final_particle':
                    has_feminine = True
                if surf2 in ['ね', 'ネ', 'ねー', 'ねえ', 'ネー'] and detail2 == 'sentence_final_particle':
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


# Set of adjectival auxiliary conjugation types that should not be followed by だ
# These auxiliaries conjugate like adjectives and already function as predicates
_ADJECTIVAL_AUXILIARY_TYPES = {
    'auxv-tai',      # ～たい (want to)
    'auxv-rashii',   # ～らしい when parsed as auxiliary (after verbs)
    'auxv-nai',      # ～ない (negation) - conjugates like i-adjective
}


def _is_adjectival_predicate_terminal(feature: Dict[str, str]) -> bool:
    """Check if a token is an adjectival predicate in terminal form.

    This identifies tokens that function as adjectival predicates and should
    not be followed by the copula だ. Covers three patterns:

    1. i-adjectives (e.g., 美しい, ない as adjective):
       - pos == "adj"
       - conjugated_type == "adjective"
       - conjugated_form starts with "terminal"

    2. Adjectival suffixes (e.g., らしい as in 学生らしい):
       - pos == "suff"
       - pos_detail1 == "adjectival"
       - conjugated_form starts with "terminal"

    3. Adjectival auxiliaries (e.g., たい as in 行きたい):
       - pos == "auxv"
       - conjugated_type in _ADJECTIVAL_AUXILIARY_TYPES
       - conjugated_form starts with "terminal"

    Args:
        feature: Feature dictionary from extract_token_features()

    Returns:
        True if the token is an adjectival predicate in terminal form
    """
    conjugated_form = feature.get('conjugated_form', '')
    if not conjugated_form.startswith('terminal'):
        return False

    pos = feature.get('pos', '')
    pos_detail1 = feature.get('pos_detail1', '')
    conjugated_type = feature.get('conjugated_type', '')
    surface = feature.get('surface', '')

    # Pattern 1: i-adjective (e.g., 美しい, ない as adjective)
    if pos == 'adj' and conjugated_type == 'adjective':
        return True

    # Pattern 2: Adjectival suffix (e.g., らしい in 学生らしい)
    # Exception: こい can be misparsed as suffix when it's part of verb こいだ (past of 漕ぐ)
    if pos == 'suff' and pos_detail1 == 'adjectival':
        # Exception for parser misparse of こいだ
        if surface == 'こい':
            return False
        return True

    # Pattern 3: Adjectival auxiliary (e.g., たい in 行きたい)
    if pos == 'auxv' and conjugated_type in _ADJECTIVAL_AUXILIARY_TYPES:
        return True

    return False


def _is_da_copula(feature: Dict[str, str]) -> bool:
    """Check if a token is the copula だ in terminal form.

    The pattern i-adjective + だ is ungrammatical, but i-adjective + だろう
    (volitional-presumptive form) is grammatically correct. We only want to
    detect the terminal form だ.

    Args:
        feature: Feature dictionary from extract_token_features()

    Returns:
        True if the token is the copula だ in terminal form (auxv-da:terminal)
    """
    return (
        feature.get('pos', '') == 'auxv' and
        feature.get('conjugated_type', '') == 'auxv-da' and
        feature.get('conjugated_form', '') == 'terminal'
    )


def _is_double_ta(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for doubled past tense marker たた.

    Args:
        curr: Current token features
        nxt: Next token features

    Returns:
        True if both tokens are auxv-ta (double past tense error)
    """
    return (
        curr.get('pos', '') == 'auxv' and
        curr.get('conjugated_type', '') == 'auxv-ta' and
        nxt.get('pos', '') == 'auxv' and
        nxt.get('conjugated_type', '') == 'auxv-ta'
    )


def _is_da_desu_redundant(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for だです redundancy (using both plain and polite copula).

    In Japanese, だ (plain copula) and です (polite copula) should not be used
    together. Using both is redundant and ungrammatical.

    Args:
        curr: Current token features
        nxt: Next token features

    Returns:
        True if だ is immediately followed by です (redundant copula error)
    """
    return (
        curr.get('pos', '') == 'auxv' and
        curr.get('conjugated_type', '') == 'auxv-da' and
        nxt.get('pos', '') == 'auxv' and
        nxt.get('conjugated_type', '') == 'auxv-desu'
    )


# Case particles that should not be doubled
_CASE_PARTICLES = {'が', 'を', 'に', 'で', 'と', 'へ', 'から', 'まで', 'より'}

# Words that can take で particle before でも in valid ででも pattern
# These are location/place words or means/instrument words
_DEDEMO_VALID_PRECEDING = {
    # Location interrogatives
    'どこ', '何処',  # where
    # Location/place nouns that commonly take で
    '場所', '所', 'ところ',  # place
    '家', '部屋', '学校', '会社', '店', '駅',  # common locations
    # Means/instrument/language (で as means particle)
    '言語', '言葉', '方法', '手段',  # language, method
    # Time-related (で as "at that time")
    'いつ', '何時',  # when
}


def _is_doubled_particle(
    curr: Dict[str, str], nxt: Dict[str, str], nxt2: Optional[Dict[str, str]] = None,
    prev: Optional[Dict[str, str]] = None
) -> bool:
    """Check for doubled case particles like がが, をを, にに.

    Exception: The pattern ででも (de de mo) is valid Japanese meaning
    "at anywhere" / "even at", but ONLY after words that naturally take
    the で particle (locations, means, instruments).

    Examples of valid ででも:
    - どこででも (where + で + でも) - "at anywhere"
    - 言語ででも (language + で + でも) - "in any language"

    Examples of invalid ででも (should be just でも):
    - どれででも (should be どれでも) - "any of these"
    - 映画ででも (should be 映画でも) - "even a movie"

    Args:
        curr: Current token features
        nxt: Next token features
        nxt2: Token after next (to check for ででも pattern), or None
        prev: Previous token features (to check for valid ででも context), or None

    Returns:
        True if both tokens are the same case particle (doubled particle error)
    """
    if curr.get('pos', '') != 'prt' or nxt.get('pos', '') != 'prt':
        return False

    curr_surface = curr.get('surface', '')
    nxt_surface = nxt.get('surface', '')

    # Check if they're the same particle and it's a case particle
    if curr_surface == nxt_surface and curr_surface in _CASE_PARTICLES:
        # Exception: ででも pattern (e.g., どこででも = at anywhere)
        # But ONLY if preceded by a word that naturally takes で
        if curr_surface == 'で' and nxt2 is not None and nxt2.get('surface', '') == 'も':
            if prev is not None:
                prev_surface = prev.get('surface', '')
                # Check if preceding word is a valid で-taking word
                if prev_surface in _DEDEMO_VALID_PRECEDING:
                    return False
                # Also allow if preceded by どんな + noun (どんな言語ででも)
                # The prev would be the noun, check if it's a valid type
                prev_pos = prev.get('pos', '')
                if prev_pos == 'n' and prev_surface in _DEDEMO_VALID_PRECEDING:
                    return False
            # If we can't verify the context, flag it as error
            return True
        # Exception: へへ (hehe) is onomatopoeia for laughter, not a doubled particle
        if curr_surface == 'へ':
            return False
        return True

    return False


def _is_double_de(
    curr: Dict[str, str], nxt: Dict[str, str], nxt2: Optional[Dict[str, str]] = None,
    prev: Optional[Dict[str, str]] = None
) -> bool:
    """Check for doubled で regardless of how each is parsed.

    In Japanese, two で particles/copulas in a row is typically ungrammatical.
    The parser may parse each で differently:
    - prt:case_particle (で as location marker)
    - auxv:auxv-da:conjunctive (で as copula conjunctive)
    - conj (で as conjunction)

    Exception: The pattern ででも (de de mo) is valid Japanese meaning
    "at anywhere" / "even at", but ONLY after words that naturally take
    the で particle (locations, means, instruments).

    Valid: どこででも (where + で + でも)
    Invalid: どれででも, 映画ででも (should be どれでも, 映画でも)

    Args:
        curr: Current token features
        nxt: Next token features
        nxt2: Token after next (to check for ででも pattern), or None
        prev: Previous token features (to check for valid ででも context), or None

    Returns:
        True if both tokens are で (doubled で error)
    """
    curr_surface = curr.get('surface', '')
    nxt_surface = nxt.get('surface', '')

    if curr_surface != 'で' or nxt_surface != 'で':
        return False

    # Exception: ででも pattern (e.g., どこででも = at anywhere)
    # The first で is location particle, second で + も = でも (even/any)
    # But ONLY if preceded by a word that naturally takes で
    if nxt2 is not None and nxt2.get('surface', '') == 'も':
        if prev is not None:
            prev_surface = prev.get('surface', '')
            # Check if preceding word is a valid で-taking word
            if prev_surface in _DEDEMO_VALID_PRECEDING:
                return False
        # If we can't verify the context or it's not a valid preceding word, flag as error
        return True

    # Accept で + で regardless of POS tags
    # The parser often misparses でで as various combinations:
    # - prt + conj
    # - auxv-da + conj
    # - prt + prt (already caught by _is_doubled_particle)
    return True


def _is_te_de_wrong_voicing(
    curr: Dict[str, str], nxt: Dict[str, str], nxt2: Optional[Dict[str, str]] = None
) -> bool:
    """Check for wrong voicing in te-form (で instead of て after certain forms).

    In Japanese, the conjunctive particle after certain verb/auxiliary forms
    should be て (te), not で (de). Using で after forms that require て is
    a voicing error.

    This detects patterns like:
    - されで (should be されて) - passive/potential conjunctive + で
    - 見えで (should be 見えて) - ichidan verb conjunctive + で
    - 見で (should be 見て) - ichidan verb + で parsed as sentence_final_particle
    - 確信しで (should be 確信して) - sa-irregular verb + で parsed as copula
    - 取っで (should be 取って) - godan verb conjunctive-geminate + で
    - 黙っで (should be 黙って) - godan verb conjunctive-geminate + で

    However, patterns like すぎではない (is not too much) are valid:
    - The で here is the copula だ in conjunctive form, not a te-form particle
    - This is indicated by で + は (binding particle) pattern

    Note: Some godan verbs DO correctly use で in te-form (e.g., 読んで from 読む).
    These are verbs ending in ぶ/む/ぬ, which use geminate んで form.
    Verbs in っ (conjunctive-geminate) form should use て.

    Args:
        curr: Current token features (verb or auxiliary in conjunctive form)
        nxt: Next token features (potential wrong で particle or misparse)
        nxt2: Token after next (to check for ではない pattern), or None

    Returns:
        True if conjunctive form is followed by で instead of て
    """
    # Check if current token is in conjunctive form
    curr_form = curr.get('conjugated_form', '')
    if not curr_form.startswith('conjunctive'):
        return False

    # Check if current is a verb or auxiliary that should use て not で
    curr_pos = curr.get('pos', '')
    curr_type = curr.get('conjugated_type', '')
    curr_surface = curr.get('surface', '')

    # These conjugation types should use て, not で:
    # - ichidan verbs (一段動詞)
    # - sa-irregular verbs (サ変)
    # - auxv-reru (れる/られる - passive/potential)
    # - auxv-seru (せる/させる - causative)
    # - auxv-masu (ます)
    te_requiring_types = {
        'auxv-reru', 'auxv-seru', 'auxv-rareru', 'auxv-saseru',
        'auxv-masu', 'auxv-tai', 'auxv-nai',
    }

    # Godan verb types that use で in te-form (ぶ/む/ぬ ending verbs)
    # These use ん + で pattern (e.g., 読む → 読んで, 飛ぶ → 飛んで)
    de_using_godan_types = {
        'godan-ba',  # 飛ぶ → 飛んで
        'godan-ma',  # 読む → 読んで
        'godan-na',  # 死ぬ → 死んで
    }

    # Exception: すぎ (excess verb) followed by で (copula indicating cause) is valid
    # e.g., 疲れすぎでそれ以上歩けなかった (couldn't walk more because of being too tired)
    # Also handle nominalized verbs that commonly take で (copula or cause particle)
    _TE_DE_EXCEPTIONS = {
        'すぎ', '過ぎ',  # "too much" - すぎで indicating cause is valid
        'ぶっつづけ', 'ぶっ続け', '打っ続け',  # "continuously" - noun
        'つづけ', '続け',  # "continuation" - can be noun
        # Nominalized verbs that take で (copula) for state/cause
        'いで',  # coming out (おいでで = being out)
        'むけ', '向け',  # aimed at (子供むけでない = not for children)
        'くずれ', '崩れ',  # collapse (がけくずれで = due to landslide)
        'かけ', '掛け',  # in the middle of (食べかけで = while eating)
        'つくり', '作り',  # made of (手作りで = handmade)
        'おくれ', '遅れ',  # delay (遅れで = due to delay)
        'あがり', '上がり',  # after finishing (風呂上がりで)
        '出',  # お出で (oide) - honorific "coming", 出で is literary form
    }
    if curr_surface in _TE_DE_EXCEPTIONS:
        return False

    # Check if it's an ichidan verb, sa-irregular verb, or one of the te-requiring auxiliaries
    is_ichidan = curr_pos == 'v' and 'ichidan' in curr_type
    is_sa_irregular = curr_pos == 'v' and 'sa-irregular' in curr_type
    is_te_aux = curr_pos == 'auxv' and curr_type in te_requiring_types

    # Check for godan verbs in conjunctive-geminate (っ) form
    # These should use て, not で
    # Excludes godan-ba/ma/na which use ん + で pattern
    is_godan_geminate = (
        curr_pos == 'v' and
        curr_form == 'conjunctive-geminate' and
        curr_surface.endswith('っ') and
        curr_type not in de_using_godan_types
    )

    # Check for godan verbs in conjunctive-i-sound (い音便) form
    # e.g., 続いで (should be 続いて) - godan-ka verbs use い + て pattern
    # Only godan-ka and godan-ga use i-sound conjunction (書く→書いて, 泳ぐ→泳いで)
    # godan-ga uses い + で (泳いで), godan-ka uses い + て (書いて)
    is_godan_i_sound_te = (
        curr_pos == 'v' and
        curr_form == 'conjunctive-i-sound' and
        curr_surface.endswith('い') and
        curr_type == 'godan-ka'  # Only godan-ka uses て, godan-ga uses で
    )

    if not (is_ichidan or is_sa_irregular or is_te_aux or is_godan_geminate or is_godan_i_sound_te):
        return False

    # Check if next token is で
    if nxt.get('surface', '') != 'で':
        return False

    nxt_pos = nxt.get('pos', '')
    nxt_detail1 = nxt.get('pos_detail1', '')
    nxt_type = nxt.get('conjugated_type', '')

    # で can appear as:
    # 1. conjunctive_particle (prt:conjunctive_particle) - most common parse
    # 2. sentence_final_particle (prt:sentence_final_particle) - misparse when at end
    # 3. copula conjunctive (auxv:auxv-da:conjunctive) - could be legitimate ではない pattern
    # 4. case_particle (prt:case_particle) - misparse in some contexts
    # 5. conjunction (conj) - misparse in some contexts
    # 6. e-ichidan-da auxiliary (auxv:e-ichidan-da) - rare misparse

    # Special case: で + は pattern indicates copula usage (ではない = is not)
    # This is grammatically correct, not a wrong voicing error
    # e.g., すぎではない (is not too much), 行きすぎではない
    if nxt_pos == 'auxv' and nxt_type == 'auxv-da':
        # Check if followed by は (binding particle) - indicates ではない pattern
        if nxt2 is not None and nxt2.get('surface', '') == 'は' and nxt2.get('pos', '') == 'prt':
            return False

    if nxt_pos == 'prt':
        if nxt_detail1 in ('conjunctive_particle', 'sentence_final_particle', 'case_particle'):
            return True
    elif nxt_pos == 'auxv' and nxt_type in ('auxv-da', 'e-ichidan-da'):
        # で parsed as copula だ in conjunctive form
        return True
    elif nxt_pos == 'conj':
        # で parsed as conjunction
        return True

    return False


def _is_verb_terminal_before_wo(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for verb in terminal/attributive form incorrectly followed by を particle.

    In Japanese, a verb directly followed by を (without a nominalizer like こと)
    is typically ungrammatical. The correct pattern is:
    - 言うことを (verb + nominalizer + を)
    - Not: 言うを, 対するを, etc.

    This detects patterns like:
    - いうを (should be いうことを)
    - 対するを (should be 対することを)

    Exceptions:
    - The idiom やむを得ず (yamu wo ezu - unavoidably) uses terminal form
    - Classical/literary Japanese allows V-terminal + を for nominalization
      (e.g., 入るを量りて出ずるを為す - measure income to control expenditure)
    - The pattern ～ているを is valid in literary style

    Args:
        curr: Current token features (potential verb in terminal/attributive form)
        nxt: Next token features (potential を particle)

    Returns:
        True if verb in terminal/attributive form is followed by を
    """
    # Check if current token is a verb in terminal or attributive form
    if curr.get('pos', '') != 'v':
        return False

    curr_form = curr.get('conjugated_form', '')
    # Both terminal and attributive forms before を are errors
    # (modern Japanese has them identical for most verbs)
    if curr_form not in ('terminal', 'attributive'):
        return False

    # Check if next token is を particle
    if nxt.get('surface', '') != 'を':
        return False

    if nxt.get('pos', '') != 'prt':
        return False

    curr_surface = curr.get('surface', '')
    curr_lemma = curr.get('lemma', '')

    # Exception: やむを得ず (yamu wo ezu) is a valid idiom
    # やむ (止む/已む) in terminal form + を is grammatically correct here
    if curr_surface in ('やむ', '止む', '已む') or curr_lemma in ('やむ', '止む', '已む'):
        return False

    # Exception: Classical/literary pattern V-terminal + を for nominalization
    # This pattern is common in:
    # - Classical proverbs and set phrases
    # - Literary/formal writing
    # - Idioms
    _LITERARY_VERBS = {
        '入る', 'はいる',  # 入るを量る (classical)
        '出る', 'でる',  # 出るを為す (classical)
        'いる',  # ～ているを (literary nominalization)
        'ある',  # ～てあるを (literary nominalization)
        '為す', 'なす',  # classical verb
        # Classical proverbs and idioms
        '足る', 'たる',  # 足るを知る (know contentment)
        '上がる', 'あがる',  # 雨は上がるを思う (expect rain to stop)
        '思う', 'おもう',  # V + を + 思う pattern
        '知る', 'しる',  # V + を + 知る pattern
        '待つ', 'まつ',  # 待つを得ず pattern
        '見る', 'みる',  # literary patterns
    }
    if curr_surface in _LITERARY_VERBS or curr_lemma in _LITERARY_VERBS:
        return False

    # Exception: Classical Japanese conjugation types (yodan, nidan, etc.)
    # In classical Japanese, V-terminal + を is valid for nominalization
    # e.g., 來たるを見て (seeing [someone] come)
    curr_type = curr.get('conjugated_type', '')
    if curr_type.startswith('yodan') or curr_type.startswith('nidan'):
        return False

    return True


def _is_verb_conjunctive_before_wo(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for verb in conjunctive form incorrectly followed by を particle.

    In Japanese, a verb in conjunctive form (連用形) followed directly by を
    is typically ungrammatical. The pattern usually indicates a typo or error.

    This detects patterns like:
    - 食べを (should be 食べること を or 食べたもの を)
    - 会いを (should be 会うこと を)

    However, many Japanese verbs in conjunctive form function as nouns
    (nominalized verbs / verbal nouns). These are grammatically correct:
    - ひきつけを起こす (have a seizure) - ひきつけ = nominalized verb
    - おいでを願う (request someone's presence) - おいで = nominalized verb
    - 受取りを諦める (give up on the receipt) - 受取り = nominalized verb

    Args:
        curr: Current token features (potential verb in conjunctive form)
        nxt: Next token features (potential を particle)

    Returns:
        True if verb in conjunctive form is followed by を
    """
    # Check if current token is a verb in conjunctive form
    if curr.get('pos', '') != 'v':
        return False

    if not curr.get('conjugated_form', '').startswith('conjunctive'):
        return False

    # Check if next token is を particle
    if nxt.get('surface', '') != 'を':
        return False

    if nxt.get('pos', '') != 'prt':
        return False

    # Exception: Many verb conjunctive forms function as nouns (nominalized verbs)
    # These are grammatically correct patterns where the verb stem acts as a noun
    _NOMINALIZED_VERBS = {
        # Medical/physical
        'ひきつけ', '引きつけ', '引き付け',  # seizure/convulsion
        'まばたき', '瞬き',  # blink
        'くしゃみ',  # sneeze
        'あくび', '欠伸',  # yawn
        'しゃっくり',  # hiccup
        # Polite/honorific forms that act as nouns
        'おいで', 'お出で',  # coming (honorific)
        'お越し',  # coming (honorific)
        'お帰り',  # return (honorific)
        'お立ち寄り',  # stopping by (honorific)
        # Common nominalized forms
        '受取り', '受け取り', 'うけとり',  # receipt
        '申し込み', '申込み',  # application
        '問い合わせ', '問合せ',  # inquiry
        '取り扱い', '取扱い',  # handling
        '見積もり', '見積り',  # estimate
        '引き取り', '引取り',  # taking over
        '差し入れ', '差入れ',  # gift (to someone in confinement)
        '持ち込み', '持込み',  # bringing in
        '買い物',  # shopping
        '売り上げ', '売上げ',  # sales
        '仕入れ',  # purchasing/stocking
        '取り消し', '取消し',  # cancellation
        '繰り返し', '繰返し',  # repetition
        '申し出', '申出',  # offer/proposal
        '思い', '想い',  # thought/feeling
        '願い',  # wish/request
        '祈り',  # prayer
        '眺め',  # view
        '流れ',  # flow
        '香り', '薫り',  # fragrance
        '光り', '輝き',  # shine/sparkle
        '響き',  # echo/resonance
        '痛み',  # pain
        '悩み',  # worry
        '喜び',  # joy
        '悲しみ',  # sadness
        '怒り',  # anger
        '驚き',  # surprise
        '恐れ',  # fear
        '憧れ',  # longing
        '焦り',  # impatience
        '苦しみ',  # suffering
        '楽しみ',  # enjoyment/pleasure
        '始まり',  # beginning
        '終わり',  # end
        '別れ',  # parting
        '出会い',  # encounter
        '集まり',  # gathering
        '並び',  # arrangement/row
        '繋がり', 'つながり',  # connection
        '関わり', 'かかわり',  # involvement
        'いで',  # coming out (archaic form used in おいで)
        # Additional nominalized verbs found in data
        'あざ笑い', 'あざわらい',  # mocking laugh
        'ほほえみ', '微笑み',  # smile
        'なまけ', '怠け',  # laziness
        '謝り', 'あやまり',  # apology
        '恥じ', 'はじ',  # shame (as noun)
        '投げ捨て', 'なげすて',  # throwing away
        '裂け', 'さけ',  # split/crack
        '越し', 'こし',  # crossing over
        'し',  # し + を pattern (often in 手つき=manner of hands)
    }

    curr_surface = curr.get('surface', '')
    curr_lemma = curr.get('lemma', '') or curr.get('base_orth', '')

    # Check if this is a known nominalized verb
    if curr_surface in _NOMINALIZED_VERBS:
        return False
    if curr_lemma in _NOMINALIZED_VERBS:
        return False

    # Heuristic: If the verb ends with り and is in conjunctive form,
    # it's likely a nominalized verb (many masu-stem nouns end in り)
    # Only apply this heuristic if it's not too short (avoid false negatives)
    if len(curr_surface) >= 3 and curr_surface.endswith('り'):
        return False

    return True


def _is_likely_compound_noun(shp_surface: str, noun_surface: str) -> bool:
    """Check if a na-adjective + noun combination is likely a compound noun.

    Sino-Japanese compound nouns (熟語/jukugo) often use na-adjectives without な.
    These are valid compound words, not missing-な errors.

    This function uses a combination of:
    1. Known compound patterns (dictionary-based)
    2. Heuristics for sino-Japanese compounds
    3. Special cases for adverb-like usage

    Args:
        shp_surface: The na-adjective surface form
        noun_surface: The following noun surface form

    Returns:
        True if likely a compound noun (should NOT be flagged as error)
    """
    # Na-adjectives that are commonly used as adverbs (not requiring な)
    # These modify the following word/phrase without な
    _ADVERB_LIKE_SHPS = {
        '大変',  # taihenn - "very" (adverb)
        '同様',  # douyou - "similarly" (adverb-like)
        '存分',  # zonbun - "to one's heart's content"
        '様々',  # samazama - "various"
        '様',    # sama/you - often grammatical pattern
        'みたい',  # mitai - "like/similar to" (often grammatical)
    }

    if shp_surface in _ADVERB_LIKE_SHPS:
        return True

    # Na-adjectives ending in 的 (-teki) are almost always used as compound prefixes
    # e.g., 性的偏見, 知的能力, 公的支出, 私的用件, 美的センス
    if shp_surface.endswith('的'):
        return True

    # Known na-adjectives that commonly form compounds without な
    # These are often used as prefixes in sino-Japanese compounds
    _COMPOUND_FORMING_SHPS = {
        # Common compound-forming na-adjectives
        '簡易', '悪質', '完全', '有名', '主要', '同一', '霊的',
        '透明', '肝心', '多様', '適宜', '高級', '低級', '新規',
        '特殊', '正規', '異常', '正常', '特別', '普通', '重要',
        '零細', '幼稚', '傍若',
        # Additional compound-formers found in data
        '不法',  # illegal (不法外国人)
        '優秀',  # excellent (優秀者)
        '優等',  # honors (優等賞)
        '単一',  # single (単一機械)
        '単独',  # solo (単独飛行)
        '可能',  # possible (可能量)
        '有力',  # influential (有力者)
        '有効',  # valid (有効券)
        '未熟',  # immature (未熟者)
        '格安',  # bargain (格安チケット)
        '見事',  # splendid (見事独立)
        '適正',  # appropriate (適正賃金)
        '重大',  # serious (重大事件)
        '高度',  # high altitude (高度一万)
        '高等',  # higher (高等教育)
        '巨大',  # giant (巨大店舗)
        '希少',  # rare (希少野生動物)
        '密',    # secret (密売買)
        '丈夫',  # healthy/sturdy
        '大丈夫',  # OK/all right
        '大事',  # important (大事なし)
        '大好き',  # love (大好き人間 = person who loves)
        '勇敢',  # brave (勇敢無比)
        '明朗',  # cheerful (明朗快活)
        '遺憾',  # regrettable (遺憾千万)
        '頑固',  # stubborn (頑固親父)
        '撩乱',  # confusion
        # Compound expressions
        'けんめい', 'けん命',  # diligently
        'いたずら',  # mischief (いたずら半分)
        'めった',  # reckless (めったうち)
        'ぴったし',  # exactly
        'まじ', 'マジ',  # seriously (slang)
        # Katakana loanwords
        'フル',  # full (フル稼動)
        'サイバー',  # cyber
        'ルンルン',  # cheerful/happy (ルンルン気分)
        'ウキウキ',  # excited (ウキウキ気分)
    }

    # Check if the shp commonly forms compounds
    if shp_surface in _COMPOUND_FORMING_SHPS:
        return True

    # Special case: よう (様) is often used in grammatical patterns
    # e.g., ～するよう + noun (asking someone to do something)
    if shp_surface == 'よう':
        return True

    # Katakana loanword compounds (ジャスト, クール, メカニカル, etc.)
    # If the shp is all katakana and noun starts with katakana, likely a loanword compound
    if shp_surface and all('\u30a0' <= c <= '\u30ff' for c in shp_surface):
        if noun_surface and '\u30a0' <= noun_surface[0] <= '\u30ff':
            return True

    # Heuristic: na-adjectives followed by kanji nouns often form compounds
    # Especially when the na-adjective is all kanji
    if shp_surface and all('\u4e00' <= c <= '\u9fff' for c in shp_surface):
        # If the following noun also starts with kanji, likely a compound
        if noun_surface and len(noun_surface) > 0 and '\u4e00' <= noun_surface[0] <= '\u9fff':
            return True

    # にぎやか通り pattern - hiragana na-adj + kanji noun can be compound
    # But be conservative - only allow specific patterns
    if shp_surface == 'にぎやか':
        return True

    return False


def _is_na_adjective_missing_na(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for na-adjective (形状詞) directly followed by noun without な.

    In Japanese, na-adjectives (形状詞/shp) must use な when modifying nouns.
    For example:
    - Correct: 静かな部屋 (shizuka na heya) - quiet room
    - Incorrect: 静か部屋 (shizuka heya) - missing な

    However, sino-Japanese compound nouns (熟語) often combine na-adjectives
    with nouns without な, and these are grammatically valid:
    - 簡易住宅 (kan'i jūtaku) - simple housing (compound noun)
    - 完全自動詞 (kanzen jidōshi) - complete intransitive verb

    Args:
        curr: Current token features (potential na-adjective)
        nxt: Next token features (potential noun without な)

    Returns:
        True if na-adjective is directly followed by noun (missing な)
    """
    # Check if current token is a na-adjective (形状詞)
    if curr.get('pos', '') != 'shp':
        return False

    # Check if next token is a noun
    nxt_pos = nxt.get('pos', '')
    if nxt_pos not in ('n', 'pn', 'pron'):
        return False

    # Check if this is likely a valid compound noun
    shp_surface = curr.get('surface', '')
    noun_surface = nxt.get('surface', '')

    if _is_likely_compound_noun(shp_surface, noun_surface):
        return False

    # This pattern indicates missing な between na-adjective and noun
    return True


def _is_i_adjective_with_na(
    curr: Dict[str, str], nxt: Dict[str, str], nxt2: Optional[Dict[str, str]]
) -> bool:
    """Check for i-adjective incorrectly followed by な before a noun.

    In Japanese, i-adjectives (形容詞) should directly modify nouns in their
    attributive form (連体形), not use な like na-adjectives. For example:
    - Correct: 新しい本 (atarashii hon)
    - Incorrect: 新しいな本 (atarashii na hon)

    Args:
        curr: Current token features (potential i-adjective)
        nxt: Next token features (potential な particle)
        nxt2: Token after next (potential noun), or None

    Returns:
        True if pattern matches i-adjective + な + noun (error pattern)
    """
    # Check if current token is an i-adjective in terminal form
    # (terminal form is used because Sudachi parses オイシイな as terminal + な)
    if curr.get('pos', '') != 'adj':
        return False

    if curr.get('conjugated_type', '') != 'adjective':
        return False

    if curr.get('conjugated_form', '') != 'terminal':
        return False

    # Check if next token is な particle
    if nxt.get('surface', '') != 'な':
        return False

    if nxt.get('pos', '') != 'prt':
        return False

    # Check if token after な is a noun (to distinguish from sentence-final な)
    if nxt2 is None:
        return False

    if nxt2.get('pos', '') not in ('n', 'pn', 'pron'):
        return False

    return True


def _is_verb_conjunctive_ta(
    curr: Dict[str, str], nxt: Dict[str, str], nxt2: Optional[Dict[str, str]] = None
) -> bool:
    """Check for verb in conjunctive form followed directly by た (missing っ).

    In Japanese, godan verbs with certain endings use the geminate form (っ)
    before た in the past tense. For example:
    - 行く → 行った (not 行きた)
    - 持つ → 持った (not 持ちた)
    - 帰る → 帰った (not 帰りた)

    This pattern detects errors where the conjunctive form (連用形) is used
    directly before た instead of the proper past form.

    Exceptions:
    - Honorific verbs like いらしゃる, おっしゃる have contracted forms
      where conjunctive + た is valid: いらした, おっしゃた
    - godan-ga verbs use い + だ (not っ + た): 泳いだ
    - Classical desiderative たし form: V-conjunctive + た + し (食いたし)

    Args:
        curr: Current token features (potential verb in conjunctive form)
        nxt: Next token features (potential た auxiliary)
        nxt2: Token after た (to check for classical たし pattern)

    Returns:
        True if verb conjunctive is followed by た (missing っ error)
    """
    # Check if current token is a verb in conjunctive form (not geminate)
    if curr.get('pos', '') != 'v':
        return False

    curr_form = curr.get('conjugated_form', '')
    curr_type = curr.get('conjugated_type', '')
    curr_surface = curr.get('surface', '')
    curr_lemma = curr.get('lemma', '')

    # Must be in simple conjunctive form (not already geminate)
    if curr_form != 'conjunctive':
        return False

    # Only godan verbs that should use geminate form before た
    # Excludes godan-ga which uses い + だ (not た)
    geminate_godan_types = {
        'godan-ka',  # 行く → 行った
        'godan-ta',  # 持つ → 持った
        'godan-ra',  # 帰る → 帰った
        'godan-waa',  # 買う → 買った
    }

    if curr_type not in geminate_godan_types:
        return False

    # Check if next token is た (past auxiliary)
    if nxt.get('surface', '') != 'た':
        return False

    if nxt.get('pos', '') != 'auxv':
        return False

    if nxt.get('conjugated_type', '') != 'auxv-ta':
        return False

    # Exception: Honorific verbs with contracted past forms
    # These are contracted forms of ～っしゃる verbs that use conjunctive + た
    # Examples: いらしゃる → いらした, おっしゃる → おっしゃた
    _HONORIFIC_CONTRACTED_VERBS = {
        # いらっしゃる contracted forms
        'いらし', 'いらっし',
        # おっしゃる contracted forms
        'おっし', 'おっしゃ',
        # くださる (くだされた archaic, くださった modern, くださた rare)
        'くださ',
        # なさる contracted forms
        'なさ',
        # ござる (archaic forms)
        'ござ',
    }

    # Check lemma for honorific verbs
    _HONORIFIC_LEMMAS = {
        'いらしゃる', 'いらっしゃる',
        'おっしゃる',
        'くださる', '下さる',
        'なさる', '為さる',
        'ござる', '御座る',
    }

    if curr_surface in _HONORIFIC_CONTRACTED_VERBS:
        return False
    if curr_lemma in _HONORIFIC_LEMMAS:
        return False

    # Exception: Literary/archaic verb forms that use conjunctive + た
    # ひとりごつ (to mutter/soliloquize) → ひとりごちた is archaic but valid
    _ARCHAIC_VERBS = {
        'ひとりごち', '独り言ち',  # ひとりごつ archaic past
        'ひとりごと',  # variant
    }

    if curr_surface in _ARCHAIC_VERBS:
        return False

    # Exception: Classical desiderative たし form
    # Pattern: V-conjunctive + た + し (conjunctive_particle)
    # Example: 河豚は食いたし命は惜しし (I want to eat pufferfish but life is precious)
    # The parser incorrectly segments this as: 食い + た + し
    # But たし is actually the classical desiderative suffix meaning "want to"
    if nxt2 is not None:
        nxt2_surface = nxt2.get('surface', '')
        nxt2_pos = nxt2.get('pos', '')
        nxt2_detail = nxt2.get('pos_detail1', '')
        # Check for た + し (conjunctive particle) pattern
        if nxt2_surface == 'し' and nxt2_pos == 'prt' and nxt2_detail == 'conjunctive_particle':
            # Also check that た is in terminal form (not attributive)
            # In classical たし, the た is not actually past tense but part of たし
            nxt_form = nxt.get('conjugated_form', '')
            if nxt_form == 'terminal':
                return False

    return True


def _is_masu_nai(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for ます + ない/ぬ (should be ません).

    In Japanese, the polite negative is formed with ません, not ますない.
    This detects errors like:
    - 開いていますない (should be 開いていません)
    - 飲みますないか (should be 飲みませんか)

    Args:
        curr: Current token features (potential ます auxiliary)
        nxt: Next token features (potential ない/ぬ)

    Returns:
        True if ます is followed by negative (error pattern)
    """
    # Check if current token is ます
    if curr.get('surface', '') != 'ます':
        return False

    if curr.get('pos', '') != 'auxv':
        return False

    # Check if next token is negative auxiliary
    nxt_surface = nxt.get('surface', '')
    nxt_pos = nxt.get('pos', '')

    # Common error patterns: ますない, ますぬ
    # Note: ますね is CORRECT (polite + confirmation particle)
    # Note: ますな can be CORRECT (archaic/literary prohibition)
    if nxt_surface == 'ない':
        # ますない is wrong (should be ません)
        return True

    # Also check for ません being misparsed as ます + ん + something
    if nxt_pos == 'auxv' and nxt.get('conjugated_type', '') == 'auxv-nai':
        return True

    return False


def _is_mashita_koto(
    curr: Dict[str, str], nxt: Dict[str, str], nxt2: Optional[Dict[str, str]] = None,
    prev: Optional[Dict[str, str]] = None
) -> bool:
    """Check for ました + こと pattern.

    NOTE: This check is DISABLED because ましたこと is actually VALID in formal
    Japanese, particularly in formal/keigo contexts like business correspondence.

    Examples of valid formal Japanese:
    - ありました事を深くお詫び致します (formal apology)
    - 選出されましたことは大変な名誉であります (formal acceptance)
    - お世話になりましたことを深く感謝いたします (formal gratitude)

    While たこと is more common in modern casual Japanese, ましたこと remains
    grammatically correct in formal registers.

    Args:
        curr: Current token features
        nxt: Next token features
        nxt2: Not used
        prev: Previous token features, or None

    Returns:
        False (check disabled - ましたこと is valid in formal Japanese)
    """
    # Check disabled - ましたこと is valid in formal Japanese
    return False


def _is_classical_adj_na(curr: Dict[str, str], nxt: Dict[str, str], nxt2: Optional[Dict[str, str]]) -> bool:
    """Check for classical adjective + な + noun (incorrect usage).

    Classical Japanese adjectives (古語形容詞) like 古き, 美しき end in き
    in attributive form and should directly modify nouns without な:
    - 古き友 (correct - classical style)
    - 古きな友 (incorrect - mixing classical with na-adjective pattern)

    Args:
        curr: Current token features (potential classical adjective)
        nxt: Next token features (potential な)
        nxt2: Token after next (potential noun), or None

    Returns:
        True if classical adjective + な + noun (error pattern)
    """
    # Check if current token is a classical adjective in attributive form
    if curr.get('pos', '') != 'adj':
        return False

    curr_type = curr.get('conjugated_type', '')
    if 'classical' not in curr_type:
        return False

    curr_form = curr.get('conjugated_form', '')
    if curr_form != 'attributive':
        return False

    # Check if next token is な (copula attributive)
    if nxt.get('surface', '') != 'な':
        return False

    # な can be particle or auxv-da attributive
    nxt_pos = nxt.get('pos', '')
    if nxt_pos not in ('prt', 'auxv'):
        return False

    # Check if followed by noun
    if nxt2 is None:
        return False

    if nxt2.get('pos', '') not in ('n', 'pn', 'pron'):
        return False

    return True


def _is_nasal_te_voicing_error(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for godan nasal conjugation + て (should be で).

    Godan verbs ending in ぬ, む, ぶ use nasal conjugation (撥音便) in te-form
    and require voiced で, not unvoiced て:
    - 死ぬ → 死んで (not 死んて)
    - 読む → 読んで (not 読んて)
    - 飛ぶ → 飛んで (not 飛んて)

    Args:
        curr: Current token features (verb in conjunctive-nasal form)
        nxt: Next token features (potential て particle)

    Returns:
        True if nasal conjugation is followed by て (voicing error)
    """
    # Check if current token is a verb in conjunctive-nasal form
    if curr.get('pos', '') != 'v':
        return False

    curr_form = curr.get('conjugated_form', '')
    if curr_form != 'conjunctive-nasal':
        return False

    # Check if next token is て (should be で for nasal conjugation)
    nxt_surface = nxt.get('surface', '')
    if nxt_surface != 'て':
        return False

    nxt_pos = nxt.get('pos', '')
    if nxt_pos != 'prt':
        return False

    return True


def _is_incomplete_te_iru(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for incomplete ている pattern (てい followed by sentence-end).

    The progressive/continuous form ている must be complete. Ending with
    just てい is ungrammatical:
    - 食べてい。 (incomplete - should be 食べている or 食べています)
    - 見てい。 (incomplete)

    Args:
        curr: Current token features (potential い verb in conjunctive)
        nxt: Next token features (potential sentence-ending punctuation)

    Returns:
        True if incomplete ている pattern is detected
    """
    # Check if current token is いる (auxiliary) in conjunctive form
    # This catches patterns like てい。 where い is the conjunctive of いる
    if curr.get('pos', '') != 'v':
        return False

    curr_surface = curr.get('surface', '')
    if curr_surface != 'い':
        return False

    curr_form = curr.get('conjugated_form', '')
    if curr_form != 'conjunctive':
        return False

    # The verb should be いる (non-self-reliant auxiliary)
    curr_detail = curr.get('pos_detail1', '')
    if curr_detail != 'non_self_reliant':
        return False

    # Check if next token is sentence-ending punctuation
    nxt_pos = nxt.get('pos', '')
    if nxt_pos != 'auxs':
        return False

    nxt_detail = nxt.get('pos_detail1', '')
    if nxt_detail not in ('period', 'question_mark', 'exclamation_mark'):
        return False

    return True


def _is_nakereba_taranai(curr: Dict[str, str], nxt: Dict[str, str],
                         nxt2: Optional[Dict[str, str]]) -> bool:
    """Check for なければたらない pattern (should be なければならない).

    This is a common typo where たる is mistakenly used instead of なる
    in the obligative construction なければならない.

    The parser interprets たら as verb たる in imperfective form when it
    appears incorrectly. The correct construction uses なら from なる.

    Args:
        curr: Current token features (potential ば particle)
        nxt: Next token features (potential たら)
        nxt2: Token after next (potential ない)

    Returns:
        True if なければたらない pattern is detected
    """
    if nxt2 is None:
        return False

    # Check if current token is ば (conjunctive particle)
    if curr.get('surface', '') != 'ば':
        return False
    if curr.get('pos', '') != 'prt':
        return False

    # Check if next token is たら parsed as verb たる in imperfective
    if nxt.get('surface', '') != 'たら':
        return False
    # It's interpreted as verb たる, not the conditional auxiliary
    if nxt.get('pos', '') != 'v':
        return False
    nxt_base = nxt.get('base_orth', '')
    if nxt_base != 'たる':
        return False

    # Check if token after is ない (negative auxiliary)
    if nxt2.get('surface', '') != 'ない':
        return False
    if nxt2.get('pos', '') != 'auxv':
        return False

    return True


def _is_ichidan_terminal_tara(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for ichidan verb in terminal form + たら (wrong conditional form).

    Ichidan verbs form the conditional by:
    - stem + たら (e.g., 食べ + たら = 食べたら)

    Using the terminal form + たら is ungrammatical:
    - できるたら (wrong - should be できたら)
    - 食べるたら (wrong - should be 食べたら)

    The parser parses this as verb(terminal) + auxv たら(conditional).

    Args:
        curr: Current token features (potential ichidan verb in terminal form)
        nxt: Next token features (potential たら auxiliary)

    Returns:
        True if ichidan terminal + たら error is detected
    """
    # Check if current token is a verb in terminal form
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'terminal':
        return False

    # Check if it's an ichidan verb (e-ichidan or i-ichidan)
    conj_type = curr.get('conjugated_type', '')
    if not ('ichidan' in conj_type):
        return False

    # Check if verb surface ends in る (terminal form of ichidan)
    curr_surface = curr.get('surface', '')
    if not curr_surface.endswith('る'):
        return False

    # Check if next token is たら auxiliary (conditional)
    if nxt.get('surface', '') != 'たら':
        return False
    if nxt.get('pos', '') != 'auxv':
        return False

    return True


def _is_ni_tsuide_error(
    prev: Optional[Dict[str, str]],
    curr: Dict[str, str],
    nxt: Dict[str, str]
) -> bool:
    """Check for についで pattern (should be について).

    The verb つく (godan-ka) conjugates as:
    - つい + て = ついて (correct: conjunctive-i-sound + て)

    The verb つぐ (godan-ga) conjugates as:
    - つい + で = ついで (from つぐ, meaning 'inherit/succeed')

    When following に (case particle), についで is almost always a typo
    for について (concerning/regarding). The parser interprets ついで as
    the conjunctive form of つぐ + で, which is wrong for this meaning.

    Args:
        prev: Previous token features (potential に particle)
        curr: Current token features (potential つい verb)
        nxt: Next token features (potential で particle)

    Returns:
        True if についで error is detected
    """
    if prev is None:
        return False

    # Check if previous token is に (case particle)
    if prev.get('surface', '') != 'に':
        return False
    if prev.get('pos', '') != 'prt':
        return False

    # Check if current token is つい parsed as verb つぐ
    if curr.get('surface', '') != 'つい':
        return False
    if curr.get('pos', '') != 'v':
        return False
    # The parser interprets it as つぐ (godan-ga) in conjunctive-i-sound form
    curr_base = curr.get('base_orth', '')
    if curr_base != 'つぐ':
        return False

    # Check if next token is で (conjunctive particle)
    if nxt.get('surface', '') != 'で':
        return False
    if nxt.get('pos', '') != 'prt':
        return False

    return True


def _is_godan_sa_de_auxiliary(
    curr: Dict[str, str],
    nxt: Dict[str, str],
    nxt2: Optional[Dict[str, str]]
) -> bool:
    """Check for godan sa-row verb conjunctive + で + auxiliary (should be て).

    Godan sa-row verbs (話す, 出す, etc.) form the te-form with て:
    - 話し + て = 話して
    - 出し + て = 出して

    Using で instead is ungrammatical:
    - 話しでいます (wrong - should be 話しています)
    - なくしでしまった (wrong - should be なくしてしまった)

    The parser usually interprets this as:
    - verb (godan-sa, conjunctive) + で (particle) + auxiliary verb

    Args:
        curr: Current token features (godan sa-row verb in conjunctive)
        nxt: Next token features (で particle)
        nxt2: Token after next (auxiliary verb いる/しまう, etc.)

    Returns:
        True if godan sa-row conjunctive + で + auxiliary error is detected
    """
    if nxt2 is None:
        return False

    # Check if current token is a godan sa-row verb in conjunctive form
    # Surface ends in し (連用形 of godan-sa verbs)
    curr_surface = curr.get('surface', '')
    if not curr_surface.endswith('し'):
        return False

    curr_pos = curr.get('pos', '')
    if curr_pos != 'v':
        return False

    curr_type = curr.get('conjugated_type', '')
    if 'godan-sa' not in curr_type:
        return False

    curr_form = curr.get('conjugated_form', '')
    if curr_form != 'conjunctive':
        return False

    # Check if next token is で (particle)
    if nxt.get('surface', '') != 'で':
        return False
    if nxt.get('pos', '') != 'prt':
        return False

    # Check if token after is an auxiliary verb (いる, しまう, etc.)
    nxt2_pos = nxt2.get('pos', '')
    if nxt2_pos != 'v':
        return False

    # The auxiliary should be non-self-reliant (いる, しまう, etc.)
    # or specific verbs that follow て form
    nxt2_detail = nxt2.get('pos_detail1', '')
    nxt2_surface = nxt2.get('surface', '')

    # Common auxiliaries after て form
    te_auxiliaries = {'い', 'いる', 'しまう', 'しまっ', 'おく', 'おい',
                      'みる', 'み', 'ある', 'あっ', 'くる', 'き', 'いく',
                      'いっ', 'あげ', 'もらう', 'もらっ', 'くれ'}

    if nxt2_surface in te_auxiliaries or nxt2_detail == 'non_self_reliant':
        return True

    return False


def _is_shi_de_auxiliary(
    curr: Dict[str, str],
    nxt: Dict[str, str],
    nxt2: Optional[Dict[str, str]]
) -> bool:
    """Check for し (particle/する) + で + auxiliary (should be て).

    The する verb te-form is して, not しで:
    - どうしてよい (correct)
    - どうしでよい (wrong)
    - 口答えしてはいけない (correct)
    - 口答えしではいけない (wrong)

    The parser often interprets しで as:
    - し (particle) + で (particle or auxv-da conjunctive)

    Args:
        curr: Current token features (し particle)
        nxt: Next token features (で particle or auxv)
        nxt2: Token after next (よい, は, いけ, etc.)

    Returns:
        True if しで + auxiliary error is detected
    """
    if nxt2 is None:
        return False

    # Check if current token is し (particle - often misparse of する conjunctive)
    if curr.get('surface', '') != 'し':
        return False

    # The parser might parse し as particle or verb
    curr_pos = curr.get('pos', '')
    if curr_pos not in ('prt', 'v'):
        return False

    # Check if next token is で
    if nxt.get('surface', '') != 'で':
        return False

    # The parser might interpret で as particle or auxv-da conjunctive
    nxt_pos = nxt.get('pos', '')
    if nxt_pos not in ('prt', 'auxv'):
        return False

    # Check if token after suggests this is a te-form context
    nxt2_surface = nxt2.get('surface', '')
    nxt2_pos = nxt2.get('pos', '')

    # Common patterns after して:
    # - してよい (してよい, していい)
    # - しては (してはいけない, してはだめ)
    # - している (progressive)
    # - してしまう

    # してよい pattern: よい/いい (adjective)
    if nxt2_pos == 'adj' and nxt2_surface in ('よい', 'いい', 'よく', 'いけ'):
        return True

    # しては pattern: は (particle)
    if nxt2_pos == 'prt' and nxt2_surface == 'は':
        return True

    # している pattern: い/いる (verb いる)
    if nxt2_pos == 'v' and nxt2_surface in ('い', 'いる', 'いけ'):
        return True

    return False


def _is_iru_terminal_masu(curr: Dict[str, str], nxt: Dict[str, str]) -> bool:
    """Check for いる (terminal) + ます pattern (should be い + ます).

    In the correct ています form, いる conjugates to い (conjunctive) before ます.
    Having いる in terminal form before ます is ungrammatical:
    - 困っているます (wrong - should be 困っています)

    The parser interprets this as:
    - いる (verb, terminal) + ます (auxv)

    Args:
        curr: Current token features (potential いる verb in terminal form)
        nxt: Next token features (potential ます auxiliary)

    Returns:
        True if いる (terminal) + ます error is detected
    """
    # Check if current token is いる in terminal form
    if curr.get('surface', '') != 'いる':
        return False
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'terminal':
        return False

    # Check if next token is ます
    if nxt.get('surface', '') not in ('ます', 'まし', 'ませ'):
        return False
    if nxt.get('pos', '') != 'auxv':
        return False

    return True


def _is_noun_te_iru_missing_geminate(
    curr: Dict[str, str],
    nxt: Dict[str, str],
    nxt2: Optional[Dict[str, str]]
) -> bool:
    """Check for noun + て + いる pattern indicating missing っ in te-form.

    When a godan verb's te-form is written without the required っ (geminate),
    the parser often interprets the verb stem as a noun:
    - 祈ています (wrong - should be 祈っています)
    - 知ている (wrong - should be 知っている)
    - 養ていけない (wrong - should be 養っていけない)
    - 帰ていました (wrong - should be 帰っていました)

    The parser interprets these as:
    - noun + て (particle) + いる/い (verb)

    We detect single-kanji nouns followed by て + いる/い as likely missing geminate.

    Args:
        curr: Current token features (potential single-kanji noun)
        nxt: Next token features (potential て particle)
        nxt2: Token after next (potential いる/い verb)

    Returns:
        True if noun + て + いる pattern indicating missing っ is detected
    """
    if nxt2 is None:
        return False

    # Check if current token is a single-kanji noun
    # (verb stems that become nouns when っ is missing)
    curr_surface = curr.get('surface', '')
    if len(curr_surface) != 1:
        return False
    if curr.get('pos', '') != 'n':
        return False

    # The kanji should be one that's commonly a verb stem
    # These are kanji for godan verbs that require っ in te-form
    common_verb_kanji = {
        '祈', '知', '養', '帰', '勝', '待', '持', '打', '立', '経',
        '建', '発', '絶', '断', '保', '買', '使', '飼', '歌', '疑',
        '払', '扱', '違', '追', '思', '言', '行', '会', '合', '習',
        '洗', '争', '戦', '笑', '救', '拾', '放', '解', '撃', '捨',
    }
    if curr_surface not in common_verb_kanji:
        return False

    # Check if next token is て (particle)
    if nxt.get('surface', '') != 'て':
        return False
    if nxt.get('pos', '') != 'prt':
        return False

    # Check if token after is いる/い (auxiliary verb for progressive)
    nxt2_surface = nxt2.get('surface', '')
    if nxt2_surface not in ('い', 'いる', 'いけ', 'いた', 'いれ'):
        return False
    if nxt2.get('pos', '') != 'v':
        return False

    return True


def _is_ta_tako_doubled_ta(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for た (auxv) + たこ pattern indicating doubled た.

    When たた is written (doubled past tense), the parser often interprets
    the second たこと as たこ (octopus/noun) + と (particle):
    - 言ったたこと (wrong - should be 言ったこと)
    - 考えたたこと (wrong - should be 考えたこと)
    - 投稿したたこと (wrong - should be 投稿したこと)

    The parser interprets these as:
    - verb + た (auxv) + たこ (noun "octopus") + と (particle)

    Args:
        curr: Current token features (potential た auxiliary verb)
        nxt: Next token features (potential たこ noun)

    Returns:
        True if た + たこ pattern indicating doubled た is detected
    """
    # Check if current token is た (past tense auxiliary)
    if curr.get('surface', '') != 'た':
        return False
    if curr.get('pos', '') != 'auxv':
        return False

    # Check if next token is たこ (misparsed from たこと)
    if nxt.get('surface', '') != 'たこ':
        return False
    if nxt.get('pos', '') != 'n':
        return False

    return True


def _is_classical_ra_terminal_sentence_end(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for classical-irregular-ra terminal form at sentence end.

    When a godan-ra verb (like ある) ends a sentence in conjunctive form,
    the parser may interpret it as classical-irregular-ra terminal:
    - あり。 (wrong - incomplete sentence, should be あります or ある)

    This pattern detects verbs parsed as classical-irregular-ra in terminal
    form immediately followed by sentence-ending punctuation.

    Args:
        curr: Current token features (potential classical-ra verb)
        nxt: Next token features (potential sentence-ending punctuation)

    Returns:
        True if classical-irregular-ra terminal + 。 pattern is detected
    """
    # Check if current token is classical-irregular-ra in terminal form
    if curr.get('conjugated_type', '') != 'classical-irregular-ra':
        return False
    if curr.get('conjugated_form', '') != 'terminal':
        return False

    # Check if next token is sentence-ending punctuation
    if nxt.get('surface', '') not in ('。', '！', '？'):
        return False
    if nxt.get('pos', '') != 'auxs':
        return False

    return True


def _is_ta_da_redundant(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for た (terminal) + だ (terminal) pattern.

    When た is in terminal form followed by だ in terminal form, this is
    ungrammatical. This catches patterns like:
    - なかっただ (wrong - should be なかった)
    - うれしかっただ (wrong - should be うれしかった)

    Note: た + だろう is grammatical (だろう is volitional-presumptive, not terminal).

    Args:
        curr: Current token features (potential た auxiliary)
        nxt: Next token features (potential だ auxiliary)

    Returns:
        True if た (terminal) + だ (terminal) pattern is detected
    """
    # Check if current token is た in terminal form
    if curr.get('pos', '') != 'auxv':
        return False
    if curr.get('conjugated_type', '') != 'auxv-ta':
        return False
    if curr.get('conjugated_form', '') != 'terminal':
        return False

    # Check if next token is だ in terminal form
    if nxt.get('pos', '') != 'auxv':
        return False
    if nxt.get('conjugated_type', '') != 'auxv-da':
        return False
    if nxt.get('conjugated_form', '') != 'terminal':
        return False

    return True


def _is_godan_imperfective_te(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for godan verb imperfective + て pattern.

    When a godan verb is in imperfective form followed by て, this indicates
    a missing っ in the te-form:
    - かかている (wrong - should be かかっている)
    - 向かて (wrong - should be 向かって)
    - もらていた (wrong - should be もらっていた)

    The parser interprets the incorrect te-form as imperfective + て particle.

    Args:
        curr: Current token features (potential godan verb in imperfective)
        nxt: Next token features (potential て particle)

    Returns:
        True if godan imperfective + て pattern is detected
    """
    # Check if current token is a verb in imperfective form
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'imperfective':
        return False

    # Check if next token is て (particle)
    if nxt.get('surface', '') != 'て':
        return False
    if nxt.get('pos', '') != 'prt':
        return False

    return True


def _is_i_adj_terminal_nai(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for i-adjective terminal + ない pattern.

    When an i-adjective is in terminal form followed by ない, this is wrong:
    - 遠いない (wrong - should be 遠くない)
    - 忙しいない (wrong - should be 忙しくない)
    - 高いない (wrong - should be 高くない)

    The correct negation uses the conjunctive (く) form + ない.

    Args:
        curr: Current token features (potential i-adjective in terminal form)
        nxt: Next token features (potential ない)

    Returns:
        True if i-adj terminal + ない pattern is detected
    """
    # Check if current token is an i-adjective in terminal form
    if curr.get('pos', '') != 'adj':
        return False
    if curr.get('conjugated_form', '') != 'terminal':
        return False

    # Check if next token is ない
    if nxt.get('surface', '') != 'ない':
        return False

    return True


def _is_shp_nai(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for shp (na-adjective) + ない pattern.

    When a na-adjective (shp) is directly followed by ない, this is wrong:
    - 好きない (wrong - should be 好きではない or 好きじゃない)
    - 明らかない (wrong - should be 明らかではない)

    The correct negation uses では/じゃ + ない.

    Args:
        curr: Current token features (potential na-adjective/shp)
        nxt: Next token features (potential ない adjective)

    Returns:
        True if shp + ない pattern is detected
    """
    # Check if current token is shp (na-adjective)
    if curr.get('pos', '') != 'shp':
        return False

    # Check if next token is ない (adjective)
    if nxt.get('surface', '') != 'ない':
        return False
    if nxt.get('pos', '') != 'adj':
        return False

    return True


def _is_adj_conjunctive_na(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for i-adjective conjunctive + な pattern.

    When an i-adjective in conjunctive form is followed by な (attributive),
    this is wrong:
    - 早くな出発 (wrong - should be 早い出発 or 早く出発)
    - ひどくな軽蔑 (wrong)
    - 危うくな溺死 (wrong)

    The i-adjective conjunctive form (-ku) should not be followed by な.

    Args:
        curr: Current token features (potential i-adjective conjunctive)
        nxt: Next token features (potential な attributive)

    Returns:
        True if i-adj conjunctive + な pattern is detected
    """
    # Check if current token is i-adjective in conjunctive form
    if curr.get('pos', '') != 'adj':
        return False
    if curr.get('conjugated_form', '') != 'conjunctive':
        return False

    # Check if next token is な in attributive form
    if nxt.get('surface', '') != 'な':
        return False
    if nxt.get('conjugated_form', '') != 'attributive':
        return False

    return True


def _is_masu_conjunctive_ta_attributive(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for ます conjunctive + た (attributive) pattern.

    When ます (in conjunctive form まし) is followed by た in attributive form,
    this is wrong - it mixes polite and plain styles in attributive position:
    - しましたことがない (wrong - should be したことがない)
    - 行っましたことがある (wrong - should be 行ったことがある)

    Note: ました in terminal form is correct (e.g., 食べました。)
    This pattern only catches た in attributive form (modifying a noun).

    Args:
        curr: Current token features (potential ます conjunctive)
        nxt: Next token features (potential た in attributive form)

    Returns:
        True if masu conjunctive + ta (attributive) pattern is detected
    """
    # Check if current token is ます in conjunctive form
    if curr.get('conjugated_type', '') != 'auxv-masu':
        return False
    if curr.get('conjugated_form', '') != 'conjunctive':
        return False

    # Check if next token is た in attributive form
    if nxt.get('conjugated_type', '') != 'auxv-ta':
        return False
    if nxt.get('conjugated_form', '') != 'attributive':
        return False

    return True


def _is_godan_ka_conjunctive_ka(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for godan-ka verb conjunctive + か pattern.

    When a godan-ka verb in conjunctive form is followed by the question
    particle か, this is wrong:
    - 行きか (wrong - should be 行くか or 行きますか)
    - 焼きか (wrong - should be 焼くか)

    The verb should be in terminal form before the question particle.

    Args:
        curr: Current token features (potential godan-ka conjunctive)
        nxt: Next token features (potential か particle)

    Returns:
        True if godan-ka conjunctive + か pattern is detected
    """
    if curr.get('conjugated_type', '') != 'godan-ka':
        return False
    if curr.get('conjugated_form', '') != 'conjunctive':
        return False
    if nxt.get('surface', '') != 'か':
        return False
    if nxt.get('pos', '') != 'prt':
        return False

    return True


def _is_verb_conjunctive_de_particle(
    curr: Dict[str, str],
    nxt: Dict[str, str],
    nxt2: Optional[Dict[str, str]],
) -> bool:
    """Check for verb conjunctive + で (particle) + verb/auxv pattern.

    When a verb in conjunctive form is followed by the particle で and then
    another verb or auxiliary verb, this is typically wrong - should use て:
    - 来でくれる (wrong - should be 来てくれる)
    - 来で下さい (wrong - should be 来てください)
    - 泳ぎで行く (wrong - should be 泳いで行く or 泳ぎに行く)
    - かけで下さい (wrong - should be かけてください)

    Note: We require a verb/auxv to follow で to avoid false positives on
    nominalized expressions like ぶっつづけで which are valid adverbials.

    Args:
        curr: Current token features (potential verb conjunctive)
        nxt: Next token features (potential で particle)
        nxt2: Next-next token features (should be verb or auxv)

    Returns:
        True if verb conjunctive + で particle + verb/auxv pattern is detected
    """
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'conjunctive':
        return False
    if nxt.get('surface', '') != 'で':
        return False
    if nxt.get('pos', '') != 'prt':
        return False

    # Must be followed by verb or auxiliary verb
    if nxt2 is None:
        return False
    nxt2_pos = nxt2.get('pos', '')
    if nxt2_pos not in ('v', 'auxv'):
        return False

    # Exception: つづけ forms valid nominalized adverbs (ぶっつづけで)
    curr_surface = curr.get('surface', '')
    if 'つづけ' in curr_surface:
        return False

    return True


def _is_verb_conjunctive_dekiru(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for verb conjunctive + できる pattern.

    When a verb in conjunctive form is directly followed by できる/来る,
    this is wrong - missing て:
    - 忘れできた (wrong - should be 忘れてきた)
    - 覚えできなさい (wrong - should be 覚えてきなさい)

    The verb should use te-form before きた/来た.

    Args:
        curr: Current token features (potential verb conjunctive)
        nxt: Next token features (potential できる)

    Returns:
        True if verb conjunctive + できる pattern is detected
    """
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'conjunctive':
        return False
    if nxt.get('pos', '') != 'v':
        return False
    if nxt.get('conjugated_type', '') != 'i-ichidan-ka':
        return False

    return True


def _is_to_i_verb(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for と + い (verb) pattern.

    When the particle と is followed by い (from いる/言う),
    this is typically wrong - missing っ:
    - といて (wrong - should be といって)
    - からといて (wrong - should be からといって)

    The pattern should use といって (quoting).

    Args:
        curr: Current token features (potential と particle)
        nxt: Next token features (potential い verb)

    Returns:
        True if と + い pattern is detected
    """
    if curr.get('surface', '') != 'と':
        return False
    if curr.get('pos', '') != 'prt':
        return False
    if nxt.get('surface', '') != 'い':
        return False
    if nxt.get('pos', '') != 'v':
        return False

    return True


def _is_verb_terminal_masu(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for verb terminal + ます pattern.

    Verbs in terminal form cannot be followed directly by ます.
    The correct form uses conjunctive (連用形):
    - 守るます (wrong) → 守ります (correct)
    - 入るます (wrong) → 入ります (correct)
    - するます (wrong) → します (correct)

    Args:
        curr: Current token features (potential verb terminal)
        nxt: Next token features (potential ます)

    Returns:
        True if verb terminal + ます pattern is detected
    """
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'terminal':
        return False
    if nxt.get('pos', '') != 'auxv':
        return False
    if nxt.get('conjugated_type', '') != 'auxv-masu':
        return False

    return True


def _is_verb_imperfective_ari(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for verb imperfective + あり pattern.

    Verbs in imperfective form followed by あり is wrong.
    This is typically a confused negative form:
    - 行かあり (wrong) → 行きます (positive) / 行きません (negative)
    - つまらあり (wrong) → つまります (positive) / つまりません (negative)
    - 知らあり (wrong) → 知ります (positive) / 知りません (negative)

    Args:
        curr: Current token features (potential verb imperfective)
        nxt: Next token features (potential あり)

    Returns:
        True if verb imperfective + あり pattern is detected
    """
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'imperfective':
        return False
    if nxt.get('surface', '') != 'あり':
        return False

    return True


def _is_shp_noun_missing_na(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for shp (na-adjective) + noun without な pattern.

    Na-adjectives (shp) must use な before nouns:
    - きれい花 (wrong) → きれいな花 (correct)

    This only detects cases where the na-adjective contains hiragana,
    to avoid false positives on valid sino-Japanese compounds like:
    - 簡易住宅 (valid compound, no な needed)
    - 性的偏見 (valid compound, no な needed)

    Args:
        curr: Current token features (potential na-adjective)
        nxt: Next token features (potential noun)

    Returns:
        True if shp + noun (missing な) pattern is detected
    """
    if curr.get('pos', '') != 'shp':
        return False
    if nxt.get('pos', '') != 'n':
        return False

    curr_surface = curr.get('surface', '')

    # Only flag if the na-adjective contains hiragana
    # This avoids false positives on sino-Japanese compounds
    # e.g., きれい花 (wrong) vs 簡易住宅 (valid compound)
    def contains_hiragana(s: str) -> bool:
        return any('\u3040' <= c <= '\u309f' for c in s) if s else False

    if not contains_hiragana(curr_surface):
        return False

    return True


def _is_verb_conjunctive_nara(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for verb conjunctive + なら pattern.

    Verb in conjunctive form followed by なら (conditional) is wrong.
    The conditional なら requires terminal or attributive form:
    - しなら (wrong) → したら/するなら (correct)
    - 行きなら (wrong) → 行ったら/行くなら (correct)

    Args:
        curr: Current token features (potential verb conjunctive)
        nxt: Next token features (potential なら)

    Returns:
        True if verb conjunctive + なら pattern is detected
    """
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'conjunctive':
        return False
    if nxt.get('pos', '') != 'auxv':
        return False
    if nxt.get('conjugated_type', '') != 'auxv-da':
        return False
    if nxt.get('conjugated_form', '') != 'conditional':
        return False

    return True


def _is_verb_terminal_wo(
    curr: Dict[str, str],
    nxt: Dict[str, str],
    prev: Optional[Dict[str, str]] = None,
    nxt2: Optional[Dict[str, str]] = None,
) -> bool:
    """Check for verb terminal + を pattern.

    Any verb in terminal form followed by を is wrong.
    Typically missing a nominalizer like の or こと:
    - いるを (wrong) → いるのを/いることを (correct)
    - あるを (wrong) → あるのを/あることを (correct)
    - 見るを (wrong) → 見るのを/見ることを (correct)

    Exceptions:
    - やむを得ず is an idiomatic expression
    - 來たるを is valid classical Japanese
    - ているを + verb is a valid literary nominalization pattern
      (e.g., かかっているを見て "seeing that it is hanging")

    Args:
        curr: Current token features (potential verb terminal)
        nxt: Next token features (potential を)
        prev: Previous token features (to check for て)
        nxt2: Next-next token features (to check for verb)

    Returns:
        True if verb terminal + を pattern is detected
    """
    if curr.get('pos', '') != 'v':
        return False
    if curr.get('conjugated_form', '') != 'terminal':
        return False
    if nxt.get('surface', '') != 'を':
        return False

    # Exception: やむを得ず is idiomatic
    curr_surface = curr.get('surface', '')
    if curr_surface in ('やむ', '來たる'):
        return False

    # Exception: ているを + verb is a valid literary pattern
    # e.g., かかっているを見て "seeing that it is hanging"
    if prev and prev.get('surface', '') == 'て' and nxt2 and nxt2.get('pos', '') == 'v':
        return False

    return True


# Alias for backwards compatibility
def _is_ichidan_terminal_wo(
    curr: Dict[str, str],
    nxt: Dict[str, str],
    prev: Optional[Dict[str, str]] = None,
    nxt2: Optional[Dict[str, str]] = None,
) -> bool:
    """Alias for _is_verb_terminal_wo for backwards compatibility."""
    return _is_verb_terminal_wo(curr, nxt, prev, nxt2)


def _is_shp_suffix_missing_na(
    curr: Dict[str, str],
    nxt: Dict[str, str],
) -> bool:
    """Check for shp (na-adjective) + suffix pattern (missing な).

    Na-adjectives followed by suffixes (except nominalization suffixes
    like さ, み, げ) need な:
    - きれい上 (wrong) → きれいな上 (correct)
    - 大切人 (wrong) → 大切な人 (correct)

    Exception: さ, み, げ are nominalization suffixes that attach directly:
    - 勇敢さ (correct) - nominalization
    - 悲しみ (correct) - nominalization

    Args:
        curr: Current token features (potential na-adjective)
        nxt: Next token features (potential suffix)

    Returns:
        True if shp + suffix (missing な) pattern is detected
    """
    if curr.get('pos', '') != 'shp':
        return False
    if nxt.get('pos', '') != 'suff':
        return False

    # Allow nominalization suffixes that attach directly
    nxt_surface = nxt.get('surface', '')
    if nxt_surface in ('さ', 'み', 'げ'):
        return False

    return True


def _rule_based_grammaticality(kotogram: str) -> bool:
    """Rule-based grammaticality check for common errors.

    Detects the following ungrammatical patterns:
    1. Adjectival predicates in terminal form followed by だ
       - 学生らしいだ, 行きたいだ, 読みやすいだ
    2. Double past tense marker (たた)
       - 食べたた, 見たた
    3. Doubled case particles
       - がが, をを, にに, でで, とと
    4. i-adjective incorrectly using な before a noun
       - 新しいな本, オイシイな話 (should be 新しい本, オイシイ話)
    5. だです redundancy (plain + polite copula together)
       - 学生だです, 幸福だです
    6. Wrong voicing in te-form (で instead of て)
       - されで (should be されて)
    7. Verb in terminal form followed by を
       - いうを (verb should be in attributive form before nominalizer)
    8. Double で (regardless of how parser interprets each)
       - 東京でで (double location marker)
    9. Verb in conjunctive form followed by を
       - 食べを, 会いを (missing nominalizer)
    10. na-adjective missing な before noun
        - 静か部屋 (should be 静かな部屋)
    11. Verb conjunctive + た (missing っ)
        - 行きた, 帰りた (should be 行った, 帰った)
    12. ます + ない (should be ません)
        - 開いていますない (should be 開いていません)
    13. ました + こと (should be たこと)
        - 食べましたことがある (should be 食べたことがある)
    14. Classical adjective + な + noun
        - 古きな友 (should be 古き友 or 古い友)
    15. Godan nasal + て (should be で)
        - 死んて, 読んて (should be 死んで, 読んで)
    16. Incomplete ている (てい。)
        - 食べてい。 (should be 食べている or 食べています)
    17. なければたらない (should be なければならない)
        - しなければたらない (should be しなければならない)
    18. Ichidan verb terminal + たら (wrong conditional)
        - できるたら (should be できたら)
    19. についで (should be について)
        - についで忠告 (should be について忠告)
    20. Godan sa-row conjunctive + で + auxiliary (should be て)
        - 話しでいます (should be 話しています)
    21. し + で + auxiliary (should be して)
        - どうしでよい (should be どうしてよい)
    22. いる (terminal) + ます (should be い + ます)
        - ているます (should be ています)
    23. Noun + て + いる (missing っ in te-form)
        - 祈ています, 知ている (should be 祈っています, 知っている)
    24. た + たこ (doubled た misparsed)
        - 言ったたこと, 考えたたこと (should be 言ったこと, 考えたこと)
    25. Classical-irregular-ra terminal + 。 (incomplete sentence)
        - あり。 (should be あります or ある)
    26. た (terminal) + だ (terminal) redundancy
        - なかっただ, うれしかっただ (should be なかった, うれしかった)
    27. Godan verb imperfective + て (missing っ)
        - かかている, 向かて (should be かかっている, 向かって)
    28. i-adjective terminal + ない (wrong negation)
        - 遠いない, 忙しいない (should be 遠くない, 忙しくない)
    29. shp (na-adjective) + ない (wrong negation)
        - 好きない, 明らかない (should be 好きではない, 明らかではない)
    30. i-adjective conjunctive + な (wrong attributive)
        - 早くな出発, ひどくな軽蔑 (should be 早い出発, ひどい軽蔑)
    31. ます conjunctive + た (attributive) - wrong style mixing
        - しましたことがない (should be したことがない)
    32. godan-ka verb conjunctive + か (wrong question form)
        - 行きか, 焼きか (should be 行くか, 焼くか)
    33. verb conjunctive + で (particle) - wrong te-form
        - 泳ぎで行く, 来でくれる (should be 泳ぎに行く, 来てくれる)
    34. verb conjunctive + できる - missing て
        - 忘れできた, 覚えできなさい (should be 忘れてきた, 覚えてきなさい)
    35. と + い (verb) - missing っ in といって
        - といて (should be といって)
    36. verb terminal + ます (wrong form)
        - 守るます, 入るます (should be 守ります, 入ります)
    37. verb imperfective + あり (wrong negation form)
        - 行かあり, つまらあり (should be 行きません, つまりません)
    38. shp (na-adjective) + noun (missing な)
        - 急激改革, 立派道路 (should be 急激な改革, 立派な道路)
    39. verb conjunctive + なら (wrong conditional)
        - しなら, 行きなら (should be したら/するなら, 行ったら/行くなら)
    40. ichidan verb terminal + を (missing nominalizer)
        - いるを, 見るを, 出るを (should be いるのを, 見るのを, 出るのを)
    41. shp + suffix (missing な, except nominalization suffixes)
        - きれい上, 大切人 (should be きれいな上, 大切な人)

    Args:
        kotogram: Kotogram string to check

    Returns:
        False if an ungrammatical pattern is detected, True otherwise
    """
    tokens = split_kotogram(kotogram)

    if not tokens:
        return True

    # Extract features from each token
    features: List[Dict[str, str]] = []
    for token in tokens:
        feature = extract_token_features(token)
        if feature:
            features.append(feature)

    # Check token patterns
    for i in range(len(features)):
        curr = features[i]

        # Get previous token if available
        prev = features[i - 1] if i > 0 else None

        # Get next tokens if available
        nxt = features[i + 1] if i + 1 < len(features) else None
        if nxt is None:
            continue
        nxt2 = features[i + 2] if i + 2 < len(features) else None

        # Pattern 1: Adjectival predicate + だ
        if _is_adjectival_predicate_terminal(curr) and _is_da_copula(nxt):
            return False

        # Pattern 2: Double past tense (たた)
        if _is_double_ta(curr, nxt):
            return False

        # Pattern 3: Doubled case particles
        if _is_doubled_particle(curr, nxt, nxt2, prev):
            return False

        # Pattern 4: i-adjective + な + noun
        if _is_i_adjective_with_na(curr, nxt, nxt2):
            return False

        # Pattern 5: だです redundancy
        if _is_da_desu_redundant(curr, nxt):
            return False

        # Pattern 6: Wrong voicing in te-form (で instead of て)
        if _is_te_de_wrong_voicing(curr, nxt, nxt2):
            return False

        # Pattern 7: Verb in terminal form followed by を
        if _is_verb_terminal_before_wo(curr, nxt):
            return False

        # Pattern 8: Double で (regardless of POS)
        if _is_double_de(curr, nxt, nxt2, prev):
            return False

        # Pattern 9: Verb in conjunctive form followed by を
        if _is_verb_conjunctive_before_wo(curr, nxt):
            return False

        # Pattern 10: na-adjective missing な before noun
        if _is_na_adjective_missing_na(curr, nxt):
            return False

        # Pattern 11: Verb conjunctive + た (missing っ)
        if _is_verb_conjunctive_ta(curr, nxt, nxt2):
            return False

        # Pattern 12: ます + ない (should be ません)
        if _is_masu_nai(curr, nxt):
            return False

        # Pattern 13: ました + こと (should be たこと)
        if _is_mashita_koto(curr, nxt, nxt2, prev):
            return False

        # Pattern 14: Classical adjective + な + noun
        if _is_classical_adj_na(curr, nxt, nxt2):
            return False

        # Pattern 15: Godan nasal + て (should be で)
        if _is_nasal_te_voicing_error(curr, nxt):
            return False

        # Pattern 16: Incomplete ている (てい。)
        if _is_incomplete_te_iru(curr, nxt):
            return False

        # Pattern 17: なければたらない (should be なければならない)
        if _is_nakereba_taranai(curr, nxt, nxt2):
            return False

        # Pattern 18: Ichidan verb terminal + たら (wrong conditional)
        if _is_ichidan_terminal_tara(curr, nxt):
            return False

        # Pattern 19: についで (should be について)
        if _is_ni_tsuide_error(prev, curr, nxt):
            return False

        # Pattern 20: Godan sa-row conjunctive + で + auxiliary (should be て)
        if _is_godan_sa_de_auxiliary(curr, nxt, nxt2):
            return False

        # Pattern 21: し + で + auxiliary (should be して)
        if _is_shi_de_auxiliary(curr, nxt, nxt2):
            return False

        # Pattern 22: いる (terminal) + ます (should be い + ます)
        if _is_iru_terminal_masu(curr, nxt):
            return False

        # Pattern 23: Noun + て + いる (missing っ in te-form)
        if _is_noun_te_iru_missing_geminate(curr, nxt, nxt2):
            return False

        # Pattern 24: た + たこ (doubled た misparsed)
        if _is_ta_tako_doubled_ta(curr, nxt):
            return False

        # Pattern 25: Classical-irregular-ra terminal + 。 (incomplete sentence)
        if _is_classical_ra_terminal_sentence_end(curr, nxt):
            return False

        # Pattern 26: た (terminal) + だ (terminal) redundancy
        if _is_ta_da_redundant(curr, nxt):
            return False

        # Pattern 27: Godan verb imperfective + て (missing っ)
        if _is_godan_imperfective_te(curr, nxt):
            return False

        # Pattern 28: i-adjective terminal + ない (wrong negation)
        if _is_i_adj_terminal_nai(curr, nxt):
            return False

        # Pattern 29: shp (na-adjective) + ない (wrong negation)
        if _is_shp_nai(curr, nxt):
            return False

        # Pattern 30: i-adjective conjunctive + な (wrong attributive)
        if _is_adj_conjunctive_na(curr, nxt):
            return False

        # Pattern 31: ます conjunctive + た (attributive) - wrong style mixing
        if _is_masu_conjunctive_ta_attributive(curr, nxt):
            return False

        # Pattern 32: godan-ka verb conjunctive + か (wrong question form)
        if _is_godan_ka_conjunctive_ka(curr, nxt):
            return False

        # Pattern 33: verb conjunctive + で (particle) + verb - wrong te-form
        if _is_verb_conjunctive_de_particle(curr, nxt, nxt2):
            return False

        # Pattern 34: verb conjunctive + できる - missing て
        if _is_verb_conjunctive_dekiru(curr, nxt):
            return False

        # Pattern 35: と + い (verb) - missing っ in といって
        if _is_to_i_verb(curr, nxt):
            return False

        # Pattern 36: verb terminal + ます (wrong form)
        if _is_verb_terminal_masu(curr, nxt):
            return False

        # Pattern 37: verb imperfective + あり (wrong negation form)
        if _is_verb_imperfective_ari(curr, nxt):
            return False

        # Pattern 38: shp (na-adjective) + noun (missing な)
        if _is_shp_noun_missing_na(curr, nxt):
            return False

        # Pattern 39: verb conjunctive + なら (wrong conditional)
        if _is_verb_conjunctive_nara(curr, nxt):
            return False

        # Pattern 40: verb terminal + を (missing nominalizer)
        if _is_ichidan_terminal_wo(curr, nxt, prev, nxt2):
            return False

        # Pattern 41: shp + suffix (missing な, except nominalization suffixes)
        if _is_shp_suffix_missing_na(curr, nxt):
            return False

    return True


def grammaticality(kotogram: str, use_model: bool = True) -> bool:
    """Analyze a Japanese sentence and return whether it is grammatically correct.

    This function uses either a trained neural model or rule-based checks to
    predict whether a sentence is grammatically correct.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information with POS tags and conjugation forms.
        use_model: If True (default), use the trained neural model for prediction.
                  If False, use rule-based checks to detect common grammatical errors
                  such as adjectival predicates in terminal form followed by だ
                  (e.g., 学生らしいだ, 行きたいだ).

    Returns:
        True if the sentence is predicted to be grammatically correct,
        False if predicted to be agrammatic (has grammatical errors).

    Examples:
        >>> # A grammatically correct sentence
        >>> kotogram1 = "⌈ˢ食べᵖv:e-ichidan-ba:conjunctive⌉⌈ˢますᵖauxv-masu:terminal⌉"
        >>> grammaticality(kotogram1, use_model=True)  # doctest: +SKIP
        True

        >>> # An agrammatic sentence (detected by model)
        >>> kotogram2 = "⌈ˢ食べᵖv:e-ichidan-ba:terminal⌉⌈ˢますᵖauxv-masu:terminal⌉"  # invalid
        >>> grammaticality(kotogram2, use_model=True)  # doctest: +SKIP
        False

        >>> # Rule-based detection of adjectival + だ pattern
        >>> # 学生らしいだ - adjectival suffix らしい in terminal form followed by だ
        >>> grammaticality("⌈ˢ学生ᵖn⌉⌈ˢらしいᵖsuff:adjectival:adjective:terminal⌉⌈ˢだᵖauxv:auxv-da:terminal⌉", use_model=False)
        False

        >>> # 行きたいだ - auxiliary たい in terminal form followed by だ
        >>> grammaticality("⌈ˢ行きᵖv:u-godan-ka:conjunctive⌉⌈ˢたいᵖauxv:auxv-tai:terminal⌉⌈ˢだᵖauxv:auxv-da:terminal⌉", use_model=False)
        False

        >>> # 行くべきだ - べき + だ is grammatically correct
        >>> grammaticality("⌈ˢ行くᵖv:u-godan-ka:terminal⌉⌈ˢべきᵖauxv:auxv-beki:terminal⌉⌈ˢだᵖauxv:auxv-da:terminal⌉", use_model=False)
        True
    """
    if not use_model:
        # Rule-based grammaticality detection
        return _rule_based_grammaticality(kotogram)

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
        _, _, grammaticality_probs = model.predict(field_inputs, attention_mask)
        grammaticality_idx = int(grammaticality_probs[0].argmax().item())

    # 1 = grammatic, 0 = agrammatic
    return grammaticality_idx == 1