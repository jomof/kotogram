"""Formality analysis for Japanese sentences in kotogram format.

This module provides tools to analyze the formality level of Japanese sentences
by examining linguistic features such as verb forms, particles, and auxiliary verbs.
"""

import re
from enum import Enum
from typing import List, Dict
from kotogram.kotogram import split_kotogram


class FormalityLevel(Enum):
    """Formality levels for Japanese sentences."""

    VERY_FORMAL = "very_formal"           # Keigo, honorific language (敬語)
    FORMAL = "formal"                     # Polite/formal (-ます/-です forms)
    NEUTRAL = "neutral"                   # Plain/dictionary form, balanced
    CASUAL = "casual"                     # Colloquial, informal contractions
    VERY_CASUAL = "very_casual"          # Highly casual, slang
    UNPRAGMATIC_FORMALITY = "unpragmatic_formality"  # Mixed/awkward formality


def formality(kotogram: str) -> FormalityLevel:
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
    """
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


def extract_token_features(token: str) -> Dict[str, str]:
    """Extract linguistic features from a single kotogram token.

    Parses a kotogram token to extract all encoded linguistic information including
    part of speech, conjugation details, and orthographic forms. This function handles
    the variable-length POS format where empty fields are omitted by the parser.

    Kotogram format uses Unicode markers to encode linguistic information:
    - ⌈⌉ : Token boundaries
    - ˢ : Surface form (the actual text)
    - ᵖ : Part of speech and grammatical features (colon-separated)
    - ᵇ : Base orthography (dictionary form spelling)
    - ᵈ : Lemma (dictionary form)
    - ʳ : Reading/pronunciation

    The POS field (ᵖ) contains colon-separated values in a specific semantic order:
    `pos:pos_detail_1:pos_detail_2:conjugated_type:conjugated_form`

    However, the parser omits empty fields, so this function identifies each field
    semantically by checking which mapping it belongs to, rather than relying on
    positional indices.

    Args:
        token: A single kotogram token string (⌈...⌉)

    Returns:
        Dictionary with extracted features:
        - surface: The surface form of the token (actual text)
        - pos: Part of speech main category (e.g., 'v', 'n', 'auxv', 'prt')
        - pos_detail1: First POS detail level (e.g., 'general', 'common_noun')
        - pos_detail2: Second POS detail level (e.g., 'general')
        - conjugated_type: Conjugation type (e.g., 'e-ichidan-ba', 'auxv-masu')
        - conjugated_form: Conjugation form (e.g., 'conjunctive', 'terminal')
        - base_orth: Base orthography (dictionary form spelling)
        - lemma: Lemma/dictionary form
        - reading: Reading/pronunciation

    Examples:
        >>> # Extract features from a verb token
        >>> token = "⌈ˢ食べᵖv:general:e-ichidan-ba:conjunctiveᵇ食べるᵈ食べるʳタベ⌉"
        >>> features = extract_token_features(token)
        >>> features['pos']
        'v'
        >>> features['conjugated_type']
        'e-ichidan-ba'
        >>> features['conjugated_form']
        'conjunctive'

        >>> # Extract features from an auxiliary verb (note: empty fields omitted)
        >>> token = "⌈ˢますᵖauxv:auxv-masu:terminalᵇますʳマス⌉"
        >>> features = extract_token_features(token)
        >>> features['pos']
        'auxv'
        >>> features['conjugated_type']
        'auxv-masu'
        >>> features['conjugated_form']
        'terminal'
        >>> features['pos_detail1']  # Empty because parser omitted it
        ''

    Note:
        All returned dictionary values are strings. Fields that are not present
        in the token will have empty string values ('').
    """
    from .japanese_parser import (
        POS1_MAP, POS2_MAP,
        CONJUGATED_TYPE_MAP, CONJUGATED_FORM_MAP
    )

    feature = {
        'surface': '',
        'pos': '',
        'pos_detail1': '',
        'pos_detail2': '',
        'conjugated_type': '',
        'conjugated_form': '',
        'base_orth': '',
        'lemma': '',
        'reading': ''
    }

    # Extract surface form (ˢ...ᵖ)
    surface_match = re.search(r'ˢ(.*?)ᵖ', token, re.DOTALL)
    if surface_match:
        feature['surface'] = surface_match.group(1)

    # Extract POS data (ᵖ...ᵇ|ᵈ|ʳ|⌉)
    pos_match = re.search(r'ᵖ([^⌉ᵇᵈʳ]+)', token)
    if pos_match:
        pos_data = pos_match.group(1)
        parts = pos_data.split(':')

        # Main POS code (always first)
        feature['pos'] = parts[0] if len(parts) > 0 else ''

        # Parse remaining fields semantically by checking which map they belong to
        # The parser skips empty fields, so we can't rely on position alone
        #
        # Parser builds: pos:pos_detail_1:pos_detail_2:conjugated_type:conjugated_form
        # But skips empty fields, so we need to identify each by checking the maps
        for i in range(1, len(parts)):
            value = parts[i]
            if not value:
                continue

            # Check which map this value belongs to
            if value in CONJUGATED_FORM_MAP.values():
                feature['conjugated_form'] = value
            elif value in CONJUGATED_TYPE_MAP.values():
                feature['conjugated_type'] = value
            elif value in POS2_MAP.values():
                # pos_detail_2 comes after pos_detail_1, so check if we already have pos_detail_1
                if feature.get('pos_detail1'):
                    feature['pos_detail2'] = value
                else:
                    feature['pos_detail1'] = value
            elif value in POS1_MAP.values():
                # pos_detail_1 comes before pos_detail_2
                if not feature.get('pos_detail1'):
                    feature['pos_detail1'] = value
                else:
                    feature['pos_detail2'] = value
            else:
                # Unknown value - try to assign by position as fallback
                if not feature.get('pos_detail1'):
                    feature['pos_detail1'] = value
                elif not feature.get('pos_detail2'):
                    feature['pos_detail2'] = value
                elif not feature.get('conjugated_type'):
                    feature['conjugated_type'] = value
                elif not feature.get('conjugated_form'):
                    feature['conjugated_form'] = value

    # Extract base orthography (ᵇ...ᵈ|ʳ|⌉)
    base_match = re.search(r'ᵇ([^⌉ᵈʳ]+)', token)
    if base_match:
        feature['base_orth'] = base_match.group(1)

    # Extract lemma/dictionary form (ᵈ...ʳ|⌉)
    lemma_match = re.search(r'ᵈ([^⌉ʳ]+)', token)
    if lemma_match:
        feature['lemma'] = lemma_match.group(1)

    # Extract reading (ʳ...⌉)
    reading_match = re.search(r'ʳ([^⌉]+)', token)
    if reading_match:
        feature['reading'] = reading_match.group(1)

    return feature


def _analyze_formality_features(features: List[Dict[str, str]]) -> FormalityLevel:
    """Analyze extracted features to determine formality level.

    Args:
        features: List of feature dictionaries from tokens

    Returns:
        FormalityLevel based on the combination of features
    """
    if not features:
        return FormalityLevel.NEUTRAL

    # Count formality indicators
    formal_count = 0
    casual_count = 0
    very_formal_count = 0

    # Track sentence-ending formality
    sentence_ender_formality = None

    for i, feature in enumerate(features):
        pos = feature.get('pos', '')
        pos_detail1 = feature.get('pos_detail1', '')
        pos_detail2 = feature.get('pos_detail2', '')
        conjugated_type = feature.get('conjugated_type', '')
        conjugated_form = feature.get('conjugated_form', '')
        surface = feature.get('surface', '')

        # Check for formal auxiliary verbs
        # Format: auxv:auxv-masu:terminal or auxv:auxv-desu:terminal
        # Note: auxv-masu/auxv-desu appear in conjugated_type, not pos_detail1
        if pos == 'auxv' and conjugated_type == 'auxv-masu':
            formal_count += 2
            sentence_ender_formality = 'formal'
        elif pos == 'auxv' and conjugated_type == 'auxv-desu':
            formal_count += 2
            sentence_ender_formality = 'formal'

        # Check for polite imperative verbs (ください, なさい)
        # These are v:general:godan-ra:imperative
        if pos == 'v' and conjugated_form == 'imperative':
            if surface in ['ください', 'なさい']:
                formal_count += 2
                sentence_ender_formality = 'formal'

        # Check for honorific/humble forms (keigo)
        if pos == 'v' and any(x in surface for x in ['いらっしゃ', 'おっしゃ', '申し上げ', 'いただ', 'お〜になる']):
            very_formal_count += 2

        # Check for casual auxiliary verbs
        # Note: These appear in conjugated_type for auxv
        if pos == 'auxv' and conjugated_type == 'auxv-ja':  # じゃ (contracted copula)
            casual_count += 1
        elif pos == 'auxv' and conjugated_type == 'auxv-nanda':  # なんだ
            casual_count += 1
        elif pos == 'auxv' and (conjugated_type == 'auxv-hen' or conjugated_type == 'auxv-hin'):  # Kansai dialect
            casual_count += 2

        # Check for casual particles at sentence end
        if i == len(features) - 1 or (i == len(features) - 2 and features[-1].get('pos', '') == 'auxs'):
            if pos == 'prt' and surface in ['よ', 'ね', 'な', 'ぜ', 'ぞ', 'さ', 'わ', 'べ', 'け', 'の']:
                # These particles are casual, but some are used with formal too
                if sentence_ender_formality == 'formal':
                    # よ, ね, わ, の, and な are acceptable with formal forms
                    # - よ/ね: common with polite speech
                    # - わ/の/な: feminine-formal register (ojou-sama kotoba)
                    if surface not in ['よ', 'ね', 'わ', 'の', 'な']:
                        casual_count += 2  # Unpragmatic combination
                else:
                    casual_count += 1

        # Check for plain/dictionary form verbs at sentence end
        if i == len(features) - 1 or (i == len(features) - 2 and features[-1].get('pos', '') == 'auxs'):
            # For verbs: v:general:general:e-ichidan-ba:terminal (conjugated_form is terminal)
            if pos == 'v' and conjugated_form == 'terminal':
                sentence_ender_formality = 'casual'
            # For adjectives: adj:general:adjective:attributive (terminal form uses attributive)
            elif pos == 'adj' and conjugated_form == 'attributive':
                sentence_ender_formality = 'casual'
            # For plain copula: auxv:auxv-da:terminal
            elif pos == 'auxv' and conjugated_type == 'auxv-da' and conjugated_form == 'terminal':
                sentence_ender_formality = 'casual'

    # Check for unpragmatic formality mixing
    # If we have both formal and casual markers, it's unpragmatic
    if formal_count > 0 and casual_count > 1:
        return FormalityLevel.UNPRAGMATIC_FORMALITY

    # If very formal markers mixed with casual
    if very_formal_count > 0 and casual_count > 0:
        return FormalityLevel.UNPRAGMATIC_FORMALITY

    # Determine final formality level
    if very_formal_count > 0:
        return FormalityLevel.VERY_FORMAL
    elif formal_count > 0:
        return FormalityLevel.FORMAL
    elif casual_count > 1:
        return FormalityLevel.CASUAL
    elif casual_count == 1:
        return FormalityLevel.NEUTRAL
    else:
        # Default to neutral for plain forms
        return FormalityLevel.NEUTRAL
