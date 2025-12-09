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
            # Check if near sentence end (within last 2 positions, allowing for auxs)
            is_near_end = i >= len(features) - 2
            casual_forms = ['conjunctive-geminate', 'volitional-presumptive']
            if conjugated_form in casual_forms:
                has_casual = True
            elif conjugated_form == 'terminal' and is_near_end:
                # Terminal だ only casual if actually at sentence end
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
    casual_particles = ['よ', 'ね', 'の', 'わ', 'な']  # Conversational particles
    # Note: These particles are acceptable with formal forms, but make plain forms casual

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