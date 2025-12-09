"""Formality analysis for Japanese sentences in kotogram format.

This module provides tools to analyze the formality level of Japanese sentences
by examining linguistic features such as verb forms, particles, and auxiliary verbs.
"""

from enum import Enum
from typing import List, Dict
from kotogram.kotogram import split_kotogram, extract_token_features


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