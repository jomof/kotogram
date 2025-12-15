"""Rule-based analysis for Japanese sentences.

This script contains the legacy rule-based logic for analyzing formality and gender
associated speech patterns. It was moved from kotogram/analysis.py to keep the
main package model-focused.
"""

from typing import List, Dict, Tuple
from kotogram.kotogram import split_kotogram, extract_token_features
from kotogram.analysis import FormalityLevel, GenderLevel


def analyze_formality(kotogram: str) -> FormalityLevel:
    """Analyze a Japanese sentence and return its formality level using rules.

    Args:
        kotogram: Kotogram compact sentence representation.

    Returns:
        FormalityLevel indicating the sentence's formality level.
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
    return rule_based_formality(features)


def rule_based_formality(features: List[Dict[str, str]]) -> FormalityLevel:
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


def analyze_gender(kotogram: str) -> GenderLevel:
    """Analyze a Japanese sentence and return its gender-associated speech level using rules.

    Args:
        kotogram: Kotogram compact sentence representation.

    Returns:
        GenderLevel indicating the sentence's gender-associated speech level.
    """
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
    return rule_based_gender(features)


def rule_based_gender(features: List[Dict[str, str]]) -> GenderLevel:
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
