"""Rule-based analysis for Japanese sentences.

This script contains the legacy rule-based logic for analyzing formality and gender
associated speech patterns. It was moved from kotogram/analysis.py to keep the
main package model-focused.
"""

# pylint: disable=too-many-lines, too-many-locals, too-many-return-statements

import glob
import os
from typing import Any, Dict, List, Set, Tuple

from kotogram.analysis import FormalityLevel, GenderLevel, RegisterLevel
from kotogram.kotogram import TokenFeatures, extract_token_features, split_kotogram
from train.tsv import parse_tsv


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


def rule_based_formality(features: List[TokenFeatures]) -> FormalityLevel:
    """Analyze extracted features to determine formality level.

    Args:
        features: List of feature dictionaries from tokens

    Returns:
        FormalityLevel based on the combination of features
    """
    if not features:
        return FormalityLevel.NEUTRAL

    # Formality indicators
    has_formal = False  # ます/です forms
    has_very_formal = False  # Honorific/humble forms (keigo)
    has_casual = False  # Plain forms with casual markers
    has_very_casual = False  # Very casual particles/forms

    # Track sentence-final particles for context
    sentence_final_particles = []

    for i, feature in enumerate(features):
        pos = feature.pos
        pos_detail_1 = feature.pos_detail_1
        conjugated_type = feature.conjugated_type
        surface = feature.surface

        # Check for formal auxiliary verbs (ます/です)
        if conjugated_type in ["aux-masu", "aux-desu"]:
            has_formal = True

        # Check for ください and なさい - formal but not very formal when imperative
        lemma = feature.lemma
        conjugated_form = feature.conjugated_form

        if lemma in ["くださる", "下さる"]:
            # ください (imperative of くださる) is standard formal/polite
            # Only mark as very formal if it's NOT the imperative form
            if conjugated_form == "imperative":
                has_formal = True
            else:
                # くださる in other forms (e.g., くださった, くださいます) is keigo
                has_very_formal = True

        if lemma in ["なさる", "為さる"]:
            # なさい (imperative of なさる) is polite imperative
            # Only mark as very formal if it's NOT the imperative form
            if conjugated_form == "imperative":
                has_formal = True
            else:
                # なさる in other forms is honorific keigo
                has_very_formal = True

        # Check for other very formal/honorific forms
        # Honorific verbs often have specific patterns or use special verb forms
        # Common indicators: いらっしゃる, おっしゃる, etc.
        if lemma in ["いらっしゃる", "おっしゃる", "ご覧になる", "お～になる"]:
            has_very_formal = True
        # Humble verbs (謙譲語)
        # Note: Sudachi may use potential forms like いただける
        if lemma in [
            "いたす",
            "致す",
            "まいる",
            "申す",
            "申し上げる",
            "お～する",
            "いただく",
            "頂く",
            "いただける",
        ]:
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
        if conjugated_type == "aux-da":
            casual_forms = ["continuative-geminate", "volitional-presumptive"]
            if conjugated_form in casual_forms:
                has_casual = True
            elif conjugated_form == "terminal":
                # Terminal だ is casual if followed only by punctuation/brackets
                # This handles quoted speech like 「好きだ。」
                is_at_clause_end = True
                for j in range(i + 1, len(features)):
                    next_pos = features[j].pos
                    next_surface = features[j].surface
                    # Skip punctuation and brackets
                    if next_pos == "aux-symbol" or next_surface in [
                        "」",
                        "』",
                        ")",
                        "）",
                    ]:
                        continue
                    # If we hit another token, だ is mid-sentence
                    is_at_clause_end = False
                    break
                if is_at_clause_end:
                    has_casual = True

        # Check for very casual auxiliary verbs
        if conjugated_type in ["aux-ja", "aux-nanda", "aux-hin", "aux-hen", "aux-nsu"]:
            has_very_casual = True

        # Sudachi may parse じゃ as conj instead of auxv-ja
        if pos == "conj" and surface == "じゃ":
            has_very_casual = True

        # Check for sentence-final particles
        if pos == "particle" and pos_detail_1 == "sentence-final-particle":
            sentence_final_particles.append(surface)

    # Analyze sentence-final particles for casual/very casual markers
    very_casual_particles = ["ぜ", "ぞ", "ぞい", "さ"]  # Masculine/rough particles
    # Casual particles include base forms and lengthened variants (なあ, ねえ, よー, etc.)
    casual_particles = [
        "よ",
        "ね",
        "の",
        "わ",
        "な",  # Base forms
        "なあ",
        "なー",
        "ねえ",
        "ねー",  # Lengthened な/ね
        "よお",
        "よー",
        "わあ",
        "わー",  # Lengthened よ/わ
        "かしら",  # Feminine wondering particle
        "かい",  # Casual question particle (masculine)
        "もの",
        "もん",  # Explanatory particle (feminine casual)
    ]
    # Note: These particles are acceptable with formal forms, but make plain forms casual

    # Combine adjacent sentence-final particles (e.g., か+い -> かい)
    combined_particles = "".join(sentence_final_particles)

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

    # Fall through to feature-based gender analysis
    return rule_based_gender(features)


def rule_based_gender(features: List[TokenFeatures]) -> GenderLevel:
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
    particle_sequence = []  # List of (index, surface, pos_detail_1)

    for i, feature in enumerate(features):
        pos = feature.pos
        pos_detail_1 = feature.pos_detail_1
        surface = feature.surface
        lemma = feature.lemma
        conjugated_type = feature.conjugated_type
        conjugated_form = feature.conjugated_form

        # Check for masculine pronouns
        # 俺 (ore) - strongly masculine
        # 僕 (boku) - masculine (but used by some women too)
        # お前 (omae) - masculine second-person pronoun
        # Check both surface form and lemma since parsers vary
        if pos == "pron":
            if surface in ["俺", "おれ", "オレ"] or lemma in ["俺", "おれ", "オレ"]:
                has_masculine = True
            if surface in ["僕", "ぼく", "ボク"] or lemma in [
                "僕",
                "ぼく",
                "ボク",
                "僕-代名詞",
            ]:
                has_masculine = True
            # お前 (omae) - rough masculine second-person pronoun
            if surface in ["お前", "おまえ", "オマエ"] or lemma in ["御前", "お前"]:
                has_masculine = True

            # Check for feminine pronouns
            # あたし (atashi) - feminine variant of 私
            # あたくし (atakushi) - very formal feminine
            # Note: lemma might be 私 for these, so check surface
            if surface in ["あたし", "アタシ", "あたくし", "アタクシ"]:
                has_feminine = True

        # Check for rough masculine auxiliary verb forms
        # ねえ (nee) - rough masculine negation (variant of ない)
        if pos == "aux-verb" and conjugated_type == "aux-nai":
            if surface in ["ねえ", "ねー", "ネエ", "ネー"]:
                has_masculine = True

        # Check for だろ (daro) - masculine sentence-final assertive
        # volitional-presumptive form of だ used assertively
        if pos == "aux-verb" and conjugated_type == "aux-da":
            if conjugated_form == "volitional-presumptive" and surface in [
                "だろ",
                "ダロ",
            ]:
                has_masculine = True

        # Track particles for pattern detection
        if pos == "particle":
            particle_sequence.append((i, surface, pos_detail_1))

        # Check for かしら (kashira) - feminine wonder/question marker
        if surface in ["かしら", "カシラ"]:
            has_feminine = True

    # Analyze particle patterns
    masculine_particles = ["ぜ", "ゼ", "ぞ", "ゾ", "ぞい", "ゾイ"]
    feminine_particles = ["わ", "ワ"]

    # Check for のよ / のね patterns (feminine sentence endings)
    # Pattern: の (pre_noun_particle) followed by よ/ね (sentence_final_particle)
    # Also match lengthened variants like のねー, のよー
    for j in range(len(particle_sequence) - 1):
        idx1, surf1, detail1 = particle_sequence[j]
        idx2, surf2, detail2 = particle_sequence[j + 1]
        # Check if consecutive particles
        if idx2 == idx1 + 1:
            if surf1 == "の" and detail1 == "nominal-particle":
                if (
                    surf2 in ["よ", "ヨ", "よー", "よお", "ヨー"]
                    and detail2 == "sentence-final-particle"
                ):
                    has_feminine = True
                if (
                    surf2 in ["ね", "ネ", "ねー", "ねえ", "ネー"]
                    and detail2 == "sentence-final-particle"
                ):
                    has_feminine = True

    # Check individual sentence-final particles
    for _, surface, pos_detail_1 in particle_sequence:
        if pos_detail_1 in ["sentence-final-particle", "adverbial-particle"]:
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


def analyze_register(kotogram: str) -> Set[RegisterLevel]:
    """Analyze a Japanese sentence and return its specific register(s)/dialect(s) using rules.

    Args:
        kotogram: Kotogram compact sentence representation.

    Returns:
        Set of RegisterLevel indicating the sentence's register(s).
    """
    # Split into tokens and extract linguistic features
    tokens = split_kotogram(kotogram)

    if not tokens:
        return {RegisterLevel.NEUTRAL}

    # Extract features from each token
    features = []
    for token in tokens:
        feature = extract_token_features(token)
        if feature:
            features.append(feature)

    # Analyze register based on features
    return rule_based_register(features)


def kotogram_str(features: List[TokenFeatures]) -> str:
    return "".join(f.surface for f in features)


def rule_based_register(features: List[TokenFeatures]) -> Set[RegisterLevel]:
    """
    Apply heuristic rules...xtracted features to determine register level(s).

    Args:
        features: List of feature dictionaries from tokens

    Returns:
        Set of RegisterLevel based on the combination of features
    """
    # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-branches, too-many-statements, too-many-boolean-expressions
    if not features:
        return {RegisterLevel.NEUTRAL}

    detected_registers: Set[RegisterLevel] = set()

    # Normalize features: ensure lemma exists (fallback to surface)
    for f in features:
        if not f.lemma:
            f.lemma = f.surface

    for i, feature in enumerate(features):  # iterate tokens
        pos = feature.pos
        pos_detail_1 = feature.pos_detail_1
        surface = feature.surface
        lemma = (
            feature.lemma
        )  # Now guaranteed to overlap with surface if lemma was missing

        # Kansaiben
        if surface in [
            "やん",
            "ねん",
            "へん",
            "ひん",
            "さかい",
            "せや",
            "せやな",
            "ほんま",
            "なんでやねん",
            "あかん",
            "ええ",
        ]:
            detected_registers.add(RegisterLevel.KANSAIBEN)
        if lemma in ["や", "ねん", "へん"]:  # Auxiliaries/particles
            # Exclude やいなや pattern (standard Japanese "as soon as")
            if lemma == "や":
                # Check if this is part of やいなや - could be first や or second や
                # First や: followed by いな
                # Second や: preceded by いな or いなや
                is_yainaya = False
                if i < len(features) - 1 and features[i + 1].surface in [
                    "いな",
                    "いなや",
                ]:
                    is_yainaya = True
                elif i > 0 and features[i - 1].surface in ["いな", "いなや"]:
                    is_yainaya = True

                if not is_yainaya:
                    detected_registers.add(RegisterLevel.KANSAIBEN)
            else:
                detected_registers.add(RegisterLevel.KANSAIBEN)
        # Check 'や' as surface if lemma missing (common in short parses)
        if surface == "や" and (
            pos_detail_1.startswith("aux") or pos.startswith("aux")
        ):
            # Exclude やいなや pattern
            is_yainaya = False
            if i < len(features) - 1 and features[i + 1].surface in ["いな", "いなや"]:
                is_yainaya = True
            elif i > 0 and features[i - 1].surface in ["いな", "いなや"]:
                is_yainaya = True

            if not is_yainaya:
                detected_registers.add(RegisterLevel.KANSAIBEN)
        # Check 'ん' (nu/negation) common in Kansai "shiran"
        # Exclude standard "masen" or "arimasen" where 'noun' is part of polite aux
        if surface == "ん" and (
            pos_detail_1.startswith("aux") or pos.startswith("aux")
        ):
            # Look behind to see if it's 'mase' + 'noun' (standard polite)
            if i > 0 and features[i - 1].surface == "ませ":
                pass
            else:
                detected_registers.add(RegisterLevel.KANSAIBEN)
        # "chau" (tigau)
        if lemma == "ちゃう" or surface == "ちゃう":
            # Only trigger if it's a verb (meaning 'chigau'/wrong).
            # If it's an auxiliary (auxv), it's likely standard casual 'te-shimau' -> 'chau'.
            if pos.startswith("verb"):
                detected_registers.add(RegisterLevel.KANSAIBEN)
        # "toki" (te-oku imperative) e.g. "shitoki"
        if surface == "とき" and i > 0 and features[i - 1].pos.startswith("verb"):
            detected_registers.add(RegisterLevel.KANSAIBEN)
        # "nanbo"
        if lemma == "なんぼ" or surface == "なんぼ":
            detected_registers.add(RegisterLevel.KANSAIBEN)
        # "wa" sentence final after adjective (meondokusai wa) - tricky but common in Kansai/Casual
        # Restrict to known phrase structure in dataset to avoid overtrigger
        if surface == "わ" and i > 0 and features[i - 1].pos.startswith("adj"):
            detected_registers.add(
                RegisterLevel.KANSAIBEN
            )  # Context dependent, but acceptable for this dataset

        # Hakataben
        if surface in [
            "と",
            "ばい",
            "けん",
            "よか",
            "すごか",
            "うまか",
            "好いとう",
            "好いとー",
            "どげん",
        ]:
            # Allow 'と' at end or before punctuation
            if surface == "と":
                if i == len(features) - 1:
                    detected_registers.add(RegisterLevel.HAKATABEN)
                elif i < len(features) - 1 and features[i + 1].surface in [
                    "？",
                    "?",
                    "！",
                    "!",
                    "。",
                ]:
                    detected_registers.add(RegisterLevel.HAKATABEN)
            else:
                detected_registers.add(RegisterLevel.HAKATABEN)

        # たい (tai) - Hakataben particle, but ONLY at sentence end or before punctuation
        # NOT the auxiliary verb たい (want to) which appears with です or と (quotative)
        if surface == "たい":
            # Check if it's the auxiliary verb (adj-aux) "want to" or the Hakataben particle
            # Hakataben 'tai' is usually parsed as a final particle or distinct from adj-aux
            is_dialect = True
            if feature.pos == "adj-aux" or feature.lemma == "たい":
                is_dialect = False

            if is_dialect:
                # Check if it's at the end or before punctuation
                if i == len(features) - 1:
                    detected_registers.add(RegisterLevel.HAKATABEN)
                elif i < len(features) - 1:
                    next_surface = features[i + 1].surface
                    # Only trigger if followed by punctuation, NOT by です/ます/と
                    if next_surface in ["？", "?", "！", "!", "。"]:
                        detected_registers.add(RegisterLevel.HAKATABEN)
        if lemma in ["好く"]:
            if i < len(features) - 1 and features[i + 1].surface == "と":
                detected_registers.add(RegisterLevel.HAKATABEN)
        # Adjective ending 'ka'/'ka-' (sugoka-) - Only at sentence end to avoid question particle か
        if (
            surface in ["か", "かー"]
            and i > 0
            and features[i - 1].pos.startswith("adj")
        ):
            # Only trigger at sentence end or before terminal punctuation
            if i == len(features) - 1 or (
                i < len(features) - 1
                and features[i + 1].surface in ["。", "！", "!", "？", "?"]
            ):
                detected_registers.add(RegisterLevel.HAKATABEN)
        # Specific token combo 'sui' + 'to' (suito-)
        if surface == "と" and i > 0 and features[i - 1].surface.startswith("好い"):
            detected_registers.add(RegisterLevel.HAKATABEN)
        if surface.startswith("好いと"):
            detected_registers.add(RegisterLevel.HAKATABEN)
        # "sogen" - handle tokenization split
        if "そげん" in surface:  # Simple surface check if token exists
            detected_registers.add(RegisterLevel.HAKATABEN)
        # If split into "so" + "gen...", check neighbors
        if (
            surface == "そ"
            and i < len(features) - 1
            and features[i + 1].surface.startswith("げん")
        ):
            detected_registers.add(RegisterLevel.HAKATABEN)
        # If split into "soge" + "noun"
        if (
            surface == "そげ"
            and i < len(features) - 1
            and features[i + 1].surface == "ん"
        ):
            detected_registers.add(RegisterLevel.HAKATABEN)

        # "samukaro", "yokaro" (adjective + ro)
        if surface.endswith("かろ") or surface.endswith("かろう"):
            detected_registers.add(RegisterLevel.HAKATABEN)
        # "kon ne" (hayaku kon ne)
        if (
            surface == "来"
            and i < len(features) - 1
            and features[i + 1].surface.startswith("ん")
        ):
            detected_registers.add(RegisterLevel.HAKATABEN)
        if surface == "来ん":
            detected_registers.add(RegisterLevel.HAKATABEN)
        # "sogen"
        if lemma == "そげん" or surface == "そげん":
            detected_registers.add(RegisterLevel.HAKATABEN)
        # "ba" (variant of 'wo' in Kyushu, sometimes 'noun' + 'ba') - careful
        # "ken" (kara/because)
        if surface == "けん" and (
            pos_detail_1.startswith("brt") or pos_detail_1.startswith("particle")
        ):
            detected_registers.add(RegisterLevel.HAKATABEN)

        # OJOUSAMA
        # Sentence ending "desu wa"
        if surface == "わ" and i > 0 and features[i - 1].surface == "です":
            detected_registers.add(RegisterLevel.OJOUSAMA)
        # "masu wa" / "masen wa"
        if surface == "わ" and i > 0 and (features[i - 1].surface in ("ます", "て")):
            # "masu wa" or "te wa" (rare, but "yoroshikute wa")
            detected_registers.add(RegisterLevel.OJOUSAMA)
        if surface == "わ" and i > 0 and "ません" in features[i - 1].surface:
            # "masen wa" is usually split as "mase" + "noun" + "wa". Check negation 'noun'.
            pass  # logic below for 'wa' after 'noun'
        if (
            surface == "わ"
            and i > 1
            and features[i - 1].surface == "ん"
            and features[i - 2].surface == "ませ"
        ):
            detected_registers.add(RegisterLevel.OJOUSAMA)

        # "masu no"
        if (
            surface == "の"
            and pos == "particle"
            and pos_detail_1 == "sentence-final-particle"
            and i > 0
            and features[i - 1].surface == "ます"
        ):
            detected_registers.add(RegisterLevel.OJOUSAMA)
        # "desu no" (could be question or assertion)
        if (
            surface == "の"
            and pos == "particle"
            and pos_detail_1 == "sentence-final-particle"
            and i > 0
            and features[i - 1].surface == "です"
        ):
            detected_registers.add(RegisterLevel.OJOUSAMA)
        # "masen no"
        if (
            surface == "の"
            and pos == "particle"
            and pos_detail_1 == "sentence-final-particle"
            and i > 1
            and features[i - 1].surface == "ん"
            and features[i - 2].surface == "ませ"
        ):
            # Already restricted by context, this is fine
            detected_registers.add(RegisterLevel.OJOUSAMA)

        # "deshita no" / "mashita no"
        if (
            surface == "の"
            and i > 0
            and (features[i - 1].surface == "た" or "た" in features[i - 1].surface)
        ):
            # Check if previous was polite 'deshi' or 'mashi'
            if i > 1 and (features[i - 2].surface in ["でし", "まし"]):
                detected_registers.add(RegisterLevel.OJOUSAMA)

        # "gokigenyou"
        if "ごきげんよう" in surface or "ごきげんよう" in lemma:
            detected_registers.add(RegisterLevel.OJOUSAMA)
        # "yoroshikute"
        if surface.startswith("よろしくて"):
            detected_registers.add(RegisterLevel.OJOUSAMA)
        # Handle split "yoroshiku" + "te" + "yo"
        if surface == "て" and i > 0 and features[i - 1].lemma == "よろしい":
            # Check if next is 'yo' or 'wa' or '?'
            if i < len(features) - 1 and features[i + 1].surface in [
                "よ",
                "わ",
                "の",
                "？",
                "?",
            ]:
                detected_registers.add(RegisterLevel.OJOUSAMA)

        # "mashite?" (Question polite)
        if (
            surface == "して" and i > 0 and features[i - 1].surface == "ま"
        ):  # "ma-shite"
            # Wait, "takemashite" -> take (v) + mashi (aux) + te (prt).
            pass
        # "masu" + "te" (mashi te)
        if surface == "て" and i > 0 and features[i - 1].surface == "まし":
            # "mashi te" at end or before ?
            if i == len(features) - 1 or (
                i < len(features) - 1
                and features[i + 1].surface in ["？", "?", "よ", "の"]
            ):
                detected_registers.add(RegisterLevel.OJOUSAMA)

        # "koto" at end (exclamatory/soft)
        if surface == "こと" and i > 0:
            # End of sentence or before punctuation
            if i == len(features) - 1 or features[i + 1].surface in [
                "。",
                "？",
                "?",
                "！",
                "!",
            ]:
                # Check previous token type (Adj or Masu/Desu) to avoid generic nouns
                prev = features[i - 1]
                if (
                    prev.pos.startswith("adj")
                    or "ませ" in prev.surface
                    or prev.surface == "です"
                    or prev.surface == "ない"
                    or prev.surface == "ん"
                ):
                    detected_registers.add(RegisterLevel.OJOUSAMA)
                # "kawairashii koto" -> adj + koto.

        # "ara ara"
        if "あらあら" in surface:
            detected_registers.add(RegisterLevel.OJOUSAMA)

        # GUNTAI
        # "de arimasu"
        if surface == "あり" and i > 0 and features[i - 1].surface == "で":
            if i < len(features) - 1 and features[i + 1].surface.startswith("ます"):
                detected_registers.add(RegisterLevel.GUNTAI)
        # Check "arimasu" directly
        if surface == "あります" and i > 0 and features[i - 1].surface == "で":
            detected_registers.add(RegisterLevel.GUNTAI)

        # "Jibun" (First person)
        if surface == "自分":
            # "Jibun wa" or "Jibun ga" (Military "I" or standard reflexive)
            # To reduce false positives, only flag as Guntai if there are OTHER military words
            # or if it's at the very start of a sentence in a formal, non-desire context.
            military_context = False
            for f in features:
                if f.surface in [
                    "了解",
                    "任務",
                    "作戦",
                    "前進",
                    "報告",
                    "異常",
                    "あります",
                    "あります！",
                ]:
                    military_context = True

            if military_context:
                if i < len(features) - 1 and features[i + 1].surface in ["は", "が"]:
                    detected_registers.add(RegisterLevel.GUNTAI)
                if (
                    i == 0
                    and i < len(features) - 1
                    and features[i + 1].surface in ["、", ",", "は", "が", "で"]
                ):
                    detected_registers.add(RegisterLevel.GUNTAI)
            elif (
                i == 0
                and i < len(features) - 1
                and features[i + 1].surface in ["は", "が"]
            ):
                # "Jibun wa..." at start is strongly indicative of military "I"
                # even without other explicit military words, BUT only in formal/stern context.
                # Exclude if it has desire forms ("tai") or soft polite markers ("desu/masu")
                # which are more common in neutral self-reflection.
                has_soft = False
                for f in features:
                    if f.surface in ["たい", "です", "ます"]:
                        has_soft = True
                if not has_soft:
                    detected_registers.add(RegisterLevel.GUNTAI)

        # "Shuugou!" (imperative noun usage)
        if "集合" in surface or "集合" in lemma:
            detected_registers.add(RegisterLevel.GUNTAI)
        # "Ryoukai"
        if surface == "了解":
            detected_registers.add(RegisterLevel.GUNTAI)
        # "Ninmu"
        if surface == "任務":
            detected_registers.add(RegisterLevel.GUNTAI)
        # "Sakusen"
        if surface == "作戦":
            detected_registers.add(RegisterLevel.GUNTAI)
        # "Zenshin"
        if surface == "前進":
            detected_registers.add(RegisterLevel.GUNTAI)
        # "Houkoku"
        if surface == "報告":
            detected_registers.add(RegisterLevel.GUNTAI)
        # "Haaku" (Grasp/Understand - common usage)
        if (
            surface == "把握"
            and i < len(features) - 1
            and features[i + 1].surface.startswith("し")
        ):
            detected_registers.add(RegisterLevel.GUNTAI)
        # "Kyuritsu" (Discipline - Kiritsu) -> "Kiritsu wo mamore"
        if surface == "規律":
            detected_registers.add(RegisterLevel.GUNTAI)

        # "Ijou nashi"
        if (
            surface == "異常"
            and i < len(features) - 1
            and "なし" in features[i + 1].surface
        ):
            detected_registers.add(RegisterLevel.GUNTAI)
        # "Ijou arimasen"
        if surface == "異常":
            if i < len(features) - 3:
                f1 = features[i + 1].surface
                f2 = features[i + 2].surface
                f3 = features[i + 3].surface
                if "あり" in f1 and "ませ" in f2 and "ん" in f3:
                    detected_registers.add(RegisterLevel.GUNTAI)
        # "Ijou arimasen"
        if surface == "異常":
            if i < len(features) - 3:
                f1 = features[i + 1].surface
                f2 = features[i + 2].surface
                f3 = features[i + 3].surface
                if "あり" in f1 and "ませ" in f2 and "ん" in f3:
                    detected_registers.add(RegisterLevel.GUNTAI)
        # "Mokuhyou" (Target) - Too common (Goal). Removed.
        # if surface == '目標':
        #      detected_registers.add(RegisterLevel.GUNTAI)

        # Imperatives
        # "MATE!", "SE-YO!", "MAMORE!"
        # Often: Verb(Imperative) + !
        if (
            pos_detail_1.endswith("imperative")
            or "imperative" in pos_detail_1
            or (feature.conjugated_form and "imperative" in feature.conjugated_form)
        ):
            # Check for exclamation or strong context
            # Exclude "Kudasai", "Nasai" (polite request/command)
            if lemma not in [
                "ください",
                "下さい",
                "下さる",
                "なさい",
                "為さる",
            ] and surface not in ["ください", "下さい", "kudasai", "なさい", "nasai"]:
                if i < len(features) - 1 and features[i + 1].surface in ["！", "!"]:
                    # Require military context for generic imperatives
                    military_context = False
                    for f in features:
                        if f.surface in [
                            "了解",
                            "任務",
                            "作戦",
                            "前進",
                            "報告",
                            "異常",
                            "あります",
                            "あります！",
                            "自分",
                        ]:
                            military_context = True
                    if military_context or lemma in [
                        "待つ",
                        "止まる",
                        "撃つ",
                        "伏せる",
                        "戻る",
                    ]:
                        detected_registers.add(RegisterLevel.GUNTAI)
                # "seyo" specifiically
                if surface == "せよ":
                    # Exclude "ni seyo" (even if)
                    if i > 0 and features[i - 1].surface == "に":
                        pass
                    else:
                        detected_registers.add(RegisterLevel.GUNTAI)
        # "da" / "aru" + ! (Plain form + !)
        # "Kaishi suru!"
        if (
            surface == "する"
            and i < len(features) - 1
            and features[i + 1].surface in ["！", "!"]
        ):
            # Only if preceded by military-ish noun (Sakusen, Ninmu, Kaishi)
            # We added noun triggers, so maybe we don't need this, but "Kaishi" isn't triggered yet.
            if i > 0 and features[i - 1].surface in ["開始"]:
                detected_registers.add(RegisterLevel.GUNTAI)
        if surface == "開始":  # Trigger on noun itself?
            detected_registers.add(RegisterLevel.GUNTAI)

        # "Nai" + ! (Strong negative assertion/imperative feel)
        # Narrowed: Only "Yurusanai" (Unforgivable) or similar strict words
        if (
            surface == "ない"
            and i < len(features) - 1
            and features[i + 1].surface in ["！", "!"]
        ):
            if i > 0 and features[i - 1].lemma in [
                "許す",
                "許せる",
                "許される",
            ]:  # Yurusenai, Yurusarenai
                detected_registers.add(RegisterLevel.GUNTAI)

        # Netslang
        if surface in ["w", "www", "草生える", "now"]:
            detected_registers.add(RegisterLevel.NETSLANG)
        if (
            surface == "なう" and feature.pos != "verb"
        ):  # Slang 'now' is usually not a verb
            detected_registers.add(RegisterLevel.NETSLANG)
        if surface == "乙":
            # Slang usage usually stand-alone or exclamation
            is_slang = True
            # Pos check: if it's a common noun, check for ordinals (A vs B)
            if feature.pos_detail_1 == "common-noun":
                # Check if it looks like an ordinal (mostly near 'の' or '甲')
                if i > 0 and features[i - 1].surface == "の":
                    is_slang = False
                if i < len(features) - 1 and features[i + 1].surface == "の":
                    is_slang = False
                if any("甲" in f.surface for f in features):
                    is_slang = False

            if is_slang:
                # Strongly favor slang if at end or with punctuation/w
                if i == len(features) - 1 or (
                    i < len(features) - 1
                    and features[i + 1].surface in ["。", "w", "W", "！", "!", "ｗ"]
                ):
                    detected_registers.add(RegisterLevel.NETSLANG)
                elif any(f.surface in ["w", "www", "乙"] for f in features):
                    detected_registers.add(RegisterLevel.NETSLANG)
        if surface in ["w", "W"] and len(surface) == 1:
            # Exclude middle initials: Check if surrounded by dots or spaces
            is_initial = False
            if i > 0 and features[i - 1].surface in [".", "．", " "]:
                is_initial = True
            if i < len(features) - 1 and features[i + 1].surface in [".", "．", " "]:
                is_initial = True

            if not is_initial:
                # Strongly favor slang if at end or with punctuation/w
                if i == len(features) - 1 or (
                    i < len(features) - 1
                    and features[i + 1].surface in ["。", "！", "!", "ｗ"]
                ):
                    detected_registers.add(RegisterLevel.NETSLANG)
                elif any(f.surface in ["www", "乙"] for f in features):
                    detected_registers.add(RegisterLevel.NETSLANG)
        if "w" in surface and len(surface) > 1 and all(c == "w" for c in surface):
            detected_registers.add(RegisterLevel.NETSLANG)
        if "W" in surface and len(surface) > 1 and all(c == "W" for c in surface):
            detected_registers.add(RegisterLevel.NETSLANG)
        if lemma in ["ktkr", "wktk", "kwsk"] or surface in ["ktkr", "wktk", "kwsk"]:
            detected_registers.add(RegisterLevel.NETSLANG)
        if "詰む" in lemma or "詰んだ" in surface:
            detected_registers.add(RegisterLevel.NETSLANG)
        if "情弱" in lemma or "情弱" in surface:
            detected_registers.add(RegisterLevel.NETSLANG)

        # "Yuushuu" for winning/LOL
        if surface == "優勝":
            # Slang usage usually stand-alone or small context
            is_slang = True
            # Exclude if direct object of competition words or part of formal titles
            if any(
                w in s
                for f in features
                for s in [f.surface]
                for w in [
                    "チーム",
                    "選手",
                    "大会",
                    "試合",
                    "個人",
                    "団体",
                    "作品",
                    "候補",
                    "選ぶ",
                    "選び",
                    "高校",
                    "大学",
                    "競技",
                    "基準",
                    "程遠い",
                    "幾度",
                    "コンテスト",
                    "タイトル",
                    "おめでとう",
                    "獲得",
                    "決定",
                    "トーナメント",
                    "優勝者",
                    "だろう",
                ]
            ):
                is_slang = False

            if is_slang:
                # Strongly favor slang if followed by 'w' or at end
                next_s = features[i + 1].surface if i < len(features) - 1 else None
                if i == len(features) - 1 or next_s in [
                    "。",
                    "w",
                    "W",
                    "！",
                    "!",
                    "ｗ",
                ]:
                    detected_registers.add(RegisterLevel.NETSLANG)
                elif next_s in ["し", "する", "した", "して"] and len(features) < 8:
                    # Exclude "shisou" (looks like) - often formal prediction
                    if i < len(features) - 2 and features[i + 2].surface == "そう":
                        pass
                    else:
                        detected_registers.add(RegisterLevel.NETSLANG)
                elif any(f.surface in ["w", "www", "乙"] for f in features):
                    detected_registers.add(RegisterLevel.NETSLANG)
        # "Kusa" for LOL
        if "草生える" in surface or "草不可避" in surface:  # Caught by substring
            detected_registers.add(RegisterLevel.NETSLANG)
        elif surface == "草" and feature.pos == "suff":  # NEVER slang as a suffix
            pass
        elif (
            surface == "草" and feature.pos_detail_1 != "common-noun"
        ):  # In slang context, it's often not parsed as a normal noun
            detected_registers.add(RegisterLevel.NETSLANG)
        elif surface == "草" and (
            i == len(features) - 1
            or (
                i < len(features) - 1
                and features[i + 1].surface in ["。", "w", "W", "！", "!", "ｗ", "ｗｗ"]
            )
        ):
            # "Kusa" at the end of a sentence
            # Check if preceded by a particle? Standard "kusa wo hiku" (pull grass)
            is_slang = True
            if i > 0 and features[i - 1].surface in ["を", "の", "に"]:
                is_slang = False
                # Exception: "sasuga ni kusa" (Indeed grass/LOL) is a common slang pattern
                if (
                    i > 1
                    and features[i - 1].surface == "に"
                    and features[i - 2].surface == "流石"
                ):
                    is_slang = True

        # TOHOKU
        # "dabe" (Copula)
        if surface.endswith("だべ"):
            detected_registers.add(RegisterLevel.TOHOKU)
        if surface == "べ" and i > 0:
            # Check previous token
            # "ikube", "surube"
            detected_registers.add(RegisterLevel.TOHOKU)

        # "nda" (It is so / Yes)
        if surface == "んだ":
            # Often "noun" + "da", so check split
            detected_registers.add(RegisterLevel.TOHOKU)
        if (
            surface == "ん"
            and i < len(features) - 1
            and features[i + 1].surface == "だ"
        ):
            # "sou na n da" -> Standard
            # "n da" at start or short answer -> Tohoku
            if i == 0:
                detected_registers.add(RegisterLevel.TOHOKU)

        # "keppare"
        if "けっぱれ" in surface:
            detected_registers.add(RegisterLevel.TOHOKU)
        # "menkoi"
        if "めんこい" in surface or "めんこい" in lemma:
            detected_registers.add(RegisterLevel.TOHOKU)
        # "warashi"
        if "わらし" in surface:
            # Check if it means child (not zashiki-warashi context specifically, but general)
            detected_registers.add(RegisterLevel.TOHOKU)
        # "ora" (I) - Tohoku version of Ore
        # Be careful of standard "Ora!" (Hey!)
        if surface == "おら" and pos == "pron":
            detected_registers.add(RegisterLevel.TOHOKU)

        # BUSHI
        # "gozaru"
        if "ござる" in surface or "ござる" in lemma:
            # Exclude standard polite "gozaimasu" / "gozaimashita"
            is_bushi_gozaru = True

            # Check current token (rarely contains masu if tokenized, but possible if unnormalized)
            if "ます" in surface or "ませ" in surface or "まし" in surface:
                is_bushi_gozaru = False

            # Check NEXT token for 'masu' / 'mase' / 'mashi' / 'mashita'
            if i < len(features) - 1:
                next_surf = features[i + 1].surface
                if next_surf.startswith("ま") or next_surf.startswith(
                    "マ"
                ):  # masu, mase, mashi
                    is_bushi_gozaru = False
                if next_surf == "て" and i < len(features) - 2:
                    # gozai mashi te -> gozai + mashi + te ?
                    # actually check i+1 'mashi'.
                    pass

            # Double check "de gozaru" (positive trigger)
            # If "de" precedes, and no "masu" follows, it's likely Bushi/Formal-Archaic
            # "de gozaimasu" (Standard) vs "de gozaru" (Bushi/Archaic)

            if is_bushi_gozaru:
                detected_registers.add(RegisterLevel.BUSHI)
        # "katajikenai"
        if "かたじけない" in surface:
            detected_registers.add(RegisterLevel.BUSHI)
        # "sessha" (I)
        if surface == "拙者":
            detected_registers.add(RegisterLevel.BUSHI)
        # "soregashi" (I)
        if surface == "某":
            detected_registers.add(RegisterLevel.BUSHI)
        # "onushi" (You)
        if surface == "お主":
            detected_registers.add(RegisterLevel.BUSHI)
        # "mairu" (Come/Go - archaic/humble)
        # Needs context to distinguish from standard Kenjogo "Mairimasu"
        if lemma == "参る":
            # "Iza, mairu" (Let's go/en garde) -> Bushi
            # "Mairimashita" (I gave up / I came) -> Standard/Kenjogo
            if surface == "参る":  # Dictionary form often archaic imperative/volitional
                detected_registers.add(RegisterLevel.BUSHI)

        # "noun" / "nu" (Negative archaic)
        if surface == "ぬ" and pos_detail_1.startswith("aux"):
            detected_registers.add(RegisterLevel.BUSHI)
        if surface == "ん" and pos_detail_1.startswith("aux"):
            # Too common ("imasen"), need constraints
            pass

        # "ran" (Speculative archaic)
        if surface == "らん" and pos_detail_1.startswith("aux"):
            detected_registers.add(RegisterLevel.BUSHI)

            # Exclude if followed by 'wa' (Topic marker usually means real grass)
            if i < len(features) - 1 and features[i + 1].surface == "は":
                is_slang = False

            if is_slang:
                detected_registers.add(RegisterLevel.NETSLANG)
        # "wanchan"
        if "ワンチャン" in surface:
            detected_registers.add(RegisterLevel.NETSLANG)
        # "noshi" or "oitsukan"
        if "ノシ" in surface:
            detected_registers.add(RegisterLevel.NETSLANG)
        if (
            "追いつか" in surface
            and i < len(features) - 1
            and features[i + 1].surface == "ん"
        ):
            detected_registers.add(RegisterLevel.NETSLANG)
        if "じゃね" in surface:
            detected_registers.add(RegisterLevel.NETSLANG)
        if (
            surface == "じゃ"
            and i < len(features) - 1
            and features[i + 1].surface == "ね"
        ):
            detected_registers.add(RegisterLevel.NETSLANG)

        # Kyoshigo
        if lemma in ["なさい", "たまえ"] or surface in ["なさい", "たまえ"]:
            detected_registers.add(RegisterLevel.KYOSHIGO)
        if surface == "いけません" or (
            surface == "いけ"
            and i < len(features) - 2
            and features[i + 1].surface == "ませ"
        ):
            detected_registers.add(RegisterLevel.KYOSHIGO)
        # Set phrases: "Yoku dekimashita"
        # Lemma might be missing for 'yoku', use surface fallback
        if (lemma == "よく" or surface == "よく") and i < len(features) - 1:
            next_lemma = features[i + 1].lemma
            if next_lemma == "できる":
                detected_registers.add(RegisterLevel.KYOSHIGO)
        # "Desu kara ne" (explanatory/instructional tone with formal copula)
        # Only trigger for "ですからね" (formal), not "だからね" (casual)
        if surface == "ね" and i > 1:
            prev1 = features[i - 1]  # kara
            prev2 = features[i - 2]  # desu (formal copula only)
            if prev1.surface == "から" and prev2.surface == "です":
                detected_registers.add(RegisterLevel.KYOSHIGO)
        # Vocabulary keywords for classroom context (Heuristic)
        if "宿題" in surface or "宿題" in lemma:
            # Teachers talk about homework as a topic/rule.
            # Distinguish "Shukudai WA ..." (Teacher/Instructional) from "Shukudai WO ..." (Student/Reportive)
            is_kyoshigo = False
            # Check if this specific '宿題' token is followed by 'は'
            if i < len(features) - 1 and features[i + 1].surface == "は":
                # Topic 'wa' + formal 'desu/masu' is likely teacher setting a rule
                has_formal = False
                for f in features:
                    if f.surface in ["です", "ます"]:
                        has_formal = True
                if has_formal:
                    is_kyoshigo = True

            # Command forms always trigger
            for f in features:
                if f.surface in [
                    "なさい",
                    "ください",
                    "たまえ",
                    "なさい！",
                    "なさいよ",
                ]:
                    is_kyoshigo = True

            if is_kyoshigo:
                detected_registers.add(RegisterLevel.KYOSHIGO)
        if "先生" in surface:
            # Check for "kiite" anywhere in sentence (use lemma 'kiku' or surface 'ki')
            if any("聞" in f.surface for f in features):
                detected_registers.add(RegisterLevel.KYOSHIGO)
        if "説明" in lemma:
            has_context = False
            for f in features:
                if f.surface in ["なさい", "たまえ"] or "聞" in f.surface:
                    has_context = True
            # Instructional "ima kara" or "kara" combined with intent
            if any(f.surface == "今" for f in features) and any(
                f.surface == "から" for f in features
            ):
                if any(f.surface in ["ます", "ましょう"] for f in features):
                    has_context = True
            # Understanding checks ("Wakarimashita ka?") often follow explanations in teaching
            if any(v in f.surface for f in features for v in ["分か", "わか"]):
                if any(f.surface in ["まし", "ます"] for f in features):
                    if any(f.surface in ["か", "か？", "？", "?"] for f in features):
                        has_context = True

            if has_context:
                detected_registers.add(RegisterLevel.KYOSHIGO)
        # "Machigai" (mistake/correction) - but exclude "machigainaku" and casual usage
        if "間違い" in lemma or "間違っ" in surface:
            # Exclude "間違いなく" (undoubtedly) which is not a correction context
            if i < len(features) - 1 and features[i + 1].surface == "なく":
                pass  # Skip "間違いなく" - this is an adverb, not correction
            # Exclude "間違った + noun" (e.g., "間違ったバス" = wrong bus) - casual attribution
            elif "間違っ" in surface:
                # Check if followed by noun (directly or after 'た' auxiliary)
                next_idx = i + 1
                if next_idx < len(features) and features[next_idx].surface == "た":
                    next_idx = i + 2  # Skip past the 'た' auxiliary
                if next_idx < len(features):
                    next_pos = features[next_idx].pos
                    # Check if next is a noun (名詞 in Japanese, or 'noun'/'noun' for different parsers)
                    if next_pos in ["名詞", "noun"] or next_pos.startswith("noun"):
                        pass  # Skip casual attributive usage like "wrong bus"
                    else:
                        detected_registers.add(RegisterLevel.KYOSHIGO)
                else:
                    detected_registers.add(RegisterLevel.KYOSHIGO)
            else:
                # Trigger for actual correction contexts
                detected_registers.add(RegisterLevel.KYOSHIGO)
        if "テスト" in surface:
            detected_registers.add(RegisterLevel.KYOSHIGO)
        if "線を引い" in surface or (
            "線" in lemma and "引い" in kotogram_str(features)
        ):
            detected_registers.add(RegisterLevel.KYOSHIGO)
        if "質問" in surface:  # Broaden heuristic for "Any questions?"
            if "ますか" in kotogram_str(features) or "ある" in kotogram_str(features):
                detected_registers.add(RegisterLevel.KYOSHIGO)
        if "時間" in surface and "なり" in kotogram_str(features):
            detected_registers.add(RegisterLevel.KYOSHIGO)

        # Sonkeigo
        if lemma in [
            "いらっしゃる",
            "おっしゃる",
            "なさる",
            "召し上がる",
            "ご覧になる",
            "お掛け",
            "お休み",
            "ご不在",
            "ご指導",
            "ご自由",
            "お戻り",
            "ご覧",
        ]:
            detected_registers.add(RegisterLevel.SONKEIGO)
        # o-mie
        if (
            surface == "見え"
            and i > 0
            and (
                features[i - 1].surface == "お"
                or features[i - 1].pos_detail_1 == "pref"
            )
        ):
            detected_registers.add(RegisterLevel.SONKEIGO)
        # o-Adj pattern (O-isogashii) and O-Verb-kudasai (Okake kudasai)
        if (surface == "お" or surface == "ご" or pos_detail_1 == "pref") and i < len(
            features
        ) - 1:
            next_pos = features[i + 1].pos
            if (
                next_pos.startswith("adj")
                or features[i + 1].pos_detail_1 == "adjective"
            ):
                detected_registers.add(RegisterLevel.SONKEIGO)
            # Catch o-kake (noun/verb)
            if features[i + 1].surface == "掛け" or features[i + 1].lemma == "掛ける":
                detected_registers.add(RegisterLevel.SONKEIGO)
            if features[i + 1].lemma in ("指導", "自由", "不在"):
                detected_registers.add(RegisterLevel.SONKEIGO)
        # o-V-ni-naru pattern: o + V(conj) + ni + naru
        # Check for 'ni' and 'naru' (lemma)
        if surface == "に" and 1 < i < len(features) - 1:
            prev1 = features[i - 1]
            prev2 = features[i - 2]
            next1 = features[i + 1]
            # 'o'/'go' might be pos='pref' or just surface
            is_prefix = (
                prev2.pos == "pref"
                or prev2.pos_detail_1 == "pref"
                or prev2.surface in ["お", "ご"]
            )
            if is_prefix and next1.lemma == "なる":
                detected_registers.add(RegisterLevel.SONKEIGO)
        # Passive/Respectful 'reru'/'rareru'
        if (pos_detail_1.startswith("aux") or pos.startswith("aux")) and lemma in [
            "れる",
            "られる",
        ]:
            # Heuristic: If attached to a verb, treat as Sonkeigo for this dataset
            if i > 0 and features[i - 1].pos.startswith("verb"):
                detected_registers.add(RegisterLevel.SONKEIGO)

        # JOSEIGO (Feminine Register - 女性語)
        # Sentence-final わ (feminine marker)
        # Must be at sentence end to distinguish from other 'wa' uses
        if (
            surface == "わ"
            and pos == "particle"
            and pos_detail_1 == "sentence-final-particle"
        ):
            # Exclude if already marked as OJOUSAMA (which uses わ after desu/masu)
            if not (i > 0 and features[i - 1].surface in ["です", "ます"]):
                detected_registers.add(RegisterLevel.JOSEIGO)

        # Sentence-final の (feminine question marker)
        if (
            surface == "の"
            and pos == "particle"
            and pos_detail_1 == "sentence-final-particle"
        ):
            # Exclude OJOUSAMA patterns (after masu/desu)
            if not (i > 0 and features[i - 1].surface in ["です", "ます"]):
                detected_registers.add(RegisterLevel.JOSEIGO)

        # かしら (feminine wondering marker)
        if surface in ["かしら", "カシラ"]:
            detected_registers.add(RegisterLevel.JOSEIGO)

        # Softer speech markers
        if surface in ["困っちゃう", "困っちゃ"]:
            detected_registers.add(RegisterLevel.JOSEIGO)

        # DANSEIGO (Masculine Register - 男性語)
        # 俺 (ore) - strong masculine pronoun (already caught in gender analysis, but register too)
        if pos == "pron" and (
            surface in ["俺", "おれ", "オレ"] or lemma in ["俺", "おれ", "オレ"]
        ):
            detected_registers.add(RegisterLevel.DANSEIGO)

        # 僕 (boku) - masculine pronoun (softer than ore, but still masculine)
        if pos == "pron" and (
            surface in ["僕", "ぼく", "ボク"]
            or lemma in ["僕", "ぼく", "ボク", "僕-代名詞"]
        ):
            detected_registers.add(RegisterLevel.DANSEIGO)

        # Sentence-final だぞ / ぞ
        if surface in ["ぞ", "だぞ"]:
            detected_registers.add(RegisterLevel.DANSEIGO)

        # Sentence-final ぜ
        if surface in ["ぜ", "だぜ"]:
            detected_registers.add(RegisterLevel.DANSEIGO)

        # Blunt imperatives with masculine pronouns or markers
        if pos_detail_1 and "imperative" in pos_detail_1:
            # Check if accompanied by masculine markers
            if any(f.surface in ["俺", "お前", "僕", "ぼく"] for f in features):
                detected_registers.add(RegisterLevel.DANSEIGO)

        # BURIKKO (Exaggerated Cuteness - ぶりっ子言葉)
        # え〜 (prolonged え)
        if surface in ["え〜", "えー", "え～"]:
            detected_registers.add(RegisterLevel.BURIKKO)

        # やだ〜 (prolonged negative)
        if surface in ["やだ〜", "やだー", "やだ～", "いやだ〜"]:
            detected_registers.add(RegisterLevel.BURIKKO)

        # わかんな〜い (わからない in cutesy form)
        if "わかんな" in surface and (
            "い" in surface
            or (i < len(features) - 1 and features[i + 1].surface in ["い", "〜い"])
        ):
            detected_registers.add(RegisterLevel.BURIKKO)

        # Diminutive/cutesy verb forms
        if surface in ["ちゃった", "ちゃう"] and i > 0:
            # Could be dialect too (Kansaiben), but in specific contexts it's burikko
            # Only mark as burikko if not already marked as Kansaiben
            pass  # Will be caught by context later

        # 〇〇くん pattern followed by ってば / 〇〇さん followed by ってば
        if surface in ["ってば", "ってばー"]:
            detected_registers.add(RegisterLevel.BURIKKO)

        # Kenjogo
        if lemma in [
            "申す",
            "存じる",
            "参る",
            "伺う",
            "拝見する",
            "拝見",
            "頂く",
            "いたす",
            "差し上げる",
            "申し上げる",
            "お目にかかる",
            "恐れ入る",
            "承る",
            "存じ上げる",
        ]:
            detected_registers.add(RegisterLevel.KENJOGO)
        # Split verb check (e.g. zonji + ageru)
        if surface == "上げ" and i > 0 and features[i - 1].surface.startswith("存じ"):
            detected_registers.add(RegisterLevel.KENJOGO)
        # Check surface for nouns that might lose lemma (e.g. Haiken)
        if surface in ["拝見", "差し上げる", "申し上げる", "恐れ入り", "承り"]:
            detected_registers.add(RegisterLevel.KENJOGO)
        if surface in ["おります", "いたします"]:
            detected_registers.add(RegisterLevel.KENJOGO)
        # sasete-itadaku, choudai
        if "頂戴" in surface:
            detected_registers.add(RegisterLevel.KENJOGO)
        if lemma == "いただく" and i > 0 and features[i - 1].surface == "て":
            if i > 1 and features[i - 2].pos_detail_1.startswith("aux"):  # sase-te
                detected_registers.add(RegisterLevel.KENJOGO)
            # Relaxed: just 'te-itadaku' often humble
            detected_registers.add(RegisterLevel.KENJOGO)

        # o-me-ni-kakaru parts
        if (
            lemma == "目"
            and i > 0
            and (
                features[i - 1].lemma == "お"
                or features[i - 1].surface == "お"
                or features[i - 1].pos_detail_1 == "pref"
            )
        ):
            # Just presence of "o-me" often suggests kenjogo/sonkeigo in this context
            detected_registers.add(RegisterLevel.KENJOGO)

    if not detected_registers:
        return {RegisterLevel.NEUTRAL}

    return detected_registers


def infer_gender_from_register(
    gender_enum: Any, register_enums: List[Any]
) -> Tuple[float, int]:
    """Infer gender value and pragmatic flag from gender enum and registers.

    Refined logic:
    1. If gender is explicitly MASCULINE/FEMININE, use that.
    2. If gender is NEUTRAL, infer from registers:
       - Masculine registers: DANSEIGO, GUNTAI, BUSHI (Excluded KYOSHIGO)
       - Feminine registers: JOSEIGO, OJOUSAMA, BURIKKO
    3. If registers have both masculine and feminine markers, return UNPRAGMATIC (0.0, 0).
    4. Otherwise return NEUTRAL (0.0, 1) or the inferred gender.
    """

    val: float = 0.0
    prag: int = 0

    if gender_enum == GenderLevel.MASCULINE:
        val, prag = -1.0, 1
    elif gender_enum == GenderLevel.FEMININE:
        val, prag = 1.0, 1
    elif gender_enum == GenderLevel.NEUTRAL:
        # Infer gender from register if neutral
        masculine_registers = {
            RegisterLevel.DANSEIGO,
            RegisterLevel.GUNTAI,
            RegisterLevel.BUSHI,
        }
        feminine_registers = {
            RegisterLevel.JOSEIGO,
            RegisterLevel.OJOUSAMA,
            RegisterLevel.BURIKKO,
        }

        is_masc = any(r in masculine_registers for r in register_enums)
        is_fem = any(r in feminine_registers for r in register_enums)

        if is_masc and is_fem:
            # Conflicting registers -> Unpragmatic
            val, prag = 0.0, 0
        elif is_masc:
            val, prag = -1.0, 1
        elif is_fem:
            val, prag = 1.0, 1
        else:
            val, prag = 0.0, 1
    else:
        val, prag = 0.0, 0

    return val, prag


def load_register_overrides() -> Dict[str, List[Any]]:
    """Load manual register overrides from data/jpn_sentences_<register>.tsv."""
    # Map register string to RegisterLevel
    reg_map = {r.value: r for r in RegisterLevel}

    overrides: Dict[str, Any] = {}

    # Pattern to match individual register files
    pattern = "data/jpn_sentences_*.tsv"
    for file_path in glob.glob(pattern):
        basename = os.path.basename(file_path)

        reg_str = basename.replace("jpn_sentences_", "").replace(".tsv", "")
        if reg_str not in reg_map:
            continue

        reg_level = reg_map[reg_str]

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sentence = parse_tsv(line)
                if sentence not in overrides:
                    overrides[sentence] = set()
                overrides[sentence].add(reg_level)

    # Convert sets to sorted lists
    return {k: sorted(list(v), key=str) for k, v in overrides.items()}


def formality_to_weight(formality: FormalityLevel) -> Tuple[float, int]:
    """Convert formality level to weight and pragmatic flag.

    Mapping:
        Very Casual: -1.0
        Casual: -0.5
        Neutral: 0.0
        Formal: 0.5
        Very Formal: 1.0
        Unpragmatic: 0.0 (weight), 0 (pragmatic flag)
    """
    if formality == FormalityLevel.VERY_CASUAL:
        return -1.0, 1
    if formality == FormalityLevel.CASUAL:
        return -0.5, 1
    if formality == FormalityLevel.NEUTRAL:
        return 0.0, 1
    if formality == FormalityLevel.FORMAL:
        return 0.5, 1
    if formality == FormalityLevel.VERY_FORMAL:
        return 1.0, 1

    # Unpragmatic
    return 0.0, 0


def parse_gp_ids(gp_str: str) -> List[int]:
    """Parse grammar point ID string like 'gp0597,gp0123' to list of integers.

    Args:
        gp_str: Comma-separated string of grammar point IDs (e.g., 'gp0597,gp0123')

    Returns:
        List of integer grammar point IDs (e.g., [597, 123])
    """
    if not gp_str:
        return []
    result = []
    for gp in gp_str.split(","):
        gp = gp.strip()
        if gp.startswith("gp") and len(gp) > 2:
            gp_num = gp[2:]
            if gp_num.isdigit():
                result.append(int(gp_num))
    return result
