"""Kotogram format utilities for parsing and reconstructing Japanese text.

This module provides core utilities for working with kotogram compact format,
a specialized encoding for Japanese text that preserves linguistic annotations
alongside the original text.

Kotogram Format Structure:
    The kotogram format uses Unicode markers to encode linguistic information:
    - ⌈⌉ : Token boundaries
    - ˢ : Surface form (the actual text)
    - ᵖ : Part of speech and grammatical features
    - ᵇ : Base orthography (dictionary form spelling)
    - ᵈ : Lemma (dictionary form)
    - ʳ : Reading/pronunciation

    Example:
        "猫を食べる" (The cat eats) becomes:
        "⌈ˢ猫ᵖnoun⌉⌈ˢをᵖparticle:case-particle⌉⌈ˢ食べるᵖverb:lower-ichidan-ba⌉"

Functions:
    kotogram_to_japanese: Convert kotogram format back to plain Japanese text
    split_kotogram: Split a kotogram sentence into individual tokens
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class TokenFeatures:
    """Linguistic features extracted from a kotogram token."""
    surface: str = ''
    pos: str = ''
    pos_detail1: str = ''
    pos_detail2: str = ''
    pos_detail3: str = ''
    conjugated_type: str = ''
    conjugated_form: str = ''
    base_orth: str = ''
    lemma: str = ''
    reading: str = ''


def kotogram_to_japanese(
    kotogram: str,
    spaces: bool = False,
    collapse_punctuation: bool = True,
    furigana: bool = False
) -> str:
    """Convert kotogram compact representation back to Japanese text.

    This function extracts the surface forms (ˢ markers) from a kotogram string
    and reconstructs the original Japanese text. It can optionally preserve
    token boundaries with spaces, handle punctuation spacing intelligently, and
    include furigana readings in parentheses.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information. Must follow the standard kotogram format
                 with ⌈⌉ token boundaries and ˢ surface markers.
        spaces: If True, insert spaces between tokens to preserve word boundaries.
               Useful for debugging or analysis. Default is False for natural
               Japanese text without spaces.
        collapse_punctuation: If True (default), remove spaces around punctuation
                            marks to ensure natural Japanese formatting. Only
                            applies when spaces=True. Handles common Japanese
                            punctuation including 。、・etc.
        furigana: If True, append IME-style readings in hiragana brackets after
                 each token when available and different from the surface form. Shows
                 what you would type in a Japanese IME to input the text. For example,
                 "漢字[かんじ]" for kanji. Default is False. Redundant readings (same
                 as surface) are omitted.

    Returns:
        Japanese text string reconstructed from the kotogram representation.
        Preserves the original character sequence and can optionally show
        token boundaries with spaces and/or furigana readings.

    Examples:
        >>> kotogram = "⌈ˢ猫ᵖnoun⌉⌈ˢをᵖparticle:case-particle⌉⌈ˢ食べるᵖverb⌉"
        >>> kotogram_to_japanese(kotogram)
        '猫を食べる'

        >>> kotogram_to_japanese(kotogram, spaces=True)
        '猫 を 食べる'

        >>> kotogram = "⌈ˢこんにちはᵖint⌉⌈ˢ。ᵖauxs⌉"
        >>> kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        'こんにちは。'

        >>> kotogram = "⌈ˢ漢字ᵖnounʳカンジ⌉⌈ˢですᵖaux-verb⌉"
        >>> kotogram_to_japanese(kotogram, furigana=True)
        '漢字[かんじ]です'

        >>> # Redundant readings are omitted (hiragana surface = hiragana reading)
        >>> kotogram = "⌈ˢひらがなᵖnounʳヒラガナ⌉"
        >>> kotogram_to_japanese(kotogram, furigana=True)
        'ひらがな'

    Note:
        Without furigana=True, this function is lossy - it only preserves the
        surface forms and discards all linguistic annotations (POS tags, readings,
        etc.). To preserve full information, keep the original kotogram string.
    """
    from kotogram.validation import ensure_string
    ensure_string(kotogram, "kotogram")

    from .japanese_parser import POS_TO_CHARS

    if not furigana:
        # Original implementation - extract surface forms only
        pattern = r'ˢ(.*?)ᵖ'
        matches = re.findall(pattern, kotogram, re.DOTALL)

        if spaces:
            # Join tokens with spaces
            result = ' '.join(matches).replace('{ ', '{').replace(' }', '}')

            if collapse_punctuation:
                # Remove spaces around Japanese punctuation for natural formatting
                for punc in POS_TO_CHARS['aux-symbol']:
                    # Skip braces as they're handled above
                    if punc == '{' or punc == '}':
                        continue
                    # Remove space before and after punctuation
                    result = result.replace(f' {punc}', punc)
                    result = result.replace(f'{punc} ', punc)

            return result
        else:
            # Concatenate all surface forms without spaces (natural Japanese)
            return ''.join(matches)
    else:
        # Furigana mode - extract surface forms and IME readings (hiragana)
        tokens = split_kotogram(kotogram)
        result_parts = []

        def to_hiragana(text: str) -> str:
            """Convert katakana to hiragana for IME-style furigana."""
            result = []
            for char in text:
                code = ord(char)
                # Katakana range: 0x30A1-0x30F6
                if 0x30A1 <= code <= 0x30F6:
                    # Convert to hiragana by subtracting offset
                    result.append(chr(code - 0x60))
                # Keep katakana length marker as hiragana equivalent
                elif char == 'ー':
                    result.append('ー')
                else:
                    result.append(char)
            return ''.join(result)

        def is_kana_only(text: str) -> bool:
            """Check if text contains only hiragana and katakana characters."""
            for char in text:
                code = ord(char)
                # Check if it's hiragana (0x3041-0x309F) or katakana (0x30A0-0x30FF)
                is_hiragana = 0x3041 <= code <= 0x309F
                is_katakana = 0x30A0 <= code <= 0x30FF

                if not (is_hiragana or is_katakana):
                    return False
            return True

        for token in tokens:
            # Extract surface form
            surface_match = re.search(r'ˢ(.*?)ᵖ', token, re.DOTALL)
            if not surface_match:
                continue
            surface = surface_match.group(1)

            # For IME-style furigana, we only add readings for kanji or mixed text
            # Pure kana (hiragana/katakana) already shows the IME input
            if is_kana_only(surface):
                # Surface is already in kana - no furigana needed
                result_parts.append(surface)
            else:
                # Surface contains kanji - extract reading for IME input
                reading_match = re.search(r'ʳ(.*?)(?:⌉|ᵇ|ᵈ)', token)
                reading_katakana = reading_match.group(1) if reading_match else None

                if reading_katakana:
                    # Convert pronunciation to hiragana for IME-style furigana
                    reading_hiragana = to_hiragana(reading_katakana)
                    result_parts.append(f"{surface}[{reading_hiragana}]")
                else:
                    # No reading available
                    result_parts.append(surface)

        if spaces:
            result = ' '.join(result_parts).replace('{ ', '{').replace(' }', '}')

            if collapse_punctuation:
                # Remove spaces around Japanese punctuation for natural formatting
                for punc in POS_TO_CHARS['aux-symbol']:
                    if punc == '{' or punc == '}':
                        continue
                    result = result.replace(f' {punc}', punc)
                    result = result.replace(f'{punc} ', punc)

            return result
        else:
            return ''.join(result_parts)


def split_kotogram(kotogram: str) -> List[str]:
    """Split a kotogram sentence into individual token representations.

    This function segments a complete kotogram string into a list of individual
    token kotograms, each representing one morphological unit. Each token
    retains its full linguistic annotation.

    Args:
        kotogram: Kotogram compact sentence representation. Should be a valid
                 kotogram string with properly matched ⌈⌉ token boundaries.

    Returns:
        List of individual token kotogram strings, each containing one complete
        token with its full annotation enclosed in ⌈⌉ boundaries. Returns empty
        list if no tokens are found.

    Examples:
        >>> kotogram = "⌈ˢ猫ᵖnoun⌉⌈ˢをᵖparticle:case-particle⌉⌈ˢ食べるᵖverb⌉"
        >>> split_kotogram(kotogram)
        ['⌈ˢ猫ᵖnoun⌉', '⌈ˢをᵖparticle:case-particle⌉', '⌈ˢ食べるᵖverb⌉']

        >>> kotogram = "⌈ˢこんにちはᵖintᵈこんにち‐はʳコンニチワ⌉⌈ˢ。ᵖauxs⌉"
        >>> tokens = split_kotogram(kotogram)
        >>> len(tokens)
        2
        >>> tokens[0]
        '⌈ˢこんにちはᵖintᵈこんにち‐はʳコンニチワ⌉'

    Note:
        This function assumes well-formed kotogram input with balanced ⌈⌉ markers.
        Malformed input may produce unexpected results. Each returned token is
        a complete, standalone kotogram representation that can be further analyzed.

    See Also:
        kotogram_to_japanese: Extract surface forms from tokens
    """
    from kotogram.validation import ensure_string
    ensure_string(kotogram, "kotogram")

    # Find all complete token annotations enclosed in ⌈⌉
    # Pattern matches: ⌈ followed by any chars (non-greedy) until ⌉
    return re.findall(r'⌈[^⌉]*⌉', kotogram)


def extract_token_features(token: str) -> TokenFeatures:
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
        TokenFeatures object with extracted features:
        - surface: The surface form of the token (actual text)
        - pos: Part of speech main category (e.g., 'v', 'n', 'auxv', 'prt')
        - pos_detail1: First POS detail level (e.g., 'general', 'common_noun')
        - pos_detail2: Second POS detail level (e.g., 'general')
        - pos_detail3: Third POS detail level (e.g., 'general')
        - conjugated_type: Conjugation type (e.g., 'lower-ichidan-ba', 'auxv-masu')
        - conjugated_form: Conjugation form (e.g., 'conjunctive', 'terminal')
        - base_orth: Base orthography (dictionary form spelling)
        - lemma: Lemma/dictionary form
        - reading: Reading/pronunciation

    Examples:
        >>> # Extract features from a verb token
        >>> token = "⌈ˢ食べᵖverb:general:lower-ichidan-ba:continuativeᵇ食べるᵈ食べるʳタベ⌉"
        >>> features = extract_token_features(token)
        >>> features.pos
        'v'
        >>> features.conjugated_type
        'lower-ichidan-ba'
        >>> features.conjugated_form
        'conjunctive'

        >>> # Extract features from an auxiliary verb (note: empty fields omitted)
        >>> token = "⌈ˢますᵖaux-verb:aux-masu:terminalᵇますʳマス⌉"
        >>> features = extract_token_features(token)
        >>> features.pos
        'auxv'
        >>> features.conjugated_type
        'auxv-masu'
        >>> features.conjugated_form
        'terminal'
        >>> features.pos_detail1  # Empty because parser omitted it
        ''

    Note:
        All returned attributes are strings. Fields that are not present
        in the token will have empty string values ('').
    """
    from kotogram.validation import ensure_string
    ensure_string(token, "token")

    from .japanese_parser import (
        POS1_MAP, POS2_MAP, POS3_MAP,
        CONJUGATED_TYPE_MAP, CONJUGATED_FORM_MAP
    )

    feature = TokenFeatures()

    # Extract surface form (ˢ...ᵖ)
    surface_match = re.search(r'ˢ(.*?)ᵖ', token, re.DOTALL)
    if surface_match:
        feature.surface = surface_match.group(1)

    # Extract POS data (ᵖ...ᵇ|ᵈ|ʳ|⌉)
    pos_match = re.search(r'ᵖ([^⌉ᵇᵈʳ]+)', token)
    if pos_match:
        pos_data = pos_match.group(1)
        parts = pos_data.split(':')

        # Main POS code (always first)
        feature.pos = parts[0] if len(parts) > 0 else ''

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
                feature.conjugated_form = value
            elif value in CONJUGATED_TYPE_MAP.values():
                feature.conjugated_type = value
            elif value in POS2_MAP.values():
                # pos_detail_2 comes after pos_detail_1, so check if we already have pos_detail_1
                if feature.pos_detail1:
                    feature.pos_detail2 = value
                else:
                    feature.pos_detail1 = value
            elif value in POS3_MAP.values():
                 # pos_detail_3 usually comes last for details
                 feature.pos_detail3 = value
            elif value in POS1_MAP.values():
                # pos_detail_1 comes before pos_detail_2
                if not feature.pos_detail1:
                    feature.pos_detail1 = value
                else:
                    feature.pos_detail2 = value
            else:
                # Unknown value - try to assign by position as fallback
                if not feature.pos_detail1:
                    feature.pos_detail1 = value
                elif not feature.pos_detail2:
                    feature.pos_detail2 = value
                elif not feature.pos_detail3:
                    feature.pos_detail3 = value
                elif not feature.conjugated_type:
                    feature.conjugated_type = value
                elif not feature.conjugated_form:
                    feature.conjugated_form = value

    # Extract base orthography (ᵇ...ᵈ|ʳ|⌉)
    base_match = re.search(r'ᵇ([^⌉ᵈʳ]+)', token)
    if base_match:
        feature.base_orth = base_match.group(1)

    # Extract lemma/dictionary form (ᵈ...ʳ|⌉)
    lemma_match = re.search(r'ᵈ([^⌉ʳ]+)', token)
    if lemma_match:
        feature.lemma = lemma_match.group(1)

    # Extract reading (ʳ...⌉)
    reading_match = re.search(r'ʳ([^⌉]+)', token)
    if reading_match:
        feature.reading = reading_match.group(1)

    return feature
