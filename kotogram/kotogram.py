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
        "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv:e-ichidan-ba⌉"

Functions:
    kotogram_to_japanese: Convert kotogram format back to plain Japanese text
    split_kotogram: Split a kotogram sentence into individual tokens
"""

import re
from typing import List


def kotogram_to_japanese(
    kotogram: str,
    spaces: bool = False,
    collapse_punctuation: bool = True
) -> str:
    """Convert kotogram compact representation back to Japanese text.

    This function extracts the surface forms (ˢ markers) from a kotogram string
    and reconstructs the original Japanese text. It can optionally preserve
    token boundaries with spaces and handle punctuation spacing intelligently.

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

    Returns:
        Japanese text string reconstructed from the kotogram representation.
        Preserves the original character sequence and can optionally show
        token boundaries with spaces.

    Examples:
        >>> kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv⌉"
        >>> kotogram_to_japanese(kotogram)
        '猫を食べる'

        >>> kotogram_to_japanese(kotogram, spaces=True)
        '猫 を 食べる'

        >>> kotogram = "⌈ˢこんにちはᵖint⌉⌈ˢ。ᵖauxs⌉"
        >>> kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        'こんにちは。'

    Note:
        This function is lossy - it only preserves the surface forms and discards
        all linguistic annotations (POS tags, readings, etc.). To preserve full
        information, keep the original kotogram string.
    """
    from .japanese_parser import POS_TO_CHARS

    # Extract all surface forms using regex pattern
    # Pattern matches: ˢ followed by any chars until ᵖ
    pattern = r'ˢ(.*?)ᵖ'
    matches = re.findall(pattern, kotogram, re.DOTALL)

    if spaces:
        # Join tokens with spaces
        result = ' '.join(matches).replace('{ ', '{').replace(' }', '}')

        if collapse_punctuation:
            # Remove spaces around Japanese punctuation for natural formatting
            for punc in POS_TO_CHARS['auxs']:
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
        >>> kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt:case_particle⌉⌈ˢ食べるᵖv⌉"
        >>> split_kotogram(kotogram)
        ['⌈ˢ猫ᵖn⌉', '⌈ˢをᵖprt:case_particle⌉', '⌈ˢ食べるᵖv⌉']

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
    # Find all complete token annotations enclosed in ⌈⌉
    # Pattern matches: ⌈ followed by any chars (non-greedy) until ⌉
    return re.findall(r'⌈[^⌉]*⌉', kotogram)
