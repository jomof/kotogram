"""Kotogram format utilities for parsing and reconstructing Japanese text.

This module provides core utilities for working with kotogram compact format,
a specialized encoding for Japanese text that preserves linguistic annotations
alongside the original text.

Kotogram Format Structure:
    The kotogram format uses Unicode markers to encode linguistic information:
    - ⌈⌉ : Token boundaries
    - ˢ : Surface form (the actual text)
    - ᵖ : Part of speech (pos)
    - ᵖ¹ : Part of speech detail 1 (pos_detail_1)
    - ᵖ² : Part of speech detail 2 (pos_detail_2)
    - ᵖ³ : Part of speech detail 3 (pos_detail_3)
    - ᵗ : Conjugated type
    - ᶜ : Conjugated form
    - ᵇ : Base orthography (dictionary form spelling)
    - ᵈ : Lemma (dictionary form)
    - ʳ : Reading/pronunciation
    - ᵍ : Reading gram (derived)

    Example:
        "食べる" (to eat) becomes:
        "⌈ˢ食べるᵖverbᵖ¹generalᵗlower-ichidan-baᶜterminalᵈ*ʳタベル⌉"

Functions:
    kotogram_to_japanese: Convert kotogram format back to plain Japanese text
    split_kotogram: Split a kotogram sentence into individual tokens
"""

import base64
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, cast

from kotogram.japanese_parser import (
    POS_MAP,
    ConjugatedFormValue,
    ConjugatedTypeValue,
    PosDetail1Value,
    PosDetail2Value,
    PosDetail3Value,
    PosValue,
)
from kotogram.masking import (
    GRAMMAR_POS_WHITELIST,
    PRESERVED_READING_MASKS,
    READING_MASK,
)

# =============================================================================
# Token Shorthands for Compression
# =============================================================================
# Common tokens can be compressed to just their surface form.
# This dict maps shorthand -> full token expansion.
# The shorthand is JUST the surface form (no ⌈⌉ markers).
TOKEN_SHORTHANDS: dict[str, str] = {
    "。": "⌈ˢ。ᵖaux-symbolᵖ¹periodᵈ*ʳ*⌉",
    "は": "⌈ˢはᵖparticleᵖ¹binding-particleᵈ*ʳハ⌉",
    "を": "⌈ˢをᵖparticleᵖ¹case-particleᵈ*ʳヲ⌉",
    "に": "⌈ˢにᵖparticleᵖ¹case-particleᵈ*ʳニ⌉",
    "の": "⌈ˢのᵖparticleᵖ¹case-particleᵈ*ʳノ⌉",
    "た": "⌈ˢたᵖaux-verbᵗaux-taᶜterminalᵈ*ʳタ⌉",
    "て": "⌈ˢてᵖparticleᵖ¹conjunctive-particleᵈ*ʳテ⌉",
    "が": "⌈ˢがᵖparticleᵖ¹case-particleᵈ*ʳガ⌉",
    "、": "⌈ˢ、ᵖaux-symbolᵖ¹commaᵈ*ʳ*⌉",
    "です": "⌈ˢですᵖaux-verbᵗaux-desuᶜterminalᵈ*ʳデス⌉",
    "し": "⌈ˢしᵖverbᵖ¹boundᵗsa-irregularᶜcontinuativeᵇするᵈするʳシ⌉",
    "と": "⌈ˢとᵖparticleᵖ¹case-particleᵈ*ʳト⌉",
    "だ": "⌈ˢだᵖaux-verbᵗaux-daᶜterminalᵈ*ʳダ⌉",
    "わ": "⌈ˢわᵖparticleᵖ¹sentence-final-particleᵈ*ʳワ⌉",
    "で": "⌈ˢでᵖparticleᵖ¹case-particleᵈ*ʳデ⌉",
    "ぜ": "⌈ˢぜᵖparticleᵖ¹sentence-final-particleᵈ*ʳゼ⌉",
    "ます": "⌈ˢますᵖaux-verbᵗaux-masuᶜterminalᵈ*ʳマス⌉",
    "いる": "⌈ˢいるᵖverbᵖ¹boundᵗupper-ichidan-aᶜterminalᵈ*ʳイル⌉",
    "も": "⌈ˢもᵖparticleᵖ¹binding-particleᵈ*ʳモ⌉",
    "ぞ": "⌈ˢぞᵖparticleᵖ¹sentence-final-particleᵈ*ʳゾ⌉",
    "こと": "⌈ˢことᵖnounᵖ¹common-nounᵖ²generalᵈ*ʳコト⌉",
    "か": "⌈ˢかᵖparticleᵖ¹sentence-final-particleᵈ*ʳカ⌉",
    "な": "⌈ˢなᵖaux-verbᵗaux-daᶜattributiveᵇだᵈだʳナ⌉",
}

# Build reverse map for compression (full token -> shorthand)
_SHORTHAND_FROM_FULL = {v: k for k, v in TOKEN_SHORTHANDS.items()}

# Pre-compiled regex patterns for performance
# Matches standard ⌈...⌉ tokens OR obfuscated [Base64] tokens
# Obfuscated tokens use brackets and strictly Base64 characters to avoid false positives with text like "foo [bar]"
_RE_KOTOGRAM_TOKEN = re.compile(r"⌈[^⌉]*⌉|\[[A-Za-z0-9+/=]+]")
_RE_SURFACE = re.compile(r"ˢ(.*?)ᵖ", re.DOTALL)

_RE_READING_FULL = re.compile(r"ʳ(.*?)[⌉ᵇᵈ]")


@dataclass
class TokenFeatures:
    """Linguistic features extracted from a kotogram token."""

    surface: str = ""
    pos: PosValue = ""
    pos_detail_1: PosDetail1Value = ""
    pos_detail_2: PosDetail2Value = ""
    pos_detail_3: PosDetail3Value = ""
    conjugated_type: ConjugatedTypeValue = ""
    conjugated_form: ConjugatedFormValue = ""
    base_orth: str = ""
    lemma: str = ""
    reading: str = ""
    reading_gram: str = ""


class Token:
    """Hashable wrapper for token features."""

    __slots__ = ("surface", "features", "_hash")

    def __init__(self, surface: str, features: TokenFeatures):
        self.surface = surface
        self.features = features
        self._hash = 0

    def __hash__(self) -> int:
        if self._hash == 0:
            if self.features:
                # Use dataclasses to converting features to tuple for hashing
                from dataclasses import astuple

                # We can't import astuple at top level due to circular deps if we aren't careful?
                # Actually astuple is standard lib.
                # But TokenFeatures fields are fixed order, so astuple is stable.
                self._hash = hash((self.surface, astuple(self.features)))
            else:
                self._hash = hash((self.surface,))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.surface == other
        if isinstance(other, Token):
            # Fast equality check could try hash first if computed, but standard way:
            if self.surface != other.surface:
                return False
            return self.features == other.features
        return False

    def __repr__(self) -> str:
        return f"Token({self.surface}, {self.features})"

    @property
    def reading(self) -> str:
        """Returns the reading in Hiragana, or surface if not available."""
        r = self.features.reading
        if not r:
            return self.surface
        # Convert Katakana to Hiragana (simple range check)
        return "".join(
            chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in str(r)
        )


def kotogram_to_japanese(
    kotogram: str,
    furigana: bool = False,
) -> str:
    # pylint: disable=too-many-locals, no-else-return
    """Convert kotogram compact representation back to Japanese text.

    This function extracts the surface forms (ˢ markers) from a kotogram string
    and reconstructs the original Japanese text. It can optionally include
    furigana readings in parentheses.

    Args:
        kotogram: Kotogram compact sentence representation containing encoded
                 linguistic information. Must follow the standard kotogram format
                 with ⌈⌉ token boundaries and ˢ surface markers.
        furigana: If True, append IME-style readings in hiragana brackets after
                 each token when available and different from the surface form. Shows
                 what you would type in a Japanese IME to input the text. For example,
                 "漢字[かんじ]" for kanji. Default is False. Redundant readings (same
                 as surface) are omitted.

    Returns:
        Japanese text string reconstructed from the kotogram representation.
        Preserves the original character sequence.

    Examples:
        >>> kotogram = "⌈ˢ猫ᵖnoun⌉⌈ˢをᵖparticle:case-particle⌉⌈ˢ食べるᵖverb⌉"
        >>> kotogram_to_japanese(kotogram)
        '猫を食べる'

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

    # Always split into tokens first to handle obfuscation correctly
    tokens = split_kotogram(kotogram)
    result_parts = []

    from kotogram.masking import katakana_to_hiragana

    def to_hiragana(text: str) -> str:
        """Convert katakana to hiragana for IME-style furigana."""
        return katakana_to_hiragana(text)

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
        # Extract features using robust API
        features: TokenFeatures = extract_token_features(token)
        surface = features.surface
        if not surface:
            continue

        if not furigana:
            # Simple surface extraction
            result_parts.append(surface)
            continue

        # Furigana mode logic
        if is_kana_only(surface):
            result_parts.append(surface)
        else:
            reading_katakana = features.reading if features.reading != surface else None

            if reading_katakana:
                reading_hiragana = to_hiragana(reading_katakana)
                result_parts.append(f"{surface}[{reading_hiragana}]")
            else:
                result_parts.append(surface)

    return "".join(result_parts)


def split_kotogram(kotogram: str) -> List[str]:
    """Split a kotogram sentence into individual token representations.

    This function segments a complete kotogram string into a list of individual
    token kotograms, each representing one morphological unit. Each token
    retains its full linguistic annotation.

    Supports shorthand compression: common tokens like "。" are stored as just
    their surface form and are returned as-is. Use extract_token_features()
    to expand shorthand tokens to their full form.

    Args:
        kotogram: Kotogram compact sentence representation. Should be a valid
                 kotogram string with properly matched ⌈⌉ token boundaries.

    Returns:
        List of individual token kotogram strings, each containing one complete
        token with its full annotation enclosed in ⌈⌉ boundaries, OR a shorthand
        token (just surface form). Returns empty list if no tokens are found.

    Examples:
        >>> kotogram = "⌈ˢ猫ᵖnoun⌉⌈ˢをᵖparticle:case-particle⌉⌈ˢ食べるᵖverb⌉"
        >>> split_kotogram(kotogram)
        ['⌈ˢ猫ᵖnoun⌉', '⌈ˢをᵖparticle:case-particle⌉', '⌈ˢ食べるᵖverb⌉']

        >>> kotogram = "⌈ˢこんにちはᵖint⌉。"  # Period is shorthand
        >>> split_kotogram(kotogram)
        ['⌈ˢこんにちはᵖint⌉', '。']

    Note:
        This function assumes well-formed kotogram input with balanced ⌈⌉ markers.
        Malformed input may produce unexpected results. Each returned token is
        a complete, standalone kotogram representation that can be further analyzed.

    See Also:
        kotogram_to_japanese: Extract surface forms from tokens
    """
    from kotogram.validation import ensure_string

    ensure_string(kotogram, "kotogram")

    tokens: List[str] = []
    i = 0
    while i < len(kotogram):
        # Check for standard ⌈...⌉ token
        if kotogram[i] == "⌈":
            end = kotogram.find("⌉", i)
            if end != -1:
                tokens.append(kotogram[i : end + 1])
                i = end + 1
                continue
        # Check for obfuscated [...] token
        if kotogram[i] == "[":
            end = kotogram.find("]", i)
            if end != -1:
                tokens.append(kotogram[i : end + 1])
                i = end + 1
                continue
        # Check for shorthand token (longest match first)
        matched = False
        for shorthand in TOKEN_SHORTHANDS:
            if kotogram[i:].startswith(shorthand):
                tokens.append(shorthand)
                i += len(shorthand)
                matched = True
                break
        if matched:
            continue
        # Skip unrecognized characters (shouldn't happen in valid input)
        i += 1

    return tokens


def obscure_kotogram_token_string(inner_content: str) -> str:
    """Obfuscate the inner content of a kotogram token if enabled.

    Args:
        inner_content: The raw string content inside ⌈⌉ (e.g. "ˢfooᵖbar")

    Returns:
        Obfuscated string (Base64) if OBSCURE_KOTOGRAM=1, else original.
    """
    from .japanese_parser import OBSCURE_KOTOGRAM

    if OBSCURE_KOTOGRAM == 1:
        # Encode to Base64
        encoded = base64.b64encode(inner_content.encode("utf-8")).decode("ascii")
        return encoded
    return inner_content


@lru_cache(maxsize=65536)
def extract_token_features(token: str) -> TokenFeatures:
    # pylint: disable=too-many-locals
    """Extract linguistic features from a single kotogram token.

    Parses a kotogram token to extract all encoded linguistic information using efficient
    string slicing instead of regex.

    Supports shorthand compression: tokens like "。" are expanded to their full form
    before parsing.

    Kotogram format uses Unicode markers:
    - ⌈⌉ : Token boundaries
    - ˢ : Surface form
    - ᵖ : POS
    - ᵇ : Base
    - ᵈ : Lemma
    - ʳ : Reading
    """

    # Expand shorthand tokens first
    token = TOKEN_SHORTHANDS.get(token, token)

    # Unobscure if needed
    # Logic is purely based on delimiters:
    # ⌈...⌉ -> Raw/Standard Kotogram (no decoding)
    # [...] -> Obfuscated (Base64 encoded)

    if token.startswith("[") and token.endswith("]"):
        # Handle bracketed obfuscated token
        inner = token[1:-1]
        # Always decode bracketed tokens as Base64
        token_content = base64.b64decode(inner).decode("utf-8")
        token = f"⌈{token_content}⌉"

    feature = TokenFeatures()

    # Find marker indices
    # Token structure: ⌈ˢ...ᵖ...ᵖ¹...ᵖ²...ᵖ³...ᵗ...ᶜ...ᵇ...ᵈ...ʳ...⌉
    # Each field has its own marker for lossless round-trip

    # ˢ Surface (always present in valid tokens)
    idx_s = token.find("ˢ")
    if idx_s == -1:
        return feature

    # Find all markers - order matters, markers are searched from idx_s
    idx_p = token.find("ᵖ", idx_s)  # POS
    idx_p1 = token.find("ᵖ¹", idx_s)  # pos_detail_1
    idx_p2 = token.find("ᵖ²", idx_s)  # pos_detail_2
    idx_p3 = token.find("ᵖ³", idx_s)  # pos_detail_3
    idx_t = token.find("ᵗ", idx_s)  # conjugated_type
    idx_c = token.find("ᶜ", idx_s)  # conjugated_form
    idx_b = token.find("ᵇ", idx_s)
    idx_d = token.find("ᵈ", idx_s)
    idx_r = token.find("ʳ", idx_s)
    idx_g = token.find("ᵍ", idx_s)
    idx_end = token.find("⌉", idx_s)

    # All possible next markers for boundary detection
    all_markers = [
        idx_p,
        idx_p1,
        idx_p2,
        idx_p3,
        idx_t,
        idx_c,
        idx_b,
        idx_d,
        idx_r,
        idx_g,
        idx_end,
    ]

    def extract_value(start_idx: int, start_offset: int) -> str:
        """Extract value from start_idx+offset to next marker."""
        if start_idx == -1:
            return ""
        start = start_idx + start_offset
        next_indices = [i for i in all_markers if i > start]
        end = min(next_indices) if next_indices else len(token)
        return token[start:end]

    # 1. Surface: ˢ to next marker
    feature.surface = extract_value(idx_s, 1)

    # Input invariant: lemma and reading must use '*' compression, not duplicate surface
    # Extract raw values to validate (before any expansion)
    def get_raw_field_value(field_marker: str, marker_len: int = 1) -> str:
        """Get raw field value from token for validation."""
        idx = token.find(field_marker, idx_s)
        if idx == -1:
            return ""
        start = idx + marker_len
        markers = [
            idx_p,
            idx_p1,
            idx_p2,
            idx_p3,
            idx_t,
            idx_c,
            idx_b,
            idx_d,
            idx_r,
            idx_g,
            idx_end,
        ]
        next_indices = [i for i in markers if i > start]
        end = min(next_indices) if next_indices else len(token)
        return token[start:end]

    raw_lemma = get_raw_field_value("ᵈ")
    raw_reading = get_raw_field_value("ʳ")

    # Check that lemma doesn't duplicate surface (should use '*' compression)
    if raw_lemma and raw_lemma == feature.surface:
        raise ValueError(
            f"Input invariant violation: lemma duplicates surface (should use '*' compression). "
            f"Token: {token!r}, surface={feature.surface!r}, lemma={raw_lemma!r}"
        )
    # Check that reading doesn't duplicate surface (should use '*' compression)
    if raw_reading and raw_reading == feature.surface:
        raise ValueError(
            f"Input invariant violation: reading duplicates surface (should use '*' compression). "
            f"Token: {token!r}, surface={feature.surface!r}, reading={raw_reading!r}"
        )

    # 2. POS: ᵖ to next marker (single char marker)
    if idx_p != -1:
        feature.pos = cast(PosValue, extract_value(idx_p, 1))

    # 3. pos_detail_1: ᵖ¹ to next marker (2-char marker)
    if idx_p1 != -1:
        feature.pos_detail_1 = cast(PosDetail1Value, extract_value(idx_p1, 2))

    # 4. pos_detail_2: ᵖ² to next marker (2-char marker)
    if idx_p2 != -1:
        feature.pos_detail_2 = cast(PosDetail2Value, extract_value(idx_p2, 2))

    # 5. pos_detail_3: ᵖ³ to next marker (2-char marker)
    if idx_p3 != -1:
        feature.pos_detail_3 = cast(PosDetail3Value, extract_value(idx_p3, 2))

    # 6. conjugated_type: ᵗ to next marker
    if idx_t != -1:
        feature.conjugated_type = cast(ConjugatedTypeValue, extract_value(idx_t, 1))

    # 7. conjugated_form: ᶜ to next marker
    if idx_c != -1:
        feature.conjugated_form = cast(ConjugatedFormValue, extract_value(idx_c, 1))

    # 4. Base: ᵇ to next marker
    if idx_b != -1:
        start = idx_b + 1
        next_indices = [i for i in [idx_d, idx_r, idx_g, idx_end] if i >= start]
        end = min(next_indices) if next_indices else len(token)
        feature.base_orth = token[start:end]

    # 8. Lemma: ᵈ to next marker
    if idx_d != -1:
        start = idx_d + 1
        next_indices = [i for i in [idx_r, idx_g, idx_end] if i >= start]
        end = min(next_indices) if next_indices else len(token)
        feature.lemma = token[start:end]
        # Expand "*" to surface (compression convention)
        if feature.lemma == "*":
            feature.lemma = feature.surface
        # Decode "<star>" escape token back to literal asterisk
        elif feature.lemma == "<star>":
            feature.lemma = "*"

    # 9. Reading: ʳ to next marker
    if idx_r != -1:
        start = idx_r + 1
        next_indices = [i for i in [idx_g, idx_end] if i >= start]
        end = min(next_indices) if next_indices else len(token)
        feature.reading = token[start:end]
        # Expand "*" to surface (compression convention)
        if feature.reading == "*":
            feature.reading = feature.surface
        # Decode "<star>" escape token back to literal asterisk
        elif feature.reading == "<star>":
            feature.reading = "*"

    # 6. Reading Gram: ᵍ to next marker
    if idx_g != -1:
        start = idx_g + 1
        next_indices = [i for i in [idx_end] if i >= start]
        end = min(next_indices) if next_indices else len(token)
        feature.reading_gram = token[start:end]

    # 7. Reading Gram (Derived from Reading if not present)
    if not feature.reading_gram:
        if feature.reading:
            # Normalize POS for whitelist check
            # POS_MAP keys are raw strings, feature.pos is a string (PosValue)
            pos_str = str(feature.pos)
            pos_norm = POS_MAP.get(pos_str, pos_str)

            if pos_norm in GRAMMAR_POS_WHITELIST:
                feature.reading_gram = feature.reading
            elif feature.reading in PRESERVED_READING_MASKS:
                feature.reading_gram = feature.reading
            else:
                feature.reading_gram = READING_MASK
        elif feature.surface:
            # No reading available but surface exists - use surface with the same logic
            pos_str = str(feature.pos)
            pos_norm = POS_MAP.get(pos_str, pos_str)

            if pos_norm in GRAMMAR_POS_WHITELIST:
                feature.reading_gram = feature.surface
            elif feature.surface in PRESERVED_READING_MASKS:
                feature.reading_gram = feature.surface
            else:
                feature.reading_gram = READING_MASK
        else:
            # No reading and no surface - use mask
            feature.reading_gram = READING_MASK

    # Final guard: reading_gram must never be empty string
    if not feature.reading_gram:
        feature.reading_gram = READING_MASK

    # Convert katakana to hiragana in reading_gram for normalization
    # Skip conversion for special mask tokens (those that start with <)
    if feature.reading_gram and not feature.reading_gram.startswith("<"):
        from kotogram.masking import katakana_to_hiragana

        feature.reading_gram = katakana_to_hiragana(feature.reading_gram)

    # Invariant: no extracted field should contain raw '*' from compression
    # '*' from compression must be expanded before returning.
    # Exception: when surface is '*', the value '*' is valid (decoded from <star>)
    for field_name in ("lemma", "reading"):
        value = getattr(feature, field_name, "")
        if value == "*" and feature.surface != "*":
            raise ValueError(
                f"Invariant violation: {field_name}='*' was not expanded. "
                f"Token: {token!r}, surface={feature.surface!r}"
            )

    return feature
