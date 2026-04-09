"""Masking strategies for training data sanitization.

This module provides utilities to mask specific tokens in a kotogram stream,
primarily for anonymization or data augmentation purposes.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from kotogram.kotogram import Token, TokenFeatures


# -------------------------------------------------------------------------
# KC Masking Constants
# -------------------------------------------------------------------------

# Whitelist for grammar-heavy POS to keep reading.
# Content-heavy POS (noun, adj, etc.) will have their reading masked.
GRAMMAR_POS_WHITELIST = {
    "particle",
    "aux-verb",
    "aux-symbol",
    "verb",
    "prefix",
    "suffix",
    "adnom",
    "conj",
    "pron",
    "adv",
    "interj",
    "adj",
    "adjectival-noun",
}

# Compound whitelist for more specific categories.
# Format: "pos:pos_detail_1:pos_detail_2" (use "*" for wildcard)
# These compound strings get reading preserved even if base POS is masked.
GRAMMAR_COMPOUND_WHITELIST = {
    "noun:common-noun:adverbial",  # Adverbial nouns like 限り, 今日, 今月
}

# Reading masks that should be preserved even if POS is masked.
# These represent specific semantic categories (e.g. names, numbers)
# that we want to distinguish in the reading grammar.
PRESERVED_READING_MASKS = {
    "<proper-noun>",
    "<person-name>",
    "<given-name>",
    "<surname>",
    "<place-name>",
    "<country>",
    "<number>",
}

READING_MASK = "<READING_MASK>"

# Fixed exemplar surfaces for each mask category. Used as:
# - Dedup key: canonicalize_sentence() replaces masked tokens with these
# - Vocab entry: apply_training_mask() uses these instead of raw surfaces
# - chiVe init: each exemplar has a real pretrained embedding
# All exemplars verified present in chiVe v1.3-mc5.
SURFACE_EXEMPLARS: Dict[str, str] = {
    "<number>": "1",
    "<surname>": "田中",
    "<given-name>": "太郎",
    "<person-name>": "田中",
    "<place-name>": "東京",
    "<country>": "日本",
    "<proper-noun>": "東京",
}


def katakana_to_hiragana(text: str) -> str:
    """Convert katakana characters to hiragana.

    Katakana range: U+30A1 to U+30F6
    Hiragana range: U+3041 to U+3096
    Offset: 0x60 (96)
    """
    result = []
    for char in text:
        code = ord(char)
        # Check if character is in katakana range
        if 0x30A1 <= code <= 0x30F6:
            # Convert to hiragana by subtracting offset
            result.append(chr(code - 0x60))
        else:
            result.append(char)
    return "".join(result)


def apply_training_mask(tokens: List["Token"]) -> List["Token"]:
    """Apply training mask: replace masked-category surfaces with exemplars.

    For tokens matching a mask category (numbers, proper nouns, names, places),
    the surface is replaced with the fixed exemplar from SURFACE_EXEMPLARS and
    the reading_gram is set to the mask tag.

    Also removes trailing "。" (Japanese period) from the end of the sentence.

    Args:
        tokens: List of kotogram.Token objects to process.

    Returns:
        New list of Token objects with masking applied.
    """
    from kotogram.kotogram import Token  # Deferred import to avoid cycle

    if tokens and tokens[-1].surface == "。":
        tokens = tokens[:-1]

    masked_tokens = []
    for token in tokens:
        new_token = token
        features = token.features

        mask_tag = get_surface_mask_for_features(features)
        if mask_tag:
            from kotogram.kotogram import TokenFeatures

            exemplar = SURFACE_EXEMPLARS[mask_tag]
            new_features = TokenFeatures(
                surface=exemplar,
                pos=features.pos,
                pos_detail_1=features.pos_detail_1,
                pos_detail_2=features.pos_detail_2,
                pos_detail_3=features.pos_detail_3,
                conjugated_type=features.conjugated_type,
                conjugated_form=features.conjugated_form,
                base_orth=features.base_orth,
                lemma="",
                reading="",
                reading_gram=mask_tag,
            )
            new_token = Token(exemplar, features=new_features)

        masked_tokens.append(new_token)

    return masked_tokens


def canonicalize_sentence(sentence: str, *, _parser: Optional[object] = None) -> str:
    """Replace masked-category tokens with exemplar surfaces in raw text.

    Used as a dedup key: two sentences that differ only by numbers, names, or
    places will produce the same canonical string.  The result is NOT stored;
    it exists only for comparison.

    Idempotent: canonicalizing an already-canonical sentence is a no-op.

    Pass ``_parser`` (a ``SudachiJapaneseParser``) to avoid re-creating
    the tokenizer on each call in hot loops.
    """
    if _parser is None:
        import importlib

        _sjp = importlib.import_module("kotogram.sudachi_japanese_parser")
        _parser = _sjp.SudachiJapaneseParser(validate=False)

    sudachi_tokens = _parser.tokenizer.tokenize(sentence)  # type: ignore[union-attr]
    kotogram_tokens = _parser._to_kotogram_tokens(sudachi_tokens)  # type: ignore[union-attr]  # pylint: disable=protected-access
    parts: List[str] = []
    for token in kotogram_tokens:
        mask = get_surface_mask_for_features(token.features)
        parts.append(SURFACE_EXEMPLARS[mask] if mask else token.surface)
    return "".join(parts)


# -------------------------------------------------------------------------
# Content character classification (shared by label.py and cc_common.py)
# -------------------------------------------------------------------------


def is_content_char(cp: int) -> bool:
    """Return True if the codepoint belongs to a content character range.

    Content ranges:
      Hiragana, Katakana, CJK Ideographs (+Extension A/B, Compat),
      ASCII/fullwidth digits, ASCII/fullwidth Latin, standard Japanese
      punctuation, 々.
    """
    return (
        0x3040 <= cp <= 0x309F  # Hiragana
        or 0x30A0 <= cp <= 0x30FF  # Katakana
        or 0x31F0 <= cp <= 0x31FF  # Katakana Phonetic Extensions
        or 0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x0030 <= cp <= 0x0039  # Digits 0-9
        or cp == 0x002E  # Full stop (period)
        or 0x0041 <= cp <= 0x005A  # Latin A-Z
        or 0x0061 <= cp <= 0x007A  # Latin a-z
        or 0xFF10 <= cp <= 0xFF19  # Fullwidth Digits
        or 0xFF21 <= cp <= 0xFF3A  # Fullwidth Latin A-Z
        or 0xFF41 <= cp <= 0xFF5A  # Fullwidth Latin a-z
        or cp == 0x3005  # 々 (Ideographic Iteration Mark)
        or cp == 0x3001  # 、 Ideographic Comma
        or cp == 0x3002  # 。 Ideographic Full Stop
        or cp == 0xFF01  # ！ Fullwidth Exclamation Mark
        or cp == 0xFF1F  # ？ Fullwidth Question Mark
        or cp == 0x300C  # 「 Left Corner Bracket
        or cp == 0x300D  # 」 Right Corner Bracket
        or cp == 0x300E  # 『 Left White Corner Bracket
        or cp == 0x300F  # 』 Right White Corner Bracket
        or cp == 0x3010  # 【 Left Black Lenticular Bracket
        or cp == 0x3011  # 】 Right Black Lenticular Bracket
        or cp == 0xFF08  # （ Fullwidth Left Parenthesis
        or cp == 0xFF09  # （ Fullwidth Right Parenthesis
        or cp == 0x30FB  # ・ Katakana Middle Dot
        or cp == 0x30FC  # ー Katakana-Hiragana Prolonged Sound Mark
        or cp == 0x2026  # … Horizontal Ellipsis
        or cp == 0xFF5E  # ～ Fullwidth Tilde
        or cp == 0xFF61  # ｡ Halfwidth Ideographic Full Stop
        or cp == 0xFF62  # ｢ Halfwidth Left Corner Bracket
        or cp == 0xFF63  # ｣ Halfwidth Right Corner Bracket
        or cp == 0xFF64  # ､ Halfwidth Ideographic Comma
        or cp == 0xFF65  # ･ Halfwidth Katakana Middle Dot
        or cp == 0xFF1A  # ： Fullwidth Colon
    )


def is_content_token(surface: str) -> bool:
    """Return True if a surface token counts as content.

    A token is content if at least half its characters are content characters
    (see ``is_content_char``).  This filters out pure-symbol tokens like
    ``~~~~~~~~~~~`` or ``====`` while keeping tokens that mix content with
    occasional punctuation.

    Special case: multi-character tokens consisting entirely of periods
    (ellipsis) are non-content, even though a single ``.`` is content.
    """
    if not surface:
        return False
    if len(surface) > 1 and all(ch == "." for ch in surface):
        return False
    content = sum(1 for ch in surface if is_content_char(ord(ch)))
    return content >= len(surface) - content


_LONG_NONCONTENT_THRESHOLD = 4


def has_majority_content(surfaces: List[str]) -> bool:
    """Return True if a token surface list has at least as many content as non-content tokens
    and contains no long non-content tokens.

    Sentences that fail this check are dominated by punctuation/symbols and
    should be excluded from training.  Any single non-content token of
    4+ characters (e.g. ``------``, ``////``) also disqualifies the sentence.

    *surfaces* can be raw surface strings from either ``Token.surface`` or
    ``get_vocab_strings(...)["surface"]``.
    """
    if not surfaces:
        return False
    content = 0
    for s in surfaces:
        if is_content_token(s):
            content += 1
        elif len(s) >= _LONG_NONCONTENT_THRESHOLD:
            return False
    return content >= len(surfaces) - content


def get_surface_mask_for_features(features: "TokenFeatures") -> Optional[str]:
    """Return a collapsed surface mask for special tokens, or None."""
    pos = features.pos
    detail1 = features.pos_detail_1
    detail2 = features.pos_detail_2
    detail3 = features.pos_detail_3

    # Proper noun hierarchy
    if pos == "noun" and detail1 == "proper-noun":
        target_surface = "<proper-noun>"

        # Person names
        if detail2 == "person-name":
            target_surface = "<person-name>"
            if detail3 == "given-name":
                target_surface = "<given-name>"
            elif detail3 == "surname":
                target_surface = "<surname>"

        # Place names
        elif detail2 == "place-name":
            target_surface = "<place-name>"
            if detail3 == "country":
                target_surface = "<country>"

        # Strict assertions for claimed subtypes
        if detail3 == "given-name" and target_surface != "<given-name>":
            raise RuntimeError(
                "Token has pos_detail_3='given-name' but failed hierarchy check. "
                f"Features: {features}"
            )
        if detail3 == "surname" and target_surface != "<surname>":
            raise RuntimeError(
                "Token has pos_detail_3='surname' but failed hierarchy check. "
                f"Features: {features}"
            )
        if detail3 == "country" and target_surface != "<country>":
            raise RuntimeError(
                "Token has pos_detail_3='country' but failed hierarchy check. "
                f"Features: {features}"
            )

        return target_surface

    # Numeral masking
    if pos == "noun" and detail1 == "numeral":
        return "<number>"

    return None
