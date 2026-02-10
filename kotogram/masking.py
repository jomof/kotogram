"""Masking strategies for training data sanitization.

This module provides utilities to mask specific tokens in a kotogram stream,
primarily for anonymization or data augmentation purposes.
"""

from typing import TYPE_CHECKING, List, Optional

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
    """Apply training mask to anonymize given names immutable.

    Replaces Japanese given names (First Names) with the placeholder "太郎" (Taro).
    Returns a new list of tokens.

    Also removes trailing "。" (Japanese period) from the end of the sentence.

    Args:
        tokens: List of kotogram.Token objects to process.

    Returns:
        New list of Token objects with masking applied.
    """
    from kotogram.kotogram import Token  # Deferred import to avoid cycle

    # Remove trailing "。" (Japanese period) if present at end of sentence
    if tokens and tokens[-1].surface == "。":
        tokens = tokens[:-1]

    masked_tokens = []
    for token in tokens:
        new_token = token
        features = token.features

        target_surface = get_surface_mask_for_features(features)
        if target_surface:
            from kotogram.kotogram import TokenFeatures

            new_features = TokenFeatures(
                surface=features.surface,
                pos=features.pos,
                pos_detail_1=features.pos_detail_1,
                pos_detail_2=features.pos_detail_2,
                pos_detail_3=features.pos_detail_3,
                conjugated_type=features.conjugated_type,
                conjugated_form=features.conjugated_form,
                base_orth=features.base_orth,
                lemma="",
                reading="",
                reading_gram=target_surface,
            )
            new_token = Token(token.surface, features=new_features)

        masked_tokens.append(new_token)

    return masked_tokens


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
