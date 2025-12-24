"""TSV parsing utilities for training data.

This module provides lightweight TSV parsing that does not require torch.
"""

import re

KANA_PUNCT_PATTERN = re.compile(r"[\u3000-\u30FF\u4E00-\u9FFF]")


def parse_tsv(line: str) -> str:
    """Parses a line from a TSV file to extract the Japanese sentence."""
    line = line.strip()
    if not line:
        raise ValueError("Empty line")

    parts = line.split("\t")

    if len(parts) >= 3:
        sentence = parts[2]
    elif len(parts) == 1 and parts[0]:
        sentence = parts[0]
    else:
        raise ValueError(f"Invalid column count: {len(parts)}. Expected 1 or >=3.")

    if " " in sentence:
        raise ValueError(f"Sentence contains space (not allowed): {sentence!r}")

    if len(sentence) <= 2:
        raise ValueError(f"Sentence is too short (<=2 chars): {sentence!r}")

    if "〇〇" in sentence:
        raise ValueError(f"Sentence contains placeholder '〇〇': {sentence!r}")

    if " jpn " in sentence:
        raise ValueError(f"Sentence contains ' jpn ' (likely malformed): {sentence!r}")

    if not KANA_PUNCT_PATTERN.search(sentence):
        raise ValueError(f"No Kana or Japanese punctuation found in: {sentence!r}")

    return sentence
