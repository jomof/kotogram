import re

# Range includes:
# \u3000-\u303F CJK Symbols and Punctuation
# \u3040-\u309F Hiragana
# \u30A0-\u30FF Katakana
# \u4E00-\u9FFF CJK Unified Ideographs (Common Kanji)
KANA_PUNCT_PATTERN = re.compile(r"[\u3000-\u30FF\u4E00-\u9FFF]")


def parse_tsv(line: str) -> str:
    """
    Parses a line from a TSV file to extract the Japanese sentence.

    Format must be either:
    - 3 or more columns (Tab separated): Sentence is in column 3 (index 2).
    - 1 column: Sentence is the only column.

    Validation:
    - Must contain at least one Kana character (including CJK punctuation).
    - Raises ValueError if format is invalid or no Kana found.
    """
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

    if " jpn " in sentence:
        raise ValueError(f"Sentence contains ' jpn ' (likely malformed): {sentence!r}")

    if not KANA_PUNCT_PATTERN.search(sentence):
        raise ValueError(f"No Kana or Japanese punctuation found in: {sentence!r}")

    return sentence
