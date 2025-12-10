#!/usr/bin/env python3
"""
Build an equivalence set YAML file that groups all equivalent Japanese sentences
under their common English key.

Reads from: .tmp-inspiration/cloze-data/resources/processed/ai-cleaned-merge-grammars/*.yaml
Outputs to: data/equivalence-set.yml

Augments equivalence sets with:
- First person pronoun substitutions (私, 僕, 俺, etc.)
- Copula substitutions (だ<punct> <-> です<punct>)

Then filters out ungrammatical sentences using the grammaticality model.
"""

import re
import yaml
from pathlib import Path
from collections import defaultdict
from itertools import product
from typing import Set


# Augmentation rules: each rule is a set of equivalent tokens
# Sentences containing any token from a set can be augmented with all others
FIRST_PERSON_PRONOUNS = {'私', '僕', '俺', 'わたし', 'ぼく', 'おれ'}

# Copula patterns: da<punct> <-> desu<punct>
# These patterns are in space-separated token format to match our sentence format
# We match at sentence boundaries to avoid mid-sentence replacements
COPULA_PATTERNS = [
    # Direct sentence endings (space-separated tokens)
    ('だ 。', 'です 。'),
    ('だ ！', 'です ！'),
    ('だ ？', 'です ？'),
    ('だ 」', 'です 」'),
    ('だ 』', 'です 』'),
    ('だ …', 'です …'),
    # With sentence-final particles
    ('だ ね 。', 'です ね 。'),
    ('だ よ 。', 'です よ 。'),
    ('だ わ 。', 'です わ 。'),
    ('だ な 。', 'です な 。'),
    # Sentence-final particles without punctuation (end of string)
    ('だ ね', 'です ね'),
    ('だ よ', 'です よ'),
    ('だ わ', 'です わ'),
    ('だ な', 'です な'),
    # With period instead of 。
    ('だ ね.', 'です ね.'),
    ('だ よ.', 'です よ.'),
    ('だ わ.', 'です わ.'),
    ('だ.', 'です.'),
]

# Sentence-initial topic patterns that can be dropped (subject omission)
# Japanese commonly omits the subject when it's clear from context
DROPPABLE_TOPIC_STARTS = [
    '私 は ',
    '僕 は ',
    '俺 は ',
    'わたし は ',
    'ぼく は ',
    'おれ は ',
]

# Progressive verb formality at sentence end: て い ます <-> て いる
# Only match at end of sentence (followed by punctuation or nothing)
PROGRESSIVE_END_PATTERNS = [
    ('て い ます 。', 'て いる 。'),
    ('て い まし た 。', 'て い た 。'),
    ('で い ます 。', 'で いる 。'),
    ('で い まし た 。', 'で い た 。'),
    # Without period
    ('て い ます', 'て いる'),
    ('て い まし た', 'て い た'),
    ('で い ます', 'で いる'),
    ('で い まし た', 'で い た'),
]

# Plural marker variations: kanji vs hiragana
PLURAL_PATTERNS = [
    ('私 達', '私 たち'),
    ('僕 達', '僕 たち'),
    ('俺 達', '俺 たち'),
]


def augment_pronouns(sentence: str) -> Set[str]:
    """Generate all pronoun-substituted variants of a sentence.

    For each first-person pronoun found, generate variants with all other
    first-person pronouns.

    Args:
        sentence: Japanese sentence (space-separated tokens)

    Returns:
        Set of all pronoun variants (including original)
    """
    result = {sentence}

    # Find which pronouns are in the sentence
    tokens = sentence.split()
    pronoun_positions = []

    for i, token in enumerate(tokens):
        if token in FIRST_PERSON_PRONOUNS:
            pronoun_positions.append(i)

    if not pronoun_positions:
        return result

    # Generate all combinations
    # For each position, we can substitute with any pronoun
    for combo in product(FIRST_PERSON_PRONOUNS, repeat=len(pronoun_positions)):
        new_tokens = tokens.copy()
        for pos_idx, new_pronoun in zip(pronoun_positions, combo):
            new_tokens[pos_idx] = new_pronoun
        result.add(' '.join(new_tokens))

    return result


def augment_copula(sentence: str) -> Set[str]:
    """Generate copula-substituted variants of a sentence.

    Swaps だ<punct> with です<punct> and vice versa.

    Args:
        sentence: Japanese sentence (space-separated tokens)

    Returns:
        Set of copula variants (including original)
    """
    result = {sentence}

    # Check each copula pattern
    # Patterns are already in space-separated format to match our tokenized sentences
    for da_form, desu_form in COPULA_PATTERNS:
        if da_form in sentence:
            result.add(sentence.replace(da_form, desu_form))
        if desu_form in sentence:
            result.add(sentence.replace(desu_form, da_form))

    return result


def augment_topic_drop(sentence: str) -> Set[str]:
    """Generate variants with sentence-initial topic dropped.

    Japanese commonly omits the subject/topic when clear from context.
    E.g., "私 は 学生 です" -> "学生 です"

    Args:
        sentence: Japanese sentence (space-separated tokens)

    Returns:
        Set including original and topic-dropped variant (if applicable)
    """
    result = {sentence}

    for topic_start in DROPPABLE_TOPIC_STARTS:
        if sentence.startswith(topic_start):
            # Drop the topic and add the shortened version
            dropped = sentence[len(topic_start):]
            if dropped:  # Make sure we don't create empty sentences
                result.add(dropped)

    return result


def augment_progressive(sentence: str) -> Set[str]:
    """Generate progressive verb formality variants at sentence end.

    Swaps て い ます with て いる and vice versa (only at sentence end).

    Args:
        sentence: Japanese sentence (space-separated tokens)

    Returns:
        Set of progressive variants (including original)
    """
    result = {sentence}

    for polite, plain in PROGRESSIVE_END_PATTERNS:
        # Check if pattern is at end of sentence
        if sentence.endswith(polite):
            result.add(sentence[:-len(polite)] + plain)
        if sentence.endswith(plain):
            result.add(sentence[:-len(plain)] + polite)

    return result


def augment_plural(sentence: str) -> Set[str]:
    """Generate plural marker variants (kanji vs hiragana).

    E.g., 私 達 <-> 私 たち

    Args:
        sentence: Japanese sentence (space-separated tokens)

    Returns:
        Set of plural variants (including original)
    """
    result = {sentence}

    for kanji, hiragana in PLURAL_PATTERNS:
        if kanji in sentence:
            result.add(sentence.replace(kanji, hiragana))
        if hiragana in sentence:
            result.add(sentence.replace(hiragana, kanji))

    return result


def augment_sentence(sentence: str) -> Set[str]:
    """Apply all augmentations to a sentence, returning cross-product of all variants.

    Args:
        sentence: Japanese sentence (space-separated tokens)

    Returns:
        Set of all augmented variants (including original)
    """
    # Start with original
    current = {sentence}

    # Apply each augmentation to all current variants (cross-product)
    # Order: pronouns -> copula -> topic drop -> progressive -> plural

    # Pronoun augmentation
    next_set = set()
    for s in current:
        next_set.update(augment_pronouns(s))
    current = next_set

    # Copula augmentation
    next_set = set()
    for s in current:
        next_set.update(augment_copula(s))
    current = next_set

    # Topic drop augmentation
    next_set = set()
    for s in current:
        next_set.update(augment_topic_drop(s))
    current = next_set

    # Progressive verb formality
    next_set = set()
    for s in current:
        next_set.update(augment_progressive(s))
    current = next_set

    # Plural marker variations
    next_set = set()
    for s in current:
        next_set.update(augment_plural(s))
    current = next_set

    return current


def filter_ungrammatical(sentences: Set[str], parser, check_grammar: bool = True) -> Set[str]:
    """Filter out ungrammatical sentences using the grammaticality model.

    Args:
        sentences: Set of Japanese sentences to filter
        parser: MecabJapaneseParser instance for converting to kotogram
        check_grammar: If True, actually check grammaticality. If False, return all.

    Returns:
        Set of grammatical sentences only
    """
    if not check_grammar:
        return sentences

    from kotogram.analysis import grammaticality

    grammatical = set()
    for sentence in sentences:
        try:
            kotogram = parser.japanese_to_kotogram(sentence)
            if grammaticality(kotogram, use_model=True):
                grammatical.add(sentence)
        except Exception:
            # If parsing fails, keep the original sentence
            grammatical.add(sentence)

    return grammatical


def main():
    input_dir = Path(__file__).parent.parent / ".tmp-inspiration/cloze-data/resources/processed/ai-cleaned-merge-grammars"
    output_file = Path(__file__).parent.parent / "data" / "equivalence-set.yml"

    # Group all Japanese sentences by their English key
    equivalences: dict[str, set[str]] = defaultdict(set)

    yaml_files = sorted(input_dir.glob("*.yaml"))
    print(f"Found {len(yaml_files)} yaml files")

    for yaml_file in yaml_files:
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "examples" not in data:
            continue

        for example in data["examples"]:
            english = example.get("english", "").strip()
            # Replace '' with " for cleaner English text
            english = english.replace("''", '"')
            japanese_list = example.get("japanese", [])

            if not english or not japanese_list:
                continue

            for jp in japanese_list:
                # Clean up the Japanese sentence (remove curly braces used for highlighting)
                jp_clean = jp.replace("{", "").replace("}", "").strip()

                # Handle parenthetical annotations
                # For optional characters like (や), create both versions
                optional_match = re.search(r'\(([ぁ-んァ-ン])\)', jp_clean)
                if optional_match:
                    # Add version with the optional character
                    jp_with = re.sub(r'\(([ぁ-んァ-ン])\)', r'\1', jp_clean)
                    equivalences[english].add(jp_with)
                    # Add version without the optional character
                    jp_without = re.sub(r'\(([ぁ-んァ-ン])\)', '', jp_clean)
                    equivalences[english].add(jp_without)
                else:
                    # Strip other parenthetical annotations (readings, explanations)
                    jp_clean = re.sub(r'\([^)]+\)', '', jp_clean)
                    equivalences[english].add(jp_clean)

    print(f"Found {len(equivalences)} unique English sentences")
    print(f"Total Japanese equivalents (before augmentation): {sum(len(v) for v in equivalences.values())}")

    # Initialize parser for grammaticality checking
    print("Loading parser and grammaticality model...")
    from kotogram.mecab_japanese_parser import MecabJapaneseParser
    parser = MecabJapaneseParser()

    # Warm up the model
    from kotogram.analysis import grammaticality
    _ = grammaticality(parser.japanese_to_kotogram("テスト"), use_model=True)
    print("Model loaded.")

    # Apply augmentations and filter
    print("Augmenting and filtering sentences...")
    result = {}
    total_before_filter = 0
    total_after_filter = 0

    for i, (eng, jps) in enumerate(sorted(equivalences.items())):
        if (i + 1) % 1000 == 0:
            print(f"  Processing {i + 1}/{len(equivalences)}...")

        # Augment all sentences in this equivalence set
        augmented = set()
        for jp in jps:
            augmented.update(augment_sentence(jp))

        total_before_filter += len(augmented)

        # Filter out ungrammatical sentences
        filtered = filter_ungrammatical(augmented, parser, check_grammar=True)
        total_after_filter += len(filtered)

        if filtered:
            result[eng] = sorted(list(filtered))

    print(f"Total Japanese equivalents (after augmentation): {total_before_filter}")
    print(f"Total Japanese equivalents (after filtering): {total_after_filter}")
    print(f"Removed {total_before_filter - total_after_filter} ungrammatical sentences")

    # Use a custom representer to prefer double-quoted strings for keys with quotes
    def str_representer(dumper, data):
        if '"' in data or "'" in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml.add_representer(str, str_representer)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(result, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
