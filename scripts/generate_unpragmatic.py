#!/usr/bin/env python3
"""Generate unpragmatic formality and gender examples from existing Japanese sentences.

This script takes sentences from jpn_sentences.tsv and programmatically modifies them
to create examples with mixed formality or gender markers that would be unusual/awkward
in natural Japanese speech. This provides training data for the UNPRAGMATIC_FORMALITY
and UNPRAGMATIC_GENDER classes.

Strategies for generating unpragmatic examples:

UNPRAGMATIC_GENDER:
1. Mix masculine and feminine first-person pronouns in the same sentence
2. Add masculine sentence-final particles to feminine sentences (and vice versa)
3. Combine masculine pronouns with feminine sentence endings

UNPRAGMATIC_FORMALITY:
1. Mix formal verb endings (ます/です) with very casual particles (ぜ/ぞ)
2. Combine keigo honorifics with casual copula (だ) forms
3. Add casual sentence-final particles to formal sentences inappropriately

Usage:
    python scripts/generate_unpragmatic.py --input data/jpn_sentences.tsv --output data/unpragmatic_sentences.tsv
    python scripts/generate_unpragmatic.py --input data/jpn_sentences.tsv --output data/unpragmatic_sentences.tsv --max-samples 1000
"""

import argparse
import csv
import random
import re
from typing import List, Tuple, Optional, Dict

# Token manipulation patterns
# Kotogram token format: ⌈ˢSURFACEᵖPOS:DETAILS...⌉

# Pronoun replacements for gender mixing
MASCULINE_PRONOUNS = {
    '俺': ('pron', 'オレ'),
    '僕': ('pron', 'ボク'),
}

FEMININE_PRONOUNS = {
    'あたし': ('pron', 'アタシ'),
}

NEUTRAL_PRONOUNS = {
    '私': ('pron', 'ワタシ'),
}

# Sentence-final particles
MASCULINE_PARTICLES = ['ぜ', 'ぞ']
FEMININE_PARTICLES = ['わ']
CASUAL_PARTICLES = ['よ', 'ね', 'な']

# Token templates
def make_pronoun_token(surface: str, reading: str) -> str:
    """Create a kotogram token for a pronoun."""
    return f"⌈ˢ{surface}ᵖpronᵈ{surface}ʳ{reading}⌉"

def make_particle_token(surface: str) -> str:
    """Create a kotogram token for a sentence-final particle."""
    return f"⌈ˢ{surface}ᵖprt:sentence_final_particle⌉"

def make_casual_copula_token() -> str:
    """Create a kotogram token for casual だ copula."""
    return "⌈ˢだᵖauxv:auxv-da:terminalᵈだʳダ⌉"


def split_kotogram(kotogram: str) -> List[str]:
    """Split a kotogram sentence into individual tokens."""
    return re.findall(r'⌈[^⌉]*⌉', kotogram)


def extract_surface(token: str) -> str:
    """Extract surface form from a kotogram token."""
    match = re.search(r'ˢ(.*?)ᵖ', token)
    return match.group(1) if match else ''


def extract_pos(token: str) -> str:
    """Extract main POS from a kotogram token."""
    match = re.search(r'ᵖ([^:⌉]+)', token)
    return match.group(1) if match else ''


def extract_pos_detail(token: str) -> str:
    """Extract POS detail (e.g., sentence_final_particle) from a kotogram token."""
    match = re.search(r'ᵖ[^:⌉]+:([^:⌉]+)', token)
    return match.group(1) if match else ''


def has_formal_ending(tokens: List[str]) -> bool:
    """Check if sentence has formal verb endings (ます/です)."""
    for token in tokens:
        pos_match = re.search(r'ᵖ([^⌉]+)', token)
        if pos_match:
            pos_data = pos_match.group(1)
            if 'auxv-masu' in pos_data or 'auxv-desu' in pos_data:
                return True
    return False


def has_casual_copula(tokens: List[str]) -> bool:
    """Check if sentence has casual copula だ at sentence end."""
    for i, token in enumerate(tokens):
        pos_match = re.search(r'ᵖ([^⌉]+)', token)
        if pos_match:
            pos_data = pos_match.group(1)
            if 'auxv-da' in pos_data and 'terminal' in pos_data:
                # Check if near end
                if i >= len(tokens) - 3:
                    return True
    return False


def find_pronoun_indices(tokens: List[str]) -> List[Tuple[int, str]]:
    """Find indices of first-person pronoun tokens and their surfaces."""
    pronouns = []
    first_person = ['私', 'わたし', 'ワタシ', '俺', 'おれ', 'オレ', '僕', 'ぼく', 'ボク',
                    'あたし', 'アタシ', 'わたくし', 'ワタクシ']
    for i, token in enumerate(tokens):
        if extract_pos(token) == 'pron':
            surface = extract_surface(token)
            if surface in first_person:
                pronouns.append((i, surface))
    return pronouns


def find_sentence_final_particles(tokens: List[str]) -> List[Tuple[int, str]]:
    """Find indices of sentence-final particles."""
    particles = []
    for i, token in enumerate(tokens):
        if extract_pos(token) == 'prt':
            detail = extract_pos_detail(token)
            if detail == 'sentence_final_particle':
                surface = extract_surface(token)
                particles.append((i, surface))
    return particles


def is_masculine_pronoun(surface: str) -> bool:
    """Check if pronoun is masculine."""
    return surface in ['俺', 'おれ', 'オレ', '僕', 'ぼく', 'ボク']


def is_feminine_pronoun(surface: str) -> bool:
    """Check if pronoun is feminine."""
    return surface in ['あたし', 'アタシ', 'あたくし', 'アタクシ']


def is_masculine_particle(surface: str) -> bool:
    """Check if particle is masculine."""
    return surface in MASCULINE_PARTICLES


def is_feminine_particle(surface: str) -> bool:
    """Check if particle is feminine."""
    return surface in FEMININE_PARTICLES


# ============================================================================
# UNPRAGMATIC GENDER GENERATORS
# ============================================================================

def generate_unpragmatic_gender_pronoun_swap(kotogram: str) -> Optional[str]:
    """Generate unpragmatic gender by swapping pronoun gender.

    Strategy: Find a masculine pronoun and add a feminine one (or vice versa).
    For example, change "俺が行く" to "俺があたしの分も行く"
    Or simply swap: "俺が行く" -> "あたしが行くぜ" (feminine pronoun with masculine particle)
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    pronouns = find_pronoun_indices(tokens)
    particles = find_sentence_final_particles(tokens)

    # Strategy 1: If has masculine pronoun, add feminine particle at end
    for idx, surface in pronouns:
        if is_masculine_pronoun(surface):
            # Add feminine particle わ at end if not already there
            if not any(is_feminine_particle(p) for _, p in particles):
                # Find position to insert (before auxs like 。)
                insert_pos = len(tokens)
                for i in range(len(tokens) - 1, -1, -1):
                    if extract_pos(tokens[i]) == 'auxs':
                        insert_pos = i
                    else:
                        break
                new_tokens = tokens[:insert_pos] + [make_particle_token('わ')] + tokens[insert_pos:]
                return ''.join(new_tokens)

    # Strategy 2: If has feminine pronoun, add masculine particle at end
    for idx, surface in pronouns:
        if is_feminine_pronoun(surface):
            # Add masculine particle ぜ at end if not already there
            if not any(is_masculine_particle(p) for _, p in particles):
                insert_pos = len(tokens)
                for i in range(len(tokens) - 1, -1, -1):
                    if extract_pos(tokens[i]) == 'auxs':
                        insert_pos = i
                    else:
                        break
                particle = random.choice(MASCULINE_PARTICLES)
                new_tokens = tokens[:insert_pos] + [make_particle_token(particle)] + tokens[insert_pos:]
                return ''.join(new_tokens)

    return None


def generate_unpragmatic_gender_mixed_pronouns(kotogram: str) -> Optional[str]:
    """Generate unpragmatic gender by replacing neutral pronoun with mixed markers.

    Strategy: Replace 私 with a gendered pronoun and add opposite-gender particle.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    pronouns = find_pronoun_indices(tokens)

    # Find neutral 私 pronoun
    for idx, surface in pronouns:
        if surface in ['私', 'わたし', 'ワタシ']:
            # Replace with masculine pronoun
            new_tokens = list(tokens)
            new_tokens[idx] = make_pronoun_token('俺', 'オレ')

            # Add feminine particle at end
            insert_pos = len(new_tokens)
            for i in range(len(new_tokens) - 1, -1, -1):
                if extract_pos(new_tokens[i]) == 'auxs':
                    insert_pos = i
                else:
                    break
            new_tokens = new_tokens[:insert_pos] + [make_particle_token('わ')] + new_tokens[insert_pos:]
            return ''.join(new_tokens)

    return None


def generate_unpragmatic_gender_particle_clash(kotogram: str) -> Optional[str]:
    """Generate unpragmatic gender by adding clashing particles.

    Strategy: Add both masculine and feminine particles to a neutral sentence.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    particles = find_sentence_final_particles(tokens)

    # Only modify if no strong gender markers already
    has_masc = any(is_masculine_particle(p) for _, p in particles)
    has_fem = any(is_feminine_particle(p) for _, p in particles)

    if has_masc or has_fem:
        return None

    # Find insert position
    insert_pos = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        if extract_pos(tokens[i]) == 'auxs':
            insert_pos = i
        else:
            break

    # Add both masculine and feminine particles (this sounds very strange)
    masc_particle = random.choice(MASCULINE_PARTICLES)
    new_tokens = tokens[:insert_pos] + [make_particle_token(masc_particle), make_particle_token('わ')] + tokens[insert_pos:]
    return ''.join(new_tokens)


# ============================================================================
# UNPRAGMATIC FORMALITY GENERATORS
# ============================================================================

def generate_unpragmatic_formality_formal_with_casual_particle(kotogram: str) -> Optional[str]:
    """Generate unpragmatic formality by adding casual particle to formal sentence.

    Strategy: Add very casual particles (ぜ/ぞ) to sentences with ます/です forms.
    This sounds very awkward - like mixing registers inappropriately.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    if not has_formal_ending(tokens):
        return None

    particles = find_sentence_final_particles(tokens)

    # Check if already has very casual particles
    if any(p in MASCULINE_PARTICLES for _, p in particles):
        return None

    # Find insert position (before final auxs punctuation)
    insert_pos = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        if extract_pos(tokens[i]) == 'auxs':
            insert_pos = i
        else:
            break

    # Add a very casual particle
    particle = random.choice(MASCULINE_PARTICLES)
    new_tokens = tokens[:insert_pos] + [make_particle_token(particle)] + tokens[insert_pos:]
    return ''.join(new_tokens)


def generate_unpragmatic_formality_casual_copula_formal_particle(kotogram: str) -> Optional[str]:
    """Generate unpragmatic formality by mixing casual copula with formal elements.

    Strategy: Find formal sentences and downgrade ending while keeping formal particles.
    This is complex - we'd need to actually replace ます with る, which requires verb knowledge.

    Simpler approach: Add casual だ after formal ending (sounds redundant/awkward).
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    if not has_formal_ending(tokens):
        return None

    # This creates awkward "ですだ" or similar patterns
    # Find position after the formal verb
    insert_pos = None
    for i, token in enumerate(tokens):
        pos_match = re.search(r'ᵖ([^⌉]+)', token)
        if pos_match:
            pos_data = pos_match.group(1)
            if 'auxv-masu' in pos_data or 'auxv-desu' in pos_data:
                if 'terminal' in pos_data:
                    insert_pos = i + 1
                    break

    if insert_pos is None:
        return None

    # Add casual particle after formal ending (sounds weird)
    particle = random.choice(MASCULINE_PARTICLES)
    new_tokens = tokens[:insert_pos] + [make_particle_token(particle)] + tokens[insert_pos:]
    return ''.join(new_tokens)


def generate_unpragmatic_formality_add_yo_to_keigo(kotogram: str) -> Optional[str]:
    """Generate slightly unpragmatic formality by adding casual よ to formal sentences.

    Note: よ with です/ます is actually acceptable in modern Japanese, so this
    generates mildly informal formal speech rather than truly unpragmatic.
    For stronger effect, we use ぜ/ぞ instead.
    """
    # This is actually often acceptable, so we'll skip this generator
    # and rely on the more obvious mixing patterns
    return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_sentence(
    kotogram: str,
    sentence: str,
    sentence_id: str,
) -> List[Tuple[str, str, str, str]]:
    """Process a single sentence and generate unpragmatic variants.

    Returns:
        List of (new_id, 'jpn', new_sentence, new_kotogram) tuples
    """
    results = []

    # Try gender unpragmatic generators
    gender_generators = [
        ('g1', generate_unpragmatic_gender_pronoun_swap),
        ('g2', generate_unpragmatic_gender_mixed_pronouns),
        ('g3', generate_unpragmatic_gender_particle_clash),
    ]

    for suffix, generator in gender_generators:
        try:
            new_kotogram = generator(kotogram)
            if new_kotogram:
                # Extract surface form for the new sentence
                new_surface = extract_all_surfaces(new_kotogram)
                new_id = f"{sentence_id}_unprag_{suffix}"
                results.append((new_id, 'jpn', new_surface, new_kotogram))
        except Exception:
            pass

    # Try formality unpragmatic generators
    formality_generators = [
        ('f1', generate_unpragmatic_formality_formal_with_casual_particle),
        ('f2', generate_unpragmatic_formality_casual_copula_formal_particle),
    ]

    for suffix, generator in formality_generators:
        try:
            new_kotogram = generator(kotogram)
            if new_kotogram:
                new_surface = extract_all_surfaces(new_kotogram)
                new_id = f"{sentence_id}_unprag_{suffix}"
                results.append((new_id, 'jpn', new_surface, new_kotogram))
        except Exception:
            pass

    return results


def extract_all_surfaces(kotogram: str) -> str:
    """Extract and concatenate all surface forms from a kotogram."""
    pattern = r'ˢ(.*?)ᵖ'
    matches = re.findall(pattern, kotogram, re.DOTALL)
    return ''.join(matches)


def main():
    parser = argparse.ArgumentParser(
        description="Generate unpragmatic formality/gender examples from Japanese sentences"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/jpn_sentences.tsv",
        help="Input TSV file with Japanese sentences"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/unpragmatic_sentences.tsv",
        help="Output TSV file for generated examples"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum number of input sentences to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # We need to import kotogram parser to convert sentences
    try:
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
        parser_instance = SudachiJapaneseParser()
    except ImportError:
        print("Error: Could not import SudachiJapaneseParser")
        print("Make sure kotogram is installed: pip install -e .")
        return

    generated = []
    processed = 0

    if args.verbose:
        print(f"Reading from {args.input}...")

    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for row in reader:
            if len(row) < 3:
                continue

            sentence_id, lang, sentence = row[0], row[1], row[2]

            if lang != 'jpn':
                continue

            try:
                # Convert to kotogram
                kotogram = parser_instance.japanese_to_kotogram(sentence)

                # Generate unpragmatic variants
                variants = process_sentence(kotogram, sentence, sentence_id)
                generated.extend(variants)

                processed += 1

                if args.verbose and processed % 5000 == 0:
                    print(f"Processed {processed} sentences, generated {len(generated)} unpragmatic examples")

                if args.max_samples and processed >= args.max_samples:
                    break

            except Exception as e:
                if args.verbose:
                    print(f"Error processing {sentence_id}: {e}")
                continue

    if args.verbose:
        print(f"\nTotal processed: {processed} sentences")
        print(f"Total generated: {len(generated)} unpragmatic examples")

    # Write output
    if args.verbose:
        print(f"Writing to {args.output}...")

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for new_id, lang, new_sentence, new_kotogram in generated:
            # Write in same format as input: id, lang, sentence
            # The kotogram will be regenerated during training, but we could also store it
            writer.writerow([new_id, lang, new_sentence])

    if args.verbose:
        print("Done!")

        # Print some statistics
        gender_count = sum(1 for x in generated if '_g' in x[0])
        formality_count = sum(1 for x in generated if '_f' in x[0])
        print(f"\nGenerated examples breakdown:")
        print(f"  Unpragmatic gender: {gender_count}")
        print(f"  Unpragmatic formality: {formality_count}")


if __name__ == "__main__":
    main()
