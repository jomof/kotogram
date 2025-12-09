#!/usr/bin/env python3
"""Generate agrammatic (grammatically incorrect) sentences from Japanese sentences.

This script takes sentences from jpn_sentences.tsv and programmatically introduces
grammatical errors to create examples of common mistakes made by Japanese learners.
This can be useful for:
- Training grammar correction models
- Creating negative examples for grammaticality classification
- Testing parsing robustness

Error categories implemented:

BEGINNER LEVEL:
- Particle errors (は/が swap, を omission, に/で confusion)
- Verb form errors (て-form mistakes, dictionary/ます mixing)
- Copula errors (だ after い-adjectives, です after plain past)
- Word order errors

INTERMEDIATE LEVEL:
- Formality mixing in subordinate clauses
- Transitivity pair confusion
- Conditional mixing

ADVANCED LEVEL:
- Incorrect pragmatic particle use
- Honorific/humble speech mixing
- Sentence-final form errors (のだ/んだ misuse)

Usage:
    python scripts/generate_agrammatic.py --input data/jpn_sentences.tsv --output data/agrammatic_sentences.tsv
    python scripts/generate_agrammatic.py --input data/jpn_sentences.tsv --output data/agrammatic_sentences.tsv --max-samples 1000
"""

import argparse
import csv
import random
import re
from typing import List, Tuple, Optional, Dict, Callable


# ============================================================================
# KOTOGRAM TOKEN UTILITIES
# ============================================================================

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
    """Extract POS detail from a kotogram token."""
    match = re.search(r'ᵖ[^:⌉]+:([^:⌉]+)', token)
    return match.group(1) if match else ''


def extract_conjugation_type(token: str) -> str:
    """Extract conjugation type from a kotogram token."""
    # Format: ᵖpos:detail1:detail2:conjugation_type:conjugation_form
    match = re.search(r'ᵖ[^⌉]+:([^:⌉]+):[^:⌉]+$', token)
    if match:
        return match.group(1)
    # Alternative: look for auxv-XXX patterns
    match = re.search(r'auxv-([^:⌉]+)', token)
    return match.group(1) if match else ''


def extract_conjugation_form(token: str) -> str:
    """Extract conjugation form from a kotogram token."""
    match = re.search(r':([^:⌉]+)⌉$', token)
    return match.group(1) if match else ''


def extract_all_surfaces(kotogram: str) -> str:
    """Extract and concatenate all surface forms from a kotogram."""
    pattern = r'ˢ(.*?)ᵖ'
    matches = re.findall(pattern, kotogram, re.DOTALL)
    return ''.join(matches)


def make_particle_token(surface: str, detail: str = 'case_particle') -> str:
    """Create a kotogram token for a particle."""
    return f"⌈ˢ{surface}ᵖprt:{detail}⌉"


def make_auxv_token(surface: str, aux_type: str, form: str = 'terminal') -> str:
    """Create a kotogram token for an auxiliary verb."""
    return f"⌈ˢ{surface}ᵖauxv:{aux_type}:{form}⌉"


# ============================================================================
# BEGINNER LEVEL ERRORS
# ============================================================================

def error_particle_wa_ga_swap(kotogram: str) -> Optional[Tuple[str, str]]:
    """Swap は and が particles inappropriately.

    Common beginner mistake: not understanding topic marker vs subject marker.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find は or が particles
    for i, token in enumerate(tokens):
        if extract_pos(token) == 'prt':
            surface = extract_surface(token)
            if surface == 'は':
                # Swap は -> が
                new_token = make_particle_token('が', 'case_particle')
                new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                return ''.join(new_tokens), 'wa_ga_swap'
            elif surface == 'が':
                # Swap が -> は
                new_token = make_particle_token('は', 'binding_particle')
                new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                return ''.join(new_tokens), 'ga_wa_swap'

    return None


def error_particle_wo_omission(kotogram: str) -> Optional[Tuple[str, str]]:
    """Remove を particle from direct objects.

    While を omission is natural in casual speech, removing it in certain
    contexts sounds ungrammatical (especially in written/formal text).
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find を particle and remove it
    for i, token in enumerate(tokens):
        if extract_pos(token) == 'prt':
            surface = extract_surface(token)
            if surface == 'を':
                # Remove を particle
                new_tokens = tokens[:i] + tokens[i+1:]
                if new_tokens:
                    return ''.join(new_tokens), 'wo_omission'

    return None


def error_particle_ni_de_confusion(kotogram: str) -> Optional[Tuple[str, str]]:
    """Confuse に and で particles.

    Common error: using に for location of action (should be で)
    or using で for destination (should be に).
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        if extract_pos(token) == 'prt':
            surface = extract_surface(token)
            if surface == 'に':
                new_token = make_particle_token('で', 'case_particle')
                new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                return ''.join(new_tokens), 'ni_de_swap'
            elif surface == 'で':
                new_token = make_particle_token('に', 'case_particle')
                new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                return ''.join(new_tokens), 'de_ni_swap'

    return None


def error_particle_he_ni_confusion(kotogram: str) -> Optional[Tuple[str, str]]:
    """Confuse へ and に for movement/direction.

    While often interchangeable, misuse in certain contexts sounds unnatural.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        if extract_pos(token) == 'prt':
            surface = extract_surface(token)
            if surface == 'へ':
                new_token = make_particle_token('に', 'case_particle')
                new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                return ''.join(new_tokens), 'he_ni_swap'

    return None


def error_copula_da_after_i_adjective(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add だ after い-adjective (incorrect).

    Common error: たかいだです or たかいだ (should be たかいです or たかい).
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find い-adjective in terminal form
    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        if pos == 'adj' and 'adjective' in token:
            form = extract_conjugation_form(token)
            surface = extract_surface(token)
            # Check if it's an い-adjective (ends in い and is terminal)
            if surface.endswith('い') and form == 'terminal':
                # Insert だ after the adjective
                da_token = make_auxv_token('だ', 'auxv-da', 'terminal')
                new_tokens = tokens[:i+1] + [da_token] + tokens[i+1:]
                return ''.join(new_tokens), 'da_after_i_adj'

    return None


def error_desu_after_plain_past(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add です after plain past form (incorrect).

    Common error: たべたです (should be たべました).
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find plain past verb (ta-form)
    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        if pos == 'auxv' and 'auxv-ta' in token:
            form = extract_conjugation_form(token)
            if form == 'terminal':
                # Insert です after the past auxiliary
                desu_token = make_auxv_token('です', 'auxv-desu', 'terminal')
                new_tokens = tokens[:i+1] + [desu_token] + tokens[i+1:]
                return ''.join(new_tokens), 'desu_after_plain_past'

    return None


def error_double_particle(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add duplicate particles (incorrect).

    Error: りんごをを食べる (double を).
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        if extract_pos(token) == 'prt':
            surface = extract_surface(token)
            if surface in ['を', 'が', 'に', 'で', 'と']:
                # Duplicate the particle
                new_tokens = tokens[:i+1] + [token] + tokens[i+1:]
                return ''.join(new_tokens), f'double_{surface}'

    return None


# ============================================================================
# INTERMEDIATE LEVEL ERRORS
# ============================================================================

def error_formality_mixing_subordinate(kotogram: str) -> Optional[Tuple[str, str]]:
    """Use polite forms in subordinate clauses (incorrect).

    Error: 食べます時 (should be 食べる時).
    Polite forms should not be used in relative/temporal clauses.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find dictionary form verb followed by time word (時, とき, etc.)
    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        if pos == 'v' and i + 1 < len(tokens):
            next_surface = extract_surface(tokens[i+1])
            # Check if followed by time/conditional marker
            if next_surface in ['時', 'とき', 'トキ', 'ところ', 'ため']:
                # This would need more complex verb conjugation to properly implement
                # For now, just add ます inappropriately
                masu_token = make_auxv_token('ます', 'auxv-masu', 'terminal')
                new_tokens = tokens[:i+1] + [masu_token] + tokens[i+1:]
                return ''.join(new_tokens), 'masu_in_subordinate'

    return None


def error_transitivity_confusion(kotogram: str) -> Optional[Tuple[str, str]]:
    """Confuse transitive/intransitive verb pairs.

    Common pairs: 開ける/開く, 付ける/付く, 落とす/落ちる
    Error: ドアを開く (should be ドアを開ける for transitive)
    """
    # This would require a dictionary of transitive/intransitive pairs
    # and understanding of the sentence context
    # Simplified implementation: swap を with が for certain verbs
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find を followed by common transitive verbs and swap to intransitive pattern
    transitive_verbs = ['開け', '閉め', '付け', '消し', '落とし', '起こし', '直し']

    for i, token in enumerate(tokens):
        if extract_pos(token) == 'prt' and extract_surface(token) == 'を':
            # Check if following token is a transitive verb
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i+1])
                for tv in transitive_verbs:
                    if next_surface.startswith(tv):
                        # Wrong: use が instead of を with transitive
                        new_token = make_particle_token('が', 'case_particle')
                        new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                        return ''.join(new_tokens), 'transitivity_confusion'

    return None


def error_conditional_mixing(kotogram: str) -> Optional[Tuple[str, str]]:
    """Mix up conditional forms inappropriately.

    Error: Using ～たら when ～なら is required, or vice versa.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find conditional markers
    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        if surface == 'たら':
            # Replace with なら (often incorrect context)
            new_token = make_particle_token('なら', 'binding_particle')
            new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
            return ''.join(new_tokens), 'tara_nara_swap'
        elif surface == 'なら':
            # Replace with たら
            # This is simplified - proper implementation would need verb stem + たら
            new_token = "⌈ˢたらᵖprt:conjunctive_particle⌉"
            new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
            return ''.join(new_tokens), 'nara_tara_swap'

    return None


def error_passive_potential_confusion(kotogram: str) -> Optional[Tuple[str, str]]:
    """Confuse passive ～られる with potential ～られる.

    Both use the same form for ichidan verbs, but context matters.
    Error: Adding extra agent marker with potential meaning.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find られる forms
    for i, token in enumerate(tokens):
        if 'rareru' in token.lower() or 'られ' in extract_surface(token):
            # Add に particle before the verb (passive-style) inappropriately
            if i > 0:
                ni_token = make_particle_token('に', 'case_particle')
                new_tokens = tokens[:i] + [ni_token] + tokens[i:]
                return ''.join(new_tokens), 'passive_potential_confusion'

    return None


# ============================================================================
# ADVANCED LEVEL ERRORS
# ============================================================================

def error_pragmatic_particle_misuse(kotogram: str) -> Optional[Tuple[str, str]]:
    """Misuse sentence-final particles (ね, よ, ぞ, etc.).

    Error: Using assertive よ when seeking agreement (should be ね)
    Or using feminine わ in masculine context.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find sentence-final particles
    for i, token in enumerate(tokens):
        if extract_pos(token) == 'prt':
            detail = extract_pos_detail(token)
            surface = extract_surface(token)
            if detail == 'sentence_final_particle':
                if surface == 'ね':
                    # Replace with inappropriate よ
                    new_token = make_particle_token('よ', 'sentence_final_particle')
                    new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                    return ''.join(new_tokens), 'ne_yo_swap'
                elif surface == 'よ':
                    # Replace with inappropriate ね
                    new_token = make_particle_token('ね', 'sentence_final_particle')
                    new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                    return ''.join(new_tokens), 'yo_ne_swap'

    return None


def error_honorific_mixing(kotogram: str) -> Optional[Tuple[str, str]]:
    """Mix 尊敬語 (respectful) and 謙譲語 (humble) forms incorrectly.

    Error: Using humble form when respectful is needed, or vice versa.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Look for honorific prefixes and humble forms
    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        # お/ご + verb stem patterns
        if surface.startswith('お') or surface.startswith('ご'):
            # Check for になる (respectful) and replace with する (humble-style but wrong)
            if i + 2 < len(tokens):
                if 'なる' in extract_surface(tokens[i+1]) or 'なり' in extract_surface(tokens[i+1]):
                    # This would replace お～になる with お～する (wrong context)
                    new_surface = surface  # Keep the surface
                    # Add する token after
                    suru_token = "⌈ˢするᵖv:non_self_reliant:sa-irregular:terminal⌉"
                    new_tokens = tokens[:i+1] + [suru_token] + tokens[i+2:]
                    return ''.join(new_tokens), 'honorific_humble_mixing'

    return None


def error_noda_misuse(kotogram: str) -> Optional[Tuple[str, str]]:
    """Misuse のだ/んだ explanatory form.

    Error: Adding のだ where it adds awkward emphasis or sounds unnatural.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find sentence-ending verbs/adjectives and add inappropriate のだ
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        pos = extract_pos(token)
        if pos in ['v', 'adj', 'auxv']:
            form = extract_conjugation_form(token)
            if form == 'terminal':
                # Check if already has のだ
                has_noda = any('のだ' in extract_surface(t) or 'んだ' in extract_surface(t)
                              for t in tokens[i+1:i+3])
                if not has_noda:
                    # Add のだ inappropriately
                    no_token = make_particle_token('の', 'pre_noun_particle')
                    da_token = make_auxv_token('だ', 'auxv-da', 'terminal')
                    new_tokens = tokens[:i+1] + [no_token, da_token] + tokens[i+1:]
                    return ''.join(new_tokens), 'noda_overuse'
                break

    return None


def error_da_after_noun_polite(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add だ after noun in polite speech (incorrect).

    Error: 先生だです (should be 先生です).
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find noun + です pattern
    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        if pos == 'n':  # Noun
            if i + 1 < len(tokens):
                next_token = tokens[i+1]
                if 'auxv-desu' in next_token:
                    # Insert だ between noun and です
                    da_token = make_auxv_token('だ', 'auxv-da', 'terminal')
                    new_tokens = tokens[:i+1] + [da_token] + tokens[i+1:]
                    return ''.join(new_tokens), 'da_desu_redundant'

    return None


def error_over_specification(kotogram: str) -> Optional[Tuple[str, str]]:
    """Over-specify subjects/objects that should be dropped.

    Japanese naturally drops pronouns when clear from context.
    Error: Adding 私は at the beginning when unnecessary.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Check if sentence starts with a pronoun
    has_initial_pronoun = False
    for token in tokens[:3]:
        if extract_pos(token) == 'pron':
            has_initial_pronoun = True
            break

    if not has_initial_pronoun and len(tokens) > 2:
        # Add unnecessary 私は at the beginning
        watashi_token = "⌈ˢ私ᵖpronʳワタシ⌉"
        wa_token = make_particle_token('は', 'binding_particle')
        new_tokens = [watashi_token, wa_token] + tokens
        return ''.join(new_tokens), 'over_specification'

    return None


def error_te_form_wrong(kotogram: str) -> Optional[Tuple[str, str]]:
    """Create incorrect て-form conjugations.

    Error: いく -> いいて (should be いって), or かう -> かうて (should be かって)
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find て-form patterns and corrupt them
    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        if surface == 'て' or surface == 'で':
            # Check preceding token for verb stem
            if i > 0:
                prev_surface = extract_surface(tokens[i-1])
                # Create wrong て-form by adding extra っ or removing small tsu
                if prev_surface.endswith('っ'):
                    # Remove the small tsu (wrong)
                    new_prev = prev_surface[:-1]
                    new_token = re.sub(r'ˢ[^ᵖ]+ᵖ', f'ˢ{new_prev}ᵖ', tokens[i-1])
                    new_tokens = tokens[:i-1] + [new_token] + tokens[i:]
                    return ''.join(new_tokens), 'te_form_missing_sokuon'

    return None


def error_wrong_verb_base(kotogram: str) -> Optional[Tuple[str, str]]:
    """Use wrong verb conjugation base.

    Error: Using dictionary form where ます-stem is needed.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find ます and replace it but keep dictionary form before it
    for i, token in enumerate(tokens):
        if 'auxv-masu' in token:
            # This would need proper verb knowledge to implement correctly
            # Simplified: just remove ます to create incomplete form
            new_tokens = tokens[:i] + tokens[i+1:]
            if new_tokens:
                return ''.join(new_tokens), 'masu_omission'

    return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

# All error generators with their categories and weights
ERROR_GENERATORS: List[Tuple[str, str, Callable, float]] = [
    # (error_code, category, function, probability_weight)

    # Beginner level errors (more common)
    ('b_wa_ga', 'beginner', error_particle_wa_ga_swap, 1.0),
    ('b_wo_omit', 'beginner', error_particle_wo_omission, 0.8),
    ('b_ni_de', 'beginner', error_particle_ni_de_confusion, 0.8),
    ('b_he_ni', 'beginner', error_particle_he_ni_confusion, 0.5),
    ('b_da_i_adj', 'beginner', error_copula_da_after_i_adjective, 0.7),
    ('b_desu_past', 'beginner', error_desu_after_plain_past, 0.6),
    ('b_double_prt', 'beginner', error_double_particle, 0.4),

    # Intermediate level errors
    ('i_masu_sub', 'intermediate', error_formality_mixing_subordinate, 0.5),
    ('i_trans', 'intermediate', error_transitivity_confusion, 0.4),
    ('i_cond', 'intermediate', error_conditional_mixing, 0.5),
    ('i_pass_pot', 'intermediate', error_passive_potential_confusion, 0.3),

    # Advanced level errors
    ('a_prag_prt', 'advanced', error_pragmatic_particle_misuse, 0.6),
    ('a_honor', 'advanced', error_honorific_mixing, 0.3),
    ('a_noda', 'advanced', error_noda_misuse, 0.4),
    ('a_da_desu', 'advanced', error_da_after_noun_polite, 0.4),
    ('a_over_spec', 'advanced', error_over_specification, 0.5),
    ('a_te_form', 'advanced', error_te_form_wrong, 0.3),
    ('a_verb_base', 'advanced', error_wrong_verb_base, 0.3),
]


def process_sentence(
    kotogram: str,
    sentence: str,
    sentence_id: str,
    max_errors_per_sentence: int = 3,
    error_probability: float = 0.3,
) -> List[Tuple[str, str, str, str, str]]:
    """Process a single sentence and generate agrammatic variants.

    Returns:
        List of (new_id, 'jpn', new_sentence, new_kotogram, error_type) tuples
    """
    results = []

    for error_code, category, generator, weight in ERROR_GENERATORS:
        # Apply probability weight
        if random.random() > error_probability * weight:
            continue

        try:
            result = generator(kotogram)
            if result:
                new_kotogram, error_type = result
                # Extract surface form for the new sentence
                new_surface = extract_all_surfaces(new_kotogram)
                # Make sure we actually changed something
                if new_surface != sentence:
                    new_id = f"{sentence_id}_agram_{error_code}"
                    results.append((new_id, 'jpn', new_surface, new_kotogram, f"{category}:{error_type}"))

                    if len(results) >= max_errors_per_sentence:
                        break
        except Exception:
            pass

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate agrammatic (grammatically incorrect) Japanese sentences"
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
        default="data/agrammatic_sentences.tsv",
        help="Output TSV file for generated examples"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help="Maximum number of input sentences to process"
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=3,
        help="Maximum number of error variants per sentence"
    )
    parser.add_argument(
        "--error-probability",
        type=float,
        default=0.3,
        help="Base probability of generating each error type (0.0-1.0)"
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
    parser.add_argument(
        "--include-kotogram",
        action="store_true",
        help="Include kotogram in output (4th column)"
    )
    parser.add_argument(
        "--include-error-type",
        action="store_true",
        help="Include error type in output (extra column)"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Import kotogram parser
    try:
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
        parser_instance = SudachiJapaneseParser()
    except ImportError:
        print("Error: Could not import SudachiJapaneseParser")
        print("Make sure kotogram is installed: pip install -e .")
        return

    generated = []
    processed = 0
    error_counts: Dict[str, int] = {}

    if args.verbose:
        print(f"Reading from {args.input}...")
        print(f"Error probability: {args.error_probability}")
        print(f"Max errors per sentence: {args.max_errors}")

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

                # Generate agrammatic variants
                variants = process_sentence(
                    kotogram,
                    sentence,
                    sentence_id,
                    max_errors_per_sentence=args.max_errors,
                    error_probability=args.error_probability,
                )

                for var in variants:
                    generated.append(var)
                    error_type = var[4]
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1

                processed += 1

                if args.verbose and processed % 5000 == 0:
                    print(f"Processed {processed} sentences, generated {len(generated)} agrammatic examples")

                if args.max_samples and processed >= args.max_samples:
                    break

            except Exception as e:
                if args.verbose:
                    print(f"Error processing {sentence_id}: {e}")
                continue

    if args.verbose:
        print(f"\nTotal processed: {processed} sentences")
        print(f"Total generated: {len(generated)} agrammatic examples")

    # Write output
    if args.verbose:
        print(f"Writing to {args.output}...")

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for new_id, lang, new_sentence, new_kotogram, error_type in generated:
            row = [new_id, lang, new_sentence]
            if args.include_kotogram:
                row.append(new_kotogram)
            if args.include_error_type:
                row.append(error_type)
            writer.writerow(row)

    if args.verbose:
        print("Done!")

        # Print statistics
        print(f"\nGenerated examples by error type:")

        # Group by category
        categories = {'beginner': [], 'intermediate': [], 'advanced': []}
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            category = error_type.split(':')[0]
            categories[category].append((error_type, count))

        for category in ['beginner', 'intermediate', 'advanced']:
            cat_total = sum(c for _, c in categories[category])
            print(f"\n  {category.upper()} ({cat_total} total):")
            for error_type, count in categories[category]:
                print(f"    {error_type}: {count}")


if __name__ == "__main__":
    main()
