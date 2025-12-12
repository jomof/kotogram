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
    python scripts/generate_agrammatic.py --input data/jpn_sentences.tsv --output data/jpn_agrammatic.tsv
    python scripts/generate_agrammatic.py --input data/jpn_sentences.tsv --output data/jpn_agrammatic.tsv --max-samples 1000
"""

import argparse
import csv
import random
import re
from typing import List, Tuple, Optional, Dict, Callable

from kotogram import split_kotogram, extract_token_features, kotogram_to_japanese


# ============================================================================
# KOTOGRAM TOKEN UTILITIES
# ============================================================================

def extract_surface(token: str) -> str:
    """Extract surface form from a kotogram token."""
    return extract_token_features(token)['surface']


def extract_pos(token: str) -> str:
    """Extract main POS from a kotogram token."""
    return extract_token_features(token)['pos']


def extract_conjugation_form(token: str) -> str:
    """Extract conjugation form from a kotogram token."""
    return extract_token_features(token)['conjugated_form']


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

    NOTE: This error type is DISABLED because swapping は/が almost always
    produces grammatically valid sentences (just with different nuance).
    Both particles are grammatically correct in most contexts.

    Keeping the function for reference but returning None.
    """
    # DISABLED - は/が swaps produce valid sentences
    return None


def error_particle_wo_omission(kotogram: str) -> Optional[Tuple[str, str]]:
    """Remove を particle from direct objects.

    NOTE: This error type is DISABLED because を omission is natural
    and grammatically acceptable in casual/spoken Japanese. It doesn't
    produce reliably ungrammatical sentences.
    """
    # DISABLED - を omission is acceptable in casual Japanese
    return None


def error_particle_ni_de_confusion(kotogram: str) -> Optional[Tuple[str, str]]:
    """Confuse に and で particles in clearly wrong contexts.

    Only swap in specific contexts where the result is ungrammatical:
    - に after time expressions -> で (wrong: 3時で会う)
    - で before いる/ある existence verbs -> に would be correct, but we want errors
      so we do the reverse: に before action verbs -> で

    Many に/で swaps produce valid sentences, so we're selective.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        if extract_pos(token) != 'prt':
            continue
        surface = extract_surface(token)

        # Case 1: に before existence verbs (いる/ある) - this is CORRECT
        # So we swap it to で which is WRONG for pure existence
        if surface == 'に' and i + 1 < len(tokens):
            next_surface = extract_surface(tokens[i + 1])
            next_pos = extract_pos(tokens[i + 1])
            # Check if followed by いる/ある (existence verbs)
            if next_pos == 'v' and next_surface in ['いる', 'い', 'ある', 'あり', 'あっ']:
                new_token = make_particle_token('で', 'case_particle')
                new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                return ''.join(new_tokens), 'ni_de_existence'

        # Case 2: で before destination verbs (行く/来る/帰る) - this is WRONG
        # に is correct for destinations. But we want to CREATE errors,
        # so find に + movement verb and swap to で
        if surface == 'に' and i + 1 < len(tokens):
            next_surface = extract_surface(tokens[i + 1])
            # Movement verbs that require に for destination
            movement_verbs = ['行', '来', '帰', '着', '届', '入', '戻']
            if any(next_surface.startswith(v) for v in movement_verbs):
                new_token = make_particle_token('で', 'case_particle')
                new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
                return ''.join(new_tokens), 'ni_de_destination'

    return None


def error_particle_he_ni_confusion(kotogram: str) -> Optional[Tuple[str, str]]:
    """Confuse へ and に for movement/direction.

    NOTE: This error type is DISABLED because へ and に are largely
    interchangeable for direction/destination. Swapping them almost
    never produces ungrammatical sentences.
    """
    # DISABLED - へ/に swaps produce valid sentences
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


def error_na_adjective_missing_na(kotogram: str) -> Optional[Tuple[str, str]]:
    """Remove な from な-adjective before noun (incorrect).

    Error: 静か部屋 (should be 静かな部屋)
    This is a clear grammatical error - な-adjectives require な before nouns.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Common な-adjectives (shape/state predicates that need な)
    na_adjectives = [
        '静か', '綺麗', '元気', '有名', '便利', '不便', '親切', '丁寧',
        '簡単', '複雑', '大切', '大事', '必要', '特別', '普通', '自由',
        '健康', '安全', '危険', '正直', '素敵', '立派', '真剣', '確か',
        '好き', '嫌い', '上手', '下手', '得意', '苦手', '暇', '無理',
    ]

    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        pos = extract_pos(token)

        # Check if this is a な-adjective (shp = shape/state predicate)
        if pos == 'shp' or surface in na_adjectives:
            # Check if followed by な + noun
            if i + 2 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                next_next_pos = extract_pos(tokens[i + 2])
                # If な followed by noun, remove the な
                if next_surface == 'な' and next_next_pos == 'n':
                    new_tokens = tokens[:i + 1] + tokens[i + 2:]
                    return ''.join(new_tokens), 'na_adj_missing_na'

    return None


def error_i_adjective_with_na(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add な after い-adjective before noun (incorrect).

    Error: 高いな山 (should be 高い山)
    い-adjectives directly modify nouns without な.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        surface = extract_surface(token)

        # Check if this is an い-adjective in attributive form
        if pos == 'adj' and 'adjective' in token:
            # Check if followed directly by a noun (correct pattern)
            if i + 1 < len(tokens):
                next_pos = extract_pos(tokens[i + 1])
                if next_pos == 'n':
                    # Insert な between adjective and noun (incorrect)
                    na_token = "⌈ˢなᵖauxv:auxv-da:conjunctive⌉"
                    new_tokens = tokens[:i + 1] + [na_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'i_adj_with_na'

    return None


def error_copula_after_verb(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add だ/です directly after verb (incorrect).

    Error: 食べるだ, 行くです (copula cannot follow verbs directly)
    Verbs already conjugate for politeness; adding copula is wrong.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        form = extract_conjugation_form(token)

        # Find verb in terminal form (sentence-ending)
        if pos == 'v' and form == 'terminal':
            # Check it's not already followed by auxiliary
            if i + 1 < len(tokens):
                next_pos = extract_pos(tokens[i + 1])
                if next_pos in ['auxv', 'prt']:
                    continue

            # Add だ after the verb (incorrect)
            da_token = make_auxv_token('だ', 'auxv-da', 'terminal')
            new_tokens = tokens[:i + 1] + [da_token] + tokens[i + 1:]
            return ''.join(new_tokens), 'copula_after_verb'

    return None


def error_dictionary_form_plus_masu(kotogram: str) -> Optional[Tuple[str, str]]:
    """Use dictionary form + ます (incorrect).

    Error: 食べるます (should be 食べます), 行くます (should be 行きます)
    ます attaches to the verb stem, not the dictionary form.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        form = extract_conjugation_form(token)

        # Find verb in terminal (dictionary) form
        if pos == 'v' and form == 'terminal':
            surface = extract_surface(token)
            # Skip する and 来る which have irregular stems
            if surface in ['する', '来る', 'くる']:
                continue

            # Add ます after dictionary form (incorrect)
            masu_token = make_auxv_token('ます', 'auxv-masu', 'terminal')
            new_tokens = tokens[:i + 1] + [masu_token] + tokens[i + 1:]
            return ''.join(new_tokens), 'dict_form_plus_masu'

    return None


def error_particle_after_verb(kotogram: str) -> Optional[Tuple[str, str]]:
    """Put case particle directly after verb (incorrect).

    Error: 食べたを見た (particle を after conjugated verb)
    Case particles attach to nouns, not verbs.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        pos = extract_pos(token)

        # Find verb
        if pos == 'v':
            # Check if followed by another element (not end of sentence)
            if i + 1 < len(tokens):
                next_pos = extract_pos(tokens[i + 1])
                # If not already followed by particle, insert one incorrectly
                if next_pos not in ['prt', 'auxv', 'auxs']:
                    # Insert を after the verb (incorrect)
                    wo_token = make_particle_token('を', 'case_particle')
                    new_tokens = tokens[:i + 1] + [wo_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'particle_after_verb'

    return None


def error_quote_missing_to(kotogram: str) -> Optional[Tuple[str, str]]:
    """Remove と after quotation (incorrect).

    Error: 「行く」言った (should be 「行く」と言った)
    Quotations require と before verbs like 言う, 思う, etc.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Quote verbs that require と
    quote_verbs = ['言', '思', '考え', '聞', '答え', '話', '書']

    for i, token in enumerate(tokens):
        surface = extract_surface(token)

        # Find と particle
        if surface == 'と' and extract_pos(token) == 'prt':
            # Check if followed by a quote verb
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if any(next_surface.startswith(qv) for qv in quote_verbs):
                    # Check if preceded by quote (」)
                    if i > 0:
                        prev_surface = extract_surface(tokens[i - 1])
                        if '」' in prev_surface or extract_pos(tokens[i - 1]) == 'auxs':
                            # Remove the と (incorrect)
                            new_tokens = tokens[:i] + tokens[i + 1:]
                            return ''.join(new_tokens), 'quote_missing_to'

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
    Only targets actual conditional particles, not parts of other words.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find conditional markers - must be particles, not parts of verbs/auxiliaries
    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        pos = extract_pos(token)

        # Only target actual particles
        if pos != 'prt':
            continue

        if surface == 'たら':
            # Replace with なら (often incorrect context)
            new_token = make_particle_token('なら', 'binding_particle')
            new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
            return ''.join(new_tokens), 'tara_nara_swap'
        elif surface == 'なら':
            # Replace with たら
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

    NOTE: This error type is DISABLED because swapping ね/よ produces
    pragmatically odd but grammatically valid sentences. Both particles
    are grammatically correct; they just convey different speaker attitudes.
    """
    # DISABLED - ね/よ swaps are grammatically valid
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

    Error: Adding のだ after polite forms (ます/です) where it's ungrammatical.
    E.g., 食べますのだ, きれいですのだ (wrong)

    Note: のだ after plain forms is grammatical (explanatory), so we only
    target polite forms where のだ cannot follow.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find polite endings (ます/です) and add inappropriate のだ after them
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        surface = extract_surface(token)

        # Only target polite forms where のだ is ungrammatical
        if 'auxv-masu' in token and surface in ['ます', 'ました']:
            # Add のだ after ます/ました (ungrammatical)
            no_token = make_particle_token('の', 'pre_noun_particle')
            da_token = make_auxv_token('だ', 'auxv-da', 'terminal')
            new_tokens = tokens[:i+1] + [no_token, da_token] + tokens[i+1:]
            return ''.join(new_tokens), 'noda_after_masu'

        if 'auxv-desu' in token and surface in ['です', 'でした']:
            # Add のだ after です/でした (ungrammatical)
            no_token = make_particle_token('の', 'pre_noun_particle')
            da_token = make_auxv_token('だ', 'auxv-da', 'terminal')
            new_tokens = tokens[:i+1] + [no_token, da_token] + tokens[i+1:]
            return ''.join(new_tokens), 'noda_after_desu'

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

    NOTE: This error type is DISABLED because adding 私は is grammatically
    valid, just stylistically redundant. It doesn't produce ungrammatical
    sentences.
    """
    # DISABLED - Adding pronouns is grammatically valid
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


def error_desu_after_dictionary_verb(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add です after dictionary form verb (incorrect).

    Error: 行くです, 食べるです (です cannot follow dictionary form verbs)
    Verbs conjugate for politeness; adding です after dictionary form is wrong.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        form = extract_conjugation_form(token)

        # Find verb in terminal (dictionary) form at sentence end or before punctuation
        if pos == 'v' and form == 'terminal':
            # Check it's not already followed by auxiliary
            if i + 1 < len(tokens):
                next_pos = extract_pos(tokens[i + 1])
                next_surface = extract_surface(tokens[i + 1])
                # Only add if followed by punctuation or sentence-final particle
                if next_pos == 'auxs' or next_surface in ['。', '！', '？', '、']:
                    desu_token = make_auxv_token('です', 'auxv-desu', 'terminal')
                    new_tokens = tokens[:i + 1] + [desu_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'desu_after_dict_verb'
            elif i == len(tokens) - 1:
                # Verb is last token
                desu_token = make_auxv_token('です', 'auxv-desu', 'terminal')
                new_tokens = tokens + [desu_token]
                return ''.join(new_tokens), 'desu_after_dict_verb'

    return None


def error_nai_desu_wrong(kotogram: str) -> Optional[Tuple[str, str]]:
    """Create incorrect negative polite form: ないです instead of ません pattern errors.

    Error: 食べますない (completely wrong order - ます before ない)
    Note: 食べないです is actually acceptable in modern Japanese, so we create
    worse errors like ますない.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find ません pattern and corrupt it
    for i, token in enumerate(tokens):
        if 'auxv-masu' in token:
            surface = extract_surface(token)
            if surface == 'ませ' or surface == 'ません':
                # Check if followed by ん (ません pattern)
                if i + 1 < len(tokens) and extract_surface(tokens[i + 1]) == 'ん':
                    # Replace ません with ますない (wrong)
                    masu_token = make_auxv_token('ます', 'auxv-masu', 'terminal')
                    nai_token = make_auxv_token('ない', 'auxv-nai', 'terminal')
                    new_tokens = tokens[:i] + [masu_token, nai_token] + tokens[i + 2:]
                    return ''.join(new_tokens), 'masu_nai_order'

    return None


def error_double_past(kotogram: str) -> Optional[Tuple[str, str]]:
    """Create double past tense marking (incorrect).

    Error: 食べましたた, 行ったた (double past marker)
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        # Find past tense た
        if 'auxv-ta' in token:
            surface = extract_surface(token)
            if surface in ['た', 'だ']:
                # Add another た after it
                ta_token = make_auxv_token('た', 'auxv-ta', 'terminal')
                new_tokens = tokens[:i + 1] + [ta_token] + tokens[i + 1:]
                return ''.join(new_tokens), 'double_past'

    return None


def error_wa_after_verb(kotogram: str) -> Optional[Tuple[str, str]]:
    """Put topic particle は directly after verb (incorrect).

    Error: 食べるは, 行ったは (topic marker cannot follow verb directly)
    Topic particle は marks nouns, not verbs.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        form = extract_conjugation_form(token)

        # Find verb in terminal form
        if pos == 'v' and form == 'terminal':
            # Check if followed by something other than は already
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if next_surface != 'は':
                    # Insert は after verb (incorrect)
                    wa_token = make_particle_token('は', 'binding_particle')
                    new_tokens = tokens[:i + 1] + [wa_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'wa_after_verb'

    return None


def error_i_adj_negative_wrong(kotogram: str) -> Optional[Tuple[str, str]]:
    """Create incorrect い-adjective negative form.

    Error: 高いくない (should be 高くない - い should change to く)
    Error: 高いない (missing く entirely)
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find い-adjective followed by negative
    for i, token in enumerate(tokens):
        pos = extract_pos(token)
        surface = extract_surface(token)

        if pos == 'adj' and surface.endswith('く'):
            # This is correct form (高く), check if followed by ない
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if next_surface in ['ない', 'なかっ', 'なく']:
                    # Change く back to い to create error (高いない)
                    wrong_surface = surface[:-1] + 'い'
                    new_token = re.sub(r'ˢ[^ᵖ]+ᵖ', f'ˢ{wrong_surface}ᵖ', token)
                    new_tokens = tokens[:i] + [new_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'i_adj_neg_wrong'

    return None


def error_na_adj_negative_wrong(kotogram: str) -> Optional[Tuple[str, str]]:
    """Create incorrect な-adjective negative form.

    Error: 静かないです (should be 静かじゃないです or 静かではないです)
    Missing じゃ/では before ない.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Common な-adjectives
    na_adjectives = [
        '静か', '綺麗', '元気', '有名', '便利', '親切', '丁寧',
        '簡単', '複雑', '大切', '大事', '必要', '特別', '普通',
        '好き', '嫌い', '上手', '下手', '得意', '苦手',
    ]

    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        pos = extract_pos(token)

        # Check if this is a な-adjective
        if pos == 'shp' or surface in na_adjectives:
            # Check if followed by じゃ/では + ない
            if i + 2 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                next_next_surface = extract_surface(tokens[i + 2])
                if next_surface in ['じゃ', 'では', 'で'] and next_next_surface in ['ない', 'なかっ', 'は']:
                    # Remove じゃ/では to create error (静かない)
                    new_tokens = tokens[:i + 1] + tokens[i + 2:]
                    return ''.join(new_tokens), 'na_adj_neg_missing_ja'

    return None


def error_te_de_confusion(kotogram: str) -> Optional[Tuple[str, str]]:
    """Confuse て and で in て-form (incorrect voicing).

    Error: 書いで (should be 書いて)
    Error: 読んて (should be 読んで)

    Only targets て/で that are conjunctive particles after verbs,
    not parts of words like なんで, そこで, etc.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        pos = extract_pos(token)

        # Only target conjunctive particles after verb stems
        if pos != 'prt':
            continue

        # Skip if preceded by interrogatives or other standalone words
        if i > 0:
            prev_surface = extract_surface(tokens[i - 1])
            prev_pos = extract_pos(tokens[i - 1])

            # Skip if preceded by pronoun/adverb (like なん, そこ, etc.)
            if prev_pos in ['pron', 'adv', 'n']:
                continue

            # Only process after verb conjunctive forms
            if prev_pos != 'v':
                continue

        if surface == 'て':
            # Check if preceded by ん (should be で after ん)
            if i > 0:
                prev_surface = extract_surface(tokens[i - 1])
                if not prev_surface.endswith('ん'):
                    # Change て to で (wrong for most verbs)
                    new_token = re.sub(r'ˢてᵖ', 'ˢでᵖ', token)
                    new_tokens = tokens[:i] + [new_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'te_de_wrong_voicing'
        elif surface == 'で':
            # Check if preceded by ん (should be で after ん, so this is correct)
            if i > 0:
                prev_surface = extract_surface(tokens[i - 1])
                if prev_surface.endswith('ん'):
                    # Change で to て (wrong after ん)
                    new_token = re.sub(r'ˢでᵖ', 'ˢてᵖ', token)
                    new_tokens = tokens[:i] + [new_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'de_te_wrong_voicing'

    return None


def error_ru_ta_confusion(kotogram: str) -> Optional[Tuple[str, str]]:
    """Create る + た error (incorrect past formation).

    Error: 食べるた (should be 食べた - dictionary form + た is wrong)
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find past tense patterns
    for i, token in enumerate(tokens):
        if 'auxv-ta' in token and i > 0:
            prev_token = tokens[i - 1]
            prev_pos = extract_pos(prev_token)
            prev_form = extract_conjugation_form(prev_token)

            # Check if verb is in conjunctive form (correct for た attachment)
            if prev_pos == 'v' and prev_form in ['conjunctive', 'conjunctive-geminate']:
                # Change verb to terminal form (dictionary) to create error
                prev_surface = extract_surface(prev_token)
                # Simple heuristic: add る for ichidan verbs
                if prev_surface.endswith(('べ', 'め', 'ね', 'け', 'せ', 'て', 'え', 'れ', 'げ', 'で', 'ぜ', 'へ', 'ぺ', 'み', 'き', 'し', 'ち', 'に', 'ひ', 'い', 'り', 'ぎ', 'じ', 'び', 'ぴ')):
                    wrong_surface = prev_surface + 'る'
                    new_token = re.sub(r'ˢ[^ᵖ]+ᵖ', f'ˢ{wrong_surface}ᵖ', prev_token)
                    new_tokens = tokens[:i - 1] + [new_token] + tokens[i:]
                    return ''.join(new_tokens), 'dict_form_plus_ta'

    return None


def error_polite_form_as_modifier(kotogram: str) -> Optional[Tuple[str, str]]:
    """Use polite form as noun modifier (incorrect).

    Error: 駐車場がありませんホテル (should be 駐車場がないホテル)
    Error: 食べましたレストラン (should be 食べたレストラン)

    In Japanese, only plain forms can modify nouns. Polite forms cannot.
    This is a common error for learners.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Common nouns that can be modified
    modifiable_nouns = [
        '人', 'もの', 'こと', 'ところ', '時', '方', '店', '家', '車', '本',
        '会社', '学校', '国', '町', '駅', '部屋', 'ホテル', 'レストラン',
        '映画', '料理', '仕事', '話', '問題', '理由', '場所', '物',
    ]

    # Find plain form that could be changed to polite
    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        pos = extract_pos(token)

        # Check if this is a plain negative (ない) followed by a noun
        if surface == 'ない' and 'auxv' in token:
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if next_surface in modifiable_nouns:
                    # Change ない to ありません (wrong for modifier)
                    # We need to insert あり + ませ + ん
                    ari_token = "⌈ˢありᵖv:existence:conjunctive⌉"
                    mase_token = make_auxv_token('ませ', 'auxv-masu', 'imperfective')
                    n_token = "⌈ˢんᵖauxv:auxv-nu:terminal⌉"
                    new_tokens = tokens[:i] + [ari_token, mase_token, n_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'polite_modifier'

        # Check for past plain (た/だ) followed by noun
        if surface in ['た', 'だ'] and 'auxv-ta' in token:
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if next_surface in modifiable_nouns:
                    # Change た to ました (wrong for modifier)
                    mashi_token = make_auxv_token('まし', 'auxv-masu', 'conjunctive')
                    ta_token = make_auxv_token('た', 'auxv-ta', 'terminal')
                    new_tokens = tokens[:i] + [mashi_token, ta_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'polite_past_modifier'

    return None


def error_past_i_adj_plus_da(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add だ after past い-adjective (incorrect).

    Error: 楽しかっただね (should be 楽しかったね)
    Error: よかっただよ (should be よかったよ)

    い-adjectives already conjugate for tense, so adding だ is redundant/wrong.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Sentence-final particles that might follow
    final_particles = ['ね', 'よ', 'な', 'わ', 'さ', 'ぞ', 'ぜ', 'かな', 'かしら', 'なあ']

    for i, token in enumerate(tokens):
        surface = extract_surface(token)

        # Look for た (past auxiliary) after い-adjective stem
        if 'auxv-ta' in token and surface == 'た':
            if i > 0:
                prev_token = tokens[i - 1]
                prev_surface = extract_surface(prev_token)
                prev_pos = extract_pos(prev_token)

                # Check if preceded by い-adjective (POS starts with 'adj')
                # The adjective surface ends in かっ for past form
                if prev_pos.startswith('adj') and prev_surface.endswith('かっ'):
                    # Check if followed by sentence-final particle or punctuation
                    if i + 1 < len(tokens):
                        next_surface = extract_surface(tokens[i + 1])
                        if next_surface in final_particles or next_surface in ['。', '！', '？', '.']:
                            # Insert だ between た and particle (wrong)
                            da_token = "⌈ˢだᵖauxv:auxv-da:terminal⌉"
                            new_tokens = tokens[:i + 1] + [da_token] + tokens[i + 1:]
                            return ''.join(new_tokens), 'past_i_adj_da'

    return None


def error_nai_plus_da(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add だ after ない auxiliary (incorrect).

    Error: 食べないだよ (should be 食べないよ)
    Error: 新鮮ではないだよ (should be 新鮮ではないよ)

    ない is an い-adjective-like auxiliary, so adding だ is incorrect.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    final_particles = ['ね', 'よ', 'な', 'わ', 'さ', 'ぞ', 'ぜ', 'かな', 'かしら', 'なあ']

    for i, token in enumerate(tokens):
        surface = extract_surface(token)

        # Look for ない (negative auxiliary) in terminal form
        if 'auxv-nai' in token and surface == 'ない':
            # Check if followed by sentence-final particle
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if next_surface in final_particles or next_surface in ['。', '！', '？', '.']:
                    # Insert だ between ない and particle (wrong)
                    da_token = "⌈ˢだᵖauxv:auxv-da:terminal⌉"
                    new_tokens = tokens[:i + 1] + [da_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'nai_da'

    return None


def error_tai_plus_da(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add だ after たい auxiliary (incorrect).

    Error: 食べたいだよ (should be 食べたいよ)
    Error: 買いたいだな (should be 買いたいな)

    たい is an い-adjective-like auxiliary, so adding だ is incorrect.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    final_particles = ['ね', 'よ', 'な', 'わ', 'さ', 'ぞ', 'ぜ', 'かな', 'かしら', 'なあ']

    for i, token in enumerate(tokens):
        surface = extract_surface(token)

        # Look for たい (desiderative auxiliary) in terminal form
        if 'auxv-tai' in token and surface == 'たい':
            # Check if followed by sentence-final particle
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if next_surface in final_particles or next_surface in ['。', '！', '？', '.']:
                    # Insert だ between たい and particle (wrong)
                    da_token = "⌈ˢだᵖauxv:auxv-da:terminal⌉"
                    new_tokens = tokens[:i + 1] + [da_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'tai_da'

    return None


def error_i_adj_terminal_plus_da(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add だ after terminal い-adjective (incorrect).

    Error: 危ないだよ (should be 危ないよ)
    Error: 少ないだね (should be 少ないね)

    い-adjectives in terminal form don't take だ.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    final_particles = ['ね', 'よ', 'な', 'わ', 'さ', 'ぞ', 'ぜ', 'かな', 'かしら', 'なあ']

    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        pos = extract_pos(token)

        # Look for terminal い-adjective (ends in い, POS is adj, form is terminal)
        if pos.startswith('adj') and 'terminal' in token and surface.endswith('い'):
            # Exclude らしい which has its own generator
            if surface == 'らしい':
                continue
            # Check if followed by sentence-final particle
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if next_surface in final_particles or next_surface in ['。', '！', '？', '.']:
                    # Insert だ between adjective and particle (wrong)
                    da_token = "⌈ˢだᵖauxv:auxv-da:terminal⌉"
                    new_tokens = tokens[:i + 1] + [da_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'i_adj_da'

    return None


def error_rashii_plus_da(kotogram: str) -> Optional[Tuple[str, str]]:
    """Add だ after らしい (incorrect).

    Error: 美味しいらしいだよ (should be 美味しいらしいよ)

    らしい is an い-adjective-like auxiliary, so adding だ is incorrect.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    final_particles = ['ね', 'よ', 'な', 'わ', 'さ', 'ぞ', 'ぜ']

    for i, token in enumerate(tokens):
        surface = extract_surface(token)

        if surface == 'らしい':
            # Check if followed by sentence-final particle
            if i + 1 < len(tokens):
                next_surface = extract_surface(tokens[i + 1])
                if next_surface in final_particles or next_surface in ['。', '！', '？']:
                    # Insert だ between らしい and particle (wrong)
                    da_token = "⌈ˢだᵖauxv:auxv-da:terminal⌉"
                    new_tokens = tokens[:i + 1] + [da_token] + tokens[i + 1:]
                    return ''.join(new_tokens), 'rashii_da'

    return None


def error_doko_instead_of_tokoro(kotogram: str) -> Optional[Tuple[str, str]]:
    """Replace ところ with どこ after relative clauses (incorrect).

    Error: 残っているどこもない (should be 残っているところもない)

    どこ is an interrogative pronoun, not a relative pronoun.
    ところ should be used for "place where X" constructions.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    for i, token in enumerate(tokens):
        surface = extract_surface(token)

        # Look for ところ after a verb
        if surface == 'ところ' and i > 0:
            prev_token = tokens[i - 1]
            # Check if preceded by a verb form (いる, た, etc.) indicating relative clause
            prev_surface = extract_surface(prev_token)
            if prev_surface in ['いる', 'た', 'ある', 'ない', 'できる', 'する']:
                # Replace ところ with どこ (wrong)
                doko_token = "⌈ˢどこᵖpron:pron-interrogative:⌉"
                new_tokens = tokens[:i] + [doko_token] + tokens[i + 1:]
                return ''.join(new_tokens), 'doko_tokoro_confusion'

    return None


def error_wrong_verb_base(kotogram: str) -> Optional[Tuple[str, str]]:
    """Use wrong verb conjugation base.

    Error: Add る to ます-stem creating invalid form like 食べまする, 行きまする.
    This creates clearly ungrammatical verb forms.
    """
    tokens = split_kotogram(kotogram)
    if not tokens:
        return None

    # Find ます in terminal form and add る after it (ungrammatical)
    for i, token in enumerate(tokens):
        surface = extract_surface(token)
        if 'auxv-masu' in token and surface == 'ます':
            # Add る after ます to create invalid form like ますร
            # Change ます to まする (ungrammatical)
            new_token = re.sub(r'ˢますᵖ', 'ˢまするᵖ', token)
            new_tokens = tokens[:i] + [new_token] + tokens[i+1:]
            return ''.join(new_tokens), 'masu_plus_ru'

    return None


# ============================================================================
# MAIN PROCESSING
# ============================================================================

# All error generators with their categories and weights
# NOTE: Several generators have been DISABLED because they produce
# grammatically valid sentences. Only truly ungrammatical errors remain.
ERROR_GENERATORS: List[Tuple[str, str, Callable, float]] = [
    # (error_code, category, function, probability_weight)

    # Beginner level errors - RELIABLE (produce ungrammatical sentences)
    # DISABLED: ('b_wa_ga', ...) - は/が swaps produce valid sentences
    # DISABLED: ('b_wo_omit', ...) - を omission is valid in casual speech
    ('b_ni_de', 'beginner', error_particle_ni_de_confusion, 1.0),  # Only in specific contexts now
    # DISABLED: ('b_he_ni', ...) - へ/に are interchangeable
    ('b_da_i_adj', 'beginner', error_copula_da_after_i_adjective, 1.0),  # Clear error: 高いだ
    ('b_desu_past', 'beginner', error_desu_after_plain_past, 1.0),  # Clear error: 食べたです
    ('b_double_prt', 'beginner', error_double_particle, 1.0),  # Clear error: をを
    ('b_na_missing', 'beginner', error_na_adjective_missing_na, 1.0),  # Clear error: 静か部屋
    ('b_i_adj_na', 'beginner', error_i_adjective_with_na, 1.0),  # Clear error: 高いな山
    ('b_copula_verb', 'beginner', error_copula_after_verb, 1.0),  # Clear error: 食べるだ
    ('b_dict_masu', 'beginner', error_dictionary_form_plus_masu, 1.0),  # Clear error: 食べるます
    ('b_desu_dict', 'beginner', error_desu_after_dictionary_verb, 1.0),  # Clear error: 行くです
    ('b_double_past', 'beginner', error_double_past, 0.8),  # Clear error: 食べたた
    ('b_wa_verb', 'beginner', error_wa_after_verb, 0.8),  # Clear error: 食べるは

    # Intermediate level errors
    ('i_masu_sub', 'intermediate', error_formality_mixing_subordinate, 0.8),
    ('i_trans', 'intermediate', error_transitivity_confusion, 0.6),
    ('i_cond', 'intermediate', error_conditional_mixing, 0.6),
    ('i_pass_pot', 'intermediate', error_passive_potential_confusion, 0.5),
    ('i_prt_verb', 'intermediate', error_particle_after_verb, 0.6),  # Clear error: 食べたを
    ('i_quote_to', 'intermediate', error_quote_missing_to, 0.8),  # Clear error: 「行く」言った
    ('i_masu_nai', 'intermediate', error_nai_desu_wrong, 0.7),  # Clear error: 食べますない
    ('i_i_adj_neg', 'intermediate', error_i_adj_negative_wrong, 0.8),  # Clear error: 高いない
    ('i_na_adj_neg', 'intermediate', error_na_adj_negative_wrong, 0.7),  # Clear error: 静かない
    ('i_te_de', 'intermediate', error_te_de_confusion, 0.6),  # Clear error: 書いで, 読んて
    ('i_dict_ta', 'intermediate', error_ru_ta_confusion, 0.6),  # Clear error: 食べるた
    ('i_polite_mod', 'intermediate', error_polite_form_as_modifier, 0.8),  # Clear error: ありませんホテル
    ('i_past_adj_da', 'intermediate', error_past_i_adj_plus_da, 0.7),  # Clear error: 楽しかっただね
    ('i_rashii_da', 'intermediate', error_rashii_plus_da, 0.6),  # Clear error: らしいだよ
    ('i_nai_da', 'intermediate', error_nai_plus_da, 0.7),  # Clear error: 食べないだよ
    ('i_tai_da', 'intermediate', error_tai_plus_da, 0.7),  # Clear error: 食べたいだよ
    ('i_i_adj_da', 'intermediate', error_i_adj_terminal_plus_da, 0.7),  # Clear error: 危ないだよ
    ('i_doko_tokoro', 'intermediate', error_doko_instead_of_tokoro, 0.7),  # Clear error: いるどこも

    # Advanced level errors
    # DISABLED: ('a_prag_prt', ...) - ね/よ swaps are grammatically valid
    ('a_honor', 'advanced', error_honorific_mixing, 0.5),
    ('a_noda', 'advanced', error_noda_misuse, 0.6),
    ('a_da_desu', 'advanced', error_da_after_noun_polite, 0.8),  # Clear error: 先生だです
    # DISABLED: ('a_over_spec', ...) - Adding pronouns is valid
    ('a_te_form', 'advanced', error_te_form_wrong, 0.5),
    ('a_verb_base', 'advanced', error_wrong_verb_base, 0.5),
]


def process_sentence(
    kotogram: str,
    sentence: str,
    sentence_id: str,
    max_errors_per_sentence: int = 3,
    error_probability: float = 0.3,
) -> List[Tuple[str, str, str, str, str, str]]:
    """Process a single sentence and generate agrammatic variants.

    Returns:
        List of (new_id, 'jpn', new_sentence, source_id, new_kotogram, error_type) tuples
        source_id is the ID of the original sentence this was derived from
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
                new_surface = kotogram_to_japanese(new_kotogram, spaces=False)
                # Make sure we actually changed something
                if new_surface != sentence:
                    # ID will be assigned later as a simple number
                    # source_id tracks which original sentence this came from
                    results.append((None, 'jpn', new_surface, sentence_id, new_kotogram, f"{category}:{error_type}"))

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
        default="data/jpn_agrammatic.tsv",
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
        default=100,
        help="Maximum number of error variants per sentence (default: 100 = effectively unlimited)"
    )
    parser.add_argument(
        "--error-probability",
        type=float,
        default=1.0,
        help="Base probability of generating each error type (0.0-1.0, default: 1.0 = try all)"
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
        help="Include error type in output (extra column). Default: False (no hints about grammar errors)"
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        default=True,
        help="Disable balancing (default: True = keep all generated examples without capping)"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Enable balancing to cap examples per error type"
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=45000,
        help="Maximum examples per error type when balancing (default: 45000)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated sentences against grammaticality model. Stop on first false positive."
    )
    parser.add_argument(
        "--disable-generator",
        type=str,
        default="",
        help="Comma-separated list of generator codes to disable (e.g., 'b_desu_past,a_verb_base,a_noda')"
    )
    parser.add_argument(
        "--include-source-id",
        action="store_true",
        help="Include source sentence ID in output (4th column before kotogram/error-type). "
             "Required for source-based train/test splitting."
    )

    args = parser.parse_args()

    # Parse disabled generators
    disabled_generators: set = set()
    if args.disable_generator:
        disabled_generators = set(g.strip() for g in args.disable_generator.split(','))
        print(f"Disabled generators: {disabled_generators}")
    random.seed(args.seed)

    # Import kotogram parser
    try:
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
        parser_instance = SudachiJapaneseParser()
    except ImportError:
        print("Error: Could not import SudachiJapaneseParser")
        print("Make sure kotogram is installed: pip install -e .")
        return

    # Validation mode: check generated sentences against grammaticality model
    if args.validate:
        from kotogram.analysis import grammaticality
        import inspect

        print("Validation mode: checking generated sentences against grammaticality model...")
        print(f"Reading from {args.input}...")

        # Load known model errors to skip
        model_errors_path = "data/jpn_model_errors.tsv"
        known_model_errors: set = set()
        try:
            with open(model_errors_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 3:
                        known_model_errors.add(row[2])  # sentence is 3rd column
            print(f"Loaded {len(known_model_errors)} known model errors to skip")
        except FileNotFoundError:
            print(f"No known model errors file found at {model_errors_path}")

        processed = 0
        checked = 0
        false_positives = 0
        skipped = 0

        with open(args.input, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                if len(row) < 3:
                    continue

                sentence_id, lang, sentence = row[0], row[1], row[2]

                if lang != 'jpn':
                    continue

                try:
                    kotogram = parser_instance.japanese_to_kotogram(sentence)

                    # Process each error generator individually to track which one produced the error
                    for error_code, category, generator, weight in ERROR_GENERATORS:
                        # Skip disabled generators
                        if error_code in disabled_generators:
                            continue

                        try:
                            result = generator(kotogram)
                            if result:
                                new_kotogram, error_type = result
                                new_surface = kotogram_to_japanese(new_kotogram, spaces=False)

                                if new_surface != sentence:
                                    # Skip known model errors
                                    if new_surface in known_model_errors:
                                        skipped += 1
                                        continue

                                    checked += 1
                                    # Check if model thinks it's grammatic (false positive)
                                    is_grammatic = grammaticality(new_kotogram, use_model=True)

                                    if is_grammatic:
                                        false_positives += 1
                                        accuracy = 100 * (checked - false_positives) / checked if checked > 0 else 0

                                        # Found a false positive - print details and stop
                                        print("\n" + "=" * 70)
                                        print("FALSE POSITIVE DETECTED")
                                        print("=" * 70)
                                        print(f"Original sentence ID: {sentence_id}")
                                        print(f"Original sentence:    {sentence}")
                                        print(f"Generated sentence:   {new_surface}")
                                        print(f"Error type:           {category}:{error_type}")
                                        print(f"Error code:           {error_code}")
                                        print(f"Generator function:   {generator.__name__}")
                                        print(f"Kotogram:             {new_kotogram[:100]}...")
                                        print()

                                        # Get source location of the generator function
                                        try:
                                            source_file = inspect.getfile(generator)
                                            source_lines, start_line = inspect.getsourcelines(generator)
                                            print(f"Source location:      {source_file}:{start_line}")
                                            print()
                                            print("Generator function source (first 30 lines):")
                                            print("-" * 70)
                                            for i, line in enumerate(source_lines[:30]):
                                                print(f"{start_line + i:4d}: {line.rstrip()}")
                                            if len(source_lines) > 30:
                                                print(f"      ... ({len(source_lines) - 30} more lines)")
                                        except Exception as e:
                                            print(f"Could not get source: {e}")

                                        print("=" * 70)
                                        print(f"\nStatistics:")
                                        print(f"  Processed sentences:  {processed}")
                                        print(f"  Checked examples:     {checked}")
                                        print(f"  Skipped (known):      {skipped}")
                                        print(f"  False positives:      {false_positives}")
                                        print(f"  Model accuracy:       {accuracy:.2f}%")
                                        print("\nStopping for debugging.")
                                        return

                        except Exception:
                            pass

                    processed += 1

                    if processed % 1000 == 0:
                        accuracy = 100 * (checked - false_positives) / checked if checked > 0 else 0
                        print(f"Processed {processed} sentences, checked {checked}, false positives: {false_positives} ({accuracy:.2f}% accuracy)...")

                    if args.max_samples and processed >= args.max_samples:
                        break

                except Exception as e:
                    if args.verbose:
                        print(f"Error processing {sentence_id}: {e}")
                    continue

        accuracy = 100 * (checked - false_positives) / checked if checked > 0 else 0
        print(f"\nValidation complete!")
        print(f"  Processed sentences:  {processed}")
        print(f"  Checked examples:     {checked}")
        print(f"  Skipped (known):      {skipped}")
        print(f"  False positives:      {false_positives}")
        print(f"  Model accuracy:       {accuracy:.2f}%")
        if false_positives == 0:
            print("\nNo false positives found - all generated sentences were correctly identified as agrammatic.")
        return

    # Normal generation mode
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
                    error_type = var[5]  # (id, lang, sentence, source_id, kotogram, error_type)
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

    # Balance error types only if --balance is specified (disabled by default)
    if args.balance:
        if args.verbose:
            print(f"\nBalancing error types (max {args.max_per_type} per type)...")

        # Group by error type (from the 6th element, index 5)
        by_error_code: Dict[str, List] = {}
        for item in generated:
            # Extract error code from error_type like "beginner:double_past" -> "b_double_past"
            error_type = item[5]  # (id, lang, sentence, source_id, kotogram, error_type)
            category, error_name = error_type.split(':')
            error_code = f"{category[0]}_{error_name}"
            if error_code not in by_error_code:
                by_error_code[error_code] = []
            by_error_code[error_code].append(item)

        # Cap each type and rebuild the list
        balanced = []
        for error_code, items in by_error_code.items():
            if len(items) > args.max_per_type:
                # Randomly sample to cap
                sampled = random.sample(items, args.max_per_type)
                balanced.extend(sampled)
                if args.verbose:
                    print(f"  {error_code}: {len(items)} -> {args.max_per_type}")
            else:
                balanced.extend(items)
                if args.verbose and len(items) < args.max_per_type:
                    print(f"  {error_code}: {len(items)} (kept all)")

        # Shuffle to mix error types
        random.shuffle(balanced)
        generated = balanced

        # Recalculate error counts
        error_counts = {}
        for item in generated:
            error_type = item[5]  # (id, lang, sentence, source_id, kotogram, error_type)
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        if args.verbose:
            print(f"\nAfter balancing: {len(generated)} examples")

    # Assign sequential numeric IDs (keep source_id intact)
    # Tuple: (id, lang, sentence, source_id, kotogram, error_type)
    for i, item in enumerate(generated):
        generated[i] = (str(i + 1), item[1], item[2], item[3], item[4], item[5])

    # Write output
    if args.verbose:
        print(f"Writing to {args.output}...")

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for new_id, lang, new_sentence, source_id, new_kotogram, error_type in generated:
            row = [new_id, lang, new_sentence]
            if args.include_source_id:
                row.append(source_id)
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
