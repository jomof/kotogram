"""Module for augmenting Japanese sentences with grammatical variations.

This module provides an extensible framework for generating variations of Japanese
sentences (e.g. changing formality, dropping topics, swapping pronouns) and verifying
their grammaticality using a neural model.
"""

from abc import ABC, abstractmethod
from typing import Set, Tuple, List, Optional, Dict, Union, Any
import time
from itertools import product
from kotogram.kotogram import split_kotogram, extract_token_features
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from dataclasses import asdict

# Type alias for tokens (either surface string or feature dict wrapper)
# Forward declaration issue? Just use class name strings or object
AugmentationToken = Union[str, 'Token']

class Token:
    """Hashable wrapper for token features."""
    def __init__(self, surface: str, features: Optional[Dict[str, str]] = None):
        self.surface = surface
        self.features = features or {}
        self._hash = hash((surface, tuple(sorted(self.features.items()))))
        
    def __hash__(self) -> int:
        return self._hash
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.surface == other
        if isinstance(other, Token):
            return self.surface == other.surface and self.features == other.features
        return False
        
    def __repr__(self) -> str:
        return f"Token({self.surface}, {self.features})"
        
    def get(self, key: str, default: Any = None) -> Any:
        if key == 'surface':
            return self.surface
        return self.features.get(key, default)

    @property
    def reading(self) -> str:
        """Returns the phonetic reading in Hiragana, or surface if not available."""
        r = self.get('reading')
        if not r:
            return self.surface
        # Convert Katakana to Hiragana
        return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in r)

# Constants and Patterns
FIRST_PERSON_PRONOUNS = {'私', '僕', '俺', 'わたし', 'ぼく', 'おれ', 'あたし', 'わたくし'}
SECOND_PERSON_PRONOUNS = {'あなた', '君', 'お前', 'あんた', '貴様'}
THIRD_PERSON_PRONOUNS = {'彼', '彼女', 'あいつ', 'こいつ', 'そいつ', '奴'}

SENTENCE_FINAL_PARTICLES = {'よ', 'ね', 'わ', 'ぞ', 'ぜ'}

LOANWORD_VARIANTS = {
    'コンピューター': 'コンピュータ',
    'センター': 'センタ',
    'エレベーター': 'エレベータ',
    'パーティー': 'パーティ',
    'マネージャー': 'マネージャ',
    'ユーザー': 'ユーザ',
}

COPULA_PATTERNS = [
    # Direct sentence endings
    (('だ', '。'), ('です', '。')),
    (('だ', '！'), ('です', '！')),
    (('だ', '？'), ('です', '？')),
    (('だ', '」'), ('です', '」')),
    (('だ', '』'), ('です', '』')),
    (('だ', '…'), ('です', '…')),
    # With sentence-final particles
    (('だ', 'ね', '。'), ('です', 'ね', '。')),
    (('だ', 'よ', '。'), ('です', 'よ', '。')),
    (('だ', 'わ', '。'), ('です', 'わ', '。')),
    (('だ', 'な', '。'), ('です', 'な', '。')),
    # Sentence-final particles without punctuation (end of string)
    (('だ', 'ね'), ('です', 'ね')),
    (('だ', 'よ'), ('です', 'よ')),
    (('だ', 'わ'), ('です', 'わ')),
    (('だ', 'な'), ('です', 'な')),
    # With period instead of 。
    (('だ', '.'), ('です', '.')),
]

PROGRESSIVE_END_PATTERNS = [
    (('て', 'い', 'ます', '。'), ('て', 'いる', '。')),
    (('て', 'い', 'まし', 'た', '。'), ('て', 'い', 'た', '。')),
    (('で', 'い', 'ます', '。'), ('で', 'いる', '。')),
    (('で', 'い', 'まし', 'た', '。'), ('で', 'い', 'た', '。')),
    # Without period
    (('て', 'い', 'ます'), ('て', 'いる')),
    (('て', 'い', 'まし', 'ta'), ('て', 'い', 'た')),
    (('で', 'い', 'ます'), ('で', 'いる')),
    (('で', 'い', 'まし', 'た'), ('で', 'い', 'た')),
]

DROPPABLE_PARTICLES = {'は', 'が', 'を'}

DROPPABLE_TOPIC_STARTS = [
    ('私', 'は'),
    ('僕達', '僕たち'),
    ('俺達', '俺たち'),
]

PLURAL_PATTERNS = [
    ('私達', '私たち'),
    ('僕達', '僕たち'),
    ('俺達', '俺たち'),
]

CONTRACTION_PATTERNS = [
    # de wa <-> ja
    (('で', 'は'), ('じゃ',)),
    # te iru <-> te ru (progressive reduction)
    (('て', 'いる'), ('てる',)),
    (('で', 'いる'), ('でる',)),
]

DROPPABLE_TOPIC_STARTS = [
    ('私', 'は'),
    ('僕', 'は'),
    ('俺', 'は'),
    ('わたし', 'は'),
    ('ぼく', 'は'),
    ('おれ', 'は'),
    ('あたし', 'は'),
    ('わたくし', 'は'),
]


def deduplicate_by_reading(candidates: Set[Tuple[AugmentationToken, ...]]) -> Set[Tuple[AugmentationToken, ...]]:
    """Groups candidates by full phonetic reading and keeps the shortest surface form for each."""
    if not candidates:
        return set()
    
    by_reading: Dict[str, Tuple[AugmentationToken, ...]] = {}
    
    for c in candidates:
        # Construct full reading
        r = "".join(t.reading if isinstance(t, Token) else str(t) for t in c)
        
        if r not in by_reading:
            by_reading[r] = c
        else:
            existing = by_reading[r]
            
            # Keep the one with shorter surface form
            curr_surface = "".join(get_surface(t) for t in c)
            exist_surface = "".join(get_surface(t) for t in existing)
            
            if len(curr_surface) < len(exist_surface) or (len(curr_surface) == len(exist_surface) and curr_surface < exist_surface):
                # Update to the new one
                by_reading[r] = c

    return set(by_reading.values())


def get_surface(token: AugmentationToken) -> str:
    """Extract surface form from a token (string or dict)."""
    if isinstance(token, Token):
        return token.surface
    if isinstance(token, dict):
        return token.get('surface', '')
    return str(token)


HUMBLE_MAP = {
    '言う': '申す',
    'いく': '参る',
    '行く': '参る',
    'くる': '参る',
    '来る': '参る',
    '食べる': 'いただく',
    '飲む': 'いただく',
    'する': 'いたす',
    '見る': '拝見する',
    '聞く': '伺う',
    '知る': '存じる',
    'もらう': 'いただく',
}

HONORIFIC_MAP = {
    '言う': 'おっしゃる',
    'いく': 'いらっしゃる',
    '行く': 'いらっしゃる',
    'くる': 'いらっしゃる',
    '来る': 'いらっしゃる',
    '食べる': '召し上がる',
    '飲む': '召し上がる',
    'する': 'なさる',
    '見る': 'ご覧になる',
    'くれる': 'くださる',
}


def conjugate_to_masu_stem(lemma: str, ctype: str) -> Optional[str]:
    """Conjugate a dictionary form verb to its masu-stem (ren'youkei)."""
    if not lemma or not ctype:
        return None

    # Ichidan: Drop 'ru'
    if 'ichidan' in ctype:
        if lemma.endswith('る'):
            return lemma[:-1]
    
    if ctype in ('suru', 'sa-irregular') or lemma.endswith('する'):
        if lemma == 'する':
            return 'し'
        if lemma.endswith('する'):
            return lemma[:-2] + 'し'

    # Kuru irregular
    if ctype in ('kuru', 'ka-irregular') or lemma == '来る':
        return '来'
    if lemma == 'くる':
        return 'き'

    # Godan: Change last vowel u -> i
    mappings = {
        'う': 'い', 'つ': 'ち', 'る': 'り', 'く': 'き', 'ぐ': 'ぎ',
        'む': 'み', 'ぶ': 'び', 'ぬ': 'に', 'す': 'し',
    }
    if 'godan' in ctype:
        last_char = lemma[-1]
        if last_char in mappings:
            return lemma[:-1] + mappings[last_char]
            
    return None


def conjugate_to_irrealis_stem(lemma: str, ctype: str) -> Optional[str]:
    """Conjugate a dictionary form verb to its irrealis-stem (mizenkei)."""
    if not lemma or not ctype:
        return None

    # Ichidan: Same as masu stem (drop 'ru')
    if 'ichidan' in ctype:
        if lemma.endswith('る'):
            return lemma[:-1]
            
    if ctype in ('suru', 'sa-irregular') or lemma.endswith('する'):
        if lemma == 'する':
            return 'し'
        if lemma.endswith('する'):
            return lemma[:-2] + 'し'

    # Kuru irregular
    if ctype in ('kuru', 'ka-irregular') or lemma == '来る':
        return '来'
    if lemma == 'くる':
        return 'こ'

    # Godan: Change last vowel u -> a (exception: u -> wa)
    mappings = {
        'う': 'わ', 'つ': 'た', 'る': 'ら', 'く': 'か', 'ぐ': 'が',
        'む': 'ま', 'ぶ': 'ば', 'ぬ': 'な', 'す': 'さ',
    }
    if 'godan' in ctype:
        last_char = lemma[-1]
        if last_char in mappings:
            return lemma[:-1] + mappings[last_char]
            
    return None


class AugmentationRule(ABC):
    """Abstract base class for sentence augmentation rules."""
    
    @abstractmethod
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        """Apply the rule to a sequence of tokens.
        
        Args:
            tokens: A tuple of tokens (strings or feature dicts).
            
        Returns:
            A set of token tuples comprising the original and any valid variations.
        """
        pass


class VerbPolitenessRule(AugmentationRule):
    """Augments sentences by swapping plain/polite verb forms."""

    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        
        # To avoid index drift issues during iteration, we collect potential replacement sites first,
        # then generate combinations relative to the original sequence.
        
        potential_swaps: List[Tuple[int, int, Tuple[AugmentationToken, ...]]] = [] # List of (index, length_to_remove, replacement_tokens)

        for i, token in enumerate(tokens):
            # Only process if we have features
            if not isinstance(token, Token):
                continue
                
            pos = token.get('pos', '')
            if pos != 'verb':
                continue
                
            lemma = token.get('lemma', get_surface(token))
            ctype = token.get('conjugated_type', '')
            cform = token.get('conjugated_form', '')
            
            # Case 1: Plain -> Polite
            # Verb (terminal) -> Verb (masu-stem) + ます
            if cform == 'terminal':
                stem = conjugate_to_masu_stem(lemma, ctype)
                if stem:
                    # Valid conjugation found
                    replacement: Tuple[AugmentationToken, ...] = (stem, 'ます')
                    potential_swaps.append((i, 1, replacement))

            # Case 2: Polite -> Plain
            # Verb (conjunctive?) + ます (auxv) -> Lemma
            # Note: "ます" might be separate token.
            # Look ahead for "ます"
            if i + 1 < len(tokens):
                next_token = tokens[i+1]
                next_surf = get_surface(next_token)
                
                # Check if next is 'ます' (auxv)
                # Ideally check POS of next token if dict, but surface 'ます' is strong signal
                if next_surf == 'ます':
                    # Replace (Verb, ます) -> (Lemma,)
                    replacement = (lemma,)
                    potential_swaps.append((i, 2, replacement))

        if potential_swaps:
            # Generate sub-combinations? 
            # Similar to ContractionRule, let's just generate "Original" and "All Swapped".
            # Mixing politeness levels in one sentence is usually grammatically weird.
            # So let's try to apply ALL consistent changes.
            
            # Group swaps by type? No, just apply them.
            # But wait, we might have conflicting overlaps? 
            # Case 1 (length 1) and Case 2 (length 2) won't overlap start indices for same token
            # unless one is subset. A token is either terminal OR followed by masu. Mutually exclusive.
            
            # Apply all possible swaps to generate ONLY the inverted politeness version?
            # Or should we output mixed? We prefer consistent.
            
            new_tokens: List[AugmentationToken] = []
            last_idx = 0
            
            # Sort swaps by index
            potential_swaps.sort(key=lambda x: x[0])
            
            for idx, length, replacement in potential_swaps:
                if idx < last_idx:
                    continue # Should not happen if mutually exclusive
                    
                # Copy preceding
                new_tokens.extend(tokens[last_idx:idx])
                # Add replacement
                new_tokens.extend(replacement)
                last_idx = idx + length
                
            if last_idx < len(tokens):
                new_tokens.extend(tokens[last_idx:])
                
            result.add(tuple(new_tokens))

        return result


class FirstPersonPronounRule(AugmentationRule):
    """Augments sentences by swapping first-person pronouns."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        token_surfaces = [get_surface(t) for t in tokens]
        pronoun_indices = [i for i, t in enumerate(token_surfaces) if t in FIRST_PERSON_PRONOUNS]
        
        if not pronoun_indices:
            return {tokens}
            
        result = set()
        for combo in product(FIRST_PERSON_PRONOUNS, repeat=len(pronoun_indices)):
            new_tokens = list(tokens)
            for idx, new_pronoun in zip(pronoun_indices, combo):
                new_tokens[idx] = new_pronoun # Replace with string
            result.add(tuple(new_tokens))
            
        return result


class SecondPersonPronounRule(AugmentationRule):
    """Augments sentences by swapping second-person pronouns (strict person matching)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        token_surfaces = [get_surface(t) for t in tokens]
        pronoun_indices = [i for i, t in enumerate(token_surfaces) if t in SECOND_PERSON_PRONOUNS]
        
        if not pronoun_indices:
            return {tokens}
            
        result = set()
        for combo in product(SECOND_PERSON_PRONOUNS, repeat=len(pronoun_indices)):
            new_tokens = list(tokens)
            for idx, new_pronoun in zip(pronoun_indices, combo):
                new_tokens[idx] = new_pronoun
            result.add(tuple(new_tokens))
            
        return result


class ThirdPersonPronounRule(AugmentationRule):
    """Augments sentences by swapping third-person pronouns (strict person matching)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        token_surfaces = [get_surface(t) for t in tokens]
        pronoun_indices = [i for i, t in enumerate(token_surfaces) if t in THIRD_PERSON_PRONOUNS]
        
        if not pronoun_indices:
            return {tokens}
            
        result = set()
        for combo in product(THIRD_PERSON_PRONOUNS, repeat=len(pronoun_indices)):
            new_tokens = list(tokens)
            for idx, new_pronoun in zip(pronoun_indices, combo):
                new_tokens[idx] = new_pronoun
            result.add(tuple(new_tokens))
            
        return result


class CopulaRule(AugmentationRule):
    """Augments sentences by changing copula formality (da <-> desu)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        token_surfaces = tuple(get_surface(t) for t in tokens)
        
        for da_toks, desu_toks in COPULA_PATTERNS:
            # Check if sentence ends with da_toks
            if len(tokens) >= len(da_toks) and token_surfaces[-len(da_toks):] == da_toks:
                new_tokens = list(tokens[:-len(da_toks)]) + list(desu_toks)
                result.add(tuple(new_tokens))
                
            # Check if sentence ends with desu_toks
            if len(tokens) >= len(desu_toks) and token_surfaces[-len(desu_toks):] == desu_toks:
                new_tokens = list(tokens[:-len(desu_toks)]) + list(da_toks)
                result.add(tuple(new_tokens))
                
        return result


class ContractionRule(AugmentationRule):
    """Augments sentences by swapping contractions (e.g. dewa <-> ja)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        token_surfaces = tuple(get_surface(t) for t in tokens)
        
        for form_a, form_b in CONTRACTION_PATTERNS:
            len_a = len(form_a)
            len_b = len(form_b)
            
            indices_a = []
            for i in range(len(token_surfaces) - len_a + 1):
                if token_surfaces[i:i+len_a] == form_a:
                    indices_a.append(i)
            
            if indices_a:
                import itertools
                for replacement_mask in itertools.product([False, True], repeat=len(indices_a)):
                    if not any(replacement_mask):
                        continue
                    
                    new_tokens: List[AugmentationToken] = []
                    last_idx = 0
                    matches = sorted(indices_a)
                    
                    for i, do_replace in zip(matches, replacement_mask):
                        new_tokens.extend(tokens[last_idx:i])
                        if do_replace:
                            new_tokens.extend(form_b)
                        else:
                            new_tokens.extend(tokens[i:i+len_a])
                        last_idx = i + len_a
                        
                    new_tokens.extend(tokens[last_idx:])
                    result.add(tuple(new_tokens))

            # Form B -> Form A
            indices_b = []
            for i in range(len(token_surfaces) - len_b + 1):
                if token_surfaces[i:i+len_b] == form_b:
                    indices_b.append(i)
            
            if indices_b:
                import itertools
                for replacement_mask in itertools.product([False, True], repeat=len(indices_b)):
                    if not any(replacement_mask):
                        continue
                        
                    new_tokens = []
                    last_idx = 0
                    matches = sorted(indices_b)
                    
                    for i, do_replace in zip(matches, replacement_mask):
                        new_tokens.extend(tokens[last_idx:i])
                        if do_replace:
                            new_tokens.extend(form_a)
                        else:
                            new_tokens.extend(tokens[i:i+len_b])
                        last_idx = i + len_b
                        
                    new_tokens.extend(tokens[last_idx:])
                    result.add(tuple(new_tokens))
                    
        return result


class ParticleDropRule(AugmentationRule):
    """Augments sentences by dropping case particles (wa, ga, o) in casual contexts."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        
        drop_indices = []
        for i, token in enumerate(tokens):
            # Only drop if we are sure it's a particle (i.e. have POS info)
            if not isinstance(token, Token):
                continue
                
            if token.get('pos') == 'particle' and token.surface in DROPPABLE_PARTICLES:
                # Safety check: Don't drop 'wa' part of split greetings (konnichi-wa, konban-wa)
                if token.surface == 'は' and i > 0:
                    prev = get_surface(tokens[i-1])
                    if prev in {'こんにち', 'こんばん'}:
                        continue
                        
                drop_indices.append(i)
                
        if drop_indices:
            import itertools
            for drop_mask in itertools.product([False, True], repeat=len(drop_indices)):
                if not any(drop_mask):
                    continue
                    
                new_tokens: List[AugmentationToken] = []
                last_idx = 0
                
                for idx, do_drop in zip(drop_indices, drop_mask):
                    new_tokens.extend(tokens[last_idx:idx])
                    if not do_drop:
                        new_tokens.append(tokens[idx])
                    last_idx = idx + 1
                    
                new_tokens.extend(tokens[last_idx:])
                result.add(tuple(new_tokens))
                
        return result


class TopicDropRule(AugmentationRule):
    """Aguments sentences by dropping clear subjects/topics at the start."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        token_surfaces = tuple(get_surface(t) for t in tokens)
        
        for topic_toks in DROPPABLE_TOPIC_STARTS:
            if len(tokens) > len(topic_toks) and token_surfaces[:len(topic_toks)] == topic_toks:
                new_tokens = tokens[len(topic_toks):]
                result.add(new_tokens)
                
        return result


class ProgressiveRule(AugmentationRule):
    """Augments sentences by changing progressive form formality (te iru <-> te i masu)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        
        # This rule logic is strictly surface end-match based in the original
        # We need to construct mixed tuples if we have dicts.
        token_surfaces = tuple(get_surface(t) for t in tokens)

        for polite_toks, plain_toks in PROGRESSIVE_END_PATTERNS:
            if len(tokens) >= len(polite_toks) and token_surfaces[-len(polite_toks):] == polite_toks:
                new_tokens = list(tokens[:-len(polite_toks)]) + list(plain_toks)
                result.add(tuple(new_tokens))
                
            if len(tokens) >= len(plain_toks) and token_surfaces[-len(plain_toks):] == plain_toks:
                new_tokens = list(tokens[:-len(plain_toks)]) + list(polite_toks)
                result.add(tuple(new_tokens))
                
        return result


class PluralRule(AugmentationRule):
    """Augments sentences by swapping kanji/hiragana for plural suffix."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        token_surfaces = [get_surface(t) for t in tokens]
        
        for kanji, hiragana in PLURAL_PATTERNS:
            indices_kanji = [i for i, t in enumerate(token_surfaces) if t == kanji]
            if indices_kanji:
                 for combo in product([kanji, hiragana], repeat=len(indices_kanji)):
                     new_tokens = list(tokens)
                     for idx, val in zip(indices_kanji, combo):
                         new_tokens[idx] = val # String replacement
                     result.add(tuple(new_tokens))

            indices_hiragana = [i for i, t in enumerate(token_surfaces) if t == hiragana]
            if indices_hiragana:
                 for combo in product([kanji, hiragana], repeat=len(indices_hiragana)):
                     new_tokens = list(tokens)
                     for idx, val in zip(indices_hiragana, combo):
                         new_tokens[idx] = val
                     result.add(tuple(new_tokens))
        return result


class PronunciationRule(AugmentationRule):
    """Augments sentences with alternate pronunciations (e.g. 日本 -> にほん, にっぽん)."""
    
    PATTERNS = [
        {'日本', 'にほん', 'にっぽん'},
    ]
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        token_surfaces = [get_surface(t) for t in tokens]
        
        for pattern_set in self.PATTERNS:
            # Find all indices where any of the pattern members match
            match_indices = [i for i, s in enumerate(token_surfaces) if s in pattern_set]
            
            if not match_indices:
                continue
                
            # Generate all combinations of pronunciation alternates for the matched indices
            # Convert set to sorted list for deterministic results
            alternates = sorted(list(pattern_set))
            for combo in product(alternates, repeat=len(match_indices)):
                new_tokens = list(tokens)
                for idx, val in zip(match_indices, combo):
                    # If the original was a Token object and the replacement is a string, 
                    # we keep it as a string for now as it's common in this module's other rules.
                    new_tokens[idx] = val
                result.add(tuple(new_tokens))
                
        return result


        return result


class RaNukiRule(AugmentationRule):
    """Augments sentences by dropping 'ra' in potential forms (e.g. 食べられる -> 食べれる)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        for i in range(len(tokens) - 1):
            v_tok = tokens[i]
            aux_tok = tokens[i+1]
            if not (is_role(v_tok, 'verb') and is_role(aux_tok, 'aux-verb')):
                continue
            
            # Check if potential ichidan
            v_f = get_features(v_tok)
            if 'ichidan' in v_f.get('conjugated_type', '') and v_f.get('conjugated_form') == 'irrealis':
                aux_surf = get_surface(aux_tok)
                if aux_surf == 'られる':
                    # Replace られる with れる
                    new_tokens = list(tokens)
                    new_tokens[i+1] = 'れる'
                    result.add(tuple(new_tokens))
                elif aux_surf == 'られた':
                    new_tokens = list(tokens)
                    new_tokens[i+1] = 'れた'
                    result.add(tuple(new_tokens))
        return result


class NegativeContractionRule(AugmentationRule):
    """Augments sentences by contracting negation (e.g. 分からない -> 分からん)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        for i in range(len(tokens) - 1):
            v_tok = tokens[i]
            neg_tok = tokens[i+1]
            
            v_f = get_features(v_tok)
            if v_f.get('conjugated_form') == 'irrealis':
                neg_surf = get_surface(neg_tok)
                if neg_surf == 'ない':
                    new_tokens = list(tokens)
                    new_tokens[i+1] = 'ん'
                    result.add(tuple(new_tokens))
        return result


class NegationSwapRule(AugmentationRule):
    """Augments sentences by swapping plain and polite negation (e.g. ない <-> ません)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        i = 0
        while i < len(tokens):
            v_tok = tokens[i]
            v_f = get_features(v_tok)
            
            # Cases:
            # 1. Plain -> Polite: (verb irrealis) + ない
            if v_f.get('conjugated_form') == 'irrealis' and i + 1 < len(tokens):
                neg_tok = tokens[i+1]
                if get_surface(neg_tok) == 'ない':
                    # Plain -> Polite
                    stem = conjugate_to_masu_stem(v_f.get('lemma', ''), v_f.get('conjugated_type', ''))
                    if stem:
                        new_tokens = list(tokens[:i]) + [stem, 'ませ', 'ん'] + list(tokens[i+2:])
                        result.add(tuple(new_tokens))
            
            # 2. Polite -> Plain: (verb continuative) + ませ + ん
            if v_f.get('conjugated_form') == 'continuative' and i + 2 < len(tokens):
                mase_tok = tokens[i+1]
                n_tok = tokens[i+2]
                if get_surface(mase_tok) == 'ませ' and get_surface(n_tok) == 'ん':
                    # Polite -> Plain
                    stem = conjugate_to_irrealis_stem(v_f.get('lemma', ''), v_f.get('conjugated_type', ''))
                    if stem:
                        new_tokens = list(tokens[:i]) + [stem, 'ない'] + list(tokens[i+3:])
                        result.add(tuple(new_tokens))
            i += 1
        return result


class SentenceFinalParticleRule(AugmentationRule):
    """Augments sentences by adding/swapping/removing sentence-final particles."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        # Find position of last punctuation or end of sentence
        last_idx = len(tokens)
        if len(tokens) > 0:
            if get_surface(tokens[-1]) in ('。', '？', '！', '!', '?'):
                last_idx = len(tokens) - 1
        
        # Check if there's already a particle at the end
        target_idx = last_idx
        has_existing = False
        if target_idx > 0:
            feat = get_features(tokens[target_idx-1])
            if feat.get('pos') == 'particle' or get_surface(tokens[target_idx-1]) in SENTENCE_FINAL_PARTICLES:
                target_idx -= 1
                has_existing = True
        
        # 1. Swap/Remove existing
        if has_existing:
            # Remove
            result.add(tuple(list(tokens[:target_idx]) + list(tokens[last_idx:])))
            # Swap
            for p in SENTENCE_FINAL_PARTICLES:
                new_tokens = list(tokens[:target_idx]) + [p] + list(tokens[last_idx:])
                result.add(tuple(new_tokens))
        else:
            # 2. Add new
            for p in SENTENCE_FINAL_PARTICLES:
                new_tokens = list(tokens[:target_idx]) + [p] + list(tokens[last_idx:])
                result.add(tuple(new_tokens))
                
        return result


class LoanwordRule(AugmentationRule):
    """Augments sentences by swapping Katakana spelling variants."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        for i, tok in enumerate(tokens):
            surf = get_surface(tok)
            if surf in LOANWORD_VARIANTS:
                new_tokens = list(tokens)
                new_tokens[i] = LOANWORD_VARIANTS[surf]
                result.add(tuple(new_tokens))
            else:
                # Reverse check
                for common, variant in LOANWORD_VARIANTS.items():
                    if surf == variant:
                        new_tokens = list(tokens)
                        new_tokens[i] = common
                        result.add(tuple(new_tokens))
        return result


class PastTenseSwapRule(AugmentationRule):
    """Augments sentences by swapping plain and polite past tense (e.g. た <-> ました)."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        i = 0
        while i < len(tokens):
            v_tok = tokens[i]
            v_f = get_features(v_tok)
            
            # v(continuative) + た(aux-ta) -> v(continuative) + まし(aux-masu) + た(aux-ta)
            if v_f.get('conjugated_form') == 'continuative' and i + 1 < len(tokens):
                ta_tok = tokens[i+1]
                if get_surface(ta_tok) == 'た' and is_role(ta_tok, 'aux-verb'):
                    new_tokens = list(tokens[:i+1]) + ['まし', 'た'] + list(tokens[i+2:])
                    result.add(tuple(new_tokens))
            
            # v(continuative) + まし(aux-masu) + た(aux-ta) -> v(continuative) + た(aux-ta)
            if v_f.get('conjugated_form') == 'continuative' and i + 2 < len(tokens):
                mashi_tok = tokens[i+1]
                ta_tok = tokens[i+2]
                if get_surface(mashi_tok) == 'まし' and get_surface(ta_tok) == 'た':
                    new_tokens = list(tokens[:i+1]) + ['た'] + list(tokens[i+3:])
                    result.add(tuple(new_tokens))
            i += 1
        return result


class AdjectivePolitenessRule(AugmentationRule):
    """Augments sentences by toggling 'desu' for i-adjectives."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        for i, tok in enumerate(tokens):
            f = get_features(tok)
            if f.get('conjugated_type') == 'i-adjective' and f.get('conjugated_form') == 'terminal':
                # Add 'desu' if not present
                if i + 1 == len(tokens) or get_surface(tokens[i+1]) != 'です':
                    new_tokens = list(tokens[:i+1]) + ['です'] + list(tokens[i+1:])
                    result.add(tuple(new_tokens))
                # Remove 'desu' if present
                elif i + 1 < len(tokens) and get_surface(tokens[i+1]) == 'です':
                    new_tokens = list(tokens[:i+1]) + list(tokens[i+2:])
                    result.add(tuple(new_tokens))
        return result


class HumbleHonorificRule(AugmentationRule):
    """Augments sentences by swapping standard verbs for humble/honorific versions."""
    
    def apply(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        result = {tokens}
        for i, tok in enumerate(tokens):
            f = get_features(tok)
            if is_role(tok, 'verb'):
                lemma = f.get('lemma') or get_surface(tok)
                cform = f.get('conjugated_form')
                
                if cform == 'terminal':
                    if lemma in HUMBLE_MAP:
                        new_tokens = list(tokens)
                        new_tokens[i] = HUMBLE_MAP[lemma]
                        result.add(tuple(new_tokens))
                    if lemma in HONORIFIC_MAP:
                        new_tokens = list(tokens)
                        new_tokens[i] = HONORIFIC_MAP[lemma]
                        result.add(tuple(new_tokens))
        return result


def is_role(token: AugmentationToken, role: str) -> bool:
    if isinstance(token, Token):
        pos = token.get('pos', '')
        if role == 'verb':
            return bool(pos == 'verb')
        if role == 'aux-verb':
            return bool(pos == 'aux-verb')
        if role == 'noun':
            return bool(pos == 'noun')
        if role == 'adj':
            return bool(pos == 'adj')
    return False

def get_features(token: AugmentationToken) -> Dict[str, str]:
    if isinstance(token, Token):
        return token.features
    return {}


class Augmenter:
    """Main class for coordinating augmentation."""
    
    _parser: Optional[SudachiJapaneseParser] = None

    def __init__(self) -> None:
        # Core rules that can interact and affect grammaticality
        self.core_rules: List[AugmentationRule] = [
            VerbPolitenessRule(),
            RaNukiRule(),
            NegativeContractionRule(),
            NegationSwapRule(),
            PastTenseSwapRule(),
            AdjectivePolitenessRule(),
            CopulaRule(),
            ProgressiveRule(),
            ContractionRule(),
            ParticleDropRule(),
            SentenceFinalParticleRule(),
            TopicDropRule(),
            PronunciationRule(),
            LoanwordRule(),
            PluralRule(),
            HumbleHonorificRule(),
        ]
        
        # Rules that are guaranteed grammatic and can explode combinatorial space
        self.late_rules: List[AugmentationRule] = [
            FirstPersonPronounRule(),
            SecondPersonPronounRule(),
            ThirdPersonPronounRule(),
        ]
    
    @classmethod
    def get_parser(cls) -> SudachiJapaneseParser:
        """Get or initialize the shared parser instance."""
        if cls._parser is None:
            # Use full dict for best accuracy
            cls._parser = SudachiJapaneseParser(dict_type='full')
        return cls._parser
        
    def augment_tokens(self, tokens: Tuple[AugmentationToken, ...], deadline: Optional[float] = None) -> Set[Tuple[AugmentationToken, ...]]:
        """Apply rules iteratively until stable or limit reached, with hysteresis deduplication."""
        current_set = {tokens}
        
        # Threshold for applying deduplication to prevent explosion
        HYSTERESIS_THRESHOLD = 5000
        
        # Apply core rules iteratively
        for _ in range(5):
            if deadline and time.time() > deadline:
                break
                
            next_set = set(current_set)
            for rule in self.core_rules:
                if deadline and time.time() > deadline:
                    break
                
                for t in current_set:
                    rule_results = rule.apply(t)
                    next_set.update(rule_results)
            
            # Hysteresis deduplication
            if len(next_set) > HYSTERESIS_THRESHOLD:
                next_set = deduplicate_by_reading(next_set)
            
            if len(next_set) == len(current_set):
                break
            current_set = next_set

        # Apply late rules (pronouns)
        late_set = set(current_set)
        for rule in self.late_rules:
            if deadline and time.time() > deadline:
                break
            for t in current_set:
                rule_results = rule.apply(t)
                late_set.update(rule_results)
            
        # Final deduplication
        return deduplicate_by_reading(late_set)
    
    def process_sentence(self, sentence: str, timeout: Optional[float] = 1.0) -> Set[str]:
        """Process a single unspaced Japanese sentence into augmented variations within a time budget."""
        if not sentence:
            return set()
            
        start_time = time.time()
        deadline = start_time + timeout if timeout else None
        
        clean_sentence = sentence.replace(" ", "")
        if not clean_sentence:
            return set()
            
        parser = self.get_parser()
        
        kotogram = parser.japanese_to_kotogram(clean_sentence)
        tokens_kotogram = split_kotogram(kotogram)
        
        # Extract features for all tokens
        token_features = []
        for t in tokens_kotogram:
            f = extract_token_features(t)
            # Ensure surface is set
            if not f.surface:
                 import re
                 match = re.search(r'ˢ(.*?)ᵖ', t)
                 f.surface = match.group(1) if match else t
            
            token_features.append(Token(f.surface, asdict(f)))
        
        if not token_features:
             return {clean_sentence}
        
        token_tuple = tuple(token_features)
        
        # 1. Generation (with deadline)
        augmented_tuples = self.augment_tokens(token_tuple, deadline=deadline)
        
        # 2. Join back surfaces
        candidate_surfaces = set()
        for aug_tuple in augmented_tuples:
            surface_list = [get_surface(token) for token in aug_tuple]
            candidate_surfaces.add("".join(surface_list))
        
        # 3. Filtration (with deadline and sharding)
        if deadline and time.time() > deadline:
            from kotogram.analysis import grammar
            k_orig = parser.japanese_to_kotogram(clean_sentence)
            if grammar(k_orig).is_grammatic:
                return {clean_sentence}
            return set()

        valid_sentences = self.filter_grammatical(candidate_surfaces, deadline=deadline)
        return set(valid_sentences)

    def filter_grammatical(self, sentences: Set[str], deadline: Optional[float] = None) -> List[str]:
        """Filter input sentences using neural model batch inference (Smart Length-Based Batching)."""
        if not sentences:
            return []
            
        from kotogram.analysis import grammars
        
        # Sort by length (descending) to group long sentences together early
        # and ensure padding waste is minimized within each batch.
        sentence_list = sorted(list(sentences), key=len, reverse=True)
        parser = self.get_parser()
        
        valid_sentences = []
        
        # BUDGET: Total character count units per batch.
        # 10000 allows for slightly larger batches to improve throughput while 
        # still maintaining good granularity for timeout checks.
        MAX_LEN_BUDGET = 10000 
        
        current_batch: List[str] = []
        current_batch_len = 0
        
        def process_shard(shard_list: List[str]) -> None:
            if not shard_list:
                return
            kotograms = [parser.japanese_to_kotogram(s) for s in shard_list]
            analyses = grammars(kotograms)
            for s, analysis in zip(shard_list, analyses):
                is_valid = (analysis.is_grammatic and 
                           analysis.formality.value != 'unpragmatic_formality' and 
                           analysis.gender.value != 'unpragmatic_gender')
                if is_valid:
                    valid_sentences.append(s)

        for s in sentence_list:
            if deadline and time.time() > deadline:
                break
                
            s_len = len(s)
            # If adding this sentence exceeds budget, process current batch first
            if current_batch and (current_batch_len + s_len > MAX_LEN_BUDGET):
                process_shard(current_batch)
                current_batch = []
                current_batch_len = 0
                # Re-check deadline after an atomic batch
                if deadline and time.time() > deadline:
                    break

            current_batch.append(s)
            current_batch_len += s_len
            
        # Process trailing batch if time remains
        if current_batch and (not deadline or time.time() < deadline):
            process_shard(current_batch)
                
        return sorted(list(set(valid_sentences)))



def augment(sentences: List[str], timeout: Optional[float] = 1.0) -> List[str]:
    """Augment a list of Japanese sentences and filter for grammaticality within a time budget.
    
    Args:
        sentences: List of input Japanese sentences (unspaced).
        timeout: Total time budget in seconds (spread across all sentences).
        
    Returns:
        Sorted unique list of augmented, grammatically valid sentences.
    """
    from kotogram.validation import ensure_list_of_strings
    ensure_list_of_strings(sentences, "sentences")

    start_time = time.time()
    deadline = start_time + timeout if timeout else None
    
    augmenter = Augmenter()
    
    # 1. Generate candidates for all sentences
    all_candidates = set()
    for s in sentences:
        if deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
        else:
            remaining = None
            
        # We pass timeout=remaining. We'll handle the final filtering ourselves for efficiency.
        # But wait, process_sentence already filters. 
        # To avoid redundancy, let's just generate candidates directly.
        # However, to keep it simple and respect the budget, let's use a modified process_sentence 
        # that doesn't filter if we want to batch filter at the end.
        # On second thought, process_sentence is the standard unit.
        # Let's just use the augmenter's low-level methods here to avoid redundant filtering.
        
        parser = augmenter.get_parser()
        kotogram = parser.japanese_to_kotogram(s)
        tokens_kotogram = split_kotogram(kotogram)
        
        token_features = []
        for t in tokens_kotogram:
            f = extract_token_features(t)
            if not f.surface:
                 import re
                 match = re.search(r'ˢ(.*?)ᵖ', t)
                 f.surface = match.group(1) if match else t
            token_features.append(Token(f.surface, asdict(f)))
        
        if not token_features:
            all_candidates.add(s)
            continue
            
        token_tuple = tuple(token_features)
        augmented_tuples = augmenter.augment_tokens(token_tuple, deadline=deadline)
        
        for aug_tuple in augmented_tuples:
            surface_list = [get_surface(token) for token in aug_tuple]
            all_candidates.add("".join(surface_list))
        
    # 2. Batch filter all candidates together (the most efficient way)
    # We pass the deadline to allow early exit during the large batch filter
    return augmenter.filter_grammatical(all_candidates, deadline=deadline)

