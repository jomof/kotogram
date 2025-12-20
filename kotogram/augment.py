"""Module for augmenting Japanese sentences with grammatical variations.

This module provides an extensible framework for generating variations of Japanese
sentences (e.g. changing formality, dropping topics, swapping pronouns) and verifying
their grammaticality using a neural model.
"""

from abc import ABC, abstractmethod
from typing import Set, Tuple, List, Optional, Dict, Union, Any
from itertools import product
from kotogram.kotogram import split_kotogram, extract_token_features
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.analysis import grammar
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

# Constants and Patterns
FIRST_PERSON_PRONOUNS = {'私', '僕', '俺', 'わたし', 'ぼく', 'おれ'}

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
]


def get_surface(token: AugmentationToken) -> str:
    """Extract surface form from a token (string or dict)."""
    if isinstance(token, Token):
        return token.surface
    if isinstance(token, dict):
        return token.get('surface', '')
    return str(token)


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

    def _conjugate_to_masu_stem(self, lemma: str, ctype: str) -> Optional[str]:
        """Conjugate a dictionary form verb to its masu-stem (ren'youkei)."""
        if not lemma or not ctype:
            return None

        # Ichidan: Drop 'ru'
        if ctype.startswith('i-ichidan') or ctype == 'e-ichidan' or 'ichidan' in ctype:
            if lemma.endswith('る'):
                return lemma[:-1]
        
        if ctype == 'suru' or lemma.endswith('する'):
            if lemma == 'する':
                return 'し'
            if lemma.endswith('する'):
                return lemma[:-2] + 'し'

        # Kuru irregular
        if ctype == 'kuru' or lemma == '来る':
            return '来'
        if lemma == 'くる':
            return 'き'

        # Godan: Change last vowel u -> i
        # u, tsu, ru, ku, gu, mu, bu, nu, su
        mappings = {
            'う': 'い',
            'つ': 'ち',
            'る': 'り',
            'く': 'き',
            'ぐ': 'ぎ',
            'む': 'み',
            'ぶ': 'び',
            'ぬ': 'に',
            'す': 'し',
        }
        if ctype.startswith('godan'):
            last_char = lemma[-1]
            if last_char in mappings:
                return lemma[:-1] + mappings[last_char]
                
        return None

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
            if pos != 'v':
                continue
                
            lemma = token.get('lemma', get_surface(token))
            ctype = token.get('conjugated_type', '')
            cform = token.get('conjugated_form', '')
            
            # Case 1: Plain -> Polite
            # Verb (terminal) -> Verb (masu-stem) + ます
            if cform == 'terminal':
                stem = self._conjugate_to_masu_stem(lemma, ctype)
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


class PronounRule(AugmentationRule):
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
                
            if token.get('pos') == 'prt' and token.surface in DROPPABLE_PARTICLES:
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


class Augmenter:
    """Main class for coordinating augmentation."""
    
    _parser: Optional[SudachiJapaneseParser] = None

    def __init__(self) -> None:
        self.rules: List[AugmentationRule] = [
            VerbPolitenessRule(), # Run first to use rich features
            PronounRule(),
            CopulaRule(),
            ContractionRule(),
            ParticleDropRule(),
            TopicDropRule(),
            ProgressiveRule(),
            PluralRule(),
        ]
    
    @classmethod
    def get_parser(cls) -> SudachiJapaneseParser:
        """Get or initialize the shared parser instance."""
        if cls._parser is None:
            # Use full dict for best accuracy
            cls._parser = SudachiJapaneseParser(dict_type='full')
        return cls._parser
        
    def augment_tokens(self, tokens: Tuple[AugmentationToken, ...]) -> Set[Tuple[AugmentationToken, ...]]:
        """Apply all rules repeatedly to generate variations."""
        current_set = {tokens}
        
        for rule in self.rules:
            next_set = set()
            for t in current_set:
                next_set.update(rule.apply(t))
            current_set = next_set
            
        return current_set
    
    def process_sentence(self, sentence: str) -> Set[str]:
        """Process a single unspaced Japanese sentence into augmented variations."""
        if not sentence:
            return set()
            
        clean_sentence = sentence.replace(" ", "")
        if not clean_sentence:
            return set()
            
        parser = self.get_parser()
        
        kotogram = parser.japanese_to_kotogram(clean_sentence)
        tokens_kotogram = split_kotogram(kotogram)
        
        # Extract features for all tokens
        # This gives us a Tuple[Token, ...]
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
        augmented_tuples = self.augment_tokens(token_tuple)
        
        # Join back surfaces
        results = set()
        for aug_tuple in augmented_tuples:
            surface_list = [get_surface(token) for token in aug_tuple]
            results.add("".join(surface_list))
            
        return results

    def filter_grammatical(self, sentences: Set[str]) -> List[str]:
        """Filter input sentences using the cached grammaticality model."""
        parser = self.get_parser()
        
        valid_sentences = []
        for sent in sentences:
            k = parser.japanese_to_kotogram(sent)
            # Use consolidated grammar() function
            if grammar(k).is_grammatic:
                valid_sentences.append(sent)
                
        return sorted(list(set(valid_sentences)))



def augment(sentences: List[str]) -> List[str]:
    """Augment a list of Japanese sentences and filter for grammaticality.
    
    This is the main entry point for the module.
    
    Args:
        sentences: List of input Japanese sentences (unspaced).
        
    Returns:
        Sorted unique list of augmented, grammatically valid sentences.
    """
    from kotogram.validation import ensure_list_of_strings
    ensure_list_of_strings(sentences, "sentences")

    augmenter = Augmenter()
    
    # 1. Augment
    candidates = set()
    for s in sentences:
        candidates.update(augmenter.process_sentence(s))
        
    # 2. Filter
    return augmenter.filter_grammatical(candidates)

