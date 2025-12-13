"""Module for augmenting Japanese sentences with grammatical variations.

This module provides an extensible framework for generating variations of Japanese
sentences (e.g. changing formality, dropping topics, swapping pronouns) and verifying
their grammaticality using a neural model.
"""

from abc import ABC, abstractmethod
from typing import Set, Tuple, List, Optional
from itertools import product
from kotogram.kotogram import split_kotogram, extract_token_features
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.analysis import grammaticality, _load_style_model

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

DROPPABLE_TOPIC_STARTS = [
    ('私', 'は'),
    ('僕', 'は'),
    ('俺', 'は'),
    ('わたし', 'は'),
    ('ぼく', 'は'),
    ('おれ', 'は'),
]

PROGRESSIVE_END_PATTERNS = [
    (('て', 'い', 'ます', '。'), ('て', 'いる', '。')),
    (('て', 'い', 'まし', 'た', '。'), ('て', 'い', 'た', '。')),
    (('で', 'い', 'ます', '。'), ('で', 'いる', '。')),
    (('で', 'い', 'まし', 'た', '。'), ('で', 'い', 'た', '。')),
    # Without period
    (('て', 'い', 'ます'), ('て', 'いる')),
    (('て', 'い', 'まし', 'た'), ('て', 'い', 'た')),
    (('で', 'い', 'ます'), ('で', 'いる')),
    (('で', 'い', 'まし', 'た'), ('で', 'い', 'た')),
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


class AugmentationRule(ABC):
    """Abstract base class for sentence augmentation rules."""
    
    @abstractmethod
    def apply(self, tokens: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
        """Apply the rule to a sequence of tokens.
        
        Args:
            tokens: A tuple of surface forms representing the sentence.
            
        Returns:
            A set of token tuples comprising the original and any valid variations.
        """
        pass


class PronounRule(AugmentationRule):
    """Augments sentences by swapping first-person pronouns."""
    
    def apply(self, tokens: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
        pronoun_indices = [i for i, t in enumerate(tokens) if t in FIRST_PERSON_PRONOUNS]
        
        if not pronoun_indices:
            return {tokens}
            
        result = set()
        for combo in product(FIRST_PERSON_PRONOUNS, repeat=len(pronoun_indices)):
            new_tokens = list(tokens)
            for idx, new_pronoun in zip(pronoun_indices, combo):
                new_tokens[idx] = new_pronoun
            result.add(tuple(new_tokens))
            
        return result


class CopulaRule(AugmentationRule):
    """Augments sentences by changing copula formality (da <-> desu)."""
    
    def apply(self, tokens: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
        result = {tokens}
        
        for da_toks, desu_toks in COPULA_PATTERNS:
            # Check if sentence ends with da_toks
            if len(tokens) >= len(da_toks) and tokens[-len(da_toks):] == da_toks:
                new_tokens = tokens[:-len(da_toks)] + desu_toks
                result.add(new_tokens)
                
            # Check if sentence ends with desu_toks
            if len(tokens) >= len(desu_toks) and tokens[-len(desu_toks):] == desu_toks:
                new_tokens = tokens[:-len(desu_toks)] + da_toks
                result.add(new_tokens)
                
        return result


class ContractionRule(AugmentationRule):
    """Augments sentences by swapping contractions (e.g. dewa <-> ja)."""
    
    def apply(self, tokens: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
        result = {tokens}
        
        for form_a, form_b in CONTRACTION_PATTERNS:
            len_a = len(form_a)
            len_b = len(form_b)
            
            # Form A -> Form B (iterate through tokens to find matches)
            # Since matches can be anywhere, we simply check sliding windows?
            # Or simpler: exact match replacement.
            # However, unlike ends-with check, this needs to scan the whole sentence.
            # For efficiency in Python with short lists, iterating indices is fine.
            
            # We must be careful about multiple occurrences.
            # A recursive or iterative approach replacing one by one is needed.
            # Let's collect all start indices first.
            
            indices_a = []
            for i in range(len(tokens) - len_a + 1):
                if tokens[i:i+len_a] == form_a:
                    indices_a.append(i)
            
            if indices_a:
                # Generate all combinations of replacing/not replacing
                # Replacing 'all' is distinct from replacing 'some'. 
                # For equivalent generation, usually ALL instances being consistent isn't required by language,
                # but mixed style might be odd ("食べているけど飲んでる"). 
                # Let's generate the version where ALL are swapped, and the original.
                # Or just do simplistic "replace one by one" -> exhaustive?
                # Given sentences are short, exhaustive combination is better coverage.
                
                # Simplified approach: just generate the version with ALL replacements?
                # Users often want full coverage. Let's do all combinations.
                import itertools
                for replacement_mask in itertools.product([False, True], repeat=len(indices_a)):
                    if not any(replacement_mask):
                        continue
                        
                    new_tokens = []
                    last_idx = 0
                    current_match_idx = 0
                    
                    matches = sorted(indices_a) # just in case
                    
                    for i, do_replace in zip(matches, replacement_mask):
                        # Copy everything up to this match from last position
                        new_tokens.extend(tokens[last_idx:i])
                        if do_replace:
                            new_tokens.extend(form_b)
                        else:
                            new_tokens.extend(form_a)
                        last_idx = i + len_a
                        
                    # Copy tail
                    new_tokens.extend(tokens[last_idx:])
                    result.add(tuple(new_tokens))

            # Form B -> Form A
            indices_b = []
            for i in range(len(tokens) - len_b + 1):
                if tokens[i:i+len_b] == form_b:
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
                            new_tokens.extend(form_b)
                        last_idx = i + len_b
                        
                    new_tokens.extend(tokens[last_idx:])
                    result.add(tuple(new_tokens))
                    
        return result


class TopicDropRule(AugmentationRule):
    """Aguments sentences by dropping clear subjects/topics at the start."""
    
    def apply(self, tokens: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
        result = {tokens}
        
        for topic_toks in DROPPABLE_TOPIC_STARTS:
            if len(tokens) > len(topic_toks) and tokens[:len(topic_toks)] == topic_toks:
                new_tokens = tokens[len(topic_toks):]
                result.add(new_tokens)
                
        return result


class ProgressiveRule(AugmentationRule):
    """Augments sentences by changing progressive form formality (te iru <-> te i masu)."""
    
    def apply(self, tokens: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
        result = {tokens}
        
        for polite_toks, plain_toks in PROGRESSIVE_END_PATTERNS:
            if len(tokens) >= len(polite_toks) and tokens[-len(polite_toks):] == polite_toks:
                new_tokens = tokens[:-len(polite_toks)] + plain_toks
                result.add(new_tokens)
                
            if len(tokens) >= len(plain_toks) and tokens[-len(plain_toks):] == plain_toks:
                new_tokens = tokens[:-len(plain_toks)] + polite_toks
                result.add(new_tokens)
                
        return result


class PluralRule(AugmentationRule):
    """Augments sentences by swapping kanji/hiragana for plural suffix."""
    
    def apply(self, tokens: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
        result = {tokens}
        
        for kanji, hiragana in PLURAL_PATTERNS:
            indices_kanji = [i for i, t in enumerate(tokens) if t == kanji]
            if indices_kanji:
                 for combo in product([kanji, hiragana], repeat=len(indices_kanji)):
                     new_tokens = list(tokens)
                     for idx, val in zip(indices_kanji, combo):
                         new_tokens[idx] = val
                     result.add(tuple(new_tokens))

            indices_hiragana = [i for i, t in enumerate(tokens) if t == hiragana]
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

    def __init__(self):
        self.rules: List[AugmentationRule] = [
            PronounRule(),
            CopulaRule(),
            ContractionRule(),
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
        
    def augment_tokens(self, tokens: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
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
        
        try:
            kotogram = parser.japanese_to_kotogram(clean_sentence)
            tokens_kotogram = split_kotogram(kotogram)
            token_surfaces = []
            for t in tokens_kotogram:
                f = extract_token_features(t)
                if f and 'surface' in f:
                    token_surfaces.append(f['surface'])
            
            if not token_surfaces:
                return {clean_sentence}
                
            token_tuple = tuple(token_surfaces)
            augmented_tuples = self.augment_tokens(token_tuple)
            
            return {"".join(t) for t in augmented_tuples}
            
        except Exception:
            # Fallback on parsing error
            return {clean_sentence}

    def filter_grammatical(self, sentences: Set[str]) -> List[str]:
        """Filter input sentences using the cached grammaticality model."""
        parser = self.get_parser()
        # Pre-load model if not loaded (warmup)
        _load_style_model()
        
        valid_sentences = []
        for sent in sentences:
            try:
                k = parser.japanese_to_kotogram(sent)
                if grammaticality(k, use_model=True):
                    valid_sentences.append(sent)
            except Exception:
                # Be conservative: assume valid if parse fails? Or drop?
                # Previous logic kept them. Let's keep them to be safe.
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
    augmenter = Augmenter()
    
    # 1. Augment
    candidates = set()
    for s in sentences:
        candidates.update(augmenter.process_sentence(s))
        
    # 2. Filter
    return augmenter.filter_grammatical(candidates)
