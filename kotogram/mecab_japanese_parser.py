"""MeCab-based implementation of Japanese parser."""

import re
from typing import Any, Dict, List, Optional

from .japanese_parser import (
    JapaneseParser,
    POS_MAP,
    POS1_MAP,
    POS2_MAP,
    CONJUGATED_TYPE_MAP,
    CONJUGATED_FORM_MAP,
)


class MecabJapaneseParser(JapaneseParser):
    """MeCab-based Japanese parser using UniDic dictionary.

    This parser uses MeCab with the UniDic dictionary to tokenize and analyze
    Japanese text, converting it into kotogram compact format.
    """

    def __init__(self, mecab_tagger: Optional[object] = None) -> None:
        """Initialize the MeCab Japanese parser.

        Args:
            mecab_tagger: Optional pre-configured MeCab tagger instance.
                         If None, will attempt to create one using unidic.
        """
        if mecab_tagger is None:
            # Lazy import to avoid requiring MeCab for the abstract interface
            import MeCab
            import unidic
            import os

            # Try common unidic dictionary locations
            unidic.DICDIR = "/usr/local/python/3.10.13/lib/python3.10/site-packages/unidic/dicdir"
            if not os.path.isdir(unidic.DICDIR):
                unidic.DICDIR = "/usr/local/lib/python3.10/dist-packages/unidic/dicdir"
            if not os.path.isdir(unidic.DICDIR):
                unidic.DICDIR = "/usr/local/python/3.12.1/lib/python3.12/site-packages/unidic/dicdir"

            self.tagger = MeCab.Tagger('-d "{}"'.format(unidic.DICDIR))
        else:
            self.tagger = mecab_tagger

    def japanese_to_kotogram(self, text: str) -> str:
        """Convert Japanese text to kotogram compact representation.

        Args:
            text: Japanese text to parse

        Returns:
            Kotogram compact sentence representation with encoded linguistic features
        """
        # Fix for special case with っ character
        text = text.replace(' っ', 'っ').replace('っ ', 'っ')
        raw = self.tagger.parse(text)
        return self._mecab_raw_to_kotogram(raw)

    def _mecab_raw_to_kotogram(self, raw: str) -> str:
        """Convert raw MeCab output to kotogram format.

        Args:
            raw: Raw output from MeCab tagger

        Returns:
            Kotogram compact sentence representation
        """
        tokens = self._parse_raw_mecab_output(raw)
        return self._raw_tokens_to_kotogram(tokens)

    def _parse_raw_mecab_output(self, raw_output: str) -> List[Dict[str, Any]]:
        """Parse raw MeCab output into structured token dictionaries.

        Args:
            raw_output: Raw text output from MeCab tagger

        Returns:
            List of token dictionaries with parsed features
        """
        tokens = []
        for line in raw_output.split("\n"):
            if line == "EOS":
                continue
            line = line.strip()
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            surface = parts[0]
            features = parts[1].split(",")
            features += [""] * (20 - len(features) - 1)
            token = {
                "surface": surface,
            }

            def add(field: str, value: str) -> None:
                """Add field to token dict if value is not empty."""
                if value == '""' or value == "":
                    return
                token[field] = value

            def get_feature(features_list: List[str], index: int, default_value: str = "") -> str:
                """Safely get a feature from the list by index."""
                if index < len(features_list):
                    return features_list[index]
                return default_value

            # Part of Speech (POS)
            add("pos", POS_MAP.get(get_feature(features, 0)))
            add("pos_detail_1", POS1_MAP.get(get_feature(features, 1)))
            add("pos_detail_2", POS2_MAP[get_feature(features, 2)])
            add("pos_detail_3", get_feature(features, 3))

            # Conjugation
            add("conjugated_type", CONJUGATED_TYPE_MAP.get(get_feature(features, 4)))
            add("conjugated_form", CONJUGATED_FORM_MAP.get(get_feature(features, 5)))

            # Lexical Information & Pronunciation
            add("lemma_pronunciation", get_feature(features, 6))
            add("lemma", get_feature(features, 7))

            # Orthography (Spelling)
            add("surface_orthography", get_feature(features, 8))
            add("base_orthography", get_feature(features, 10))

            # Pronunciation
            add("surface_pronunciation", get_feature(features, 9))
            add("base_pronunciation", get_feature(features, 11))
            add("surface_kana", get_feature(features, 20))
            add("base_kana", get_feature(features, 21))

            # Semantic and Grammatical Info
            add("word_origin", get_feature(features, 12))
            add("initial_inflection_type", get_feature(features, 13))
            add("initial_inflection_form", get_feature(features, 14))
            add("final_inflection_type", get_feature(features, 15))
            add("final_inflection_form", get_feature(features, 16))
            add("initial_connection_type", get_feature(features, 17))
            add("final_connection_type", get_feature(features, 18))
            add("entry_type", get_feature(features, 19))

            # Word Form (Pronunciation-based)
            add("surface_form_pronunciation", get_feature(features, 22))
            add("base_form_pronunciation", get_feature(features, 23))

            # Accent Information
            add("accent_type", get_feature(features, 24))
            add("accent_connection_type", get_feature(features, 25))
            add("accent_modification_type", get_feature(features, 26))

            # Unique Identifiers
            add("internal_id", get_feature(features, 27))
            add("lemma_id", get_feature(features, 28))

            tokens.append(token)
        return tokens

    def _raw_token_to_kotogram(self, token: Dict[str, Any]) -> str:
        """Convert a single parsed token to kotogram format.

        Args:
            token: Dictionary containing parsed token features

        Returns:
            Kotogram representation of the token
        """
        recombined = ""
        surface = token["surface"]
        pos = token["pos"]
        pos_detail_1 = token.get("pos_detail_1")
        pos_detail_2 = token.get("pos_detail_2")

        conjugated_type = token.get("conjugated_type")
        conjugated_form = token.get("conjugated_form")
        lemma = token.get("lemma")
        base = token.get("base_orthography", None)
        pronunciation = token.get("surface_pronunciation", None)

        pos_code = POS_MAP.get(pos, pos)

        recombined += f"⌈ˢ{surface}ᵖ{pos_code}"
        if pos_detail_1:
            recombined += f":{pos_detail_1}"
        if pos_detail_2 and pos_detail_2 != "general":
            recombined += f":{pos_detail_2}"
        if conjugated_type:
            recombined += f":{conjugated_type}"
        if conjugated_form:
            recombined += f":{conjugated_form}"
        if base:
            recombined += f"ᵇ{base}"
        if lemma and lemma != surface:
            recombined += f"ᵈ{lemma}"
        if pronunciation and pronunciation != surface:
            recombined += f"ʳ{pronunciation}"
        recombined += "⌉"
        return recombined

    def _raw_tokens_to_kotogram(self, tokens: List[Dict[str, Any]]) -> str:
        """Convert a list of parsed tokens to kotogram format.

        Args:
            tokens: List of token dictionaries

        Returns:
            Kotogram representation of the full sentence
        """
        recombined = ""
        for token in tokens:
            recombined += self._raw_token_to_kotogram(token)
        return recombined


def kotogram_to_japanese(kotogram: str, spaces: bool = False, collapse_punctuation: bool = True) -> str:
    """Convert kotogram compact representation back to Japanese text.

    Args:
        kotogram: Kotogram compact sentence representation
        spaces: Whether to add spaces between tokens
        collapse_punctuation: Whether to remove spaces around punctuation

    Returns:
        Japanese text reconstructed from kotogram
    """
    from .japanese_parser import POS_TO_CHARS

    pattern = r'ˢ(.*?)ᵖ'
    matches = re.findall(pattern, kotogram, re.DOTALL)
    if spaces:
        result = ' '.join(matches).replace('{ ', '{').replace(' }', '}')
        if collapse_punctuation:
            for punc in POS_TO_CHARS['auxs']:
                if punc == '{' or punc == '}':
                    continue
                result = result.replace(f' {punc}', punc)
                result = result.replace(f'{punc} ', punc)
        return result
    else:
        return ''.join(matches)


def split_kotogram(kotogram: str) -> List[str]:
    """Split a kotogram sentence into individual token representations.

    Args:
        kotogram: Kotogram compact sentence representation

    Returns:
        List of individual token kotogram representations
    """
    return re.findall(r'⌈[^⌉]*⌉', kotogram)
