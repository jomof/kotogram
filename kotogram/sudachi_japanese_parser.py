"""Sudachi-based implementation of Japanese parser."""

from typing import Any, Dict, List, Optional
from kotogram.exceptions import MissingMappingError

from .japanese_parser import (
    JapaneseParser,
    POS_MAP,
    POS1_MAP,
    POS2_MAP,
    POS3_MAP,
    CONJUGATED_TYPE_MAP,
    CONJUGATED_FORM_MAP,
)


class SudachiJapaneseParser(JapaneseParser):
    """Sudachi-based Japanese parser using SudachiDict.

    This parser uses SudachiPy with the SudachiDict dictionary to tokenize and analyze
    Japanese text, converting it into kotogram compact format.
    """

    def __init__(self, dict_type: str = 'full', validate: bool = False) -> None:
        """Initialize the Sudachi Japanese parser.

        Args:
            dict_type: Dictionary type to use ('small', 'core', or 'full').
                      Default is 'full' for maximum coverage.
            validate: If True, raises descriptive exceptions when mapping lookups fail.
                     Useful for debugging unmapped linguistic features.
        """
        # Lazy import to avoid requiring Sudachi for the abstract interface
        from sudachipy import dictionary

        self.dict_obj = dictionary.Dictionary(dict=dict_type)
        self.tokenizer = self.dict_obj.create()
        self.validate = validate

    def japanese_to_kotogram(self, text: str) -> str:
        """Convert Japanese text to kotogram compact representation.

        Args:
            text: Japanese text to parse

        Returns:
            Kotogram compact sentence representation with encoded linguistic features
        """
        from kotogram.validation import ensure_string
        ensure_string(text, "text")

        # Fix for special case with っ character
        text = text.replace(' っ', 'っ').replace('っ ', 'っ')

        tokens = self.tokenizer.tokenize(text)
        return self._tokens_to_kotogram(tokens)

    def _tokens_to_kotogram(self, tokens: List[Any]) -> str:
        """Convert Sudachi tokens to kotogram format.

        Args:
            tokens: List of Sudachi token objects

        Returns:
            Kotogram compact sentence representation
        """
        parsed_tokens = []

        for token in tokens:
            # Extract token features
            surface = token.surface()
            pos_tuple = token.part_of_speech()  # Tuple of 6 elements
            dictionary_form = token.dictionary_form()
            reading_form = token.reading_form()

            # Parse POS tuple
            # Format: (POS, POS1, POS2, POS3, conjugation_type, conjugation_form)
            parsed_token = {
                "surface": surface,
            }

            def add(field: str, value: Optional[str]) -> None:
                """Add field to token dict if value is not empty."""
                if value is None or value == '""' or value == "" or value == "*":
                    return
                parsed_token[field] = value

            def validated_lookup(mapping: Dict[str, str], key: str, map_name: str) -> Optional[str]:
                """Lookup with validation support."""
                if key == "" or key == "*":
                    return mapping.get(key, None)

                result = mapping.get(key)
                if self.validate and result is None and key != "" and key != "*":
                    raise MissingMappingError(
                        map_name=map_name,
                        key=key,
                        context=f"Sudachi token: surface='{surface}', pos={pos_tuple}"
                    )
                return result

            # Part of Speech (0, 1, 2 are POS levels, 3 is detail)
            if len(pos_tuple) >= 1:
                add("pos", validated_lookup(POS_MAP, pos_tuple[0], "POS_MAP"))
            if len(pos_tuple) >= 2:
                add("pos_detail_1", validated_lookup(POS1_MAP, pos_tuple[1], "POS1_MAP"))
            if len(pos_tuple) >= 3:
                add("pos_detail_2", validated_lookup(POS2_MAP, pos_tuple[2], "POS2_MAP"))
            if len(pos_tuple) >= 4:
                add("pos_detail_3", validated_lookup(POS3_MAP, pos_tuple[3], "POS3_MAP"))

            # Conjugation (4 is conjugation type, 5 is conjugation form)
            if len(pos_tuple) >= 5:
                add("conjugated_type", validated_lookup(CONJUGATED_TYPE_MAP, pos_tuple[4], "CONJUGATED_TYPE_MAP"))
            if len(pos_tuple) >= 6:
                add("conjugated_form", validated_lookup(CONJUGATED_FORM_MAP, pos_tuple[5], "CONJUGATED_FORM_MAP"))

            # Lexical information
            add("lemma", dictionary_form if dictionary_form != surface else None)
            add("base_orthography", dictionary_form if dictionary_form != surface else None)
            add("surface_pronunciation", reading_form if reading_form != surface else None)

            parsed_tokens.append(parsed_token)

        # Convert parsed tokens to kotogram format
        return self._raw_tokens_to_kotogram(parsed_tokens)

    def _raw_token_to_kotogram(self, token: Dict[str, Any]) -> str:
        """Convert a single parsed token to kotogram format.

        Args:
            token: Dictionary containing parsed token features

        Returns:
            Kotogram representation of the token
        """
        recombined = ""
        surface = token["surface"]
        pos = token.get("pos", "")
        pos_detail_1 = token.get("pos_detail_1")
        pos_detail_2 = token.get("pos_detail_2")

        conjugated_type = token.get("conjugated_type")
        conjugated_form = token.get("conjugated_form")
        lemma = token.get("lemma")
        base = token.get("base_orthography", None)
        pronunciation = token.get("surface_pronunciation", None)

        pos_detail_3 = token.get("pos_detail_3")
        pos_code = pos if pos else ""

        recombined += f"⌈ˢ{surface}ᵖ{pos_code}"
        if pos_detail_1:
            recombined += f":{pos_detail_1}"
        if pos_detail_2 and pos_detail_2 != "general":
            recombined += f":{pos_detail_2}"
        if pos_detail_3 and pos_detail_3 != "general":
            recombined += f":{pos_detail_3}"
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
