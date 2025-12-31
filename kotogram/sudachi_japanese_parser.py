"""Sudachi-based implementation of Japanese parser."""

from typing import Any, Dict, List, Literal, Mapping, Optional

from kotogram.exceptions import MissingMappingError
from kotogram.kotogram import Token

from .japanese_parser import (
    CONJUGATED_FORM_MAP,
    CONJUGATED_TYPE_MAP,
    POS1_MAP,
    POS2_MAP,
    POS3_MAP,
    POS_MAP,
    JapaneseParser,
    KotogramFormat,
)


class SudachiJapaneseParser(JapaneseParser):
    """Sudachi-based Japanese parser using SudachiDict.

    This parser uses SudachiPy with the SudachiDict dictionary to tokenize and analyze
    Japanese text, converting it into kotogram compact format.
    """

    def __init__(self, dict_type: str = "full", validate: bool = False) -> None:
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

    def japanese_to_kotogram(
        self, text: str, fmt: Literal["Default", "TrainingMask"] = "Default"
    ) -> str:
        """Convert Japanese text to kotogram compact representation.

        Args:
            text: Raw Japanese input string.
            fmt: Format option (e.g. KotogramFormat.TRAINING_MASK for anonymization).

        Returns:
            Kotogram compact sentence representation with encoded linguistic features
        """
        # from kotogram.japanese_parser import KotogramFormat # Removed, moved to top-level
        from kotogram.validation import ensure_string

        ensure_string(text, "text")

        # Fix for special case with っ character
        text = text.replace(" っ", "っ").replace("っ ", "っ")

        # Fix for phantom tokens caused by ellipsis in Sudachi (maps … -> ...)
        text = text.replace("…", "...")

        # Normalize double exclamation mark (‼ -> !!) for consistent tokenization
        text = text.replace("‼", "!!")

        sudachi_tokens = self.tokenizer.tokenize(text)

        # Convert to Kotogram Token objects first
        tokens = self._to_kotogram_tokens(sudachi_tokens)

        # Handle Training Mask (Given Name Replacement)
        if fmt == KotogramFormat.TRAINING_MASK:
            from kotogram.masking import apply_training_mask

            tokens = apply_training_mask(tokens)

        # Serialize to kotogram string
        return self._tokens_to_string(tokens)

    def _to_kotogram_tokens(self, tokens: List[Any]) -> List["Token"]:
        # pylint: disable=cell-var-from-loop
        """Convert Sudachi tokens to Kotogram Token objects.

        Args:
            tokens: List of Sudachi token objects

        Returns:
            List of kotogram.Token objects
        """
        from kotogram.kotogram import TokenFeatures

        k_tokens = []

        for token in tokens:
            # Extract token features
            surface = token.surface()
            pos_tuple = token.part_of_speech()  # Tuple of 6 elements
            dictionary_form = token.dictionary_form()
            reading_form = token.reading_form()

            # Parse POS tuple
            # Format: (POS, POS1, POS2, POS3, conjugation_type, conjugation_form)
            # Use a dict temporarily to accumulate features before constructing TokenFeatures
            feature_dict: Dict[str, Any] = {}

            def add(field: str, value: Optional[str]) -> None:
                """Add field to token dict if value is not empty."""
                if value is None or value == '""' or value == "":
                    return
                # Allow "*" specifically for lemma to indicate explicit surface identity
                if value == "*" and field != "lemma":
                    return
                feature_dict[field] = value

            def validated_lookup(
                mapping: Mapping[str, str], key: str, map_name: str
            ) -> Optional[str]:
                """Lookup with validation support."""
                if key in ("", "*"):
                    return mapping.get(key, None)

                result = mapping.get(key)
                if self.validate and result is None and key not in ("", "*"):
                    raise MissingMappingError(
                        map_name=map_name,
                        key=key,
                        context=f"Sudachi token: surface='{surface}', pos={pos_tuple}",
                    )
                return result

            # Part of Speech (0, 1, 2 are POS levels, 3 is detail)
            if len(pos_tuple) >= 1:
                add("pos", validated_lookup(POS_MAP, pos_tuple[0], "POS_MAP"))
            if len(pos_tuple) >= 2:
                add(
                    "pos_detail_1", validated_lookup(POS1_MAP, pos_tuple[1], "POS1_MAP")
                )
            if len(pos_tuple) >= 3:
                add(
                    "pos_detail_2", validated_lookup(POS2_MAP, pos_tuple[2], "POS2_MAP")
                )
            if len(pos_tuple) >= 4:
                add(
                    "pos_detail_3", validated_lookup(POS3_MAP, pos_tuple[3], "POS3_MAP")
                )

            # Conjugation (4 is conjugation type, 5 is conjugation form)
            if len(pos_tuple) >= 5:
                add(
                    "conjugated_type",
                    validated_lookup(
                        CONJUGATED_TYPE_MAP, pos_tuple[4], "CONJUGATED_TYPE_MAP"
                    ),
                )
            if len(pos_tuple) >= 6:
                add(
                    "conjugated_form",
                    validated_lookup(
                        CONJUGATED_FORM_MAP, pos_tuple[5], "CONJUGATED_FORM_MAP"
                    ),
                )

            # Lexical information
            # Explicitly use "*" if lemma matches surface
            add("lemma", dictionary_form if dictionary_form != surface else "*")
            add(
                "base_orth",
                dictionary_form if dictionary_form != surface else None,
            )
            add(
                "reading",
                reading_form if reading_form != surface else None,
            )

            # Construct TokenFeatures dataclass from dict
            features = TokenFeatures(**feature_dict)
            k_tokens.append(Token(surface, features=features))

        return k_tokens

    def _tokens_to_string(self, tokens: List["Token"]) -> str:
        """Serialize a list of Kotogram Tokens to compact string format."""
        parts = []
        for token in tokens:
            parts.append(self._token_to_string(token))
        return "".join(parts)

    def _token_to_string(self, token: "Token") -> str:
        # pylint: disable=too-many-locals
        """Convert a single Token object to kotogram fragment."""
        from kotogram.japanese_parser import (
            OBSCURE_KOTOGRAM,
        )

        surface = token.surface
        features = token.features

        # Access features with defaults via named properties (TokenFeatures default is "")
        pos = features.pos
        pos_detail_1 = features.pos_detail_1
        pos_detail_2 = features.pos_detail_2
        pos_detail_3 = features.pos_detail_3

        conjugated_type = features.conjugated_type
        conjugated_form = features.conjugated_form
        lemma = features.lemma
        base = features.base_orth
        pronunciation = features.reading

        pos_code = pos if pos else ""

        inner = f"ˢ{surface}ᵖ{pos_code}"

        if pos_detail_1:
            inner += f":{pos_detail_1}"
        if pos_detail_2 and pos_detail_2 != "general":
            inner += f":{pos_detail_2}"
        if pos_detail_3 and pos_detail_3 != "general":
            inner += f":{pos_detail_3}"
        if conjugated_type:
            inner += f":{conjugated_type}"
        if conjugated_form:
            inner += f":{conjugated_form}"
        if base:
            inner += f"ᵇ{base}"
        if lemma and lemma != surface:
            inner += f"ᵈ{lemma}"
        if pronunciation and pronunciation != surface:
            inner += f"ʳ{pronunciation}"

        # Obfuscate if enabled
        from kotogram.kotogram import obscure_kotogram_token_string

        inner = obscure_kotogram_token_string(inner)

        if OBSCURE_KOTOGRAM == 1:
            return f"[{inner}]"

        return f"⌈{inner}⌉"
