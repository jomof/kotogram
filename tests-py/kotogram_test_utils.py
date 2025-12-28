"""Utilities for testing kotogram functionality."""

import json
from typing import Any, List

from kotogram.analysis import (
    FormalityLevel,
    GenderLevel,
    GrammarAnalysis,
    RegisterLevel,
)
from kotogram.kotogram import Token, extract_token_features, split_kotogram


class KotogramTestUtils:
    """Utility methods for testing kotogram."""

    @staticmethod
    def grammar_analysis_from_json(json_str: str) -> GrammarAnalysis:
        """Deserialize analysis result from JSON string."""
        d = json.loads(json_str)

        # Map strings back to Enums
        d["formality"] = FormalityLevel(d["formality"])
        d["gender"] = GenderLevel(d["gender"])
        d["registers"] = {RegisterLevel(r) for r in d["registers"]}
        d["register_scores"] = {
            RegisterLevel(k): v for k, v in d["register_scores"].items()
        }

        if "kc_top" not in d:
            d["kc_top"] = None
        else:
            # JSON keys are always strings, map back to int
            d["kc_top"] = {int(k): v for k, v in d["kc_top"].items()}

        return GrammarAnalysis(**d)

    @staticmethod
    def tokenize_sentence(sentence: str, parser: Any) -> List[Token]:
        """Tokenize a sentence into a list of Token objects using the provided parser."""
        from dataclasses import asdict

        k = parser.japanese_to_kotogram(sentence)
        return [
            Token(
                extract_token_features(t).surface or t,
                asdict(extract_token_features(t)),
            )
            for t in split_kotogram(k)
        ]
