"""Kotogram - A dual Python/TypeScript library for Japanese text parsing and encoding."""

__version__ = "0.0.20"

from .japanese_parser import JapaneseParser
from .sudachi_japanese_parser import SudachiJapaneseParser
from .kotogram import kotogram_to_japanese, split_kotogram, extract_token_features
from .analysis import grammar, GrammarAnalysis
from .constants import FormalityLevel, GenderLevel, RegisterLevel
from .augment import augment


__all__ = [
    "JapaneseParser",
    "SudachiJapaneseParser",
    "kotogram_to_japanese",
    "split_kotogram",
    "grammar",
    "GrammarAnalysis",
    "FormalityLevel",
    "GenderLevel",
    "RegisterLevel",
    "extract_token_features",
    "augment",
]
