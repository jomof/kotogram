"""Kotogram - A dual Python/TypeScript library for Japanese text parsing and encoding."""

__version__ = "0.0.15"

from .japanese_parser import JapaneseParser
from .sudachi_japanese_parser import SudachiJapaneseParser
from .kotogram import kotogram_to_japanese, split_kotogram, extract_token_features
from .analysis import formality, FormalityLevel, gender, GenderLevel, style, grammaticality
from .augment import augment


__all__ = [
    "JapaneseParser",
    "SudachiJapaneseParser",
    "kotogram_to_japanese",
    "split_kotogram",
    "formality",
    "FormalityLevel",
    "gender",
    "GenderLevel",
    "style",
    "grammaticality",
    "extract_token_features",
    "augment",
]
