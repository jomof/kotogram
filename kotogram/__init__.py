"""Kotogram - A dual Python/TypeScript library for Japanese text parsing and encoding."""

__version__ = "0.0.3"

from .codec import Codec
from .reversing_codec import ReversingCodec
from .japanese_parser import JapaneseParser
from .mecab_japanese_parser import MecabJapaneseParser
from .sudachi_japanese_parser import SudachiJapaneseParser
from .kotogram import kotogram_to_japanese, split_kotogram, extract_token_features
from .analysis import formality, FormalityLevel, gender, GenderLevel, style, grammaticality

__all__ = [
    "Codec",
    "ReversingCodec",
    "JapaneseParser",
    "MecabJapaneseParser",
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
]
