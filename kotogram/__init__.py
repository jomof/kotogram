"""Kotogram - A dual Python/TypeScript library for Japanese text parsing and encoding."""

__version__ = "0.0.21"

from .analysis import GrammarAnalysis, grammar
from .augment import augment
from .constants import FormalityLevel, GenderLevel, RegisterLevel
from .japanese_parser import JapaneseParser
from .kotogram import extract_token_features, kotogram_to_japanese, split_kotogram
from .profile import ProfileReport, get_profile_report, increment_profile_counter
from .sudachi_japanese_parser import SudachiJapaneseParser

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
    "increment_profile_counter",
    "get_profile_report",
    "ProfileReport",
]
