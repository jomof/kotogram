#!/usr/bin/env python3
"""Command-line tool for working with kotograms."""

import argparse
import json
import sys

from kotogram.japanese_parser import KotogramFormat
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser


def get_parser() -> SudachiJapaneseParser:
    """Get the Sudachi parser instance."""
    return SudachiJapaneseParser()


def _get_kotogram_from_args(args: argparse.Namespace) -> str:
    """Helper to extract/parse kotogram from arguments."""
    text = str(args.text)
    if text == "-":
        text = sys.stdin.read().strip()

    # If it doesn't look like a kotogram, parse it first
    if not text.startswith("⌈"):
        parser = get_parser()
        # Always use training mask for inference/analysis commands (grammar, etc)
        # to ensure names are anonymized and match training distribution.
        return parser.japanese_to_kotogram(text, fmt=KotogramFormat.TRAINING_MASK)
    return text


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse Japanese text to kotogram format."""
    parser = get_parser()
    text = str(args.text)

    if text == "-":
        text = sys.stdin.read().strip()

    fmt = KotogramFormat.DEFAULT
    if getattr(args, "format_training_mask", False):
        fmt = KotogramFormat.TRAINING_MASK

    kotogram = parser.japanese_to_kotogram(text, fmt=fmt)
    print(kotogram)
    return 0


def cmd_raw(args: argparse.Namespace) -> int:
    """Show raw parser output for inspection."""
    text = str(args.text)

    if text == "-":
        text = sys.stdin.read().strip()

    # Print original sentence
    print(f"Input: {text}")
    print()

    from sudachipy import dictionary

    dict_obj = dictionary.Dictionary(dict="full")
    tokenizer = dict_obj.create()
    tokens = tokenizer.tokenize(text)

    print("Sudachi raw output:")
    for token in tokens:
        print(f"Surface: {token.surface()}")
        print(f"  POS: {token.part_of_speech()}")
        print(f"  Dictionary form: {token.dictionary_form()}")
        print(f"  Reading form: {token.reading_form()}")
        print(f"  Normalized form: {token.normalized_form()}")
        print()

    return 0


def _check_model() -> bool:
    """Check if model exists and print error if not."""
    from kotogram.analysis import check_model_available

    if not check_model_available():
        sys.stderr.write("\nError: Model file not found.\n")
        sys.stderr.write("Please ensure the style model is trained or installed.\n")
        return False
    return True


def cmd_grammar(args: argparse.Namespace) -> int:
    """Analyze grammar of Japanese text."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)
    from kotogram.analysis import grammar

    result = grammar(kotogram)

    # Use to_json() then load/dump for pretty printing
    data = json.loads(result.to_json())
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def cmd_formality_score(args: argparse.Namespace) -> int:
    """Get formality score."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)
    from kotogram.analysis import grammar

    result = grammar(kotogram)
    print(result.formality_score)
    return 0


def cmd_formality_is_pragmatic(args: argparse.Namespace) -> int:
    """Get formality pragmatic status."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)
    from kotogram.analysis import grammar

    result = grammar(kotogram)
    print(str(result.formality_is_pragmatic).lower())
    return 0


def cmd_grammaticality(args: argparse.Namespace) -> int:
    """Get grammaticality score."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)
    from kotogram.analysis import grammar

    result = grammar(kotogram)
    # The user asked for "grammaticality", but Vulture flagged "grammaticality_score".
    # Printing the score is more informative.
    print(result.grammaticality_score)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="kotogram",
        description="Command-line tool for working with kotograms",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # parse command
    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse Japanese text to kotogram format",
    )
    parse_parser.add_argument(
        "text",
        help="Japanese text to parse (use '-' to read from stdin)",
    )
    parse_parser.add_argument(
        "--format-training-mask",
        action="store_true",
        help="Apply training mask (replace given names with placeholder)",
    )
    parse_parser.set_defaults(func=cmd_parse)

    # raw command
    raw_parser = subparsers.add_parser(
        "raw",
        help="Show raw parser output for inspection",
    )
    raw_parser.add_argument(
        "text",
        help="Japanese text to parse (use '-' to read from stdin)",
    )
    raw_parser.set_defaults(func=cmd_raw)

    # grammar command
    grammar_parser = subparsers.add_parser(
        "grammar",
        help="Analyze grammar of Japanese text",
    )
    grammar_parser.add_argument(
        "text",
        help="Japanese text or kotogram to analyze (use '-' to read from stdin)",
    )
    grammar_parser.set_defaults(func=cmd_grammar)

    # New commands to expose analysis properties
    # formality_score
    fs_parser = subparsers.add_parser(
        "formality_score",
        help="Get formality score (-1.0 to 1.0)",
    )
    fs_parser.add_argument("text", help="Text to analyze")
    fs_parser.set_defaults(func=cmd_formality_score)

    # formality_is_pragmatic
    fp_parser = subparsers.add_parser(
        "formality_is_pragmatic",
        help="Check if formality is pragmatically determined",
    )
    fp_parser.add_argument("text", help="Text to analyze")
    fp_parser.set_defaults(func=cmd_formality_is_pragmatic)

    # grammaticality
    g_parser = subparsers.add_parser(
        "grammaticality",
        help="Get grammaticality score (0.0 to 1.0)",
    )
    g_parser.add_argument("text", help="Text to analyze")
    g_parser.set_defaults(func=cmd_grammaticality)

    args = parser.parse_args()

    try:
        result = args.func(args)
        if isinstance(result, int):
            return result
        return 0
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
