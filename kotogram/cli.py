#!/usr/bin/env python3
"""Command-line tool for working with kotograms."""

import argparse
import json
import sys

from kotogram.sudachi_japanese_parser import SudachiJapaneseParser


def get_parser() -> SudachiJapaneseParser:
    """Get the Sudachi parser instance."""
    return SudachiJapaneseParser()


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse Japanese text to kotogram format."""
    parser = get_parser()
    text = str(args.text)

    if text == "-":
        text = sys.stdin.read().strip()

    kotogram = parser.japanese_to_kotogram(text)
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


def cmd_grammar(args: argparse.Namespace) -> int:
    """Analyze grammar of Japanese text."""
    text = str(args.text)

    if text == "-":
        text = sys.stdin.read().strip()

    # If it doesn't look like a kotogram, parse it first
    if not text.startswith("⌈"):
        parser = get_parser()
        kotogram = parser.japanese_to_kotogram(text)
    else:
        kotogram = text

    from kotogram.analysis import grammar

    result = grammar(kotogram)

    # Use to_json() then load/dump for pretty printing
    data = json.loads(result.to_json())
    print(json.dumps(data, indent=2, ensure_ascii=False))
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
