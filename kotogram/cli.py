#!/usr/bin/env python3
"""Command-line tool for working with kotograms."""

import argparse
import json
import sys
from typing import Any, Dict, Literal

from sudachipy import dictionary

from kotogram.analysis import _ANALYZER, check_model_available, grammar
from kotogram.constants import (
    FORMALITY_ID_TO_LABEL,
    GENDER_ID_TO_LABEL,
    REGISTER_ID_TO_LABEL,
)
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

    fmt: Literal["Default", "TrainingMask"] = KotogramFormat.DEFAULT
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

    result = grammar(kotogram)

    # Use to_json() then load/dump for pretty printing
    data = json.loads(result.to_json())

    # Filter register_probs: only include >= 50% and round to 2 decimal places
    if "register_probs" in data and data["register_probs"]:
        data["register_probs"] = {
            name: round(prob, 2)
            for name, prob in data["register_probs"].items()
            if prob >= 0.5
        }

    # Filter grammar_point_probs: only include >= 50%, round to 2 decimal places,
    # and sort by probability descending
    if "grammar_point_probs" in data and data["grammar_point_probs"]:
        filtered = {
            gp_id: round(prob, 2)
            for gp_id, prob in data["grammar_point_probs"].items()
            if prob >= 0.5
        }
        # Sort by probability descending
        data["grammar_point_probs"] = dict(
            sorted(filtered.items(), key=lambda x: -x[1])
        )

    # Round kc_top to 2 decimal places and sort by probability descending
    if "kc_top" in data and data["kc_top"]:
        rounded = {kc_id: round(prob, 2) for kc_id, prob in data["kc_top"].items()}
        # Sort by probability descending
        data["kc_top"] = dict(sorted(rounded.items(), key=lambda x: -x[1]))

    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def cmd_formality_score(args: argparse.Namespace) -> int:
    """Get formality score."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)

    result = grammar(kotogram)
    print(result.formality_score)
    return 0


def cmd_formality_is_pragmatic(args: argparse.Namespace) -> int:
    """Get formality pragmatic status."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)

    result = grammar(kotogram)
    print(str(result.formality_is_pragmatic).lower())
    return 0


def cmd_grammaticality(args: argparse.Namespace) -> int:
    """Get grammaticality score."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)

    result = grammar(kotogram)
    # The user asked for "grammaticality", but the property is "grammaticality_score".
    # Printing the score is more informative.
    print(result.grammaticality_score)
    return 0


def cmd_gender_score(args: argparse.Namespace) -> int:
    """Get gender score."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)

    result = grammar(kotogram)
    print(result.gender_score)
    return 0


def cmd_gender_is_pragmatic(args: argparse.Namespace) -> int:
    """Get gender pragmatic status."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)

    result = grammar(kotogram)
    print(str(result.gender_is_pragmatic).lower())
    return 0


def cmd_vocab(
    _args: Any,
) -> int:
    """Show vocabulary sizes."""
    if not _check_model():
        return 1

    _, tokenizer = _ANALYZER.load()
    # Cast to Tokenizer to appease Pylint if inference fails
    # (though explicit import should match if load returns Tokenizer)
    # The error was E1101: Instance of 'Tokenizer' has no 'vocab_sizes' member
    # If _ANALYZER.load() returns Tokenizer, Pylint should see it if it scans tokenizer.py properly.
    # But maybe it doesn't.
    print(json.dumps(tokenizer.get_vocab_sizes(), indent=2))
    return 0


def cmd_config(_args: Any) -> int:
    """Show model configuration."""
    if not _check_model():
        return 1

    model, _ = _ANALYZER.load()
    # pylint: disable=protected-access
    print(json.dumps(model.config.to_dict(), indent=2))
    return 0


def cmd_labels(_args: Any) -> int:
    """Show label mappings."""
    # Reconstruct inverse maps if needed or just dump what's available

    # helper to serializable
    def serialize(d: Dict[Any, Any]) -> Dict[str, Any]:
        return {str(k): str(v) for k, v in d.items()}

    data = {
        "formality": serialize(FORMALITY_ID_TO_LABEL),
        "register": serialize(REGISTER_ID_TO_LABEL),
        "gender": serialize(GENDER_ID_TO_LABEL),
    }

    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run a simple benchmark/smoke test."""
    if not _check_model():
        return 1

    kotogram = _get_kotogram_from_args(args)
    import time

    start = time.time()
    for _ in range(args.iterations):
        grammar(kotogram)
    end = time.time()

    print(f"Processed {args.iterations} iterations in {end - start:.4f}s")
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    """Augment sentences via CLI."""
    if not _check_model():
        return 1

    from kotogram.augment import augment

    text = args.text
    if text == "-":
        text = sys.stdin.read().strip()

    if not text:
        return 0

    sentences = [text]  # augment expects list of strings

    # We might want to support bulk augmentation from file later,
    # but for now single input to match other commands.

    variations = augment(sentences, timeout=args.timeout)
    print(json.dumps(variations, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    # pylint: disable=too-many-locals
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

    # gender_score
    gs_parser = subparsers.add_parser(
        "gender_score",
        help="Get gender score (-1.0 to 1.0)",
    )
    gs_parser.add_argument("text", help="Text to analyze")
    gs_parser.set_defaults(func=cmd_gender_score)

    # gender_is_pragmatic
    gp_parser = subparsers.add_parser(
        "gender_is_pragmatic",
        help="Check if gender is pragmatically determined",
    )
    gp_parser.add_argument("text", help="Text to analyze")
    gp_parser.set_defaults(func=cmd_gender_is_pragmatic)

    # augment
    augment_parser = subparsers.add_parser(
        "augment",
        help="Augment Japanese text with grammatical variations",
    )
    augment_parser.add_argument("text", help="Text to augment")
    augment_parser.add_argument(
        "--timeout", type=float, default=1.0, help="Timeout in seconds"
    )
    augment_parser.set_defaults(func=cmd_augment)

    # vocab
    vocab_parser = subparsers.add_parser("vocab", help="Show vocabulary sizes")
    vocab_parser.set_defaults(func=cmd_vocab)

    # config
    config_parser = subparsers.add_parser("config", help="Show model configuration")
    config_parser.set_defaults(func=cmd_config)

    # labels
    labels_parser = subparsers.add_parser("labels", help="Show label mappings")
    labels_parser.set_defaults(func=cmd_labels)

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run smoke test")
    bench_parser.add_argument("text", help="Text to analyze")
    bench_parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations"
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    parser.add_argument(
        "--model-dir",
        help="Path to custom model directory containing model.pt",
    )

    args = parser.parse_args()

    # Configure analyzer with model dir if provided
    if args.model_dir:
        _ANALYZER.set_model_dir(args.model_dir)

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
