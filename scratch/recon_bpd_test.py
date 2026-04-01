#!/usr/bin/env python3
"""Reconstruction spot-check for BPD training.

Test file format (recon_bpd_test.txt):
    <full_sentence>[<alt_1>,<alt_2>,...]: <masked_variant_1> <masked_variant_2> ...

The bracketed alternatives are optional. When present, they list
full-sentence reconstructions that are considered acceptable in
addition to the exact original. For example:
    学校に行く[学校へ行く]: 学校x行く
means that reconstructing 学校へ行く is as good as 学校に行く.

Each 'x' in a masked variant replaces exactly one surface token.
Two adjacent 'x' characters replace two consecutive tokens.

Usage:
    python -m scratch.recon_bpd_test --check   # validate test file parsing
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from kotogram.kotogram import TokenFeatures, extract_token_features, split_kotogram
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.tokenizer import Tokenizer, get_vocab_strings


def _get_parser() -> SudachiJapaneseParser:
    """Lazy-load the Sudachi parser singleton."""
    if not hasattr(_get_parser, "_instance"):
        _get_parser._instance = SudachiJapaneseParser()  # type: ignore[attr-defined]
    return _get_parser._instance  # type: ignore[attr-defined]


@dataclass
class TestCase:
    """A single reconstruction test case."""

    full_sentence: str
    acceptable_alternatives: List[str] = field(default_factory=list)
    masked_variants: List[str] = field(default_factory=list)


@dataclass
class ReconTestResults:
    """Results from running reconstruction tests."""

    metrics: Dict[str, float]
    failure_path: str  # path to failures file (empty if no failures)
    verbose_path: str  # path to verbose report (empty if not written)


def _test_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), "recon_bpd_test.txt")


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from file.

    Supports optional alternative reconstructions in brackets:
        学校に行く[学校へ行く]: 学校x行く
    Multiple alternatives are comma-separated:
        水を飲む[茶を飲む,酒を飲む]: xを飲む
    """
    cases: List[TestCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            colon_idx = line.index(": ")
            full_part = line[:colon_idx].strip()
            variants = line[colon_idx + 2 :].strip().split()

            # Parse optional [alt1,alt2,...] bracket syntax
            alternatives: List[str] = []
            if "[" in full_part:
                bracket_start = full_part.index("[")
                bracket_end = full_part.index("]")
                alt_str = full_part[bracket_start + 1 : bracket_end]
                alternatives = [a.strip() for a in alt_str.split(",") if a.strip()]
                full = full_part[:bracket_start].strip()
            else:
                full = full_part

            cases.append(
                TestCase(
                    full_sentence=full,
                    acceptable_alternatives=alternatives,
                    masked_variants=variants,
                )
            )
    return cases


def _tokenize_to_surfaces(sentence: str) -> List[str]:
    """Tokenize a raw Japanese sentence to surface strings via Sudachi."""
    parser = _get_parser()
    kotogram = parser.japanese_to_kotogram(sentence)
    tokens = split_kotogram(kotogram)
    surfaces: List[str] = []
    for tok in tokens:
        features: TokenFeatures = extract_token_features(tok)
        # Use raw surface (what appears in the text) for alignment,
        # NOT normalized_surface (dictionary form like 食べる for 食べ).
        if features.surface:
            surfaces.append(features.surface)
    return surfaces


def align_tokens_to_masked(
    token_surfaces: List[str],
    masked_string: str,
) -> Optional[List[int]]:
    """Find which token positions are masked (replaced by 'x').

    Returns list of masked token indices (0-based into token_surfaces),
    or None if alignment fails.
    """
    masked_positions: List[int] = []
    char_idx = 0

    for token_idx, surface in enumerate(token_surfaces):
        if char_idx >= len(masked_string):
            return None
        if masked_string[char_idx] == "x":
            masked_positions.append(token_idx)
            char_idx += 1
        elif masked_string[char_idx : char_idx + len(surface)] == surface:
            char_idx += len(surface)
        else:
            return None

    if char_idx != len(masked_string):
        return None
    return masked_positions


def check_test_file() -> bool:
    """Validate that all test cases parse correctly against the tokenizer."""
    path = _test_file_path()
    cases = load_test_cases(path)
    all_ok = True

    for case in cases:
        surfaces = _tokenize_to_surfaces(case.full_sentence)
        alt_display = ""
        if case.acceptable_alternatives:
            alt_display = f"  (also accepts: {', '.join(case.acceptable_alternatives)})"
        print(f"  Sentence: {case.full_sentence}{alt_display}")
        print(f"  Tokens: {surfaces}")

        for variant in case.masked_variants:
            masked_pos = align_tokens_to_masked(surfaces, variant)
            if masked_pos is None:
                print(f"    \u2717 FAIL: Cannot align '{variant}'")
                all_ok = False
            else:
                masked_tokens = [surfaces[i] for i in masked_pos]
                print(
                    f"    \u2713 OK: '{variant}' \u2192 masked {masked_pos} ({masked_tokens})"
                )
        print()

    status = "ALL OK" if all_ok else "SOME FAILURES"
    print(f"Result: {status}")
    return all_ok


def run_reconstruction_test(
    ctx: "EpochContext",
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    """Run reconstruction spot-check after an epoch.

    Merges test metrics into ``metrics`` and appends any artifact
    file paths to ``ctx.artifact_paths``.  The model is switched to
    eval mode internally and restored afterwards.
    """
    from scratch.recon_bpd_checkpoint import EpochContext as _EC  # noqa: F811
    assert isinstance(ctx, _EC)

    path = _test_file_path()
    if not os.path.exists(path):
        return

    cases = load_test_cases(path)
    if not cases:
        return

    output_dir = (
        os.path.join(os.path.dirname(ctx.checkpoint_path), "recon_test")
        if ctx.checkpoint_path
        else ""
    )
    model = ctx.model
    tokenizer = ctx.tokenizer
    device = ctx.device
    temperature = ctx.temperature

    id_to_surface = {v: k for k, v in tokenizer.field_vocabs["surface"].items()}

    was_training = model.training
    model.eval()

    total = 0
    passed_strict = 0
    passed_alt = 0
    failure_lines: List[str] = []
    verbose_lines: List[str] = []

    with torch.no_grad():
        for case in cases:
            surfaces = _tokenize_to_surfaces(case.full_sentence)

            # Encode via tokenizer (produces CLS + token IDs)
            parser = _get_parser()
            kotogram = parser.japanese_to_kotogram(case.full_sentence)
            features_list = [
                extract_token_features(tok) for tok in split_kotogram(kotogram)
            ]
            encoded = tokenizer.encode_features(features_list)
            surface_ids = encoded["surface"]  # [CLS, tok0, tok1, ...]

            # Pre-tokenize acceptable alternatives for this case
            alt_surface_ids_list: List[list] = []
            for alt_sentence in case.acceptable_alternatives:
                try:
                    alt_kotogram = parser.japanese_to_kotogram(alt_sentence)
                    alt_features = [
                        extract_token_features(tok)
                        for tok in split_kotogram(alt_kotogram)
                    ]
                    alt_encoded = tokenizer.encode_features(alt_features)
                    alt_ids = alt_encoded["surface"]
                    # Only use alternatives with matching token count
                    if len(alt_ids) == len(surface_ids):
                        alt_surface_ids_list.append(alt_ids)
                except Exception:
                    pass  # skip malformed alternatives silently

            # Track per-case outcome: strict, alt, or fail
            case_strict = True  # all variants match exact original
            case_alt = True  # all variants match original OR alternative
            first_fail_line = ""

            for variant in case.masked_variants:
                masked_pos = align_tokens_to_masked(surfaces, variant)
                if masked_pos is None:
                    case_strict = False
                    case_alt = False
                    continue

                # +1 offset for CLS token at position 0
                masked_id_positions = [p + 1 for p in masked_pos]

                # Create input with masked positions zeroed out (PAD)
                ids_tensor = torch.tensor([surface_ids], device=device)
                for pos in masked_id_positions:
                    ids_tensor[0, pos] = 0

                attn_mask = torch.ones(
                    1, len(surface_ids), device=device, dtype=torch.long
                )

                # Full forward pass: encode -> KC -> recon
                pooled = model.encode(ids_tensor, attn_mask)
                kc_logits_raw, _ = model.kc_head.forward_with_raw(pooled)
                kc_probs = torch.sigmoid(kc_logits_raw / temperature)
                h_recon = model.recon.forward_hidden(kc_probs, attn_mask)
                logits = F.linear(h_recon, model.recon.output_head.weight)

                variant_strict = True
                variant_alt = True
                for pos in masked_id_positions:
                    pred_id = int(logits[0, pos].argmax().item())
                    if pred_id != surface_ids[pos]:
                        variant_strict = False
                        # Check alternatives
                        acceptable_ids = {surface_ids[pos]}
                        for alt_ids in alt_surface_ids_list:
                            acceptable_ids.add(alt_ids[pos])
                        if pred_id not in acceptable_ids:
                            variant_alt = False

                if not variant_strict:
                    case_strict = False
                if not variant_alt:
                    case_alt = False
                    if not first_fail_line:
                        # Build actual reconstruction for report
                        actual_surfaces = list(surfaces)
                        # Bracket masked tokens in expected sentence
                        expected_surfaces = list(surfaces)
                        for orig_pos in masked_pos:
                            pred_id = int(logits[0, orig_pos + 1].argmax().item())
                            actual_surfaces[orig_pos] = id_to_surface.get(pred_id, "?")
                            expected_surfaces[orig_pos] = f"[{surfaces[orig_pos]}]"
                        actual_recon = "".join(actual_surfaces)
                        expected_display = "".join(expected_surfaces)
                        also = ""
                        if case.acceptable_alternatives:
                            also = f" [also accepts: {', '.join(case.acceptable_alternatives)}]"
                        first_fail_line = (
                            f"{expected_display}{also}: {actual_recon} (from {variant})"
                        )

            total += 1
            if case_strict:
                passed_strict += 1
                passed_alt += 1
                verbose_lines.append(f"STRICT {case.full_sentence}")
            elif case_alt:
                passed_alt += 1
                verbose_lines.append(f"ALT    {case.full_sentence}")
            else:
                verbose_lines.append(f"FAIL   {first_fail_line}")
                failure_lines.append(first_fail_line)

    if was_training:
        model.train()

    failure_path = ""
    verbose_path = ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Always write verbose report
        verbose_path = os.path.join(output_dir, f"epoch {epoch + 1:03d} verbose.txt")
        with open(verbose_path, "w", encoding="utf-8") as f:
            for line in verbose_lines:
                f.write(line + "\n")
        # Write failures only if there are any
        if failure_lines:
            failure_path = os.path.join(output_dir, f"epoch {epoch + 1:03d} failures.txt")
            with open(failure_path, "w", encoding="utf-8") as f:
                for line in failure_lines:
                    f.write(line + "\n")

    passed = passed_alt  # "pass" means considering alternatives
    failed = total - passed
    metrics.update({
        "recon_test_pct": 100.0 * passed / max(1, total),
        "recon_test_total": float(total),
        "recon_test_pass": float(passed),
        "recon_test_pass_strict": float(passed_strict),
        "recon_test_fail": float(failed),
    })
    if verbose_path:
        ctx.artifact_paths.append(verbose_path)
    if failure_path:
        ctx.artifact_paths.append(failure_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruction spot-check for BPD training"
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate test file parsing"
    )
    args = parser.parse_args()

    if args.check:
        ok = check_test_file()
        raise SystemExit(0 if ok else 1)
    else:
        parser.print_help()
