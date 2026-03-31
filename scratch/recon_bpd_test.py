#!/usr/bin/env python3
"""Reconstruction spot-check for BPD training.

Test file format (recon_bpd_test.txt):
    <full_sentence>: <masked_variant_1> <masked_variant_2> ...

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
    masked_variants: List[str] = field(default_factory=list)


@dataclass
class ReconTestResults:
    """Results from running reconstruction tests."""

    metrics: Dict[str, float]
    failure_path: str  # path to failures file (empty if no failures)


def _test_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), "recon_bpd_test.txt")


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from file."""
    cases: List[TestCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            colon_idx = line.index(": ")
            full = line[:colon_idx].strip()
            variants = line[colon_idx + 2 :].strip().split()
            cases.append(TestCase(full_sentence=full, masked_variants=variants))
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
        print(f"  Sentence: {case.full_sentence}")
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
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    device: torch.device,
    temperature: float,
    epoch: int,
    output_dir: str,
) -> ReconTestResults:
    """Run reconstruction spot-check after an epoch.

    The model is switched to eval mode internally and restored afterwards.
    Returns metrics dict and path to failure file (if any failures).
    """
    path = _test_file_path()
    if not os.path.exists(path):
        return ReconTestResults(metrics={}, failure_path="")

    cases = load_test_cases(path)
    if not cases:
        return ReconTestResults(metrics={}, failure_path="")

    id_to_surface = {v: k for k, v in tokenizer.field_vocabs["surface"].items()}

    was_training = model.training
    model.eval()

    total = 0
    passed = 0
    failure_lines: List[str] = []

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

            case_passed = True

            for variant in case.masked_variants:
                masked_pos = align_tokens_to_masked(surfaces, variant)
                if masked_pos is None:
                    case_passed = False
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

                variant_passed = True
                for pos in masked_id_positions:
                    pred_id = int(logits[0, pos].argmax().item())
                    true_id = surface_ids[pos]
                    if pred_id != true_id:
                        variant_passed = False

                if not variant_passed:
                    case_passed = False
                    # Build actual reconstruction for failure report
                    actual_surfaces = list(surfaces)
                    for orig_pos in masked_pos:
                        pred_id = int(logits[0, orig_pos + 1].argmax().item())
                        actual_surfaces[orig_pos] = id_to_surface.get(pred_id, "?")
                    actual_recon = "".join(actual_surfaces)
                    failure_lines.append(
                        f"{case.full_sentence}: {actual_recon} (from {variant})"
                    )
                    break  # report first failing variant per case

            total += 1
            if case_passed:
                passed += 1

    if was_training:
        model.train()

    failure_path = ""
    if failure_lines and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        failure_path = os.path.join(output_dir, f"epoch {epoch + 1} failures.txt")
        with open(failure_path, "w", encoding="utf-8") as f:
            for line in failure_lines:
                f.write(line + "\n")

    failed = total - passed
    metrics: Dict[str, float] = {
        "recon_test_pct": 100.0 * passed / max(1, total),
        "recon_test_total": float(total),
        "recon_test_pass": float(passed),
        "recon_test_fail": float(failed),
    }

    return ReconTestResults(metrics=metrics, failure_path=failure_path)


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
