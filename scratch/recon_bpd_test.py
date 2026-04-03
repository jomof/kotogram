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
class _CaseRecord:
    """Per-case result record for reporting."""

    outcome: str  # "STRICT", "ALT", or "FAIL"
    report_line: str  # human-readable line for the report
    sim: float  # avg cosine sim: predicted embedding vs main-target embedding


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
    """Validate that all test cases parse correctly against the tokenizer.

    Three checks are performed:

    0. Distinctness check: the main sentence and all alternatives are unique
       strings.  Duplicates silently waste test budget.

    1. Alignment check: every masked variant aligns correctly to the main
       sentence's token list (each 'x' corresponds to exactly one token).

    2. Coverage check: for every acceptable alternative, at least one masked
       variant "covers" it.  A variant covers an alternative when the
       alternative's tokens at all *unmasked* positions match the main
       sentence's tokens.  If they differ at an unmasked position the
       alternative can never be produced by filling in the x's — the test
       case is broken.
    """
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

        # ── Check 0: all sentences (main + alternatives) are distinct ─────
        all_sentences = [case.full_sentence] + case.acceptable_alternatives
        seen: set = set()
        for s in all_sentences:
            if s in seen:
                print(f"    \u2717 FAIL (distinct): '{s}' appears more than once")
                all_ok = False
            seen.add(s)

        # ── Check 1: alignment ────────────────────────────────────────────
        valid_masks: List[List[int]] = []
        for variant in case.masked_variants:
            masked_pos = align_tokens_to_masked(surfaces, variant)
            if masked_pos is None:
                print(f"    \u2717 FAIL (align): Cannot align '{variant}'")
                all_ok = False
            else:
                masked_tokens = [surfaces[i] for i in masked_pos]
                print(
                    f"    \u2713 OK: '{variant}' \u2192 masked {masked_pos} ({masked_tokens})"
                )
                valid_masks.append(masked_pos)

        # ── Check 2: each alternative is coverable by at least one variant ─
        for alt in case.acceptable_alternatives:
            alt_surfaces = _tokenize_to_surfaces(alt)
            if len(alt_surfaces) != len(surfaces):
                print(
                    f"    \u2717 FAIL (coverage): '{alt}' has {len(alt_surfaces)} tokens"
                    f" but main has {len(surfaces)} — length mismatch, no mask can cover it"
                )
                all_ok = False
                continue

            covered = False
            for masked_pos in valid_masks:
                masked_set = set(masked_pos)
                # All unmasked positions must be identical between main and alt
                if all(
                    alt_surfaces[p] == surfaces[p]
                    for p in range(len(surfaces))
                    if p not in masked_set
                ):
                    covered = True
                    break

            if covered:
                print(f"    \u2713 OK (coverage): '{alt}' is coverable")
            else:
                diff = [
                    p for p in range(len(surfaces)) if alt_surfaces[p] != surfaces[p]
                ]
                print(
                    f"    \u2717 FAIL (coverage): '{alt}' differs at positions {diff} "
                    f"({[surfaces[p] for p in diff]} \u2192 {[alt_surfaces[p] for p in diff]})"
                    f" but no variant masks those positions"
                )
                all_ok = False

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

    For each masked test case the model predicts the hidden token(s).  The
    result is recorded as STRICT (exact match), ALT (accepted alternative),
    or FAIL.  Two report files are written:

    * verbose.txt — FAILURES section then PASSED section, each with a
      per-case cosine-similarity tag and min/median/max statistics.
    * failures.txt — same layout (so the failure file alone is self-
      contained without needing to open the verbose file).

    Cosine similarity is computed between the predicted token’s output
    embedding and the main-target token’s output embedding, averaged over
    all masked positions across all variants of the case.
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
    case_records: List[_CaseRecord] = []

    with torch.no_grad():
        embed_weight = model.recon.output_head.weight  # [V, H]

        for case in cases:
            surfaces = _tokenize_to_surfaces(case.full_sentence)

            parser = _get_parser()
            kotogram = parser.japanese_to_kotogram(case.full_sentence)
            features_list = [
                extract_token_features(tok) for tok in split_kotogram(kotogram)
            ]
            encoded = tokenizer.encode_features(features_list)
            surface_ids = encoded["surface"]  # [CLS, tok0, tok1, ...]

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
                    if len(alt_ids) == len(surface_ids):
                        alt_surface_ids_list.append(alt_ids)
                except Exception:
                    pass

            case_strict = True
            case_alt = True
            # Per-position cosine sims across all variants
            case_sims: List[float] = []
            first_fail_line = ""
            # Predictions from the most recently processed variant
            case_last_actual: List[str] = list(surfaces)

            for variant in case.masked_variants:
                masked_pos = align_tokens_to_masked(surfaces, variant)
                if masked_pos is None:
                    case_strict = False
                    case_alt = False
                    continue

                masked_id_positions = [p + 1 for p in masked_pos]

                ids_tensor = torch.tensor([surface_ids], device=device)
                for pos in masked_id_positions:
                    ids_tensor[0, pos] = 0

                attn_mask = torch.ones(
                    1, len(surface_ids), device=device, dtype=torch.long
                )

                pooled = model.encode(ids_tensor, attn_mask)
                kc_logits_raw, _ = model.kc_head.forward_with_raw(pooled)
                kc_probs = torch.sigmoid(kc_logits_raw / temperature)
                h_recon = model.recon.forward_hidden(kc_probs, attn_mask)
                logits = F.linear(h_recon, model.recon.output_head.weight)

                variant_strict = True
                variant_alt = True
                actual_surfaces = list(surfaces)
                expected_surfaces = list(surfaces)

                for orig_pos in masked_pos:
                    pos = orig_pos + 1
                    pred_id = int(logits[0, pos].argmax().item())
                    exp_id = surface_ids[pos]

                    # Cosine similarity: predicted embedding vs main-target embedding
                    sim = F.cosine_similarity(
                        embed_weight[pred_id].unsqueeze(0),
                        embed_weight[exp_id].unsqueeze(0),
                    ).item()
                    case_sims.append(sim)

                    actual_surfaces[orig_pos] = id_to_surface.get(pred_id, "?")
                    expected_surfaces[orig_pos] = f"[{surfaces[orig_pos]}]"

                    if pred_id != exp_id:
                        variant_strict = False
                        acceptable_ids: set = {exp_id}
                        for alt_ids in alt_surface_ids_list:
                            acceptable_ids.add(alt_ids[pos])
                        if pred_id not in acceptable_ids:
                            variant_alt = False

                case_last_actual = actual_surfaces

                if not variant_strict:
                    case_strict = False
                if not variant_alt:
                    case_alt = False
                    if not first_fail_line:
                        actual_recon = "".join(actual_surfaces)
                        expected_display = "".join(expected_surfaces)
                        also = ""
                        if case.acceptable_alternatives:
                            also = (
                                f" [also accepts:"
                                f" {', '.join(case.acceptable_alternatives)}]"
                            )
                        first_fail_line = (
                            f"{expected_display}{also}:"
                            f" {actual_recon} (from {variant})"
                        )

            total += 1
            case_sim = sum(case_sims) / max(1, len(case_sims))
            sim_tag = f"[sim={case_sim:.2f}]"
            actual_recon = "".join(case_last_actual)

            if case_strict:
                passed_strict += 1
                passed_alt += 1
                report_line = f"STRICT {case.full_sentence} {sim_tag}"
                case_records.append(
                    _CaseRecord("STRICT", report_line, case_sim)
                )
            elif case_alt:
                passed_alt += 1
                report_line = f"ALT    {case.full_sentence} \u2192 {actual_recon} {sim_tag}"
                case_records.append(
                    _CaseRecord("ALT", report_line, case_sim)
                )
            else:
                fail_report = f"FAIL   {first_fail_line} {sim_tag}"
                case_records.append(
                    _CaseRecord("FAIL", fail_report, case_sim)
                )

    if was_training:
        model.train()

    def _sim_stats(sims: List[float]) -> str:
        if not sims:
            return "n/a"
        sims_s = sorted(sims)
        n = len(sims_s)
        median = (
            sims_s[n // 2]
            if n % 2 == 1
            else (sims_s[n // 2 - 1] + sims_s[n // 2]) / 2.0
        )
        return f"min={sims_s[0]:.2f}  median={median:.2f}  max={sims_s[-1]:.2f}"

    def _build_section(
        title: str, records: List[_CaseRecord]
    ) -> List[str]:
        sorted_records = sorted(records, key=lambda r: r.sim)
        sims = [r.sim for r in sorted_records]
        out = [f"=== {title} ({len(sorted_records)}) ==="]
        for r in sorted_records:
            out.append(r.report_line)
        out.append(f"Stats: {_sim_stats(sims)}")
        return out

    fail_records = [r for r in case_records if r.outcome == "FAIL"]
    pass_records = [r for r in case_records if r.outcome != "FAIL"]

    report_path = ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        report_sections: List[str] = (
            _build_section("FAILURES", fail_records)
            + [""]
            + _build_section("PASSED", pass_records)
        )

        report_path = os.path.join(
            output_dir, f"epoch {epoch + 1:03d} failures.txt"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            for line in report_sections:
                f.write(line + "\n")

    passed = passed_alt
    failed = total - passed
    metrics.update({
        "recon_test_pct": 100.0 * passed / max(1, total),
        "recon_test_total": float(total),
        "recon_test_pass": float(passed),
        "recon_test_pass_strict": float(passed_strict),
        "recon_test_fail": float(failed),
    })
    if report_path:
        ctx.artifact_paths.append(report_path)


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
