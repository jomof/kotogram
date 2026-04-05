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
import dataclasses
import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from kotogram.kotogram import TokenFeatures, extract_token_features, split_kotogram
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.tokenizer import Tokenizer


def _get_parser() -> SudachiJapaneseParser:
    """Lazy-load the Sudachi parser singleton."""
    if not hasattr(_get_parser, "_instance"):
        _get_parser._instance = SudachiJapaneseParser()  # type: ignore[attr-defined]
    from typing import cast

    return cast(SudachiJapaneseParser, getattr(_get_parser, "_instance"))


@dataclass
class TestCase:
    """A single reconstruction test case."""

    full_sentence: str
    acceptable_alternatives: List[str] = field(default_factory=list)
    masked_variants: List[str] = field(default_factory=list)
    section: str = ""


@dataclass
class _CaseRecord:
    """Per-case result record for reporting."""

    outcome: str  # "STRICT", "ALT", or "FAIL"
    report_line: str  # human-readable line for the report
    sim: float  # avg cosine sim: predicted embedding vs main-target embedding


def _test_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), "recon_bpd_test.txt")


@dataclass
class VariantData:
    """Captured variant data for batching."""

    case_idx: int
    variant_idx: int
    variant_str: str
    surfaces: List[str]
    surface_ids: List[int]
    masked_pos: List[int]
    masked_id_positions: List[int]
    alt_surface_ids_list: List[List[int]]


@dataclass
class TestBatch:
    """A single batch of test variants."""

    ids_tensor: torch.Tensor
    attention_mask: torch.Tensor
    variants: List[VariantData]


def _build_test_cache(
    txt_path: str, cache_path: str, tokenizer: Tokenizer, cases: List[TestCase]
) -> None:
    """Build and save the test cache to disk."""
    print(f"Building reconstruction test cache: {cache_path}")
    variants_all = []

    parser = _get_parser()
    for case_idx, case in enumerate(cases):
        surfaces = _tokenize_to_surfaces(case.full_sentence)
        try:
            kotogram = parser.japanese_to_kotogram(case.full_sentence)
            features_list = [
                extract_token_features(tok) for tok in split_kotogram(kotogram)
            ]
            encoded = tokenizer.encode_features(features_list)
            surface_ids = encoded["surface"]
        except Exception:
            continue

        alt_surface_ids_list: List[List[int]] = []
        for alt_sentence in case.acceptable_alternatives:
            try:
                alt_kotogram = parser.japanese_to_kotogram(alt_sentence)
                alt_features = [
                    extract_token_features(tok) for tok in split_kotogram(alt_kotogram)
                ]
                alt_encoded = tokenizer.encode_features(alt_features)
                alt_ids = alt_encoded["surface"]
                if len(alt_ids) == len(surface_ids):
                    alt_surface_ids_list.append(alt_ids)
            except Exception:
                pass

        for variant_idx, variant in enumerate(case.masked_variants):
            masked_pos = align_tokens_to_masked(surfaces, variant)
            if masked_pos is None:
                continue
            masked_id_positions = [p + 1 for p in masked_pos]
            variants_all.append(
                VariantData(
                    case_idx=case_idx,
                    variant_idx=variant_idx,
                    variant_str=variant,
                    surfaces=surfaces,
                    surface_ids=surface_ids,
                    masked_pos=masked_pos,
                    masked_id_positions=masked_id_positions,
                    alt_surface_ids_list=alt_surface_ids_list,
                )
            )

    # Randomize to pack small/large sentences together
    random.seed(42)
    random.shuffle(variants_all)

    batches = []
    batch_size = 32
    for i in range(0, len(variants_all), batch_size):
        chunk = variants_all[i : i + batch_size]
        max_len = max(len(v.surface_ids) for v in chunk)

        ids_tensors = []
        mask_tensors = []
        for v in chunk:
            ids = list(v.surface_ids)
            for pos in v.masked_id_positions:
                ids[pos] = 0
            pad_len = max_len - len(ids)
            ids_tensors.append(ids + [0] * pad_len)
            mask_tensors.append([1] * len(ids) + [0] * pad_len)

        batches.append(
            TestBatch(
                ids_tensor=torch.tensor(ids_tensors, dtype=torch.long),
                attention_mask=torch.tensor(mask_tensors, dtype=torch.long),
                variants=chunk,
            )
        )

    torch.save(batches, cache_path)


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from file.

    Supports optional alternative reconstructions in brackets:
        学校に行く[学校へ行く]: 学校x行く
    Multiple alternatives are comma-separated:
        水を飲む[茶を飲む,酒を飲む]: xを飲む
    """
    cases: List[TestCase] = []
    current_section = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Track section headers: # ── <name> ─────
            if line.startswith("# ── "):
                current_section = line[5:].rstrip("\u2500- ").strip()
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
                    section=current_section,
                )
            )
    return cases


def _is_japanese_char(c: str) -> bool:
    """Return True if c is hiragana, katakana, a CJK ideograph, or Japanese punctuation.

    Covers:
    - Hiragana (U+3040–309F)
    - Katakana (U+30A0–30FF)
    - CJK Symbols & Punctuation (U+3000–303F) — includes 。、〜 etc.
    - CJK Unified Ideographs (U+4E00–9FFF)
    - CJK Extension A (U+3400–4DBF)
    - CJK Extension B (U+20000–2A6DF)
    - CJK Compatibility Ideographs (U+F900–FAFF)
    - Halfwidth & Fullwidth Forms (U+FF00–FFEF) — includes ！？ etc.
    - Horizontal Ellipsis (U+2026) — …
    """
    cp = ord(c)
    return (
        0x3000 <= cp <= 0x303F  # CJK symbols & punctuation (includes 。、)
        or 0x3040 <= cp <= 0x309F  # hiragana
        or 0x30A0 <= cp <= 0x30FF  # katakana
        or 0x4E00 <= cp <= 0x9FFF  # CJK unified ideographs (common)
        or 0x3400 <= cp <= 0x4DBF  # CJK extension A
        or 0x20000 <= cp <= 0x2A6DF  # CJK extension B
        or 0xF900 <= cp <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= cp <= 0xFFEF  # halfwidth & fullwidth forms (includes ！？)
        or cp == 0x2026  # horizontal ellipsis …
    )


def _tokenize_to_surfaces(sentence: str) -> List[str]:
    """Tokenize a raw Japanese sentence to surface strings via Sudachi."""
    parser = _get_parser()
    kotogram = parser.japanese_to_kotogram(sentence)
    tokens = split_kotogram(kotogram)
    surfaces: List[str] = []
    for tok in tokens:
        features: TokenFeatures = extract_token_features(tok)
        # Use raw surface (what appears in the text) for alignment.
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

    Four checks are performed:

    -1. Process-comment check: comment lines must not contain editorial
        phrases left over from editing (e.g. "--check", "TOKENIZATION
        WARNING", "Adjust", "Verify").  These are authoring notes that
        should be removed before committing.

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

    3. Relevance check: every sentence (main + alternatives) must contain
       at least one kanji/kana character that also appears in the section
       title.  This ensures each test case is actually exercising the
       grammar/vocabulary pattern the section is named for.  Skipped when
       the section title contains no CJK characters.
    """
    path = _test_file_path()
    all_ok = True

    # ── Check -1: no process-internal comment phrases ─────────────────
    _PROCESS_PHRASES = [
        "TOKENIZATION WARNING",
        "--check",
        "Adjust per",
        "Adjust x",
        "Verify token",
        "If --check",
        "If so,",
        "DELETE them",
        "may normalize to",
        "per --check",
    ]
    with open(path, encoding="utf-8") as _f:
        for _lineno, _raw in enumerate(_f, 1):
            _stripped = _raw.strip()
            if not _stripped.startswith("#"):
                continue
            for _phrase in _PROCESS_PHRASES:
                if _phrase in _stripped:
                    print(
                        f"  ✗ FAIL (process-comment) line {_lineno}: "
                        f"'{_phrase}' found in comment — remove before committing"
                    )
                    all_ok = False
                    break

    cases = load_test_cases(path)

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
        sec = f"  # {case.section}" if case.section else ""

        for variant in case.masked_variants:
            masked_pos = align_tokens_to_masked(surfaces, variant)
            if masked_pos is None:
                print(f"    \u2717 FAIL (align): Cannot align '{variant}'{sec}")
                all_ok = False
            else:
                masked_tokens = [surfaces[i] for i in masked_pos]
                print(
                    f"    \u2713 OK: '{variant}' \u2192 masked {masked_pos} ({masked_tokens}){sec}"
                )
                valid_masks.append(masked_pos)

        # ── Check 2: each alternative is coverable by at least one variant ─
        for alt in case.acceptable_alternatives:
            alt_surfaces = _tokenize_to_surfaces(alt)
            if len(alt_surfaces) != len(surfaces):
                print(
                    f"    \u2717 FAIL (coverage): '{alt}' has {len(alt_surfaces)} tokens"
                    f" but main has {len(surfaces)} — length mismatch, no mask can cover it{sec}"
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
                print(f"    \u2713 OK (coverage): '{alt}' is coverable{sec}")
            else:
                diff = [
                    p for p in range(len(surfaces)) if alt_surfaces[p] != surfaces[p]
                ]
                print(
                    f"    \u2717 FAIL (coverage): '{alt}' differs at positions {diff} "
                    f"({[surfaces[p] for p in diff]} \u2192 {[alt_surfaces[p] for p in diff]})"
                    f" but no variant masks those positions{sec}"
                )
                all_ok = False

        # ── Check 3: section relevance ─────────────────────────────────
        section_japanese: set = {c for c in case.section if _is_japanese_char(c)}
        if section_japanese:
            if not any(c in section_japanese for c in case.full_sentence):
                print(
                    f"    \u2717 FAIL (relevance): '{case.full_sentence}' shares no"
                    f" Japanese chars with section title{sec}"
                )
                all_ok = False

        # ── Check 4: each x masks exactly one token in all sentences ──
        for variant in case.masked_variants:
            # Count x positions in the mask pattern
            x_positions = [i for i, c in enumerate(variant) if c == "x"]
            n_x = len(x_positions)

            # Verify main sentence
            main_masked = align_tokens_to_masked(surfaces, variant)
            if main_masked is not None and len(main_masked) != n_x:
                print(
                    f"    \u2717 FAIL (x-count): '{variant}' has {n_x} x's"
                    f" but masks {len(main_masked)} tokens in"
                    f" main sentence{sec}"
                )
                all_ok = False

            # Verify each alternative
            for alt in case.acceptable_alternatives:
                alt_surfaces = _tokenize_to_surfaces(alt)
                alt_masked = align_tokens_to_masked(alt_surfaces, variant)
                if alt_masked is not None and len(alt_masked) != n_x:
                    print(
                        f"    \u2717 FAIL (x-count): '{variant}' has {n_x} x's"
                        f" but masks {len(alt_masked)} tokens in"
                        f" '{alt}'{sec}"
                    )
                    all_ok = False

        print()

    status = "ALL OK" if all_ok else "SOME FAILURES"
    print(f"Result: {status}")
    return all_ok


# ── Module-level cache to prevent memory leaks from torch.load ──
_RAM_CACHE_DIGEST: str = ""
_RAM_CACHE_BATCHES: List[TestBatch] = []


def run_reconstruction_test(
    ctx: Any,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    """Run reconstruction spot-check after an epoch."""
    global _RAM_CACHE_DIGEST, _RAM_CACHE_BATCHES
    from scratch.recon_bpd_checkpoint import EpochContext as _EC  # noqa: F811

    assert isinstance(ctx, _EC)
    t0 = time.perf_counter()

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
    from typing import Any, cast

    model = cast(Any, ctx.model)
    tokenizer = cast(Any, ctx.tokenizer)
    device = ctx.device
    temperature = ctx.temperature

    id_to_surface = {v: k for k, v in tokenizer.field_vocabs["surface"].items()}

    # Compute hash digest of relevant files for cache invalidation
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    with open(__file__, "rb") as f:
        h.update(f.read())
    # Add tokenizer json config backing to the hash explicitly
    from kotogram import locations

    tok_path = os.path.join(locations.get_style_output_dir(), "tokenizer.json")
    if os.path.exists(tok_path):
        with open(tok_path, "rb") as f:
            h.update(f.read())

    digest = h.hexdigest()[:16]

    if _RAM_CACHE_DIGEST == digest and _RAM_CACHE_BATCHES:
        batches = _RAM_CACHE_BATCHES
    else:
        os.makedirs(".cache/optuna", exist_ok=True)
        cache_path = os.path.join(".cache/optuna", f"recon_test_cache_{digest}.pt")

        if not os.path.exists(cache_path):
            _build_test_cache(path, cache_path, tokenizer, cases)

        batches = torch.load(cache_path, map_location="cpu", weights_only=False)
        _RAM_CACHE_DIGEST = digest
        _RAM_CACHE_BATCHES = batches

    was_training = model.training
    model.eval()

    # Predictions from variant_results
    variant_results: List[dict] = []
    global_sim_sum = 0.0
    global_t1_correct = 0
    global_token_count = 0

    with torch.no_grad():
        embed_weight = model.recon.output_head.weight  # [V, H]

        for batch in batches:
            ids_tensor = batch.ids_tensor.to(device, non_blocking=True)
            attn_mask = batch.attention_mask.to(device, non_blocking=True)

            pooled = model.encode(ids_tensor, attn_mask)
            kc_logits_raw, _ = model.kc_head.forward_with_raw(pooled)
            kc_probs = torch.sigmoid(kc_logits_raw / temperature)
            h_recon = model.recon.forward_hidden(kc_probs, attn_mask)

            # Predict top-1 for the whole batch, but chunked by tokens to save VRAM
            # [B, T, H] -> [B, T]
            T = ids_tensor.size(1)
            pred_ids = torch.zeros((ids_tensor.size(0), T), dtype=torch.long)
            recon_chunk = 4  # Small chunk to prevent [B, chunk, V] spike
            for c0 in range(0, T, recon_chunk):
                c1 = min(c0 + recon_chunk, T)
                # Manifesting [B, recon_chunk, V] is ~1/16th of the previous memory spike
                chunk_logits = F.linear(h_recon[:, c0:c1, :], embed_weight)
                pred_ids[:, c0:c1] = chunk_logits.argmax(dim=-1).cpu()

            for b_idx, variant in enumerate(batch.variants):
                variant_strict = True
                variant_alt = True
                variant_t1_correct = 0
                variant_tokens = 0
                actual_surfaces = list(variant.surfaces)
                expected_surfaces = list(variant.surfaces)
                sims = []

                for orig_pos, pos in zip(
                    variant.masked_pos, variant.masked_id_positions
                ):
                    pred_id = int(pred_ids[b_idx, pos].item())
                    exp_id = variant.surface_ids[pos]

                    # Cosine similarity: predicted embedding vs main-target embedding
                    sim = F.cosine_similarity(
                        embed_weight[pred_id].unsqueeze(0),
                        embed_weight[exp_id].unsqueeze(0),
                    ).item()
                    sims.append(sim)
                    global_sim_sum += sim
                    global_token_count += 1
                    variant_tokens += 1
                    if pred_id == exp_id:
                        global_t1_correct += 1
                        variant_t1_correct += 1

                    actual_surfaces[orig_pos] = id_to_surface.get(pred_id, "?")
                    expected_surfaces[orig_pos] = f"[{variant.surfaces[orig_pos]}]"

                    if pred_id != exp_id:
                        variant_strict = False
                        acceptable_ids: set = {exp_id}
                        for alt_ids in variant.alt_surface_ids_list:
                            if pos < len(alt_ids):
                                acceptable_ids.add(alt_ids[pos])
                        if pred_id not in acceptable_ids:
                            variant_alt = False

                variant_results.append(
                    {
                        "case_idx": variant.case_idx,
                        "variant_idx": variant.variant_idx,
                        "variant_str": variant.variant_str,
                        "strict": variant_strict,
                        "alt": variant_alt,
                        "sims": sims,
                        "t1_correct": variant_t1_correct,
                        "tokens": variant_tokens,
                        "actual_surfaces": actual_surfaces,
                        "expected_surfaces": expected_surfaces,
                    }
                )

    # Group by case
    from collections import defaultdict

    case_to_variants = defaultdict(list)
    for res in variant_results:
        case_to_variants[res["case_idx"]].append(res)

    case_records: List[_CaseRecord] = []
    total = 0
    passed_strict = 0
    passed_alt = 0

    sim_all = []
    sim_pass = []
    sim_fail = []
    to1_all = []
    to1_pass = []
    to1_fail = []

    for case_idx, case in enumerate(cases):
        results = case_to_variants.get(case_idx, [])
        if not results:
            continue
        results.sort(key=lambda x: x["variant_idx"])

        case_strict = True
        case_alt = True
        case_sims = []
        case_t1_correct = 0
        case_tokens = 0
        first_fail_line = ""
        last_actual = []

        for res in results:
            case_sims.extend(res["sims"])
            case_t1_correct += res["t1_correct"]
            case_tokens += res["tokens"]
            last_actual = res["actual_surfaces"]
            if not res["strict"]:
                case_strict = False
            if not res["alt"]:
                case_alt = False
                if not first_fail_line:
                    actual_recon = "".join(res["actual_surfaces"])
                    expected_display = "".join(res["expected_surfaces"])
                    also = ""
                    if case.acceptable_alternatives:
                        also = (
                            f" [also accepts:"
                            f" {', '.join(case.acceptable_alternatives)}]"
                        )
                    first_fail_line = (
                        f"{expected_display}{also}:"
                        f" {actual_recon} (from {res['variant_str']})"
                    )

        total += 1
        case_sim = sum(case_sims) / max(1, len(case_sims))
        case_to1 = 100.0 * case_t1_correct / max(1, case_tokens)
        sim_tag = f"[sim={case_sim:.2f}]"
        actual_recon = "".join(last_actual)

        is_pass = case_strict or case_alt

        sim_all.append(case_sim)
        to1_all.append(case_to1)
        if is_pass:
            sim_pass.append(case_sim)
            to1_pass.append(case_to1)
        else:
            sim_fail.append(case_sim)
            to1_fail.append(case_to1)

        if case_strict:
            passed_strict += 1
            passed_alt += 1
            report_line = f"STRICT {case.full_sentence} {sim_tag}  # {case.section}"
            case_records.append(_CaseRecord("STRICT", report_line, case_sim))
        elif case_alt:
            passed_alt += 1
            report_line = (
                f"ALT    {case.full_sentence} \u2192 {actual_recon} {sim_tag}"
                f"  # {case.section}"
            )
            case_records.append(_CaseRecord("ALT", report_line, case_sim))
        else:
            fail_report = f"FAIL   {first_fail_line} {sim_tag}  # {case.section}"
            case_records.append(_CaseRecord("FAIL", fail_report, case_sim))

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

    def _build_section(title: str, records: List[_CaseRecord]) -> List[str]:
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

        # ── Header ───────────────────────────────────────────────────
        sep = "=" * 72
        experiment = ctx.run_name or os.path.basename(ctx.checkpoint_path)
        header: List[str] = [
            sep,
            f"Experiment : {experiment}",
            f"Epoch      : {epoch + 1}",
            "",
            "Model parameters:",
        ]
        if ctx.config is not None:
            for k, v in sorted(
                dataclasses.asdict(cast(Any, ctx.config)).items()  # type: ignore[arg-type]
            ):
                header.append(f"  {k:<40} {v}")
        header += [
            "",
            "Similarity metric (sim):",
            "  sim = cosine_similarity(embed[pred_id], embed[main_target_id])",
            "  where embed[i] = output_head.weight[i]  (row vector, shape: [hidden_dim])",
            "  Interpretation:",
            "    sim = 1.00  → STRICT pass (correct token, same embedding)",
            "    sim < 1.00  → ALT pass (accepted alternative; measures how close",
            "                   the alt's embedding is to the main target's embedding)",
            "    sim ≈ 0.00  → FAIL: prediction nearly orthogonal to target",
            "    sim < 0.00  → FAIL: prediction antipodal to target",
            "  Averaged over all masked positions across all variants per case.",
            sep,
            "",
        ]

        report_sections: List[str] = (
            header
            + _build_section("FAILURES", fail_records)
            + [""]
            + _build_section("PASSED", pass_records)
        )

        report_path = os.path.join(output_dir, f"epoch {epoch + 1:03d} failures.txt")
        with open(report_path, "w", encoding="utf-8") as f_out:
            for line in report_sections:
                f_out.write(line + "\n")

    passed = passed_alt
    failed = total - passed
    t1 = time.perf_counter()

    def pctl(arr: List[float], p: float) -> float:
        if not arr:
            return 0.0
        s = sorted(arr)
        return s[int((len(s) - 1) * p / 100.0)]

    metrics.update(
        {
            "test/cos": global_sim_sum / max(1, global_token_count),
            "test/To-1": 100.0 * global_t1_correct / max(1, global_token_count),
            "test/pct": 100.0 * passed / max(1, total),
            "test/total": float(total),
            "test/pass": float(passed),
            "test/pass_strict": float(passed_strict),
            "test/fail": float(failed),
            "test/ms": (t1 - t0) * 1000.0,
        }
    )

    for prefix, arr_sim, arr_t1 in [
        ("test/pass", sim_pass, to1_pass),
        ("test/fail", sim_fail, to1_fail),
        ("test/all", sim_all, to1_all),
    ]:
        metrics[f"{prefix}/cos/p10"] = pctl(arr_sim, 10)
        metrics[f"{prefix}/cos/p50"] = pctl(arr_sim, 50)
        metrics[f"{prefix}/cos/p90"] = pctl(arr_sim, 90)
        metrics[f"{prefix}/to1/p10"] = pctl(arr_t1, 10)
        metrics[f"{prefix}/to1/p50"] = pctl(arr_t1, 50)
        metrics[f"{prefix}/to1/p90"] = pctl(arr_t1, 90)
    if report_path:
        ctx.artifact_paths.append(report_path)

    # Explicitly clear GPU memory after massive reconstruction inference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


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
