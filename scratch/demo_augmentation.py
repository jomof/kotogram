#!/usr/bin/env python3
"""One-off demo: visualize the data augmentation stages in recon_bpd.

Shows how a single sentence transforms at each stage:
  1. Raw dataset sample (feature_ids from StyleDataset.__getitem__)
  2. Content drop (BundledStyleDataset.__getitem__, content_drop_ratio=0.5)
  3. Collation + padding (collate_fn → attention_mask)
  4. Consistency doubling (batch duplicated for dual-mask regularization)
  5. Input masking (_apply_mask, input_mask_ratio=0.15)
"""

import re
import sys, os
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from kotogram.tokenizer import Tokenizer
from scripts.dataset import BundledStyleDataset, resolve_dataset
from train.dataset import collate_fn

# Matches tokens that contain a mix of Japanese (Hiragana/Katakana/CJK) and
# non-standard characters (Latin, digits, symbols, emoji, etc.)
_JP_RE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
_LATIN_RE = re.compile(r"[A-Za-z]{2,}")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAD6\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF]")
_SYMBOL_RE = re.compile(r"[♪♫♬★☆●○◎◆◇■□▲△▼▽※→←↑↓♡♥＊〜～…‥§†‡¶]")
_FULLWIDTH_LATIN_RE = re.compile(r"[\uFF21-\uFF3A\uFF41-\uFF5A]")
_HALFWIDTH_KANA_RE = re.compile(r"[\uFF65-\uFF9F]")


def _dirt_score(tokens: list[str]) -> float:
    """Score how 'dirty' a sentence is. Higher = more non-standard mixing."""
    if not tokens:
        return 0.0
    score = 0.0
    has_jp = any(_JP_RE.search(t) for t in tokens)
    if not has_jp:
        return 0.0
    for t in tokens:
        if t in ("<PAD>", "<UNK>", "<CLS>", "<MASK>"):
            continue
        if _LATIN_RE.search(t):
            score += 3.0  # multi-char Latin is very dirty
        if _EMOJI_RE.search(t):
            score += 5.0
        if _SYMBOL_RE.search(t):
            score += 2.0
        if _FULLWIDTH_LATIN_RE.search(t):
            score += 1.5
        if _HALFWIDTH_KANA_RE.search(t):
            score += 2.0
        # UNK tokens hint at truly weird characters
        if t == "<UNK>":
            score += 4.0
    return score / len(tokens)


def decode_ids(inv_vocab: dict, ids: torch.Tensor) -> str:
    return " ".join(inv_vocab.get(int(i), f"[{i}]") for i in ids)


def show_ids(label: str, ids: torch.Tensor, inv_vocab: dict):
    print(f"\n  {label}")
    print(f"    IDs  : {ids.tolist()}")
    print(f"    Toks : {decode_ids(inv_vocab, ids)}")
    print(f"    Len  : {len(ids)}")


def main():
    torch.manual_seed(42)

    print("Loading dataset from dataset.lock...")
    bundle, _chive = resolve_dataset()
    print(f"  Dataset: {bundle['dataset_id']}  sentences: {bundle['sentence_count']:,}")

    tokenizer = Tokenizer()
    tokenizer.load_state({"field_vocabs": bundle["vocab"], "frozen": True})
    inv_vocab = {v: k for k, v in tokenizer.field_vocabs["surface"].items()}

    content_mask = bundle.get("content_mask")

    # ── Find non-Japanese punctuation tokens ──────────────────────
    # Western punctuation that should be Japanese equivalents
    NON_JP_PUNCT = set(",.:;!?()[]\"'`~-/\\@#$%^&*+=|<>{}")
    JP_EQUIV = {
        ",": "、", ".": "。", "!": "！", "?": "？",
        ":": "：", ";": "；", "(": "（", ")": "）",
        "~": "～", "-": "ー", "\"": "「」",
    }

    print("\nScanning vocabulary for non-Japanese punctuation tokens...")
    nonjp_punct_ids: dict[int, str] = {}
    for tok, tid in tokenizer.field_vocabs["surface"].items():
        if tid < 4:
            continue
        if any(ch in NON_JP_PUNCT for ch in tok):
            nonjp_punct_ids[tid] = tok

    print(f"  {len(nonjp_punct_ids)} tokens contain non-JP punctuation")
    by_char: dict[str, list[str]] = {}
    for tok in nonjp_punct_ids.values():
        for ch in tok:
            if ch in NON_JP_PUNCT:
                by_char.setdefault(ch, []).append(tok)
    for ch in sorted(by_char):
        examples = sorted(set(by_char[ch]))[:8]
        jp = JP_EQUIV.get(ch, "?")
        print(f"    '{ch}' (JP: {jp})  →  {examples}")

    # ── Scan sentences for non-JP punctuation ────────────────────
    print("\nScanning for sentences with non-JP punctuation (10% sample)...")
    ds_scan = BundledStyleDataset.from_bundle(bundle, sample_ratio=0.10)
    ds_scan.content_drop_ratio = 0.0
    gram_scan = ds_scan.filter_by_grammaticality(label=1)

    punct_ids_set = set(nonjp_punct_ids.keys())
    hits = []
    for i in range(len(gram_scan)):
        sample = gram_scan[i]
        surface = sample.feature_ids["surface"]
        ids_list = [int(x) for x in surface]
        matched = [nonjp_punct_ids[x] for x in ids_list if x in punct_ids_set]
        if not matched:
            continue
        tokens = [inv_vocab.get(int(tid), "") for tid in surface]
        hits.append((i, len(matched), matched, " ".join(tokens)))

    hits.sort(key=lambda x: (-x[1], x[0]))
    print(f"  Found {len(hits)} sentences with non-JP punctuation")
    print(f"\n  Top 20:")
    for rank, (idx, cnt, matched, text) in enumerate(hits[:20]):
        print(f"    #{rank:2d} [{', '.join(matched)}]  {text[:90]}")

    # Pick a spread
    picks_idx = []
    n = len(hits)
    if n >= 5:
        picks_idx = [0, 1, 2, n // 3, 2 * n // 3]
    else:
        picks_idx = list(range(min(n, 5)))

    gram_nodrop = gram_scan
    gram_drop = gram_scan
    gram_drop.content_drop_ratio = 0.5

    demo_indices = [hits[i][0] for i in picks_idx]
    print(f"\n  Selected {len(demo_indices)} samples for full augmentation walkthrough")

    for demo_idx in demo_indices:
        print("\n" + "=" * 72)
        print(f"SAMPLE idx={demo_idx}")
        print("=" * 72)

        # ── 1. Raw sample (no content drop) ──────────────────────
        raw_sample = gram_nodrop[demo_idx]
        raw_surface = raw_sample.feature_ids["surface"]
        show_ids("1. RAW (no augmentation)", raw_surface, inv_vocab)

        if content_mask is not None and len(content_mask) > 0:
            is_content = content_mask[raw_surface]
            markers = "".join("C" if c else "F" for c in is_content)
            print(f"    C/F  : {markers}  (C=content, F=function)")

        # ── 2. Content drop ──────────────────────────────────────
        torch.manual_seed(demo_idx)
        drop_sample = gram_drop[demo_idx]
        drop_surface = drop_sample.feature_ids["surface"]
        show_ids("2. CONTENT DROP (ratio=0.5)", drop_surface, inv_vocab)
        delta = len(raw_surface) - len(drop_surface)
        print(f"    Dropped {delta} function token(s)")

        print("\n    Stochastic variations (same sentence, different random seeds):")
        for trial in range(3):
            torch.manual_seed(demo_idx * 100 + trial + 1)
            var_sample = gram_drop[demo_idx]
            var_surface = var_sample.feature_ids["surface"]
            print(
                f"      trial {trial}: "
                f"{decode_ids(inv_vocab, var_surface)}  (len={len(var_surface)})"
            )

        # ── 3. Collation (padding + attention_mask) ──────────────
        # Get 3 samples with different content-drop randomness
        mini_batch = []
        for i in range(3):
            torch.manual_seed(demo_idx * 100 + i + 10)
            mini_batch.append(gram_drop[demo_idx])

        collated = collate_fn(mini_batch)
        surface_t = collated.feature_inputs["input_ids_surface"]
        attn_mask = collated.attention_mask

        print(
            f"\n  3. COLLATED BATCH "
            f"(3 samples, padded to max_len={surface_t.shape[1]})"
        )
        for i in range(surface_t.shape[0]):
            seq_len = int(attn_mask[i].sum().item())
            toks = decode_ids(inv_vocab, surface_t[i, :seq_len])
            pad_count = surface_t.shape[1] - seq_len
            print(f"    [{i}] {toks}  | +{pad_count} pad")
        print(f"    Attention mask shape: {attn_mask.shape}")

        # ── 4. Consistency doubling ──────────────────────────────
        ids = surface_t
        mask = attn_mask
        B_actual = ids.size(0)
        recon_targets = torch.cat([ids, ids], dim=0)
        mask_doubled = torch.cat([mask, mask], dim=0)

        print(
            f"\n  4. CONSISTENCY DOUBLING "
            f"(B {B_actual} → {recon_targets.size(0)})"
        )
        print(f"    recon_targets shape: {recon_targets.shape}")
        print(f"    attention_mask shape: {mask_doubled.shape}")
        print(
            f"    row[0] == row[{B_actual}]: "
            f"{torch.equal(recon_targets[0], recon_targets[B_actual])}"
        )

        # ── 5. Input masking ─────────────────────────────────────
        INPUT_MASK_RATIO = 0.15
        maskable = mask_doubled.bool()

        def _apply_mask(ids_in: torch.Tensor) -> torch.Tensor:
            rand_mask = (
                torch.rand_like(ids_in.float()) < INPUT_MASK_RATIO
            ) & maskable
            return ids_in.masked_fill(rand_mask, 0)

        torch.manual_seed(42)
        masked_v1 = _apply_mask(recon_targets)
        torch.manual_seed(43)
        masked_v2 = _apply_mask(recon_targets)

        print(f"\n  5. INPUT MASKING (ratio={INPUT_MASK_RATIO})")
        print("    Two independent masks on the SAME doubled batch:")
        for row in range(min(2, B_actual)):
            seq_len = int(mask_doubled[row].sum().item())
            orig = recon_targets[row, :seq_len]
            m1 = masked_v1[row, :seq_len]
            m2 = masked_v2[row, :seq_len]

            n_masked_1 = int((m1 == 0).sum().item()) - int((orig == 0).sum().item())
            n_masked_2 = int((m2 == 0).sum().item()) - int((orig == 0).sum().item())

            print(f"\n    row[{row}] (len={seq_len}):")
            print(f"      original: {decode_ids(inv_vocab, orig)}")
            print(f"      mask_v1 : {decode_ids(inv_vocab, m1)}  ({n_masked_1} masked)")
            print(f"      mask_v2 : {decode_ids(inv_vocab, m2)}  ({n_masked_2} masked)")

            twin_row = row + B_actual
            m1_twin = masked_v1[twin_row, :seq_len]
            differs = int((m1[:seq_len] != m1_twin[:seq_len]).sum().item())
            print(
                f"      twin[{twin_row}]: "
                f"{decode_ids(inv_vocab, m1_twin)}  "
                f"(same seed → {differs} position(s) differ from row[{row}])"
            )

        # ── Summary diagram ──────────────────────────────────────
        print(f"\n  PIPELINE SUMMARY for this sample:")
        print(f"    raw tokens          → {len(raw_surface)} tokens")
        print(
            f"    content drop (0.5)  → {len(drop_surface)} tokens  "
            f"({delta} function words dropped)"
        )
        print(
            f"    collate + pad       → {surface_t.shape[1]} positions  "
            f"(batch-max aligned)"
        )
        print(
            f"    consistency double  → 2x batch "
            f"({B_actual}→{2*B_actual} rows)"
        )
        print(
            f"    input mask (15%)    → ~{int(len(drop_surface)*0.15)} "
            f"positions zeroed per view"
        )

    print("\n" + "=" * 72)
    print("DONE — These augmentations happen every batch, every epoch.")
    print("Content drop and input masking are stochastic → the model never")
    print("sees the exact same input twice.")
    print("=" * 72)


if __name__ == "__main__":
    main()
