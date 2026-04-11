"""Debug pristine data augmentation pipeline.

Loads the same dataset as recon_bpd training runs (with --percent 99
filtering), applies the pristine mapping, and gathers statistics to
reveal potential failure modes.
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Dict

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dataset import BundledStyleDataset, resolve_dataset
from scripts.recon_bpd.token_remap import (
    NUM_SPECIAL,
    apply_pristine,
    apply_remap_to_bundle,
    build_pristine_id_mapping,
    pristine_surface,
)


def main() -> None:  # noqa: C901
    # ── Load dataset bundle (same path as training) ─────────────────
    bundle, chive = resolve_dataset()
    original_vocab: Dict[str, int] = dict(bundle["vocab"]["surface"])
    original_vocab_size = len(original_vocab)
    print(f"Original vocab size: {original_vocab_size:,}")
    print(f"ChiVe shape: {chive.shape}")

    # ── Apply token_percentile=99 remap (same as training) ──────────
    bundle, chive, remap = apply_remap_to_bundle(bundle, chive, percentile=99.0)
    post_remap_vocab: Dict[str, int] = dict(bundle["vocab"]["surface"])
    post_remap_vocab_size = len(post_remap_vocab)
    inv_post_vocab = {v: k for k, v in post_remap_vocab.items()}
    print(f"Post-remap vocab size: {post_remap_vocab_size:,}")
    print(f"UNK ID: {remap.unk_id}")

    # ── Build pristine mapping ON POST-REMAP VOCAB ──────────────────
    # This is what the training pipeline does: vocab = dataset_bundle["vocab"]["surface"]
    # AFTER apply_remap_to_bundle has already modified the bundle.
    pristine_mapping = build_pristine_id_mapping(post_remap_vocab)
    print(f"Pristine mapping shape: {pristine_mapping.shape}")

    # ── Analyze the static pristine mapping ─────────────────────────
    n_identity = 0
    n_remapped = 0
    n_to_pad = 0
    remap_pairs: list = []
    pad_tokens: list = []
    for tid in range(len(pristine_mapping)):
        dst = int(pristine_mapping[tid])
        if dst == tid:
            n_identity += 1
        elif dst == 0:
            n_to_pad += 1
            tok = inv_post_vocab.get(tid, f"<id_{tid}>")
            pad_tokens.append(tok)
        else:
            n_remapped += 1
            src_tok = inv_post_vocab.get(tid, f"<id_{tid}>")
            dst_tok = inv_post_vocab.get(dst, f"<id_{dst}>")
            remap_pairs.append((src_tok, dst_tok, tid, dst))

    print("\n=== Static Pristine Mapping ===")
    print(f"  Identity (unchanged): {n_identity:,}")
    print(f"  Remapped (dirty→pristine): {n_remapped}")
    print(f"  Mapped to PAD (non-content): {n_to_pad}")

    if remap_pairs:
        print("\n  Remap pairs (dirty → pristine):")
        for src, dst, sid, did in remap_pairs:
            print(f"    '{src}' (id={sid}) → '{dst}' (id={did})")

    if pad_tokens:
        print("\n  Tokens mapped to PAD (first 50):")
        for tok in pad_tokens[:50]:
            print(f"    '{tok}'")
        if len(pad_tokens) > 50:
            print(f"    ... and {len(pad_tokens) - 50} more")

    # ── Check for MISSING PRISTINE TARGETS ──────────────────────────
    # If a dirty token maps to a pristine surface that doesn't exist
    # in the post-remap vocab, the static mapping cannot remap it.
    # It either stays dirty (identity) or falls through to PAD.
    print("\n=== Missing Pristine Target Check ===")
    missing_targets = []
    for tok, tid in post_remap_vocab.items():
        if tid < NUM_SPECIAL:
            continue
        p = pristine_surface(tok)
        if p != tok:
            pid = post_remap_vocab.get(p)
            if pid is None:
                missing_targets.append((tok, p, tid))
    if missing_targets:
        print(
            f"  ⚠️  {len(missing_targets)} tokens have pristine targets MISSING from vocab:"
        )
        for src, dst, sid in missing_targets[:30]:
            print(f"    '{src}' → '{dst}' (target not in vocab!)")
        if len(missing_targets) > 30:
            print(f"    ... and {len(missing_targets) - 30} more")
    else:
        print("  ✓ All pristine targets exist in the post-remap vocab")

    # ── Sample sentences and run apply_pristine ─────────────────────
    print("\n=== Sentence-Level Analysis ===")
    dataset = BundledStyleDataset.from_bundle(bundle, sample_ratio=1.0)
    dataset.pristine_static_mapping = pristine_mapping
    dataset.pristine_vocab = dict(post_remap_vocab)
    dataset.content_drop_ratio = 0.0  # disable for analysis
    gram_ds = dataset.filter_by_grammaticality(label=1)
    print(f"Grammatical sentences: {len(gram_ds):,}")

    # Gather per-sentence statistics
    total_tokens = 0
    total_changed = 0
    total_to_pad = 0
    total_to_unk = 0
    change_histogram = Counter()  # (src_tok, dst_tok) → count
    pad_position_counts = Counter()

    n_samples = min(len(gram_ds), 50000)
    unk_id = remap.unk_id

    # Detect sentences where ALL content becomes PAD or UNK
    catastrophic_sentences = []
    pristine_pad_count = 0
    dirty_has_unk = 0
    pristine_introduces_unk = 0

    for i in range(n_samples):
        sample = gram_ds[i]
        surface = sample.feature_ids.get("surface")
        if surface is None:
            continue

        dirty = surface
        pristine_result = apply_pristine(
            dirty, post_remap_vocab, static_mapping=pristine_mapping
        )

        seq_len = len(dirty)
        total_tokens += seq_len

        dirty_unk_count = (dirty == unk_id).sum().item()
        if dirty_unk_count > 0:
            dirty_has_unk += 1

        changed_mask = dirty != pristine_result
        n_changed = changed_mask.sum().item()
        total_changed += n_changed

        # Check for PAD in pristine output (at non-padding positions)
        pristine_pad_mask = pristine_result == 0
        n_pad = pristine_pad_mask.sum().item()
        total_to_pad += n_pad
        if n_pad > 0:
            pristine_pad_count += 1

        # Check for UNK in pristine result that wasn't in dirty
        pristine_unk = (pristine_result == unk_id).sum().item()
        if pristine_unk > dirty_unk_count:
            pristine_introduces_unk += 1
            total_to_unk += pristine_unk - dirty_unk_count

        # Track changes
        if n_changed > 0:
            for j in range(seq_len):
                if changed_mask[j]:
                    src_id = int(dirty[j])
                    dst_id = int(pristine_result[j])
                    src_tok = inv_post_vocab.get(src_id, f"<id_{src_id}>")
                    dst_tok = inv_post_vocab.get(dst_id, f"<id_{dst_id}>")
                    change_histogram[(src_tok, dst_tok)] += 1
                    if dst_id == 0:
                        pad_position_counts[j] += 1

        # Catastrophic: more than half the tokens changed to PAD or UNK
        n_bad = n_pad + (
            pristine_unk - dirty_unk_count if pristine_unk > dirty_unk_count else 0
        )
        if n_bad > seq_len * 0.3 and seq_len > 3:
            catastrophic_sentences.append((i, seq_len, n_bad, n_pad, pristine_unk))

    pct_changed = 100.0 * total_changed / max(1, total_tokens)
    pct_pad = 100.0 * total_to_pad / max(1, total_tokens)
    pct_unk = 100.0 * total_to_unk / max(1, total_tokens)
    print(f"  Analyzed {n_samples:,} sentences, {total_tokens:,} total tokens")
    print(f"  Tokens changed by pristine:  {total_changed:,} ({pct_changed:.3f}%)")
    print(f"  Tokens mapped to PAD:        {total_to_pad:,} ({pct_pad:.3f}%)")
    print(f"  Tokens mapped to UNK (new):  {total_to_unk:,} ({pct_unk:.3f}%)")
    print(f"  Sentences with dirty UNK:    {dirty_has_unk:,} / {n_samples:,}")
    print(f"  Sentences with pristine PAD: {pristine_pad_count:,} / {n_samples:,}")
    print(f"  Sentences introducing UNK:   {pristine_introduces_unk:,} / {n_samples:,}")

    print("\n=== Top Change Pairs (dirty → pristine) ===")
    for (src, dst), cnt in change_histogram.most_common(30):
        print(f"  {cnt:6d}x  '{src}' → '{dst}'")

    if catastrophic_sentences:
        print(f"\n=== ⚠️  Catastrophic Sentences ({len(catastrophic_sentences)}) ===")
        print("  (>30% of tokens become PAD/UNK)")
        for idx, slen, nbad, npad, nunk in catastrophic_sentences[:20]:
            sample = gram_ds[idx]
            dirty_ids = sample.feature_ids["surface"]
            pristine_ids = apply_pristine(
                dirty_ids, post_remap_vocab, static_mapping=pristine_mapping
            )
            dirty_str = " ".join(inv_post_vocab.get(int(t), "?") for t in dirty_ids)
            prist_str = " ".join(inv_post_vocab.get(int(t), "?") for t in pristine_ids)
            print(
                f"\n  Sentence #{idx} (len={slen}, bad={nbad}, pad={npad}, unk={nunk}):"
            )
            print(f"    Dirty:    {dirty_str}")
            print(f"    Pristine: {prist_str}")

    # ── Content mask stats ───────────────────────────────────────────
    print("\n=== Content Mask Stats ===")
    cm = bundle["content_mask"]
    print(f"  Content mask size: {len(cm)}")
    print(f"  Content (True): {cm.sum().item()}")
    print(f"  Non-content (False): {(~cm).sum().item()}")

    # ── KEY INVARIANT CHECK: dirty→pristine→CE target alignment ─────
    print("\n=== CE Target Alignment Check ===")
    # After removing Step 4, pristine should NEVER introduce PAD.
    # Verify that directly.
    trivial_pad_target = 0
    nontrivial_changed = 0
    for i in range(min(n_samples, 10000)):
        sample = gram_ds[i]
        surface = sample.feature_ids.get("surface")
        if surface is None:
            continue
        pristine_result = apply_pristine(
            surface, post_remap_vocab, static_mapping=pristine_mapping
        )

        # Count positions where dirty != PAD but pristine == PAD
        for j in range(len(surface)):
            if int(surface[j]) != 0 and int(pristine_result[j]) == 0:
                trivial_pad_target += 1
            elif int(surface[j]) != int(pristine_result[j]):
                nontrivial_changed += 1

    print(
        f"  Positions where dirty!=0 but pristine==0 (trivial PAD): {trivial_pad_target:,}"
    )
    print(f"  Positions with nontrivial pristine changes: {nontrivial_changed:,}")
    if trivial_pad_target == 0:
        print("  ✓ No PAD introduced by pristine pipeline")
    else:
        print(
            f"  ⚠️  Ratio trivial/(trivial+nontrivial): "
            f"{trivial_pad_target / max(1, trivial_pad_target + nontrivial_changed):.4f}"
        )


if __name__ == "__main__":
    main()
