#!/usr/bin/env python3
"""Demo: token-level pristine mapping as it would work in the dataloader.

Shows input (dirty) vs output (pristine) token sequences for real sentences.
Non-content tokens with no pristine equivalent are mapped to <PAD>.
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from kotogram.tokenizer import PAD_ID, Tokenizer
from scripts.dataset import resolve_dataset
from scripts.recon_bpd.token_remap import (
    apply_pristine,
    build_pristine_id_mapping,
)

# Corpus rows added for pristine regression (see --mode regression).
# Note: SQLite trigger rejects ASCII letters in sentences; use 甲/乙 not A/B for =/- tests.
PRISTINE_REGRESSION: tuple[tuple[str, str], ...] = (
    ("!", "テストです!"),
    ("?", "本当ですか?"),
    ("ASCII ,", "果物,野菜を食べよう。"),
    (":", "結論:今日は休みです。"),
    ("~", "今日は休み~明日だ。"),
    ('"', '彼は"猫が好き"と言った。'),
    ("U+FF0E", "終わります．"),
    ("｡", "句点は｡で書く。"),
    ("､", "読点は､で区切る。"),
    ("｢｣", "｢引用｣の例です。"),
    ("... token", "もう少し...待って。"),
    (". . . run + ?", "そうかな...?"),
    ("final .", "これで完結です."),
    ("mid . (list)", "3.本文を読んでください。"),
    ("&", "醤油&味噌は合う。"),
    ("%", "割引は50%です。"),
    ("=", "式=甲と乙。"),
    ("-", "試験-甲と乙。"),
    ("*", "注*読むこと。"),
    ("+", "合計+税で百円。"),
    ("two . run", "あ..あ。"),
    ("fullwidth ％", "料金は５０％です。"),
    ("already …", "それは…本当ですか。"),
    ("already 「」", "彼は「猫が好きです」と言った。"),
)


def tokenize_sentence(parser, s):
    from kotogram.kotogram import extract_token_features, split_kotogram
    from kotogram.tokenizer import get_vocab_strings

    kotogram = parser.japanese_to_kotogram(s, fmt="TrainingMask")
    tokens = split_kotogram(kotogram)
    surfs = []
    for t in tokens:
        feats = extract_token_features(t)
        vs = get_vocab_strings(feats)
        surfs.append(vs["surface"])
    return surfs


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        default="demo",
        choices=["dots", "demo", "regression"],
    )
    args = ap.parse_args()

    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

    parser = SudachiJapaneseParser(validate=False)

    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.db")

    if args.mode == "dots":
        # Investigate how Sudachi tokenizes '...' across all corpus sentences
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT sentence FROM sentences WHERE sentence LIKE '%...%' ORDER BY RANDOM() LIMIT 500"
        )
        rows = [r[0] for r in cur.fetchall()]
        conn.close()

        from collections import Counter

        patterns: Counter[str] = Counter()
        examples: dict[str, list[str]] = {}

        for s in rows:
            surfs = tokenize_sentence(parser, s)
            # Find runs containing '.' or '...' tokens
            i = 0
            while i < len(surfs):
                if surfs[i] in (".", "...", "…", ".."):
                    run = []
                    while i < len(surfs) and surfs[i] in (".", "...", "…", ".."):
                        run.append(surfs[i])
                        i += 1
                    pat = " ".join(run)
                    patterns[pat] += 1
                    if pat not in examples or len(examples[pat]) < 3:
                        examples.setdefault(pat, []).append(
                            f"{s[:60]}  →  {' '.join(surfs)}"
                        )
                else:
                    i += 1

        print(f"Checked {len(rows)} sentences containing '...'")
        print("\nDot-token patterns found:")
        for pat, cnt in patterns.most_common():
            print(f"  {cnt:5d}x  [{pat}]")
            for ex in examples[pat][:2]:
                print(f"         {ex}")
        return

    if args.mode == "regression":
        print("Loading dataset bundle (vocab)...")
        bundle, _ = resolve_dataset()
        tokenizer = Tokenizer()
        tokenizer.load_state({"field_vocabs": bundle["vocab"], "frozen": True})
        vocab = tokenizer.field_vocabs["surface"]
        inv_vocab = {v: k for k, v in vocab.items()}
        static_mapping = build_pristine_id_mapping(vocab)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        missing: list[str] = []
        print(f"\nPristine regression ({len(PRISTINE_REGRESSION)} cases)\n")
        for label, s in PRISTINE_REGRESSION:
            row = cur.execute(
                "SELECT 1 FROM sentences WHERE sentence = ?", (s,)
            ).fetchone()
            if not row:
                missing.append(s)
                print(f"[MISSING FROM corpus.db] ({label}) {s}")
                continue

            surfs = tokenize_sentence(parser, s)
            dirty_ids = [vocab.get(sf, 1) for sf in surfs]
            dirty_t = torch.tensor(dirty_ids, dtype=torch.long)
            pristine_t = apply_pristine(dirty_t, vocab, static_mapping=static_mapping)

            dirty_str = " ".join(inv_vocab.get(int(x), "?") for x in dirty_t)
            pristine_str = " ".join(
                inv_vocab.get(int(x), "?") if int(x) != PAD_ID else "___"
                for x in pristine_t
            )
            changes = []
            for d, p in zip(dirty_t.tolist(), pristine_t.tolist()):
                if d != p:
                    dt = inv_vocab.get(d, "?")
                    pt = "PAD" if p == PAD_ID else inv_vocab.get(p, "?")
                    changes.append(f"{dt}->{pt}")
            ch = ", ".join(changes) if changes else "(no change)"
            print(f"({label})")
            print(f"  {s}")
            print(f"  DIRTY:    {dirty_str}")
            print(f"  PRISTINE: {pristine_str}")
            print(f"  {ch}\n")
        conn.close()
        if missing:
            print(f"Missing {len(missing)} row(s); insert into data/corpus.db to fix.")
        return

    # --- demo mode ---
    print("Loading dataset...")
    bundle, _ = resolve_dataset()
    tokenizer = Tokenizer()
    tokenizer.load_state({"field_vocabs": bundle["vocab"], "frozen": True})
    vocab = tokenizer.field_vocabs["surface"]
    inv_vocab = {v: k for k, v in vocab.items()}

    static_mapping = build_pristine_id_mapping(vocab)
    v = len(static_mapping)

    n_pristine = 0
    n_pad = 0
    pad_toks = []
    for tid in range(4, v):
        if static_mapping[tid] == PAD_ID and tid != PAD_ID:
            n_pad += 1
            pad_toks.append((tid, inv_vocab.get(tid, "?")))
        elif int(static_mapping[tid]) != tid:
            n_pristine += 1

    print(f"  Vocab size: {v:,}")
    print(f"  Static pristine rewrites: {n_pristine}")
    print(f"  Non-content -> PAD: {n_pad}")
    print("  Context-dependent: '.' (last-token->。, dot-runs->…+PAD)")

    print("\n  Tokens mapped to PAD (non-content, no pristine equiv):")
    for tid, tok in pad_toks[:30]:
        print(f"    id={tid:6d}  '{tok}'")
    if len(pad_toks) > 30:
        print(f"    ... and {len(pad_toks) - 30} more")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT sentence FROM sentences ORDER BY RANDOM() LIMIT 5000")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()

    changed_sentences = []
    for s in rows:
        surfs = tokenize_sentence(parser, s)

        dirty_ids = [vocab.get(sf, 1) for sf in surfs]
        dirty_t = torch.tensor(dirty_ids, dtype=torch.long)
        pristine_t = apply_pristine(dirty_t, vocab, static_mapping=static_mapping)

        if not torch.equal(dirty_t, pristine_t):
            changes = []
            for i, (d, p) in enumerate(zip(dirty_t.tolist(), pristine_t.tolist())):
                if d != p:
                    d_tok = inv_vocab.get(d, f"[{d}]")
                    if p == PAD_ID:
                        changes.append(f"'{d_tok}'->PAD")
                    else:
                        p_tok = inv_vocab.get(p, f"[{p}]")
                        changes.append(f"'{d_tok}'->'{p_tok}'")
            changed_sentences.append((s, surfs, dirty_t, pristine_t, changes))

    print(f"\n{'=' * 72}")
    print(
        f"Sampled {len(rows):,} sentences, {len(changed_sentences):,} changed by pristine mapping"
    )
    print(f"{'=' * 72}\n")

    for i, (s, surfs, dirty_t, pristine_t, changes) in enumerate(
        changed_sentences[:40]
    ):
        dirty_str = " ".join(inv_vocab.get(int(x), "?") for x in dirty_t)
        pristine_str = " ".join(
            inv_vocab.get(int(x), "?") if int(x) != PAD_ID else "___"
            for x in pristine_t
        )
        print(f"#{i:3d} {s}")
        print(f"     DIRTY:    {dirty_str}")
        print(f"     PRISTINE: {pristine_str}")
        print(f"     CHANGES:  {', '.join(changes)}")
        print()


if __name__ == "__main__":
    main()
