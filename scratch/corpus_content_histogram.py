#!/usr/bin/env python3
"""Histogram of non-content characters per sentence in corpus.db.

Usage: .venv/bin/python scratch/corpus_content_histogram.py
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kotogram.masking import is_content_char

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sentences")
    total = c.fetchone()[0]

    c.execute("SELECT sentence FROM sentences")
    counts: dict[int, int] = {}
    examples: dict[int, str] = {}
    for (sent,) in c:
        n = sum(1 for ch in sent if not is_content_char(ord(ch)))
        counts[n] = counts.get(n, 0) + 1
        if n not in examples or len(sent) < len(examples[n]):
            examples[n] = sent

    conn.close()

    max_n = max(counts.keys())
    bar_width = 50

    print(f"Total sentences: {total:,}")
    print("Non-content chars per is_content_char (JP punctuation = content)")
    print()
    print(f"{'n+':>4s}  {'count':>10s}  {'%':>6s}  {'':>{bar_width}s}  example")
    print(f"{'':->4s}  {'':->10s}  {'':->6s}  {'':->{bar_width}s}  {'':->50s}")

    cumulative = total
    max_cum = total
    for t in range(0, min(max_n + 1, 21)):
        at = counts.get(t, 0)
        pct = 100 * cumulative / total
        filled = int(cumulative / max_cum * bar_width) if max_cum else 0
        bar = "█" * filled

        ex = examples.get(t, "")
        marked = []
        for ch in ex:
            if not is_content_char(ord(ch)):
                marked.append(f"«{ch}»")
            else:
                marked.append(ch)
        ex_str = "".join(marked)
        if len(ex_str) > 80:
            ex_str = ex_str[:77] + "..."

        print(
            f"{t:>3d}+  {cumulative:>10,}  {pct:>5.1f}%  {bar:<{bar_width}s}  {ex_str}"
        )
        cumulative -= at

    print()
    print("Non-content chars shown as «x»")


if __name__ == "__main__":
    main()
