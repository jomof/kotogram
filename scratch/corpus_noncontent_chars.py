#!/usr/bin/env python3
"""Frequency histogram of non-content characters in corpus.db.

Usage: .venv/bin/python scratch/corpus_noncontent_chars.py
"""

import os
import sqlite3
import sys
import unicodedata
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kotogram.masking import is_content_char

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.db")


def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sentences")
    total = c.fetchone()[0]

    c.execute("SELECT sentence FROM sentences")
    char_freq: Counter[str] = Counter()
    sent_freq: Counter[str] = Counter()
    for (sent,) in c:
        seen: set[str] = set()
        for ch in sent:
            if not is_content_char(ord(ch)):
                char_freq[ch] += 1
                if ch not in seen:
                    sent_freq[ch] += 1
                    seen.add(ch)

    conn.close()

    max_sent = max(sent_freq.values()) if sent_freq else 1
    bar_width = 40

    print(f"Total sentences: {total:,}")
    print()
    print(
        f"  {'char':>4s}  {'code':>8s}  {'name':<40s}"
        f"  {'occur':>10s}  {'sents':>10s}  {'%':>6s}  bar"
    )
    print(
        f"  {'':->4s}  {'':->8s}  {'':-<40s}"
        f"  {'':->10s}  {'':->10s}  {'':->6s}  {'':->40s}"
    )
    for ch, n in char_freq.most_common():
        name = unicodedata.name(ch, "?")
        sn = sent_freq[ch]
        pct = 100 * sn / total
        filled = max(1, int(sn / max_sent * bar_width))
        bar = "█" * filled
        print(
            f"  {ch!r:>4s}  U+{ord(ch):04X}  {name:<40s}"
            f"  {n:>10,}  {sn:>10,}  {pct:>5.2f}%  {bar}"
        )


if __name__ == "__main__":
    main()
