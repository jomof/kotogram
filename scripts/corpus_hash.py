"""Deterministic content hash for corpus.db.

Computes a SHA-256 over all content tables (register, grammar, sentences,
corpus_gp_pos, corpus_gp_neg) in primary-key order.  The ``metadata`` table
is deliberately excluded so that writing hash bookkeeping into it does not
invalidate the hash itself.
"""

import hashlib
import sqlite3
from typing import Any, Dict, Optional

_CONTENT_TABLES = [
    ("register", "SELECT id, label FROM register ORDER BY id"),
    ("grammar", "SELECT id, name FROM grammar ORDER BY id"),
    (
        "sentences",
        "SELECT sentence, formality, gender, grammatic, register_ids "
        "FROM sentences ORDER BY sentence",
    ),
    (
        "corpus_gp_pos",
        "SELECT sentence, gp_id FROM corpus_gp_pos ORDER BY sentence, gp_id",
    ),
    (
        "corpus_gp_neg",
        "SELECT sentence, gp_id FROM corpus_gp_neg ORDER BY sentence, gp_id",
    ),
]


def corpus_content_hash(db_path: str) -> str:
    """Deterministic SHA-256 over all content tables (excludes metadata)."""
    h = hashlib.sha256()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for table_name, query in _CONTENT_TABLES:
            h.update(table_name.encode("utf-8"))
            for row in conn.execute(query):
                for col in row:
                    if col is None:
                        h.update(b"\x00")
                    elif isinstance(col, str):
                        h.update(col.encode("utf-8"))
                    elif isinstance(col, (int, float)):
                        h.update(repr(col).encode("ascii"))
                    else:
                        h.update(bytes(col))
                h.update(b"\x1e")  # record separator
    finally:
        conn.close()
    return h.hexdigest()


def read_metadata(db_path: str, key: str) -> Optional[str]:
    """Read a single value from the metadata table. Returns None if missing."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone()
        if not has_table:
            return None
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def write_metadata(db_path: str, key: str, value: str) -> None:
    """Upsert a single value into the metadata table."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata"
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def corpus_summary(db_path: str) -> Dict[str, Any]:
    """Return a dict of corpus statistics for display during dataset build."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        grammatic = conn.execute(
            "SELECT COUNT(*) FROM sentences WHERE grammatic = 1"
        ).fetchone()[0]
        agrammatic = total - grammatic

        formality_set = conn.execute(
            "SELECT COUNT(*) FROM sentences WHERE formality IS NOT NULL"
        ).fetchone()[0]
        gender_set = conn.execute(
            "SELECT COUNT(*) FROM sentences WHERE gender IS NOT NULL"
        ).fetchone()[0]

        gp_count = conn.execute("SELECT COUNT(*) FROM grammar").fetchone()[0]
        gp_pos_count = conn.execute("SELECT COUNT(*) FROM corpus_gp_pos").fetchone()[0]
        gp_neg_count = conn.execute("SELECT COUNT(*) FROM corpus_gp_neg").fetchone()[0]

        register_dist: Dict[str, int] = {}
        for label, cnt in conn.execute(
            "SELECT r.label, COUNT(*) "
            "FROM sentences s "
            "JOIN register r ON (',' || s.register_ids || ',') LIKE ('%,' || r.id || ',%') "
            "GROUP BY r.label ORDER BY COUNT(*) DESC"
        ):
            register_dist[label] = cnt

        return {
            "total_sentences": total,
            "grammatic": grammatic,
            "agrammatic": agrammatic,
            "formality_labeled": formality_set,
            "gender_labeled": gender_set,
            "grammar_points": gp_count,
            "gp_pos_annotations": gp_pos_count,
            "gp_neg_annotations": gp_neg_count,
            "register_distribution": register_dist,
        }
    finally:
        conn.close()
