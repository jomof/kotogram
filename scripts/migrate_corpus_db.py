import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Set

# pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-return-statements

# Configuration
OLD_DB_PATH = "data/corpus.db"
MIGRATED_DB_PATH = "data/corpus-migrated.db"
BATCH_SIZE = 20000

# SQL Definitions - Compact Design
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- dictionaries
CREATE TABLE register(
  id INTEGER PRIMARY KEY,
  label TEXT
);

-- Keep grammar dictionary as-is (text PK, compact and inspectable)
CREATE TABLE grammar(
  id TEXT PRIMARY KEY,   -- "gp0001"
  name TEXT NOT NULL
) WITHOUT ROWID;

-- sentences is the canonical storage for per-sentence fields (including register_ids CSV)
CREATE TABLE sentences(
  sentence TEXT PRIMARY KEY,

  formality REAL CHECK(formality IS NULL OR formality IN(-1.0, -0.5, 0.0, 0.5, 1.0)),
  gender    REAL CHECK(gender    IS NULL OR gender    IN(-1.0, 0.0, 1.0)),
  grammatic INTEGER NOT NULL CHECK(grammatic IN(0, 1)),

  register_ids TEXT NOT NULL,

  -- Preserve old semantic constraints (same as old corpus table)
  CHECK(grammatic = 0 OR (formality IS NOT NULL AND gender IS NOT NULL)),
  CHECK(grammatic = 1 OR (formality IS NULL AND gender IS NULL))
) WITHOUT ROWID;

-- Normalized relations (sentence TEXT FK)
CREATE TABLE corpus_gp_pos(
  sentence TEXT NOT NULL,
  gp_id    TEXT NOT NULL,
  PRIMARY KEY(sentence, gp_id),
  FOREIGN KEY(sentence) REFERENCES sentences(sentence) ON DELETE CASCADE,
  FOREIGN KEY(gp_id)    REFERENCES grammar(id)
) WITHOUT ROWID;

CREATE TABLE corpus_gp_neg(
  sentence TEXT NOT NULL,
  gp_id    TEXT NOT NULL,
  PRIMARY KEY(sentence, gp_id),
  FOREIGN KEY(sentence) REFERENCES sentences(sentence) ON DELETE CASCADE,
  FOREIGN KEY(gp_id)    REFERENCES grammar(id)
) WITHOUT ROWID;

-- key-value metadata (excluded from content hashing)
CREATE TABLE metadata(
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
) WITHOUT ROWID;

-- corpus VIEW: mimics old corpus table layout EXACTLY
CREATE VIEW corpus AS
SELECT
  s.sentence,
  s.formality,
  s.gender,
  s.grammatic,
  s.register_ids,
  COALESCE((
    SELECT group_concat(gp_id, ',')
    FROM (
      SELECT gp_id
      FROM corpus_gp_pos
      WHERE sentence = s.sentence
      ORDER BY gp_id
    )
  ), '') AS grammar,
  COALESCE((
    SELECT group_concat(gp_id, ',')
    FROM (
      SELECT gp_id
      FROM corpus_gp_neg
      WHERE sentence = s.sentence
      ORDER BY gp_id
    )
  ), '') AS grammar_negative
FROM sentences s;
"""

TRIGGERS_SQL = """
-- 1) Enforce no labels on ungrammatic sentences in normalized tables
CREATE TRIGGER corpus_gp_pos_require_grammatic
BEFORE INSERT ON corpus_gp_pos
FOR EACH ROW
WHEN (SELECT grammatic FROM sentences WHERE sentence = NEW.sentence) != 1
BEGIN
  SELECT RAISE(ABORT, 'Cannot add positive grammar labels to ungrammatic sentence');
END;

CREATE TRIGGER corpus_gp_neg_require_grammatic
BEFORE INSERT ON corpus_gp_neg
FOR EACH ROW
WHEN (SELECT grammatic FROM sentences WHERE sentence = NEW.sentence) != 1
BEGIN
  SELECT RAISE(ABORT, 'Cannot add negative grammar labels to ungrammatic sentence');
END;

-- 2) Enforce no gp in both pos and neg (normalized)
CREATE TRIGGER prevent_gp_in_both_polarities_pos
BEFORE INSERT ON corpus_gp_pos
FOR EACH ROW
WHEN EXISTS (
  SELECT 1 FROM corpus_gp_neg
  WHERE sentence = NEW.sentence AND gp_id = NEW.gp_id
)
BEGIN
  SELECT RAISE(ABORT, 'gp cannot be both positive and negative for the same sentence');
END;

CREATE TRIGGER prevent_gp_in_both_polarities_neg
BEFORE INSERT ON corpus_gp_neg
FOR EACH ROW
WHEN EXISTS (
  SELECT 1 FROM corpus_gp_pos
  WHERE sentence = NEW.sentence AND gp_id = NEW.gp_id
)
BEGIN
  SELECT RAISE(ABORT, 'gp cannot be both positive and negative for the same sentence');
END;

-- 3) If a sentence is updated to ungrammatic, clear normalized labels too
CREATE TRIGGER corpus_clear_labels_when_ungrammatic
AFTER UPDATE OF grammatic ON sentences
FOR EACH ROW
WHEN NEW.grammatic = 0
BEGIN
  DELETE FROM corpus_gp_pos WHERE sentence = NEW.sentence;
  DELETE FROM corpus_gp_neg WHERE sentence = NEW.sentence;
END;

-- 4) Enforce register 0 cannot be mixed with others
CREATE TRIGGER sentences_register_ids_no_mix
BEFORE INSERT ON sentences
FOR EACH ROW
WHEN (
  (',' || NEW.register_ids || ',') LIKE '%,0,%'
  AND NEW.register_ids LIKE '%,%'
)
BEGIN
  SELECT RAISE(ABORT, 'register_ids cannot include 0 with other register ids');
END;

CREATE TRIGGER sentences_register_ids_no_mix_update
BEFORE UPDATE OF register_ids ON sentences
FOR EACH ROW
WHEN (
  (',' || NEW.register_ids || ',') LIKE '%,0,%'
  AND NEW.register_ids LIKE '%,%'
)
BEGIN
  SELECT RAISE(ABORT, 'register_ids cannot include 0 with other register ids');
END;
"""


def parse_csv_set(s: str) -> Set[str]:
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def migrate(
    limit: int = 0, dry_run: bool = False, allow_ungrammatic_destruction: bool = False
) -> None:
    print(f"Starting migration{' (DRY RUN)' if dry_run else ''}...")
    start_time = time.time()

    if os.path.exists(MIGRATED_DB_PATH):
        os.remove(MIGRATED_DB_PATH)

    # 1. Setup New DB
    mig_conn = sqlite3.connect(MIGRATED_DB_PATH)
    mig_c = mig_conn.cursor()

    mig_c.executescript(SCHEMA_SQL)
    mig_c.executescript(TRIGGERS_SQL)
    mig_c.execute("PRAGMA foreign_keys=ON")

    # 2. Attach Old DB (Read-Only)
    old_conn = sqlite3.connect(f"file:{OLD_DB_PATH}?mode=ro", uri=True)
    old_conn.execute("PRAGMA foreign_keys=ON")
    old_c = old_conn.cursor()

    mig_c.execute("BEGIN TRANSACTION")

    # Copy Dictionaries
    print("Copying registers...")
    old_c.execute("SELECT id, label FROM register")
    mig_c.executemany("INSERT INTO register(id, label) VALUES (?, ?)", old_c.fetchall())

    print("Copying grammar...")
    old_c.execute("SELECT id, name FROM grammar")
    mig_c.executemany("INSERT INTO grammar(id, name) VALUES (?, ?)", old_c.fetchall())

    # Validation Sets
    mig_c.execute("SELECT id FROM grammar")
    valid_gp_codes = {row[0] for row in mig_c.fetchall()}

    # Copy Sentences & Relations
    print("Copying sentences and relations...")

    old_c.execute("SELECT count(*) FROM corpus")
    total_rows = old_c.fetchone()[0]
    if limit > 0:
        total_rows = min(total_rows, limit)

    offset = 0
    processed = 0

    sql_insert_sentence = """
        INSERT INTO sentences(sentence, formality, gender, grammatic, register_ids)
        VALUES (?, ?, ?, ?, ?)
    """
    sql_insert_pos = "INSERT INTO corpus_gp_pos(sentence, gp_id) VALUES (?, ?)"
    sql_insert_neg = "INSERT INTO corpus_gp_neg(sentence, gp_id) VALUES (?, ?)"

    while processed < total_rows:
        batch_limit = min(BATCH_SIZE, total_rows - processed)
        old_c.execute(f"""
            SELECT sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative
            FROM corpus LIMIT {batch_limit} OFFSET {offset}
        """)
        rows = old_c.fetchall()

        if not rows:
            break

        batch_sentences = []
        batch_pos = []
        batch_neg = []

        for row in rows:
            # row: sentence, formality, gender, grammatic, reg_str, gram_str, neg_str
            s_txt = row[0]
            formality = row[1]
            gender = row[2]
            grammatic = row[3]
            reg_str = row[4]
            gram_str = row[5]
            neg_str = row[6]

            # Semantic Checks & optional sanitization
            if grammatic == 0:
                if formality is not None or gender is not None:
                    if allow_ungrammatic_destruction:
                        formality = None
                        gender = None
                    else:
                        raise RuntimeError(
                            f"Ungrammatic sentence '{s_txt}' has non-null attributes: "
                            f"formality={formality}, gender={gender}. Migration aborted."
                        )
                # Ensure no grammar labels -> if strictly invalid, we must clear them if destruction allowed,
                # or abort. Design: grammar tables are normalized, so we just don't insert them.
                # But we should check if they exist in input and allow/disallow.
                if gram_str or neg_str:
                    if allow_ungrammatic_destruction:
                        gram_str = ""
                        neg_str = ""
                    else:
                        raise RuntimeError(
                            f"Ungrammatic sentence '{s_txt}' has grammar labels. Migration aborted."
                        )

            # Registers Processing
            reg_ids_set = parse_csv_set(reg_str)
            reg_ints = []
            for rid in reg_ids_set:
                if not rid.isdigit():
                    raise RuntimeError(
                        f"Invalid register ID '{rid}' in sentence '{s_txt}'"
                    )
                reg_ints.append(int(rid))

            if not reg_ints:
                # Assuming empty means error or default to 0?
                # User said: "register_ids must not be empty; if neutral, it must be exactly '0'".
                # If old DB had empty, we insert '0'.
                reg_ints = [0]

            if 0 in reg_ints and len(reg_ints) > 1:
                raise RuntimeError(
                    f"Register 0 mixed with others in sentence '{s_txt}': {reg_ints}"
                )

            reg_ints.sort()
            canonical_reg_str = ",".join(str(i) for i in reg_ints)

            batch_sentences.append(
                (s_txt, formality, gender, grammatic, canonical_reg_str)
            )

            # Grammar Positive
            pos_codes = parse_csv_set(gram_str)
            for code in pos_codes:
                if code not in valid_gp_codes:
                    raise RuntimeError(
                        f"Unknown grammar code {code!r} in sentence {s_txt!r}"
                    )
                batch_pos.append((s_txt, code))

            # Grammar Negative
            neg_codes = parse_csv_set(neg_str)
            for code in neg_codes:
                if code not in valid_gp_codes:
                    raise RuntimeError(
                        f"Unknown grammar code {code!r} in sentence {s_txt!r} (negative)"
                    )
                batch_neg.append((s_txt, code))

            # Overlap Check
            overlap = pos_codes.intersection(neg_codes)
            if overlap:
                raise RuntimeError(
                    f"Overlap detected for sentence '{s_txt}': {overlap}"
                )

        mig_c.executemany(sql_insert_sentence, batch_sentences)
        mig_c.executemany(sql_insert_pos, batch_pos)
        mig_c.executemany(sql_insert_neg, batch_neg)

        processed += len(rows)
        offset += len(rows)
        print(f"Processed {processed} rows...", end="\r")

    print("\nCommitting transaction...")
    mig_conn.commit()

    # 3. Validation
    validate_migration(limit, allow_ungrammatic_destruction, old_conn, mig_conn)

    print("Compacting database...")
    mig_conn.execute("VACUUM")

    old_conn.close()
    mig_conn.close()

    print(f"Migration phase complete in {time.time() - start_time:.2f}s.")

    if not dry_run:
        swap_and_test()
    else:
        print("Dry run complete. No swap performed.")


def validate_migration(
    limit: int,
    allow_destruction: bool,
    old_conn: sqlite3.Connection,
    mig_conn: sqlite3.Connection,
) -> None:
    print("Validating...")
    old_c = old_conn.cursor()
    mig_c = mig_conn.cursor()

    # 1. Counts
    old_c.execute("SELECT count(*) FROM corpus")
    old_count = old_c.fetchone()[0]

    # Check VIEW count
    mig_c.execute("SELECT count(*) FROM corpus")
    new_count = mig_c.fetchone()[0]

    if old_count != new_count and limit == 0:
        raise ValueError(f"Count mismatch: Old {old_count}, New {new_count}")

    # 2. Ungrammatic Safety
    mig_c.execute("""
        SELECT COUNT(*) FROM corpus_gp_pos p JOIN sentences s ON s.sentence=p.sentence
        WHERE s.grammatic=0
    """)
    if mig_c.fetchone()[0] > 0:
        raise ValueError("Found positive grammar labels on ungrammatic sentences")

    mig_c.execute("""
        SELECT COUNT(*) FROM corpus_gp_neg p JOIN sentences s ON s.sentence=p.sentence
        WHERE s.grammatic=0
    """)
    if mig_c.fetchone()[0] > 0:
        raise ValueError("Found negative grammar labels on ungrammatic sentences")

    # 3. Full 1:1 Verification (Streaming)
    print("Running full 1:1 verification (matching migration scope)...")

    # We must iterate the exact same slice of the Old DB that we migrated
    old_query = "SELECT sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative FROM corpus"
    if limit > 0:
        old_query += f" LIMIT {limit}"

    old_c.execute(old_query)

    count = 0
    while True:
        row = old_c.fetchone()
        if not row:
            break

        s_txt = row[0]
        old_form = row[1]
        old_gend = row[2]
        old_gram = row[3]
        old_reg_str = row[4]
        old_pos_str = row[5]
        old_neg_str = row[6]

        # Query New View
        mig_c.execute(
            """
            SELECT formality, gender, grammatic, register_ids, grammar, grammar_negative
            FROM corpus WHERE sentence=?
        """,
            (s_txt,),
        )
        new_row = mig_c.fetchone()

        if not new_row:
            raise ValueError(f"Sentence missing in new DB: {s_txt}")

        new_form = new_row[0]
        new_gend = new_row[1]
        new_gram = new_row[2]
        new_reg_str = new_row[3]
        new_pos_str = new_row[4]
        new_neg_str = new_row[5]

        # Check Fields
        if old_form != new_form:
            if not (allow_destruction and old_gram == 0 and new_form is None):
                raise ValueError(
                    f"Mismatch formality for '{s_txt}': {old_form} != {new_form}"
                )
        if old_gend != new_gend:
            if not (allow_destruction and old_gram == 0 and new_gend is None):
                raise ValueError(
                    f"Mismatch gender for '{s_txt}': {old_gend} != {new_gend}"
                )
        if old_gram != new_gram:
            raise ValueError(
                f"Mismatch grammatic for '{s_txt}': {old_gram} != {new_gram}"
            )

        # Check Sets for CSVs
        # Note: Canonicalization handles the comparison
        old_reg_set = parse_csv_set(old_reg_str)
        # Normalize old empty to '0' logic if needed?
        if not old_reg_set:
            old_reg_set = {"0"}  # as per migration logic

        new_reg_set = parse_csv_set(new_reg_str)

        if old_reg_set != new_reg_set:
            # Handle string->int->string roundtrip differences if 0 vs '0' etc
            # Normalize to ints for set comparison
            old_regs_int = {int(x) for x in old_reg_set if x.isdigit()}
            new_regs_int = {int(x) for x in new_reg_set if x.isdigit()}
            if old_regs_int != new_regs_int:
                raise ValueError(
                    f"Mismatch register_ids for '{s_txt}': {old_regs_int} vs {new_regs_int}"
                )

        old_pos_set = parse_csv_set(old_pos_str)
        if allow_destruction and old_gram == 0:
            old_pos_set = set()

        new_pos_set = parse_csv_set(new_pos_str)
        if old_pos_set != new_pos_set:
            raise ValueError(f"Mismatch grammar for '{s_txt}'")

        old_neg_set = parse_csv_set(old_neg_str)
        if allow_destruction and old_gram == 0:
            old_neg_set = set()
        new_neg_set = parse_csv_set(new_neg_str)
        if old_neg_set != new_neg_set:
            raise ValueError(f"Mismatch grammar_negative for '{s_txt}'")

        # 4. Normalized Relation Consistency Check
        mig_c.execute("SELECT gp_id FROM corpus_gp_pos WHERE sentence=?", (s_txt,))
        norm_pos = {r[0] for r in mig_c.fetchall()}
        if norm_pos != new_pos_set:
            raise ValueError(f"Normalized POS table mismatch for '{s_txt}'")

        mig_c.execute("SELECT gp_id FROM corpus_gp_neg WHERE sentence=?", (s_txt,))
        norm_neg = {r[0] for r in mig_c.fetchall()}
        if norm_neg != new_neg_set:
            raise ValueError(f"Normalized NEG table mismatch for '{s_txt}'")

        count += 1
        if count % 1000 == 0:
            print(f"Verified {count} sentences...", end="\r")

    print(f"\nVerification passed for {count} sentences.")


def swap_and_test() -> None:
    print("Performing Swap...")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = f"{OLD_DB_PATH}.bak.{timestamp}"

    shutil.move(OLD_DB_PATH, backup_path)
    shutil.move(MIGRATED_DB_PATH, OLD_DB_PATH)

    # Clean up WAL/SHM if they exist
    if os.path.exists(f"{OLD_DB_PATH}-wal"):
        os.remove(f"{OLD_DB_PATH}-wal")
    if os.path.exists(f"{OLD_DB_PATH}-shm"):
        os.remove(f"{OLD_DB_PATH}-shm")

    print(f"Swap complete. Backup at {backup_path}.")
    print("Running tests...")

    try:
        subprocess.run(["./test.sh"], check=True)
        print("Tests passed!")
    except subprocess.CalledProcessError:
        print("TESTS FAILED. Rolling back...")
        if os.path.exists(OLD_DB_PATH):
            os.remove(OLD_DB_PATH)
        shutil.move(backup_path, OLD_DB_PATH)
        print("Rollback complete.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate corpus.db to normalized schema (Compact Design)."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Perform migration without swapping DBs"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of rows for testing"
    )
    parser.add_argument(
        "--allow-ungrammatic-destruction",
        action="store_true",
        help="Allow clearing attributes for ungrammatic sentences",
    )
    args = parser.parse_args()

    migrate(
        limit=args.limit,
        dry_run=args.dry_run,
        allow_ungrammatic_destruction=args.allow_ungrammatic_destruction,
    )
