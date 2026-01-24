"""
Tests for scripts/curate_upsert_sentence.py via scripts/curate CLI.
"""
# pylint: disable=redefined-outer-name

import os
import sqlite3
import subprocess
import sys

import pytest


# Fixture to create a temporary DB with the same schema as data/corpus.db
@pytest.fixture
def temp_corpus_db(tmp_path):
    # pylint: disable=redefined-outer-name
    source_db_path = "data/corpus.db"
    if not os.path.exists(source_db_path):
        pytest.skip(f"Source DB not found at {source_db_path}")

    db_path = tmp_path / "test_corpus.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    src_conn = sqlite3.connect(source_db_path)
    src_c = src_conn.cursor()

    tables_to_clone = ["register", "corpus", "grammar"]

    try:
        c.execute("PRAGMA foreign_keys = OFF")

        for table in tables_to_clone:
            src_c.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            res = src_c.fetchone()
            if res:
                create_sql = res[0]
                c.execute(create_sql)
            else:
                if table == "corpus":
                    raise RuntimeError("Corpus table not found in source DB")

        c.execute("PRAGMA foreign_keys = ON")

        # Populate dummy grammar data for validation
        # Only if the table exists (it should, from cloning)
        c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='grammar'"
        )
        if c.fetchone():
            # Insert some dummy points
            dummy_gps = [
                ("gp0001", "Point 1"),
                ("gp0002", "Point 2"),
                ("gp0005", "Point 5"),
            ]
            c.executemany(
                "INSERT OR IGNORE INTO grammar (id, name) VALUES (?, ?)", dummy_gps
            )

        conn.commit()
    finally:
        src_conn.close()

    conn.close()
    return str(db_path)


def run_curate(args):
    """Run scripts/curate with the given arguments."""
    # Run using the same python interpreter to ensure environment consistency
    cmd = [sys.executable, "scripts/curate"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"scripts/curate failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    return result


def test_upsert_insert_new(temp_corpus_db):
    """Test inserting a new sentence defaulting to None/None/0."""
    run_curate(
        ["upsert", "New sentence", "--allow-insert", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT formality, gender, grammatic FROM corpus WHERE sentence='New sentence'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] is None  # Formality
    assert row[1] is None  # Gender
    assert row[2] == 0  # Grammatic (None/None -> 0)


def test_upsert_insert_explicit_valid(temp_corpus_db):
    """Test inserting with explicit valid flags."""
    run_curate(
        [
            "upsert",
            "Formal sentence",
            "--formality=formal",
            "--gender=feminine",
            "--allow-insert",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT formality, gender, grammatic FROM corpus WHERE sentence='Formal sentence'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 0.5  # Formal
    assert row[1] == 1.0  # Feminine
    assert row[2] == 1  # Grammatic


def test_upsert_insert_unpragmatic(temp_corpus_db):
    """Test inserting unpragmatic formality."""
    run_curate(
        [
            "upsert",
            "Weird sentence",
            "--formality=unpragmatic",
            "--gender=masculine",
            "--allow-insert",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT formality, gender, grammatic FROM corpus WHERE sentence='Weird sentence'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] is None  # Formality
    assert row[1] == -1.0  # Masculine
    assert row[2] == 0  # Grammatic (Unpragmatic F forces 0)


def test_upsert_insert_forbidden_by_default(temp_corpus_db):
    """Test inserting a new sentence fails without --allow-insert."""
    # Should fail because "New Forbidden" doesn't exist
    with pytest.raises(RuntimeError, match="Use --allow-insert"):
        run_curate(["upsert", "New Forbidden", "--db-path", temp_corpus_db])


def test_upsert_update_existing(temp_corpus_db):
    """Test updating an existing sentence."""
    conn = sqlite3.connect(temp_corpus_db)
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids) VALUES (?, ?, ?, ?, ?)",
        ("Update me", 0.0, 0.0, 1, ""),
    )
    conn.commit()
    conn.close()

    # Update Formality only
    run_curate(
        ["upsert", "Update me", "--formality=very_formal", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT formality, gender, grammatic FROM corpus WHERE sentence='Update me'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 1.0  # Updated
    assert row[1] == 0.0  # Unchanged
    assert row[2] == 1  # Still grammatic


def test_upsert_update_to_unpragmatic(temp_corpus_db):
    """Test updating existing sentence to unpragmatic."""
    conn = sqlite3.connect(temp_corpus_db)
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids) VALUES (?, ?, ?, ?, ?)",
        ("Good sentence", 0.0, 0.0, 1, ""),
    )
    conn.commit()
    conn.close()

    # Update Gender to Unpragmatic
    run_curate(
        ["upsert", "Good sentence", "--gender=unpragmatic", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT formality, gender, grammatic FROM corpus WHERE sentence='Good sentence'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 0.0  # Unchanged
    assert row[1] is None  # Updated to Unpragmatic
    assert row[2] == 0  # Grammatic flipped to 0


def test_upsert_update_to_valid_from_unpragmatic(temp_corpus_db):
    """Test making an agrammatic sentence grammatic."""
    conn = sqlite3.connect(temp_corpus_db)
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids) VALUES (?, ?, ?, ?, ?)",
        ("Bad sentence", None, None, 0, ""),
    )
    conn.commit()
    conn.close()

    # Update
    run_curate(
        [
            "upsert",
            "Bad sentence",
            "--formality=neutral",
            "--gender=neutral",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT formality, gender, grammatic FROM corpus WHERE sentence='Bad sentence'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 0.0
    assert row[1] == 0.0
    assert row[2] == 1  # Now grammatic


def test_grammar_invalid_id(temp_corpus_db):
    """Test using an invalid grammar ID fails."""
    # This should return non-zero exit code due to raise ValueError
    with pytest.raises(RuntimeError, match="Invalid grammar point ID: 'gp9999'"):
        run_curate(
            [
                "upsert",
                "Invalid grammar check",
                "--grammar=+gp9999",
                "--db-path",
                temp_corpus_db,
            ]
        )


def test_grammar_add_positive(temp_corpus_db):
    """Test adding positive grammar point to existing list."""
    conn = sqlite3.connect(temp_corpus_db)
    # Existing: gp0005. Add: gp0001. Expect: gp0001,gp0005 (sorted)
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("PosAdd", 0.0, 0.0, 1, "", "gp0005", ""),
    )
    conn.commit()
    conn.close()

    run_curate(["upsert", "PosAdd", "--grammar=+gp0001", "--db-path", temp_corpus_db])

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT grammar, grammar_negative, grammatic FROM corpus WHERE sentence='PosAdd'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == "gp0001,gp0005"
    assert row[1] == ""
    assert row[2] == 1


def test_grammar_add_negative(temp_corpus_db):
    """Test adding negative grammar point."""
    conn = sqlite3.connect(temp_corpus_db)
    # Existing neg: gp0002. Add neg: gp0001. Expect sorted.
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("NegAdd", 0.0, 0.0, 1, "", "", "gp0002"),
    )
    conn.commit()
    conn.close()

    run_curate(["upsert", "NegAdd", "--grammar=-gp0001", "--db-path", temp_corpus_db])

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT grammar, grammar_negative FROM corpus WHERE sentence='NegAdd'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == ""
    assert row[1] == "gp0001,gp0002"


def test_grammar_move_neg_to_pos(temp_corpus_db):
    """Test moving a negative label to positive."""
    conn = sqlite3.connect(temp_corpus_db)
    # Existing: neg=gp0001. Op: +gp0001.
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("MoveNegToPos", 0.0, 0.0, 1, "", "", "gp0001"),
    )
    conn.commit()
    conn.close()

    run_curate(
        ["upsert", "MoveNegToPos", "--grammar=+gp0001", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT grammar, grammar_negative FROM corpus WHERE sentence='MoveNegToPos'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == "gp0001"
    assert row[1] == ""


def test_grammar_move_pos_to_neg(temp_corpus_db):
    """Test moving a positive label to negative."""
    conn = sqlite3.connect(temp_corpus_db)
    # Existing: pos=gp0001. Op: -gp0001.
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("MovePosToNeg", 0.0, 0.0, 1, "", "gp0001", ""),
    )
    conn.commit()
    conn.close()

    run_curate(
        ["upsert", "MovePosToNeg", "--grammar=-gp0001", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT grammar, grammar_negative FROM corpus WHERE sentence='MovePosToNeg'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == ""
    assert row[1] == "gp0001"


def test_unpragmatic_formality_resets_grammatic(temp_corpus_db):
    """Regression check: unpragmatic formality resets grammatic to 0."""
    conn = sqlite3.connect(temp_corpus_db)
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("UnpragF", 0.0, 0.0, 1, "", "", ""),
    )
    conn.commit()
    conn.close()

    run_curate(
        ["upsert", "UnpragF", "--formality=unpragmatic", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute("SELECT formality, grammatic FROM corpus WHERE sentence='UnpragF'")
    row = cursor.fetchone()
    conn.close()

    assert row[0] is None
    assert row[1] == 0


def test_unpragmatic_gender_resets_grammatic(temp_corpus_db):
    """Regression check: unpragmatic gender resets grammatic to 0."""
    conn = sqlite3.connect(temp_corpus_db)
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("UnpragG", 0.0, 0.0, 1, "", "", ""),
    )
    conn.commit()
    conn.close()

    run_curate(
        ["upsert", "UnpragG", "--gender=unpragmatic", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute("SELECT gender, grammatic FROM corpus WHERE sentence='UnpragG'")
    row = cursor.fetchone()
    conn.close()

    assert row[0] is None
    assert row[1] == 0


def test_grammar_conflict_error(temp_corpus_db):
    """Corner Case: Conflicting operations (+gp1,-gp1). Should fail."""
    # This should return non-zero exit code due to conflict
    with pytest.raises(
        RuntimeError, match="Conflicting grammar operations for ID 'gp0001'"
    ):
        run_curate(
            [
                "upsert",
                "Conflict",
                "--formality=neutral",
                "--gender=neutral",
                "--grammar=+gp0001,-gp0001",
                "--allow-insert",
                "--db-path",
                temp_corpus_db,
            ]
        )

    # Reverse order should also fail
    with pytest.raises(
        RuntimeError, match="Conflicting grammar operations for ID 'gp0001'"
    ):
        run_curate(
            [
                "upsert",
                "Conflict2",
                "--formality=neutral",
                "--gender=neutral",
                "--grammar=-gp0001,+gp0001",
                "--allow-insert",
                "--db-path",
                temp_corpus_db,
            ]
        )


def test_grammar_redundant_ops(temp_corpus_db):
    """Corner Case: Redundant operations (+gp1, +gp1). Results should be unique set."""
    run_curate(
        [
            "upsert",
            "Redundant",
            "--formality=neutral",
            "--gender=neutral",
            "--grammar=+gp0001,+gp0001",
            "--allow-insert",
            "--db-path",
            temp_corpus_db,
        ]
    )
    conn = sqlite3.connect(temp_corpus_db)
    row = conn.execute(
        "SELECT grammar FROM corpus WHERE sentence='Redundant'"
    ).fetchone()
    conn.close()
    assert row[0] == "gp0001"  # Not "gp0001,gp0001"


def test_grammar_malformed_string(temp_corpus_db):
    """Corner Case: Whitespace and empty segments in argument."""
    # "+gp0001, , -gp0002"
    run_curate(
        [
            "upsert",
            "Malformed",
            "--formality=neutral",
            "--gender=neutral",
            "--grammar=+gp0001, , -gp0002",
            "--allow-insert",
            "--db-path",
            temp_corpus_db,
        ]
    )
    conn = sqlite3.connect(temp_corpus_db)
    row = conn.execute(
        "SELECT grammar, grammar_negative FROM corpus WHERE sentence='Malformed'"
    ).fetchone()
    conn.close()
    assert row[0] == "gp0001"
    assert row[1] == "gp0002"


def test_upsert_batch_from_file(temp_corpus_db):
    """Test batch upsert from file (single transaction)."""
    # Create batch file
    batch_file = "batch_sentences.txt"
    with open(batch_file, "w", encoding="utf-8") as f:
        f.write("Batch 1\n")
        f.write("# Comment line\n")
        f.write("\n")  # Empty line
        f.write("Batch 2\n")

    try:
        run_curate(
            [
                "upsert",
                "--sentences",
                batch_file,
                "--allow-insert",
                "--formality=neutral",
                "--gender=neutral",
                "--db-path",
                temp_corpus_db,
            ]
        )

        conn = sqlite3.connect(temp_corpus_db)
        cursor = conn.cursor()

        # Verify Batch 1
        cursor.execute("SELECT formality FROM corpus WHERE sentence='Batch 1'")
        row1 = cursor.fetchone()
        assert row1 is not None
        assert row1[0] == 0.0

        # Verify Batch 2
        cursor.execute("SELECT formality FROM corpus WHERE sentence='Batch 2'")
        row2 = cursor.fetchone()
        assert row2 is not None
        assert row2[0] == 0.0

        conn.close()
    finally:
        if os.path.exists(batch_file):
            os.remove(batch_file)


def test_grammar_invalid_format_no_sign(temp_corpus_db):
    """Corner Case: Invalid format (missing +/-)."""
    with pytest.raises(RuntimeError, match="Invalid grammar operation: 'gp0001'"):
        run_curate(
            ["upsert", "InvalidFmt", "--grammar=gp0001", "--db-path", temp_corpus_db]
        )


def test_force_grammatic_defaults(temp_corpus_db):
    """Test forcing grammatic=1 sets defaults for missing style."""
    run_curate(
        [
            "upsert",
            "Forced Grammatic",
            "--grammatic=1",
            "--allow-insert",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT formality, gender, grammatic FROM corpus WHERE sentence='Forced Grammatic'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 0.0  # Formality defaults to Neutral
    assert row[1] == 0.0  # Gender defaults to Neutral
    assert row[2] == 1  # Grammatic


def test_force_grammatic_preserves_style(temp_corpus_db):
    """Test forcing grammatic=1 preserves existing style."""
    conn = sqlite3.connect(temp_corpus_db)
    # Existing sentence: Formality=Formal, Gender=Feminine, Grammatic=0
    # This state (F/G set but Gram=0) is valid in DB provided constraint is met?
    # Constraint: grammatic = 0 OR (formality IS NOT NULL AND gender IS NOT NULL) <- Wait.
    # Constraint: CHECK (grammatic = 0 OR (formality IS NOT NULL AND gender IS NOT NULL))
    # No, that's not the constraint.
    # Constraint: CHECK (grammatic = 1 OR (grammar = '' AND grammar_negative = ''))
    # Constraint: CHECK (grammatic = 0 OR (formality IS NOT NULL AND gender IS NOT NULL))
    # -> If Grammatic=1, F/G can be anything? No.
    # Let's check schema in scripts/curate:
    # CHECK (grammatic = 0 OR (formality IS NOT NULL AND gender IS NOT NULL))
    # So if Grammatic=1, F and G MUST be NOT NULL.
    # If Grammatic=0, F and G CAN be NULL (or anything).

    # Let's insert Agrammatic with specific style (should be impossible if we strictly follow "agrammatic implies no style"? No, we can have agrammatic with style?)
    # Wait, usually agrammatic sentences don't have style labels.
    # But let's verify "changing ungrammatic to grammatic shouldn't modify any other fields".
    # User Request: "changing an ungrammatic sentence to grammatic shouldn't modify any other fields"
    # This implies the ungrammatic sentence MIGHT have fields?
    # Or maybe it has NULL fields, and we are adding style?
    # If it has NULL fields, our logic defaults them to 0.0.
    # If it has fields, we should keep them.

    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Ungrammatic With Style", 0.5, 1.0, 0, "", "", ""),
    )
    conn.commit()
    conn.close()

    run_curate(
        [
            "upsert",
            "Ungrammatic With Style",
            "--grammatic=1",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT formality, gender, grammatic FROM corpus WHERE sentence='Ungrammatic With Style'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 0.5  # Preserved
    assert row[1] == 1.0  # Preserved
    assert row[2] == 1  # Updated


def test_force_agrammatic_clears_grammar(temp_corpus_db):
    """Test forcing grammatic=0 clears grammar tags."""
    conn = sqlite3.connect(temp_corpus_db)
    # Existing Grammatic with tags
    conn.execute(
        "INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("Grammatic With Tags", 0.0, 0.0, 1, "", "gp0001", "gp0002"),
    )
    conn.commit()
    conn.close()

    run_curate(
        [
            "upsert",
            "Grammatic With Tags",
            "--grammatic=0",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT grammatic, grammar, grammar_negative FROM corpus WHERE sentence='Grammatic With Tags'"
    )
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 0
    assert row[1] == ""
    assert row[2] == ""
