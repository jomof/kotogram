"""
Tests for scripts/curate_upsert_sentence.py via scripts/curate CLI.
"""
# pylint: disable=redefined-outer-name, too-many-positional-arguments

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

    # Clone tables in dependency order, then view
    tables_to_clone = [
        "register",
        "grammar",
        "sentences",
        "corpus_gp_pos",
        "corpus_gp_neg",
        "corpus",
    ]

    try:
        c.execute("PRAGMA foreign_keys = OFF")

        for table in tables_to_clone:
            # Check for Table OR View
            src_c.execute(
                "SELECT sql FROM sqlite_master WHERE (type='table' OR type='view') AND name=?",
                (table,),
            )
            res = src_c.fetchone()
            if res:
                create_sql = res[0]
                c.execute(create_sql)
            else:
                if table == "corpus":
                    # This implies valid view was not found, which is critical for tests
                    raise RuntimeError("Corpus view (or table) not found in source DB")

        c.execute("PRAGMA foreign_keys = ON")

        # Populate dummy grammar data for validation
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


def insert_test_row(
    db_path,
    sentence,
    formality,
    gender,
    grammatic,
    register_ids="",
    grammar="",
    grammar_negative="",
):
    """Helper to insert a row into the normalized schema."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 1. Insert into sentences
    c.execute(
        "INSERT INTO sentences (sentence, formality, gender, grammatic, register_ids) VALUES (?, ?, ?, ?, ?)",
        (sentence, formality, gender, grammatic, register_ids),
    )

    # 2. Insert grammar relations
    if grammar:
        gps = [gp.strip() for gp in grammar.split(",") if gp.strip()]
        if gps:
            c.executemany(
                "INSERT INTO corpus_gp_pos (sentence, gp_id) VALUES (?, ?)",
                [(sentence, gp) for gp in gps],
            )

    if grammar_negative:
        gps = [gp.strip() for gp in grammar_negative.split(",") if gp.strip()]
        if gps:
            c.executemany(
                "INSERT INTO corpus_gp_neg (sentence, gp_id) VALUES (?, ?)",
                [(sentence, gp) for gp in gps],
            )

    conn.commit()
    conn.close()


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
    assert row is not None
    assert row[0] == 0.0  # Formality (Default Neutral)
    assert row[1] == 0.0  # Gender (Default Neutral)
    assert row[2] == 1  # Grammatic (Default 1)


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
    assert (
        row[1] is None
    )  # Masculine -> Cleared to None because Unpragmatic Formality -> Gram=0
    assert row[2] == 0  # Grammatic (Unpragmatic F forces 0)


def test_upsert_insert_forbidden_by_default(temp_corpus_db):
    """Test inserting a new sentence fails without --allow-insert."""
    # Should fail because "New Forbidden" doesn't exist
    with pytest.raises(RuntimeError, match="Use --allow-insert"):
        run_curate(["upsert", "New Forbidden", "--db-path", temp_corpus_db])


def test_upsert_update_existing(temp_corpus_db):
    """Test updating an existing sentence."""
    insert_test_row(temp_corpus_db, "Update me", 0.0, 0.0, 1)

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
    insert_test_row(temp_corpus_db, "Good sentence", 0.0, 0.0, 1)

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

    assert row[0] is None  # Updated to Unpragmatic (cleared)
    assert row[1] is None  # Updated to Unpragmatic
    assert row[2] == 0  # Grammatic flipped to 0


def test_upsert_update_to_valid_from_unpragmatic(temp_corpus_db):
    """Test making an agrammatic sentence grammatic."""
    insert_test_row(temp_corpus_db, "Bad sentence", None, None, 0)

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
    with pytest.raises(RuntimeError, match="Invalid grammar point ID: 'gp9999'"):
        run_curate(
            [
                "upsert",
                "Invalid grammar check",
                "--grammar=+gp9999",
                "--allow-insert",
                "--formality=neutral",
                "--gender=neutral",
                "--db-path",
                temp_corpus_db,
            ]
        )


def test_grammar_add_positive(temp_corpus_db):
    """Test adding positive grammar point to existing list."""
    # Existing: gp0005. Add: gp0001. Expect: gp0001,gp0005 (sorted)
    insert_test_row(temp_corpus_db, "PosAdd", 0.0, 0.0, 1, grammar="gp0005")

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
    # Existing neg: gp0002. Add neg: gp0001. Expect sorted.
    insert_test_row(temp_corpus_db, "NegAdd", 0.0, 0.0, 1, grammar_negative="gp0002")

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
    # Existing: neg=gp0001. Op: +gp0001.
    insert_test_row(
        temp_corpus_db, "MoveNegToPos", 0.0, 0.0, 1, grammar_negative="gp0001"
    )

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
    # Existing: pos=gp0001. Op: -gp0001.
    insert_test_row(temp_corpus_db, "MovePosToNeg", 0.0, 0.0, 1, grammar="gp0001")

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
    insert_test_row(temp_corpus_db, "UnpragF", 0.0, 0.0, 1)

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
    insert_test_row(temp_corpus_db, "UnpragG", 0.0, 0.0, 1)

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


def test_select_existing_positive(temp_corpus_db):
    """--select-existing=+gpXXXX selects only positive-labeled sentences."""
    insert_test_row(temp_corpus_db, "SelPos-1", 0.0, 0.0, 1, grammar="gp0001")
    insert_test_row(temp_corpus_db, "SelPos-2", 0.0, 0.0, 1, grammar="gp0001,gp0002")
    insert_test_row(temp_corpus_db, "SelPos-3", 0.0, 0.0, 1, grammar_negative="gp0001")
    insert_test_row(temp_corpus_db, "SelPos-4", 0.0, 0.0, 1)

    run_curate(
        [
            "upsert",
            "--select-existing=+gp0001",
            "--formality=very_formal",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    try:
        rows = dict(
            conn.execute(
                "SELECT sentence, formality FROM corpus WHERE sentence LIKE 'SelPos-%'"
            ).fetchall()
        )
    finally:
        conn.close()

    assert rows["SelPos-1"] == 1.0
    assert rows["SelPos-2"] == 1.0
    assert rows["SelPos-3"] == 0.0
    assert rows["SelPos-4"] == 0.0


def test_select_existing_negative(temp_corpus_db):
    """--select-existing=-gpXXXX selects only negative-labeled sentences."""
    insert_test_row(temp_corpus_db, "SelNeg-1", 0.0, 0.0, 1, grammar="gp0001")
    insert_test_row(temp_corpus_db, "SelNeg-2", 0.0, 0.0, 1, grammar_negative="gp0001")
    insert_test_row(
        temp_corpus_db, "SelNeg-3", 0.0, 0.0, 1, grammar_negative="gp0001,gp0002"
    )
    insert_test_row(temp_corpus_db, "SelNeg-4", 0.0, 0.0, 1)

    run_curate(
        [
            "upsert",
            "--select-existing=-gp0001",
            "--gender=feminine",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    try:
        rows = dict(
            conn.execute(
                "SELECT sentence, gender FROM corpus WHERE sentence LIKE 'SelNeg-%'"
            ).fetchall()
        )
    finally:
        conn.close()

    # Feminine maps to 1.0
    assert rows["SelNeg-1"] == 0.0
    assert rows["SelNeg-2"] == 1.0
    assert rows["SelNeg-3"] == 1.0
    assert rows["SelNeg-4"] == 0.0


def test_select_existing_either(temp_corpus_db):
    """--select-existing=!gpXXXX selects sentences with positive OR negative label."""
    insert_test_row(temp_corpus_db, "SelEither-1", 0.0, 0.0, 1, grammar="gp0001")
    insert_test_row(
        temp_corpus_db, "SelEither-2", 0.0, 0.0, 1, grammar_negative="gp0001"
    )
    insert_test_row(temp_corpus_db, "SelEither-3", 0.0, 0.0, 1, grammar="gp0002")

    run_curate(
        [
            "upsert",
            "--select-existing=!gp0001",
            "--formality=casual",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    try:
        rows = dict(
            conn.execute(
                "SELECT sentence, formality FROM corpus WHERE sentence LIKE 'SelEither-%'"
            ).fetchall()
        )
    finally:
        conn.close()

    # Casual maps to -0.5
    assert rows["SelEither-1"] == -0.5
    assert rows["SelEither-2"] == -0.5
    assert rows["SelEither-3"] == 0.0


def test_select_existing_conflicts_with_sentence(temp_corpus_db):
    """--select-existing cannot be combined with a positional sentence."""
    insert_test_row(
        temp_corpus_db, "ConflictSelExisting", 0.0, 0.0, 1, grammar="gp0001"
    )
    with pytest.raises(
        RuntimeError,
        match="Cannot specify both '--select-existing' and 'sentence/--sentences'",
    ):
        run_curate(
            [
                "upsert",
                "Some sentence",
                "--select-existing=+gp0001",
                "--db-path",
                temp_corpus_db,
            ]
        )


def test_select_existing_conflicts_with_sentences_file(temp_corpus_db):
    """--select-existing cannot be combined with --sentences."""
    batch_file = "batch_sentences_for_select_existing_conflict.txt"
    with open(batch_file, "w", encoding="utf-8") as f:
        f.write("Any sentence\n")
    try:
        with pytest.raises(
            RuntimeError,
            match="Cannot specify both '--select-existing' and 'sentence/--sentences'",
        ):
            run_curate(
                [
                    "upsert",
                    "--sentences",
                    batch_file,
                    "--select-existing=+gp0001",
                    "--db-path",
                    temp_corpus_db,
                ]
            )
    finally:
        if os.path.exists(batch_file):
            os.remove(batch_file)


def test_select_existing_invalid_format_missing_sign(temp_corpus_db):
    """--select-existing must start with +, -, or !."""
    with pytest.raises(RuntimeError, match="Invalid '--select-existing' value"):
        run_curate(["upsert", "--select-existing=gp0001", "--db-path", temp_corpus_db])


def test_select_existing_invalid_format_missing_id(temp_corpus_db):
    """--select-existing must include a grammar id after the sign."""
    with pytest.raises(RuntimeError, match="Missing grammar point id"):
        run_curate(["upsert", "--select-existing=+", "--db-path", temp_corpus_db])


def test_grammar_invalid_format_no_sign(temp_corpus_db):
    """Corner Case: Invalid format (missing +/-)."""
    with pytest.raises(RuntimeError, match="Invalid grammar operation: 'gp0001'"):
        run_curate(
            [
                "upsert",
                "InvalidFmt",
                "--grammar=gp0001",
                "--allow-insert",
                "--formality=neutral",
                "--gender=neutral",
                "--db-path",
                temp_corpus_db,
            ]
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


def test_force_agrammatic_clears_grammar(temp_corpus_db):
    """Test forcing grammatic=0 clears grammar tags."""
    insert_test_row(
        temp_corpus_db,
        "Grammatic With Tags",
        0.0,
        0.0,
        1,
        grammar="gp0001",
        grammar_negative="gp0002",
    )

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


def test_grammar_remove_positive(temp_corpus_db):
    """Test removing positive grammar label with !."""
    insert_test_row(temp_corpus_db, "RemovePos", 0.0, 0.0, 1, grammar="gp0001,gp0005")

    run_curate(
        ["upsert", "RemovePos", "--grammar=!gp0001", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    row = conn.execute(
        "SELECT grammar FROM corpus WHERE sentence='RemovePos'"
    ).fetchone()
    conn.close()

    assert row[0] == "gp0005"


def test_grammar_remove_negative(temp_corpus_db):
    """Test removing negative grammar label with !."""
    insert_test_row(
        temp_corpus_db, "RemoveNeg", 0.0, 0.0, 1, grammar_negative="gp0002,gp0005"
    )

    run_curate(
        ["upsert", "RemoveNeg", "--grammar=!gp0002", "--db-path", temp_corpus_db]
    )

    conn = sqlite3.connect(temp_corpus_db)
    row = conn.execute(
        "SELECT grammar_negative FROM corpus WHERE sentence='RemoveNeg'"
    ).fetchone()
    conn.close()

    assert row[0] == "gp0005"


def test_grammar_remove_mixed_ops(temp_corpus_db):
    """Test mixing +, -, and ! in one command."""
    # Start: Pos=gp0001, Neg=gp0002.
    # Op: !gp0001, !gp0002, +gp0005
    insert_test_row(
        temp_corpus_db,
        "MixedOps",
        0.0,
        0.0,
        1,
        grammar="gp0001",
        grammar_negative="gp0002",
    )

    run_curate(
        [
            "upsert",
            "MixedOps",
            "--grammar=!gp0001,!gp0002,+gp0005",
            "--db-path",
            temp_corpus_db,
        ]
    )

    conn = sqlite3.connect(temp_corpus_db)
    row = conn.execute(
        "SELECT grammar, grammar_negative FROM corpus WHERE sentence='MixedOps'"
    ).fetchone()
    conn.close()

    assert row[0] == "gp0005"
    assert row[1] == ""
