"""Tests for register label-to-ID mapping consistency.

These tests ensure that the register label mappings in kotogram/constants.py
(the source of truth used at inference time) remain consistent and match
the data/corpus.db register table.
"""

import os
import sqlite3

import pytest

from kotogram.constants import (
    REGISTER_ID_TO_LABEL,
    REGISTER_LABEL_TO_ID,
    RegisterLevel,
)


def test_register_mapping_is_bijection():
    """Ensure label→ID and ID→label mappings are inverses."""
    assert len(REGISTER_LABEL_TO_ID) == len(REGISTER_ID_TO_LABEL), (
        "Forward and reverse mappings must have same length"
    )

    for label, id_val in REGISTER_LABEL_TO_ID.items():
        assert REGISTER_ID_TO_LABEL[id_val] == label, (
            f"Reverse mapping broken for {label} → {id_val}"
        )

    for id_val, label in REGISTER_ID_TO_LABEL.items():
        assert REGISTER_LABEL_TO_ID[label] == id_val, (
            f"Forward mapping broken for {id_val} → {label}"
        )


def test_register_mapping_matches_db():
    """Ensure code mappings match the source of truth in corpus.db.

    Note: The code (kotogram/constants.py) IS the source of truth for inference.
    This test validates that corpus.db is in sync with the code.
    """
    db_path = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.db")

    if not os.path.exists(db_path):
        pytest.skip("corpus.db not found")

    # Use helper to check if table exists and get data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='register'"
    )
    table_exists = cursor.fetchone()

    if not table_exists:
        conn.close()
        pytest.skip("register table not found in corpus.db")

    cursor.execute("SELECT id, label FROM register ORDER BY id")
    db_mapping = {row[0]: row[1].upper() for row in cursor.fetchall()}
    conn.close()

    # Verify all DB entries are in code
    for db_id, db_label_upper in db_mapping.items():
        assert db_id in REGISTER_ID_TO_LABEL, (
            f"DB has ID {db_id} ('{db_label_upper}') but code doesn't"
        )
        code_label = REGISTER_ID_TO_LABEL[db_id]
        assert code_label.name == db_label_upper, (
            f"ID {db_id}: DB has '{db_label_upper}' but code has '{code_label.name}'"
        )

    # Verify all code entries are in DB
    for code_id, code_label in REGISTER_ID_TO_LABEL.items():
        assert code_id in db_mapping, (
            f"Code has ID {code_id} ('{code_label.name}') but DB doesn't"
        )
        assert code_label.name == db_mapping[code_id], (
            f"ID {code_id}: Code has '{code_label.name}' but DB has '{db_mapping[code_id]}'"
        )


def test_register_enum_has_all_labels():
    """Ensure RegisterLevel enum has exactly the labels we expect."""
    expected = {
        "NEUTRAL",
        "SONKEIGO",
        "KENJOGO",
        "KANSAIBEN",
        "HAKATABEN",
        "KYOSHIGO",
        "NETSLANG",
        "OJOUSAMA",
        "GUNTAI",
        "JOSEIGO",
        "DANSEIGO",
        "BURIKKO",
        "TOHOKU",
        "BUSHI",
    }
    actual = {label.name for label in RegisterLevel}
    assert actual == expected, (
        f"Enum labels mismatch: {actual.symmetric_difference(expected)}"
    )


def test_register_ids_are_contiguous():
    """Ensure register IDs form a contiguous sequence 0..N-1."""
    ids = sorted(REGISTER_ID_TO_LABEL.keys())
    expected_ids = list(range(len(ids)))
    assert ids == expected_ids, (
        f"Register IDs must be contiguous starting from 0, got {ids}"
    )


def test_register_mapping_specific_values():
    """Test specific register ID mappings that are critical for the system.

    These are the mappings that were causing the bug described in the issue.
    """
    # Test the specific values that were wrong in the bug
    assert REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL] == 0, "NEUTRAL must be ID 0"
    assert REGISTER_LABEL_TO_ID[RegisterLevel.DANSEIGO] == 10, "DANSEIGO must be ID 10"
    assert REGISTER_LABEL_TO_ID[RegisterLevel.BURIKKO] == 11, "BURIKKO must be ID 11"
    assert REGISTER_LABEL_TO_ID[RegisterLevel.TOHOKU] == 12, "TOHOKU must be ID 12"

    # Verify reverse mapping
    assert REGISTER_ID_TO_LABEL[0] == RegisterLevel.NEUTRAL
    assert REGISTER_ID_TO_LABEL[10] == RegisterLevel.DANSEIGO
    assert REGISTER_ID_TO_LABEL[11] == RegisterLevel.BURIKKO
    assert REGISTER_ID_TO_LABEL[12] == RegisterLevel.TOHOKU


def test_register_mapping_not_using_enum_order():
    """Verify that the mapping doesn't accidentally use enum iteration order.

    This test would fail with the old buggy implementation that used:
    REGISTER_LABEL_TO_ID = {v: i for i, v in enumerate(RegisterLevel)}
    """
    # Get the enum members in definition order
    enum_members = list(RegisterLevel)

    # If we used enumerate(RegisterLevel), NEUTRAL would be at position 11
    # (since it's defined 12th in the enum)
    enum_order_position = enum_members.index(RegisterLevel.NEUTRAL)

    # But in our mapping, NEUTRAL should be ID 0, not 11
    actual_id = REGISTER_LABEL_TO_ID[RegisterLevel.NEUTRAL]

    assert actual_id == 0, (
        f"NEUTRAL should have ID 0, not enum position {enum_order_position}"
    )
    assert actual_id != enum_order_position, (
        "Mapping appears to be using enum iteration order (the bug!)"
    )


def test_all_register_labels_have_mapping():
    """Ensure every RegisterLevel enum member has a mapping."""
    for register in RegisterLevel:
        assert register in REGISTER_LABEL_TO_ID, (
            f"RegisterLevel.{register.name} is missing from REGISTER_LABEL_TO_ID"
        )

    assert len(REGISTER_LABEL_TO_ID) == len(list(RegisterLevel)), (
        "Number of mappings doesn't match number of enum members"
    )
