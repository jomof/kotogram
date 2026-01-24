"""
Upsert functionality for manual corpus curation.
"""
# pylint: disable=too-many-locals

import os
import sqlite3
from typing import Dict, Optional, Tuple, cast

from rich.console import Console

console = Console()

FORMALITY_MAP = {
    "very_casual": -1.0,
    "casual": -0.5,
    "neutral": 0.0,
    "formal": 0.5,
    "very_formal": 1.0,
    "unpragmatic": None,  # Special case: Maps to SQL NULL
}

GENDER_MAP = {
    "masculine": -1.0,
    "feminine": 1.0,
    "neutral": 0.0,
    "unpragmatic": None,  # Special case: Maps to SQL NULL
}


def get_current_row(cursor: sqlite3.Cursor, sentence: str) -> Optional[Tuple]:
    """Fetch current row for sentence."""
    cursor.execute(
        "SELECT formality, gender, grammatic, register_ids, grammar, grammar_negative FROM corpus WHERE sentence = ?",
        (sentence,),
    )
    return cast(Optional[Tuple], cursor.fetchone())


def get_valid_grammar_ids(cursor: sqlite3.Cursor) -> set[str]:
    """Fetch all valid grammar point IDs from the grammar table."""
    cursor.execute("SELECT id FROM grammar")
    return {row[0] for row in cursor.fetchall()}


def apply_grammar_diff(
    current_grammar_str: str,
    current_negative_str: str,
    diff_str: str,
    valid_ids: set[str],
) -> Tuple[str, str]:
    """
    Apply a diff string (e.g., '+gp1,-gp2') to current grammar sets.
    Ensures disjointness and validation.
    """
    # Parse current state
    grammar_set = set(filter(None, current_grammar_str.split(",")))
    negative_set = set(filter(None, current_negative_str.split(",")))

    # Parse diff
    ops = [op.strip() for op in diff_str.split(",") if op.strip()]

    # Check for conflicts
    seen_ops: Dict[str, str] = {}  # gp_id -> '+' or '-'
    for op in ops:
        if not (op.startswith("+") or op.startswith("-")):
            raise ValueError(
                f"Invalid grammar operation: '{op}'. Must start with + or -"
            )

        sign = op[0]
        gp_id = op[1:]

        if gp_id in seen_ops and seen_ops[gp_id] != sign:
            raise ValueError(
                f"Conflicting grammar operations for ID '{gp_id}': {seen_ops[gp_id]} and {sign}"
            )

        seen_ops[gp_id] = sign

    for op in ops:
        # Re-parsing safe because verified above
        gp_id = op[1:]
        if gp_id not in valid_ids:
            # We fail immediately for invalid IDs as requested
            raise ValueError(
                f"Invalid grammar point ID: '{gp_id}'. Not found in grammar table."
            )

        if op.startswith("+"):
            # Add to grammar, remove from negative
            grammar_set.add(gp_id)
            negative_set.discard(gp_id)
        else:
            # Add to negative, remove from grammar
            negative_set.add(gp_id)
            grammar_set.discard(gp_id)

    # Sort and join
    new_grammar = ",".join(sorted(grammar_set))
    new_negative = ",".join(sorted(negative_set))

    return new_grammar, new_negative


def resolve_value(
    new_label: Optional[str],
    current_val: Optional[float],
    mapping: Dict[str, Optional[float]],
    label_name: str,
) -> Optional[float]:
    """
    Resolve the new value for a field.
    Args:
        new_label: The string label provided by CLI (or None if not provided).
        current_val: The current value in the DB (or None).
        mapping: Dictionary mapping labels to values (including None).
        label_name: Name of the field for error reporting.
    Returns:
        The resolved float value or None.
    """
    if new_label:
        if new_label not in mapping:
            raise ValueError(
                f"Invalid {label_name}: {new_label}. Valid: {list(mapping.keys())}"
            )
        return mapping[new_label]

    # If not provided, keep current value (which might be None)
    return current_val


def get_current_values(row: Optional[Tuple]) -> Tuple:
    """Extract values from row with defaults."""
    if not row:
        return None, None, "", "", ""
    return (
        row[0],  # formality
        row[1],  # gender
        # row[2] is grammatic (unused directly)
        row[3] or "",  # register_ids
        row[4] or "" if len(row) > 4 else "",  # grammar
        row[5] or "" if len(row) > 5 else "",  # grammar_negative
    )


def perform_db_write(
    cursor: sqlite3.Cursor,
    sentence: str,
    formality: Optional[float],
    gender: Optional[float],
    grammatic: int,
    grammar: str,
    grammar_negative: str,
    register_ids: str,
    is_update: bool,
) -> str:
    """Execute the INSERT or UPDATE query."""
    # pylint: disable=too-many-positional-arguments
    if is_update:
        query = """
            UPDATE corpus
            SET formality = ?, gender = ?, grammatic = ?, grammar = ?, grammar_negative = ?
            WHERE sentence = ?
        """
        cursor.execute(
            query, (formality, gender, grammatic, grammar, grammar_negative, sentence)
        )
        return "Updated"

    query = """
        INSERT INTO corpus (sentence, formality, gender, grammatic, register_ids, grammar, grammar_negative)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    cursor.execute(
        query,
        (
            sentence,
            formality,
            gender,
            grammatic,
            register_ids,
            grammar,
            grammar_negative,
        ),
    )
    return "Inserted"


def calculate_upsert_values(
    cursor: sqlite3.Cursor,
    sentence: str,
    formality_str: Optional[str],
    gender_str: Optional[str],
    grammar_diff_str: Optional[str],
    target_grammatic: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float], int, str, str, str, bool]:
    """Calculate all values needed for upsert."""
    # pylint: disable=too-many-positional-arguments
    # 0. Validate grammar IDs if diff provided
    valid_grammar_ids = set()
    if grammar_diff_str:
        valid_grammar_ids = get_valid_grammar_ids(cursor)

    # 1. Fetch and Prepare State
    row = get_current_row(cursor, sentence)
    curr_f, curr_g, curr_r, curr_gram, curr_nav = get_current_values(row)

    # 2. Resolve new values
    new_f = resolve_value(formality_str, curr_f, FORMALITY_MAP, "formality")
    new_g = resolve_value(gender_str, curr_g, GENDER_MAP, "gender")

    # 3. Calculate Grammaticality and Styles
    new_is_gram = 0

    if target_grammatic is not None:
        new_is_gram = target_grammatic
        if new_is_gram == 1:
            # Enforce constraints: Grammatic=1 requires Formality/Gender IS NOT NULL
            if new_f is None:
                new_f = 0.0  # Default to Neutral
            if new_g is None:
                new_g = 0.0  # Default to Neutral
        # If target_grammatic == 0, we don't force styles to None, they can stay.
        # But we must clear grammar tags later.
    else:
        # Default logic
        new_is_gram = 1 if (new_f is not None and new_g is not None) else 0

    # 4. Resolve Grammar Logic
    new_gram_str = curr_gram
    new_nav_str = curr_nav

    if target_grammatic == 0:
        # If explicitly setting to agrammatic, must clear grammar tags to satisfy CHECK constraint
        new_gram_str = ""
        new_nav_str = ""
    elif grammar_diff_str:
        new_gram_str, new_nav_str = apply_grammar_diff(
            curr_gram, curr_nav, grammar_diff_str, valid_grammar_ids
        )

    return (
        new_f,
        new_g,
        new_is_gram,
        new_gram_str,
        new_nav_str,
        curr_r,
        bool(row),
    )


def _upsert_sentence_logic(
    cursor: sqlite3.Cursor,
    sentence: str,
    formality_str: Optional[str],
    gender_str: Optional[str],
    grammar_diff_str: Optional[str],
    allow_insert: bool,
    target_grammatic: Optional[int] = None,
) -> Tuple[str, Optional[float], Optional[float], int, str, str]:
    """Internal logic for upserting a single sentence."""
    # pylint: disable=too-many-positional-arguments
    (
        new_f,
        new_g,
        new_is_gram,
        new_gram_str,
        new_nav_str,
        curr_r,
        is_update,
    ) = calculate_upsert_values(
        cursor, sentence, formality_str, gender_str, grammar_diff_str, target_grammatic
    )

    if not is_update and not allow_insert:
        raise ValueError(
            f"Sentence '{sentence}' not found. Use --allow-insert to create it."
        )

    action = perform_db_write(
        cursor,
        sentence,
        new_f,
        new_g,
        new_is_gram,
        new_gram_str,
        new_nav_str,
        curr_r,
        is_update,
    )
    return action, new_f, new_g, new_is_gram, new_gram_str, new_nav_str


# pylint: disable=too-many-locals
def curate_upsert(
    sentence: str,
    formality_str: Optional[str],
    gender_str: Optional[str],
    grammar_diff_str: Optional[str] = None,
    db_path: str = "data/corpus.db",
    allow_insert: bool = False,
    grammatic: Optional[int] = None,
) -> None:
    """
    Upsert a sentence into the corpus database.

    Args:
        sentence: The Japanese sentence.
        formality_str: 'formal', 'casual', 'unpragmatic', etc.
        gender_str: 'masculine', 'feminine', etc.
        grammar_diff_str: comma-separated diffs e.g. '+gp01,-gp02'
        db_path: Path to SQLite DB.
        allow_insert: If True, allow inserting new sentences.
    """
    # pylint: disable=too-many-locals, too-many-positional-arguments
    if not os.path.exists(db_path):
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    try:
        (
            action,
            new_f,
            new_g,
            new_is_gram,
            new_gram_str,
            new_nav_str,
        ) = _upsert_sentence_logic(
            c,
            sentence,
            formality_str,
            gender_str,
            grammar_diff_str,
            allow_insert,
            target_grammatic=grammatic,
        )
        conn.commit()

        # Feedback
        console.print(f"[green]{action} sentence:[/green] {sentence}")
        console.print(f"  Formality: {new_f} ({formality_str or 'unchanged'})")
        console.print(f"  Gender:    {new_g} ({gender_str or 'unchanged'})")
        console.print(f"  Grammatic: {new_is_gram}")
        if grammar_diff_str:
            console.print(f"  Grammar +: {new_gram_str}")
            console.print(f"  Grammar -: {new_nav_str}")

    finally:
        conn.close()


def curate_upsert_batch(
    sentences: list[str],
    formality_str: Optional[str],
    gender_str: Optional[str],
    grammar_diff_str: Optional[str] = None,
    db_path: str = "data/corpus.db",
    allow_insert: bool = False,
    grammatic: Optional[int] = None,
) -> None:
    """Batch upsert sentences in a single transaction."""
    # pylint: disable=too-many-positional-arguments
    if not os.path.exists(db_path):
        console.print(f"[red]Database not found at {db_path}[/red]")
        return

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    try:
        updated_count = 0
        inserted_count = 0

        for sentence in sentences:
            (
                action,
                _,
                _,
                _,
                _,
                _,
            ) = _upsert_sentence_logic(
                c,
                sentence,
                formality_str,
                gender_str,
                grammar_diff_str,
                allow_insert,
                target_grammatic=grammatic,
            )
            if action == "Updated":
                updated_count += 1
            else:
                inserted_count += 1

        conn.commit()
        console.print("[green]Batch upsert complete.[/green]")
        console.print(f"  Inserted: {inserted_count}")
        console.print(f"  Updated:  {updated_count}")

    finally:
        conn.close()
