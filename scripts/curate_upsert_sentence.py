"""
Upsert functionality for manual corpus curation.
"""
# pylint: disable=too-many-locals

import os
import sqlite3
from collections import Counter
from typing import Any, Dict, Optional, Tuple, cast

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
    """Fetch current row for sentence (from VIEW)."""
    # The view 'corpus' provides the exact interface we need for reading.
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
        if not (op.startswith("+") or op.startswith("-") or op.startswith("!")):
            raise ValueError(
                f"Invalid grammar operation: '{op}'. Must start with +, -, or !"
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
        elif op.startswith("-"):
            # Add to negative, remove from grammar
            negative_set.add(gp_id)
            grammar_set.discard(gp_id)
        else:
            # Remove from both ('!' case)
            grammar_set.discard(gp_id)
            negative_set.discard(gp_id)

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


def normalize_register_ids(ids_str: str) -> str:
    """Normalize register IDs to a sorted CSV of distinct integers."""
    if not ids_str:
        return "0"

    parts = [int(x.strip()) for x in ids_str.split(",") if x.strip()]

    if not parts:
        return "0"

    # Enforce rules: if 0 is present, it must be the only one.
    unique_ids = sorted(list(set(parts)))
    if 0 in unique_ids and len(unique_ids) > 1:
        raise ValueError(
            f"Register 0 cannot be combined with other registers: {unique_ids}"
        )

    return ",".join(str(p) for p in unique_ids)


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
    """Execute the INSERT or UPDATE query (normalized tables)."""
    # pylint: disable=too-many-positional-arguments

    # Normalize register_ids
    clean_regs = normalize_register_ids(register_ids)

    # 1. Update/Insert main sentences table
    if is_update:
        # Check if exists
        cursor.execute("SELECT 1 FROM sentences WHERE sentence=?", (sentence,))
        if not cursor.fetchone():
            # Fallback to insert if not found? But is_update implies we thought it existed.
            # Or maybe it was partial? Let's assume strict update.
            pass

        query = """
            UPDATE sentences
            SET formality = ?, gender = ?, grammatic = ?, register_ids = ?
            WHERE sentence = ?
        """
        cursor.execute(query, (formality, gender, grammatic, clean_regs, sentence))

        # Clear existing relations to replace them
        cursor.execute("DELETE FROM corpus_gp_pos WHERE sentence=?", (sentence,))
        cursor.execute("DELETE FROM corpus_gp_neg WHERE sentence=?", (sentence,))

    else:
        query = """
            INSERT INTO sentences (sentence, formality, gender, grammatic, register_ids)
            VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(query, (sentence, formality, gender, grammatic, clean_regs))

    # 2. Insert new relations (if grammatical)
    # If ungrammatic, we ensure labels are empty (already done by triggers usually,
    # but application logic ensures we don't try to insert them).

    if grammatic == 1:
        if grammar:
            gp_list = [gp.strip() for gp in grammar.split(",") if gp.strip()]
            cursor.executemany(
                "INSERT INTO corpus_gp_pos(sentence, gp_id) VALUES (?, ?)",
                [(sentence, gp) for gp in gp_list],
            )

        if grammar_negative:
            gn_list = [gn.strip() for gn in grammar_negative.split(",") if gn.strip()]
            cursor.executemany(
                "INSERT INTO corpus_gp_neg(sentence, gp_id) VALUES (?, ?)",
                [(sentence, gn) for gn in gn_list],
            )

    return "Updated" if is_update else "Inserted"


_FETCH_SENTINEL = object()


def calculate_upsert_values(
    cursor: sqlite3.Cursor,
    sentence: str,
    formality_str: Optional[str],
    gender_str: Optional[str],
    grammar_diff_str: Optional[str],
    target_grammatic: Optional[int] = None,
    current_row_state: object = _FETCH_SENTINEL,
) -> Tuple[
    Optional[float],
    Optional[float],
    int,
    str,
    str,
    str,
    bool,
    Optional[float],
    Optional[float],
    str,
    str,
]:
    """Calculate all values needed for upsert."""
    # pylint: disable=too-many-positional-arguments, too-many-locals
    # 0. Validate grammar IDs if diff provided
    valid_grammar_ids = set()
    if grammar_diff_str:
        valid_grammar_ids = get_valid_grammar_ids(cursor)

    # 1. Fetch and Prepare State
    if current_row_state is not _FETCH_SENTINEL:
        row = cast(Optional[Tuple], current_row_state)
    else:
        row = get_current_row(cursor, sentence)
    curr_f, curr_g, curr_r, curr_gram, curr_nav = get_current_values(row)

    # 1a. Handle defaults for NEW sentences (if row is None)
    is_new = row is None
    if is_new:
        # Default behavior: If styles are provided, respect them.
        # If styles are NOT provided, default to NEUTRAL (0.0).
        # Implicitly Grammatic=1 unless overridden or impossible.
        if formality_str is None:
            # Only default if not explicit
            # Wait, do we default to 0.0 only?
            # User requirement: "--grammatic=1 --formality=neutral --gender=neutral should be implied if not specified"
            # So if not specified, pretend they passed "neutral".
            formality_str = "neutral"

        if gender_str is None:
            gender_str = "neutral"

        # We DO NOT force target_grammatic to 1.
        # By defaulting styles to Neutral (0.0), the natural calculation
        # (new_f is not None and new_g is not None) will result in Grammatic=1.
        # This allows --formality=unpragmatic (None) to correctly result in Grammatic=0.

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

    # Enforce constraints: Grammatic=0 requires Formality/Gender IS NULL and no grammar tags
    if new_is_gram == 0:
        new_f = None
        new_g = None
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
        curr_f,
        curr_g,
        curr_gram,
        curr_nav,
    )


def _upsert_sentence_logic(
    cursor: sqlite3.Cursor,
    sentence: str,
    formality_str: Optional[str],
    gender_str: Optional[str],
    grammar_diff_str: Optional[str],
    allow_insert: bool,
    target_grammatic: Optional[int] = None,
) -> Tuple[str, Optional[float], Optional[float], int, str, str, Dict[str, Tuple]]:
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
        old_f,
        old_g,
        old_gram,
        old_nav,
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

    # Calculate Diff
    diffs: Dict[str, Any] = {}
    if old_f != new_f:
        diffs["formality"] = (old_f, new_f)
    if old_g != new_g:
        diffs["gender"] = (old_g, new_g)
    # Grammatic change?
    # Old grammatic is implicitly 1 if old_f/old_g is set, else 0?
    # Actually row exists check handles 'New'
    old_is_gram = 1 if (old_f is not None and old_g is not None) else 0
    if not is_update:
        # New sentence: everything changed effectively, but diffs vs None is fine
        pass
    else:
        if old_is_gram != new_is_gram:
            diffs["grammatic"] = (old_is_gram, new_is_gram)

    if old_gram != new_gram_str:
        diffs["grammar"] = (old_gram, new_gram_str)
    if old_nav != new_nav_str:
        diffs["grammar_negative"] = (old_nav, new_nav_str)

    return action, new_f, new_g, new_is_gram, new_gram_str, new_nav_str, diffs


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
    # Important: Enable FK constraints
    conn.execute("PRAGMA foreign_keys=ON")
    c = conn.cursor()

    try:
        (action, new_f, new_g, new_is_gram, new_gram_str, new_nav_str, diffs) = (
            _upsert_sentence_logic(
                c,
                sentence,
                formality_str,
                gender_str,
                grammar_diff_str,
                allow_insert,
                target_grammatic=grammatic,
            )
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

        # Diff report (Single)
        if diffs:
            console.print("[dim]Changes:[/dim]")
            for k, (o, n) in diffs.items():
                console.print(f"  {k}: {o} -> {n}")

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
) -> Dict[str, int]:
    """Batch upsert sentences in a single transaction with pre-validation.
    Returns change_desc -> count for reporting."""
    # pylint: disable=too-many-positional-arguments, too-many-locals
    if not os.path.exists(db_path):
        console.print(f"[red]Database not found at {db_path}[/red]")
        return {}

    # Load canonical index for dedup gating on new inserts
    from scripts.canonical_index import CanonicalIndex

    canon_index = CanonicalIndex.corpus().load_or_build()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    c = conn.cursor()

    change_stats: Counter[str] = Counter()
    canon_skipped = 0

    try:
        # Pass 1: Calculation and Validation (Dry Run)

        upsert_plans = []  # List of (sentence, write_args, diffs, action_label)
        batch_state_map: Dict[
            str, Any
        ] = {}  # sentence -> row_tuple (mocked or fetched)

        for sentence in sentences:
            # 1. Determine current state (from local context or DB)
            current_row_state = batch_state_map.get(sentence, _FETCH_SENTINEL)

            # 2. Calculate new state
            (
                new_f,
                new_g,
                new_is_gram,
                new_gram_str,
                new_nav_str,
                curr_r,
                is_existing_row,  # Boolean indicating if row existed (in DB or mock)
                old_f,
                old_g,
                old_gram,
                old_nav,
            ) = calculate_upsert_values(
                c,
                sentence,
                formality_str,
                gender_str,
                grammar_diff_str,
                target_grammatic=grammatic,
                current_row_state=current_row_state,
            )

            # 3. Logic Validation
            is_update_context = is_existing_row

            if not is_update_context and not allow_insert:
                raise ValueError(
                    f"Sentence '{sentence}' not found. Use --allow-insert to create it."
                )

            # Canonical dedup gate: skip new inserts if a canonical equivalent
            # already exists in corpus.db.
            if not is_update_context and allow_insert:
                if canon_index.might_contain(sentence):
                    existing = canon_index.get_existing(sentence)
                    if existing:
                        canon_skipped += 1
                        continue

            write_args = (
                sentence,
                new_f,
                new_g,
                new_is_gram,
                new_gram_str,
                new_nav_str,
                curr_r,
                is_update_context,
            )

            # Calculate diffs for reporting
            diffs: Dict[str, Any] = {}
            if old_f != new_f:
                diffs["formality"] = (old_f, new_f)
            if old_g != new_g:
                diffs["gender"] = (old_g, new_g)

            old_is_gram = 1 if (old_f is not None and old_g is not None) else 0
            if is_update_context:
                if old_is_gram != new_is_gram:
                    diffs["grammatic"] = (old_is_gram, new_is_gram)

            if old_gram != new_gram_str:
                diffs["grammar"] = (old_gram, new_gram_str)
            if old_nav != new_nav_str:
                diffs["grammar_negative"] = (old_nav, new_nav_str)

            action_label = "Updated" if is_update_context else "Inserted"

            upsert_plans.append((sentence, write_args, diffs, action_label))

            # 5. Update local state map for next iteration
            mock_row = (new_f, new_g, new_is_gram, curr_r, new_gram_str, new_nav_str)
            batch_state_map[sentence] = mock_row

        # Pass 2: Execution

        updated_count = 0
        inserted_count = 0

        with conn:
            for sentence, args, diffs, action in upsert_plans:
                perform_db_write(c, *args)

                if action == "Updated":
                    updated_count += 1
                else:
                    inserted_count += 1
                    change_stats["Inserted"] += 1
                    canon_index.add(sentence)

                for k, (o, n) in diffs.items():
                    change_desc = f"{k}: {o} -> {n}"
                    change_stats[change_desc] += 1

        canon_index.save()
        canon_index.close()

        console.print("[green]Batch upsert complete.[/green]")
        console.print(f"  Inserted: {inserted_count}")
        console.print(f"  Updated:  {updated_count}")
        if canon_skipped:
            console.print(f"  Skipped (canonical duplicate): {canon_skipped}")

        if change_stats:
            console.print("\n[bold]Change Report:[/bold]")
            for desc, count in change_stats.most_common():
                console.print(f"  {count} sentences changed {desc}")

        return dict(change_stats)

    finally:
        conn.close()
