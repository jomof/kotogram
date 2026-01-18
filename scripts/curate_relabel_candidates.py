#!/usr/bin/env python3
# pylint: disable=too-many-lines
"""
Find likely mislabeled sentences in corpus.db based on model predictions.

This module identifies sentences where the model's predictions strongly disagree
with the current database labels for formality, gender, grammar_point, and register families.
"""

import heapq
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.progress import track

from kotogram.constants import (
    REGISTER_ID_TO_LABEL,
    REGISTER_LABEL_TO_ID,
    FormalityThresholds,
    GenderThresholds,
)
from scripts.progress_utils import create_progress
from scripts.rule_based_analysis import parse_gp_ids

console = Console()

# Type aliases for clarity
PredictionDict = Dict[str, Tuple[float, bool]]
ValueDict = Dict[str, Optional[float]]
GPLabelsDict = Dict[
    str, Tuple[List[int], List[int]]
]  # sentence -> (pos_gp_ids, neg_gp_ids)
GPProbsDict = Dict[str, List[float]]  # sentence -> grammar_point_probs
RegisterLabelsDict = Dict[str, List[int]]  # sentence -> register_ids
RegisterProbsDict = Dict[str, List[float]]  # sentence -> register_probs


# Thresholds for relabel candidate suggestions (script-specific, not used in production)
class _RelabelThresholds:
    """Thresholds for suggesting relabel candidates."""

    MSE_MIN_DISAGREEMENT = 0.3  # Minimum score difference to suggest relabel
    GP_MIN_PROB = 0.6  # Minimum probability to consider as high-confidence prediction
    REG_MIN_PROB = 0.7  # Minimum probability for register prediction


@dataclass
class RelabelCandidate:
    """A sentence that may be mislabeled."""

    family: str
    sentence: str
    current_value: str
    predicted_value: str
    confidence: float
    # For grammar_point, this holds the specific GP ID
    gp_id: Optional[str] = None
    gp_name: Optional[str] = None
    # For register, this holds the specific register ID/name
    register_id: Optional[int] = None
    register_name: Optional[str] = None

    def to_command_line(self) -> str:
        """Format as machine-readable command with human-readable comment."""
        # Escape sentence for shell
        escaped_sent = self.sentence.replace('"', '\\"')

        # Machine-readable part on left
        if self.family == "grammar_point":
            machine = (
                f'relabel {self.family} "{escaped_sent}" '
                f"gp={self.gp_id} current={self.current_value} "
                f"suggested={self.predicted_value}"
            )
            # Human-readable comment
            gp_label = self.gp_name or self.gp_id or ""
            comment = (
                f"# Currently {self.current_value}, suggesting {self.predicted_value} "
                f"for {gp_label} (confidence={self.confidence:.2f})"
            )
        elif self.family == "register":
            # Register format: suggested shows complete new label set
            reg_label = self.register_name or f"reg{self.register_id}"
            # Quote suggested if it contains commas
            suggested_quoted = (
                f'"{self.predicted_value}"'
                if "," in self.predicted_value
                else self.predicted_value
            )
            machine = (
                f'relabel {self.family} "{escaped_sent}" '
                f"current={self.current_value} suggested={suggested_quoted}"
            )
            comment = (
                f"# Currently {self.current_value}, suggesting {self.predicted_value} "
                f"(adding {reg_label}, confidence={self.confidence:.2f})"
            )
        else:
            # Human-readable labels for both current and suggested
            current_label = self._human_readable_current()
            proposed_label = self._human_readable_label()
            # Quote labels if they contain spaces
            current_quoted = (
                f'"{current_label}"' if " " in current_label else current_label
            )
            proposed_quoted = (
                f'"{proposed_label}"' if " " in proposed_label else proposed_label
            )
            machine = (
                f'relabel {self.family} "{escaped_sent}" '
                f"current={current_quoted} suggested={proposed_quoted}"
            )
            comment = (
                f"# Currently {current_label}, suggesting {proposed_label} "
                f"(confidence={self.confidence:.2f})"
            )

        return f"{machine}  {comment}"

    def _human_readable_label(self) -> str:
        """Convert predicted value to human-readable label."""
        # Handle non-numeric labels first
        if self.predicted_value in ("unpragmatic", "add", "negative", "positive"):
            return self.predicted_value
        pred = float(self.predicted_value)
        if self.family == "formality":
            # Use same thresholds as model classification
            return (
                "very formal"
                if pred >= FormalityThresholds.VERY_FORMAL_MIN
                else "formal"
                if pred >= FormalityThresholds.FORMAL_MIN
                else "neutral"
                if pred >= FormalityThresholds.NEUTRAL_MIN
                else "casual"
                if pred >= FormalityThresholds.CASUAL_MIN
                else "very casual"
            )
        if self.family == "gender":
            # Negative = masculine, Positive = feminine
            # Use same thresholds as model classification
            return (
                "masculine"
                if pred <= GenderThresholds.MASCULINE_MAX
                else "feminine"
                if pred >= GenderThresholds.FEMININE_MIN
                else "neutral"
            )
        return self.predicted_value

    def _human_readable_current(self) -> str:
        """Convert current_value to human-readable label."""
        if self.current_value == "unpragmatic":
            return "unpragmatic"
        val = float(self.current_value)
        if self.family == "formality":
            # Use same thresholds as model classification
            return (
                "very formal"
                if val >= FormalityThresholds.VERY_FORMAL_MIN
                else "formal"
                if val >= FormalityThresholds.FORMAL_MIN
                else "neutral"
                if val >= FormalityThresholds.NEUTRAL_MIN
                else "casual"
                if val >= FormalityThresholds.CASUAL_MIN
                else "very casual"
            )
        if self.family == "gender":
            # Negative = masculine, Positive = feminine
            # Use same thresholds as model classification
            return (
                "masculine"
                if val <= GenderThresholds.MASCULINE_MAX
                else "feminine"
                if val >= GenderThresholds.FEMININE_MIN
                else "neutral"
            )
        return self.current_value


def _shuffle_by_confidence_and_dedupe(
    candidates: List[RelabelCandidate],
) -> List[RelabelCandidate]:
    """Group by confidence, shuffle within groups, and remove duplicate sentences.

    Also ensures diversity by interleaving different current_value types.
    """
    from collections import defaultdict

    if not candidates:
        return []

    # Group by current_value to ensure diversity
    by_current_value: Dict[str, List[RelabelCandidate]] = defaultdict(list)
    for c in candidates:
        by_current_value[c.current_value].append(c)

    # Within each current_value group, sort by confidence and shuffle ties
    for group in by_current_value.values():
        # Group by rounded confidence
        conf_groups: Dict[float, List[RelabelCandidate]] = defaultdict(list)
        for c in group:
            conf_groups[round(c.confidence, 2)].append(c)
        # Shuffle within confidence groups
        for conf_group in conf_groups.values():
            random.shuffle(conf_group)
        # Flatten sorted by confidence descending
        group.clear()
        for conf in sorted(conf_groups.keys(), reverse=True):
            group.extend(conf_groups[conf])

    # Interleave from each current_value group for diversity
    result: List[RelabelCandidate] = []
    seen_sentences: Set[str] = set()
    groups = list(by_current_value.values())
    indices = [0] * len(groups)

    # Round-robin through groups until all are exhausted
    while True:
        made_progress = False
        for i, group in enumerate(groups):
            while indices[i] < len(group):
                candidate = group[indices[i]]
                indices[i] += 1
                if candidate.sentence not in seen_sentences:
                    seen_sentences.add(candidate.sentence)
                    result.append(candidate)
                    made_progress = True
                    break  # Move to next group
        if not made_progress:
            break

    return result


def _compute_mse_candidates(
    family_name: str,
    sentences: List[str],
    db_values: ValueDict,
    model_predictions: PredictionDict,
) -> List[RelabelCandidate]:
    """
    Find MSE family mislabeling candidates (formality or gender).

    Args:
        family_name: "formality" or "gender"
        sentences: List of sentences
        db_values: sentence -> value from DB (None if unpragmatic)
        model_predictions: sentence -> (predicted_value, is_pragmatic)

    Returns:
        List of RelabelCandidate sorted by confidence descending
    """
    candidates = []

    for sent in sentences:
        db_val = db_values.get(sent)
        pred = model_predictions.get(sent)
        if pred is None:
            continue

        pred_val, pred_is_pragmatic = pred

        # Skip if both agree on unpragmatic
        if db_val is None and not pred_is_pragmatic:
            continue

        # Case 1: DB says unpragmatic, model says pragmatic with strong value
        if db_val is None and pred_is_pragmatic:
            confidence = abs(pred_val)
            candidates.append(
                RelabelCandidate(
                    family=family_name,
                    sentence=sent,
                    current_value="unpragmatic",
                    predicted_value=f"{pred_val:.2f}",
                    confidence=confidence,
                )
            )
            continue

        # Case 2: DB has value, model says unpragmatic
        if db_val is not None and not pred_is_pragmatic:
            confidence = abs(db_val)  # Strong DB value but model says unpragmatic
            candidates.append(
                RelabelCandidate(
                    family=family_name,
                    sentence=sent,
                    current_value=f"{db_val:.2f}",
                    predicted_value="unpragmatic",
                    confidence=confidence,
                )
            )
            continue

        # Case 3: Both pragmatic but values disagree
        if db_val is not None and pred_is_pragmatic:
            diff = abs(pred_val - db_val)
            if diff > _RelabelThresholds.MSE_MIN_DISAGREEMENT:
                confidence = diff
                candidates.append(
                    RelabelCandidate(
                        family=family_name,
                        sentence=sent,
                        current_value=f"{db_val:.2f}",
                        predicted_value=f"{pred_val:.2f}",
                        confidence=confidence,
                    )
                )

    return _shuffle_by_confidence_and_dedupe(candidates)


def _parse_register_ids(reg_str: str) -> List[int]:
    """Parse register IDs from comma-separated string (e.g., '0,5,7')."""
    if not reg_str:
        return []
    result = []
    for part in reg_str.split(","):
        part = part.strip()
        if part.isdigit():
            result.append(int(part))
    return result


def _load_sentences_from_db(
    db_path: str,
) -> Tuple[List[str], ValueDict, ValueDict, GPLabelsDict, RegisterLabelsDict]:
    """Load grammatic sentences from corpus.db."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "SELECT sentence, formality, gender, grammar, grammar_negative, register_ids "
        "FROM corpus WHERE grammatic = 1"
    )
    rows = c.fetchall()
    conn.close()

    sentences = []
    db_formality: ValueDict = {}
    db_gender: ValueDict = {}
    db_grammar: GPLabelsDict = {}
    db_register: RegisterLabelsDict = {}

    for sent, f_val, g_val, gp_pos_str, gp_neg_str, reg_str in rows:
        sentences.append(sent)
        db_formality[sent] = f_val
        db_gender[sent] = g_val
        db_grammar[sent] = (
            parse_gp_ids(gp_pos_str or ""),
            parse_gp_ids(gp_neg_str or ""),
        )
        db_register[sent] = _parse_register_ids(reg_str or "")

    return sentences, db_formality, db_gender, db_grammar, db_register


def _load_gp_names(db_path: str) -> Dict[str, str]:
    """Load grammar point names from corpus.db grammar table."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, name FROM grammar")
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def _init_inference_cache(cache_path: str) -> None:
    """Initialize inference cache database."""
    conn = sqlite3.connect(cache_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS inference (
            sentence TEXT PRIMARY KEY,
            formality_score REAL NOT NULL,
            formality_pragmatic INTEGER NOT NULL,
            gender_score REAL NOT NULL,
            gender_pragmatic INTEGER NOT NULL,
            gp_probs TEXT,
            register_probs TEXT
        ) WITHOUT ROWID
    """)
    # Migrate old table if needed (add missing columns)
    c.execute("PRAGMA table_info(inference)")
    columns = {row[1] for row in c.fetchall()}
    if "gp_probs" not in columns:
        c.execute("ALTER TABLE inference ADD COLUMN gp_probs TEXT")
    if "register_probs" not in columns:
        c.execute("ALTER TABLE inference ADD COLUMN register_probs TEXT")
    conn.commit()
    conn.close()


def _load_cached_inference(
    cache_path: str, sentences: List[str], load_gp_probs: bool = True
) -> Tuple[PredictionDict, PredictionDict, GPProbsDict, RegisterProbsDict, List[str]]:
    """Load cached inference results.

    Returns:
        (formality, gender, gp_probs, register_probs, uncached)

    Args:
        load_gp_probs: If False, skip loading/parsing gp_probs (faster if not needed)
    """
    import json

    if not os.path.exists(cache_path):
        return {}, {}, {}, {}, sentences

    conn = sqlite3.connect(cache_path)
    c = conn.cursor()

    # Create index if it doesn't exist (speeds up lookups)
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_inference_sentence ON inference(sentence)"
    )
    conn.commit()

    model_formality: PredictionDict = {}
    model_gender: PredictionDict = {}
    model_gp_probs: GPProbsDict = {}
    model_register_probs: RegisterProbsDict = {}

    sent_set = set(sentences)
    console.print("  Loading cached results...")

    # Use fetchmany for better performance - always load register_probs
    cols = (
        "sentence, formality_score, formality_pragmatic, "
        "gender_score, gender_pragmatic, gp_probs, register_probs"
    )
    c.execute(f"SELECT {cols} FROM inference")

    while True:
        rows = c.fetchmany(100000)
        if not rows:
            break

        for row in rows:
            if row[0] not in sent_set:
                continue

            model_formality[row[0]] = (row[1], bool(row[2]))
            model_gender[row[0]] = (row[3], bool(row[4]))
            if load_gp_probs and row[5]:
                model_gp_probs[row[0]] = json.loads(row[5])
            if row[6]:
                model_register_probs[row[0]] = json.loads(row[6])

    conn.close()

    # Find uncached sentences
    if load_gp_probs:
        uncached = [sent for sent in sentences if sent not in model_gp_probs]
    else:
        uncached = [sent for sent in sentences if sent not in model_formality]

    return model_formality, model_gender, model_gp_probs, model_register_probs, uncached


def _save_inference_cache(  # pylint: disable=too-many-positional-arguments,too-many-locals
    cache_path: str,
    sentences: List[str],
    model_formality: PredictionDict,
    model_gender: PredictionDict,
    model_gp_probs: GPProbsDict,
    model_register_probs: RegisterProbsDict,
) -> None:
    """Save inference results to cache."""
    import json

    _init_inference_cache(cache_path)

    conn = sqlite3.connect(cache_path)
    c = conn.cursor()

    for sent in sentences:
        f_score, f_prag = model_formality[sent]
        g_score, g_prag = model_gender[sent]
        gp_probs_json = (
            json.dumps(model_gp_probs.get(sent)) if sent in model_gp_probs else None
        )
        reg_probs_json = (
            json.dumps(model_register_probs.get(sent))
            if sent in model_register_probs
            else None
        )
        c.execute(
            "INSERT OR REPLACE INTO inference VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                sent,
                f_score,
                int(f_prag),
                g_score,
                int(g_prag),
                gp_probs_json,
                reg_probs_json,
            ),
        )

    conn.commit()
    conn.close()


def _tokenize_sentences(sentences: List[str]) -> List[str]:
    """Tokenize sentences to kotograms."""
    from kotogram.japanese_parser import KotogramFormat
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

    parser = SudachiJapaneseParser()
    return [
        parser.japanese_to_kotogram(sent, KotogramFormat.TRAINING_MASK)
        for sent in track(sentences, description="Tokenizing...")
    ]


def _run_batch_inference(  # pylint: disable=too-many-locals
    sentences: List[str],
    kotograms: List[str],
) -> Tuple[PredictionDict, PredictionDict, GPProbsDict, RegisterProbsDict]:
    """Run model inference in batches."""
    from kotogram.analysis import grammars

    batch_size = 256
    model_formality: PredictionDict = {}
    model_gender: PredictionDict = {}
    model_gp_probs: GPProbsDict = {}
    model_register_probs: RegisterProbsDict = {}

    for batch_idx in track(
        range((len(kotograms) + batch_size - 1) // batch_size),
        description="Inference...",
    ):
        batch_koto = kotograms[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_sent = sentences[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        for sent, res in zip(batch_sent, grammars(batch_koto)):
            model_formality[sent] = (res.formality_score, res.formality_is_pragmatic)
            model_gender[sent] = (res.gender_score, res.gender_is_pragmatic)
            if res.grammar_point_probs is not None:
                model_gp_probs[sent] = res.grammar_point_probs
            # Use raw register probabilities from model
            if res.register_probs is not None:
                model_register_probs[sent] = res.register_probs

    return model_formality, model_gender, model_gp_probs, model_register_probs


def _run_inference_with_cache(  # pylint: disable=too-many-locals
    sentences: List[str], cache_path: str
) -> Tuple[PredictionDict, PredictionDict, GPProbsDict, RegisterProbsDict]:
    """Load from cache or run inference for uncached sentences.

    Note: For grammar_point and register, we don't load probs into memory upfront -
    they are streamed on-demand when processing candidates.
    """
    console.print(f"[bold blue]Checking inference cache ({cache_path})...[/bold blue]")

    # Never load gp_probs upfront - we stream them for grammar_point/register
    (
        model_formality,
        model_gender,
        model_gp_probs,
        model_register_probs,
        uncached,
    ) = _load_cached_inference(cache_path, sentences, load_gp_probs=False)
    cached_count = len(sentences) - len(uncached)
    console.print(f"  Cached: {cached_count:,}, Need inference: {len(uncached):,}")

    if uncached:
        console.print("[bold blue]Tokenizing uncached sentences...[/bold blue]")
        kotograms = _tokenize_sentences(uncached)

        console.print("[bold blue]Running model inference...[/bold blue]")
        new_formality, new_gender, new_gp_probs, new_register_probs = (
            _run_batch_inference(uncached, kotograms)
        )

        model_formality.update(new_formality)
        model_gender.update(new_gender)
        model_gp_probs.update(new_gp_probs)
        model_register_probs.update(new_register_probs)

        console.print("[bold blue]Saving to inference cache...[/bold blue]")
        _save_inference_cache(
            cache_path,
            uncached,
            new_formality,
            new_gender,
            new_gp_probs,
            new_register_probs,
        )

    return model_formality, model_gender, model_gp_probs, model_register_probs


def _compute_and_collect_candidates(  # pylint: disable=too-many-locals
    *,
    families: List[str],
    sentences: List[str],
    db_formality: ValueDict,
    db_gender: ValueDict,
    db_grammar: GPLabelsDict,
    db_register: RegisterLabelsDict,
    model_formality: PredictionDict,
    model_gender: PredictionDict,
    cache_path: str,
    gp_names: Dict[str, str],
    top_n: int,
    max_sentence_len: int = 30,
    seen_sentences: Optional[Set[str]] = None,
) -> List[RelabelCandidate]:
    """Compute and collect candidates for each family."""
    if seen_sentences is None:
        seen_sentences = set()

    results: List[RelabelCandidate] = []
    short_sents = [s for s in sentences if len(s) <= max_sentence_len]

    def add_top_candidates(candidates: List[RelabelCandidate]) -> None:
        # Filter out sentences we've already seen
        filtered = [c for c in candidates if c.sentence not in seen_sentences]
        console.print(
            f"  Found {len(candidates)} total, "
            f"{len(filtered)} new, showing top {min(top_n, len(filtered))}"
        )
        results.extend(filtered[:top_n])

    if "formality" in families:
        console.print("[bold blue]Computing formality candidates...[/bold blue]")
        add_top_candidates(
            _compute_mse_candidates(
                "formality", short_sents, db_formality, model_formality
            ),
        )

    if "gender" in families:
        console.print("[bold blue]Computing gender candidates...[/bold blue]")
        add_top_candidates(
            _compute_mse_candidates("gender", short_sents, db_gender, model_gender),
        )

    if "grammar_point" in families:
        console.print("[bold blue]Computing grammar_point candidates...[/bold blue]")
        add_top_candidates(
            _compute_grammar_point_candidates(
                sentences,
                db_grammar,
                cache_path,
                gp_names,
                max_sentence_len,
                target_count=top_n,
                seen_sentences=seen_sentences,
            ),
        )

    if "register" in families:
        console.print("[bold blue]Computing register candidates...[/bold blue]")
        add_top_candidates(
            _compute_register_candidates(
                sentences,
                db_register,
                cache_path,
                max_sentence_len,
                target_count=top_n,
                seen_sentences=seen_sentences,
            ),
        )

    return results


class _GPCandidateHeap:
    """Candidate heap for grammar points with diversity enforcement.

    Ensures no single GP dominates by limiting candidates per GP.
    Uses per-GP heaps to track the best candidates for each GP.
    """

    # Max candidates per GP to ensure diversity
    MAX_PER_GP = 3

    def __init__(self, target_count: int):
        # Per-GP heaps: gp_id -> [(prob, tiebreaker, candidate), ...]
        self.per_gp_heaps: Dict[int, List[Tuple[float, int, RelabelCandidate]]] = {}
        self.seen_pairs: Set[Tuple[str, int]] = set()
        self.unique_sentences: Set[str] = set()
        self.tiebreaker = 0
        self.target_count = target_count

    def add_candidate(self, candidate: RelabelCandidate, gp_id_int: int) -> None:
        """Add candidate, enforcing per-GP limits."""
        pair = (candidate.sentence, gp_id_int)
        if pair in self.seen_pairs:
            return

        self.seen_pairs.add(pair)
        self.unique_sentences.add(candidate.sentence)
        prob = candidate.confidence
        self.tiebreaker += 1

        # Get or create heap for this GP
        if gp_id_int not in self.per_gp_heaps:
            self.per_gp_heaps[gp_id_int] = []

        gp_heap = self.per_gp_heaps[gp_id_int]

        # Add to per-GP heap (limited to MAX_PER_GP)
        if len(gp_heap) < self.MAX_PER_GP:
            heapq.heappush(gp_heap, (prob, self.tiebreaker, candidate))
        elif prob > gp_heap[0][0]:
            heapq.heapreplace(gp_heap, (prob, self.tiebreaker, candidate))

    def has_enough_sentences(self) -> bool:
        """Check if we have enough unique sentences."""
        return len(self.unique_sentences) >= self.target_count

    def get_candidates_sorted(self) -> List[RelabelCandidate]:
        """Return candidates sorted by confidence descending, distributed across GPs."""
        # Collect all candidates from per-GP heaps
        all_candidates: List[Tuple[float, int, RelabelCandidate]] = []
        for gp_heap in self.per_gp_heaps.values():
            all_candidates.extend(gp_heap)

        # Sort by confidence descending
        all_candidates.sort(key=lambda x: -x[0])
        return [c for _, _, c in all_candidates]


def _compute_grammar_point_candidates(  # pylint: disable=too-many-positional-arguments,too-many-locals
    sentences: List[str],
    db_grammar: GPLabelsDict,
    cache_path: str,
    gp_names: Dict[str, str],
    max_sentence_len: int = 30,
    target_count: int = 20,
    seen_sentences: Optional[Set[str]] = None,
) -> List[RelabelCandidate]:
    """Find grammar point candidates for negative labeling.

    Finds the highest-confidence predictions that aren't already positively
    labeled. Suggests all as negatives - user will manually filter out any
    that are actually valid positives.

    Args:
        cache_path: Path to inference cache database
        max_sentence_len: Exclude sentences longer than this (character count)
        target_count: Target number of candidates to find (stops early once satisfied)
        seen_sentences: Sentences to exclude (already processed in previous runs)

    Streams from the cache on-demand for efficiency - stops as soon as we find
    enough unique sentences.
    """
    import json

    import numpy as np

    if seen_sentences is None:
        seen_sentences = set()

    # Filter by length and seen upfront
    eligible_sents = [
        s for s in sentences if len(s) <= max_sentence_len and s not in seen_sentences
    ]
    if not eligible_sents:
        return []

    random.shuffle(eligible_sents)  # Shuffle in place for randomization

    min_prob = _RelabelThresholds.GP_MIN_PROB
    heap = _GPCandidateHeap(target_count)

    console.print(f"  Sampling from {len(eligible_sents):,} eligible sentences...")

    # Stream from cache on-demand
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    processed = 0

    try:
        with create_progress(console) as progress:
            task = progress.add_task("Sampling...", total=target_count)

            for sent in eligible_sents:
                # Query cache for this sentence
                cursor.execute(
                    "SELECT gp_probs FROM inference WHERE sentence = ?", (sent,)
                )
                row = cursor.fetchone()
                if not row or not row[0]:
                    continue  # No cache entry

                processed += 1
                gp_probs = json.loads(row[0])

                pos_gp_ids, _ = db_grammar.get(sent, ([], []))
                positive_ids = set(pos_gp_ids)

                probs_array = np.array(gp_probs, dtype=np.float32)

                # Find high confidence predictions not already positively labeled
                high_prob_indices = np.where(probs_array > min_prob)[0]
                for gp_id in high_prob_indices:
                    gp_id_int = int(gp_id)
                    if gp_id_int in positive_ids:
                        continue
                    prob = float(probs_array[gp_id])
                    gp_id_str = f"gp{gp_id_int:04d}"
                    candidate = RelabelCandidate(
                        family="grammar_point",
                        sentence=sent,
                        current_value="unlabeled",
                        predicted_value="negative",
                        confidence=prob,
                        gp_id=gp_id_str,
                        gp_name=gp_names.get(gp_id_str),
                    )
                    heap.add_candidate(candidate, gp_id_int)

                progress.update(task, completed=len(heap.unique_sentences))

                # Stop early once we have enough unique sentences
                if heap.has_enough_sentences():
                    console.print(
                        f"  [dim]Early stop: found {len(heap.unique_sentences)} unique "
                        f"sentences after {processed:,} cache hits[/dim]"
                    )
                    break
    finally:
        conn.close()

    candidates = heap.get_candidates_sorted()
    console.print(
        f"  Processed {processed:,} sentences, found {len(candidates)} candidates"
    )

    return candidates


def _compute_register_candidates(  # pylint: disable=too-many-positional-arguments,too-many-locals
    sentences: List[str],
    db_register: RegisterLabelsDict,
    cache_path: str,
    max_sentence_len: int = 30,
    target_count: int = 20,
    seen_sentences: Optional[Set[str]] = None,
) -> List[RelabelCandidate]:
    """Find register candidates for labeling.

    Finds the highest-confidence register predictions that aren't already
    labeled in the database. Suggests adding the register label.

    Args:
        sentences: All sentences to consider
        db_register: Current register labels from DB (sentence -> list of register IDs)
        cache_path: Path to inference cache database
        max_sentence_len: Exclude sentences longer than this (character count)
        target_count: Target number of candidates to find (stops early once satisfied)
        seen_sentences: Sentences to exclude (already processed in previous runs)

    Streams from the cache on-demand for efficiency - stops as soon as we find
    enough unique sentences.
    """
    import json

    import numpy as np

    if seen_sentences is None:
        seen_sentences = set()

    # Filter by length and seen upfront
    eligible_sents = [
        s for s in sentences if len(s) <= max_sentence_len and s not in seen_sentences
    ]
    if not eligible_sents:
        return []

    random.shuffle(eligible_sents)  # Shuffle in place for randomization

    min_prob = _RelabelThresholds.REG_MIN_PROB
    heap = _GPCandidateHeap(target_count)  # Reuse the heap class

    console.print(f"  Sampling from {len(eligible_sents):,} eligible sentences...")

    # Stream from cache on-demand
    conn = sqlite3.connect(cache_path)
    cursor = conn.cursor()
    processed = 0

    try:
        with create_progress(console) as progress:
            task = progress.add_task("Sampling...", total=target_count)

            for sent in eligible_sents:
                # Query cache for this sentence
                cursor.execute(
                    "SELECT register_probs FROM inference WHERE sentence = ?", (sent,)
                )
                row = cursor.fetchone()
                if not row or not row[0]:
                    continue  # No cache entry

                processed += 1
                reg_probs = json.loads(row[0])

                current_reg_ids = set(db_register.get(sent, []))

                probs_array = np.array(reg_probs, dtype=np.float32)

                # Find high confidence predictions not already labeled
                high_prob_indices = np.where(probs_array > min_prob)[0]
                for reg_id in high_prob_indices:
                    reg_id_int = int(reg_id)
                    if reg_id_int in current_reg_ids:
                        continue
                    prob = float(probs_array[reg_id])
                    # Get register label name from ID
                    reg_label = REGISTER_ID_TO_LABEL.get(reg_id_int)
                    if reg_label is None:
                        continue
                    reg_name = (
                        reg_label.value
                        if hasattr(reg_label, "value")
                        else str(reg_label)
                    )
                    # Build human-readable current labels
                    current_labels = []
                    for rid in sorted(current_reg_ids):
                        rlabel = REGISTER_ID_TO_LABEL.get(rid)
                        if rlabel:
                            current_labels.append(
                                rlabel.value
                                if hasattr(rlabel, "value")
                                else str(rlabel)
                            )
                    current_str = ",".join(current_labels) if current_labels else "none"

                    # Build complete new label set (current + new label)
                    new_reg_ids = sorted(current_reg_ids | {reg_id_int})
                    new_labels = []
                    for rid in new_reg_ids:
                        rlabel = REGISTER_ID_TO_LABEL.get(rid)
                        if rlabel:
                            new_labels.append(
                                rlabel.value
                                if hasattr(rlabel, "value")
                                else str(rlabel)
                            )
                    suggested_str = ",".join(new_labels)

                    candidate = RelabelCandidate(
                        family="register",
                        sentence=sent,
                        current_value=current_str,
                        predicted_value=suggested_str,
                        confidence=prob,
                        register_id=reg_id_int,
                        register_name=reg_name,
                    )
                    heap.add_candidate(candidate, reg_id_int)

                progress.update(task, completed=len(heap.unique_sentences))

                # Stop early once we have enough unique sentences
                if heap.has_enough_sentences():
                    console.print(
                        f"  [dim]Early stop: found {len(heap.unique_sentences)} unique "
                        f"sentences after {processed:,} cache hits[/dim]"
                    )
                    break
    finally:
        conn.close()

    candidates = heap.get_candidates_sorted()

    # Keep only top candidate per sentence
    seen_sents: Set[str] = set()
    unique: List[RelabelCandidate] = []
    for c in candidates:
        if c.sentence not in seen_sents:
            seen_sents.add(c.sentence)
            unique.append(c)

    console.print(
        f"  Processed {processed:,} sentences, found {len(unique)} candidates"
    )

    return unique


def _load_seen_sentences(seen_file: str) -> Set[str]:
    """Load previously seen sentences from file."""
    seen_sentences: Set[str] = set()
    if os.path.exists(seen_file):
        with open(seen_file, encoding="utf-8") as f:
            for line in f:
                sentence = line.strip()
                if sentence:
                    seen_sentences.add(sentence)
        console.print(f"Loaded {len(seen_sentences):,} previously seen sentences.")
    return seen_sentences


def _write_candidates_file(
    candidates_file: str, all_candidates: List[RelabelCandidate]
) -> None:
    """Write candidates to file with header comments."""
    with open(candidates_file, "w", encoding="utf-8") as f:
        # Write header comments explaining the scales (data-driven from constants)
        f.write(
            f"# Formality scale: "
            f"very casual (-1.0 to {FormalityThresholds.CASUAL_MIN}) | "
            f"casual ({FormalityThresholds.CASUAL_MIN} to {FormalityThresholds.NEUTRAL_MIN}) | "
            f"neutral ({FormalityThresholds.NEUTRAL_MIN} to {FormalityThresholds.FORMAL_MIN}) | "
            f"formal ({FormalityThresholds.FORMAL_MIN} to {FormalityThresholds.VERY_FORMAL_MIN}) | "
            f"very formal ({FormalityThresholds.VERY_FORMAL_MIN} to 1.0)\n"
        )
        f.write(
            f"# Gender scale: "
            f"masculine (-1.0 to {GenderThresholds.MASCULINE_MAX}) | "
            f"neutral ({GenderThresholds.MASCULINE_MAX} to {GenderThresholds.FEMININE_MIN}) | "
            f"feminine ({GenderThresholds.FEMININE_MIN} to 1.0)\n"
        )
        for candidate in all_candidates:
            f.write(candidate.to_command_line() + "\n")


def _append_to_seen_file(
    seen_file: str, all_candidates: List[RelabelCandidate]
) -> None:
    """Append candidate sentences to seen file.

    Also validates that no proposed sentence was already in the seen file,
    which would indicate a bug in the filtering logic.
    """
    if not all_candidates:
        return

    # Validation: check that we're not proposing sentences that are already seen
    if os.path.exists(seen_file):
        with open(seen_file, encoding="utf-8") as f:
            existing_seen = {line.strip() for line in f if line.strip()}
        proposed_sentences = {c.sentence for c in all_candidates}
        already_seen = proposed_sentences & existing_seen
        if already_seen:
            examples = list(already_seen)[:3]
            raise RuntimeError(
                f"BUG: {len(already_seen)} proposed sentences were already in "
                f"relabel-candidates-seen.txt. Examples: {examples}"
            )

    with open(seen_file, "a", encoding="utf-8") as f:
        for candidate in all_candidates:
            f.write(candidate.sentence + "\n")
    console.print(
        f"[dim]Appended {len(all_candidates)} sentences to "
        f"relabel-candidates-seen.txt[/dim]"
    )


def find_relabel_candidates(  # pylint: disable=too-many-locals
    db_path: str,
    top_n: int = 20,
    families: Optional[List[str]] = None,
) -> None:
    """
    Find and print likely mislabeled sentences.

    Args:
        db_path: Path to corpus.db
        top_n: Number of candidates to show per family
        families: List of families to check (default: all)
    """
    if families is None:
        families = ["formality", "gender", "grammar_point", "register"]

    if not os.path.exists(db_path):
        console.print(f"[red]Database not found: {db_path}[/red]")
        return

    data_dir = os.path.dirname(db_path)
    seen_file = os.path.join(data_dir, "relabel-candidates-seen.txt")
    candidates_file = os.path.join(data_dir, "relabel-candidates.txt")

    # Load previously seen sentences
    seen_sentences = _load_seen_sentences(seen_file)

    # Load sentences from DB
    console.print(f"[bold blue]Loading sentences from {db_path}...[/bold blue]")
    sentences, db_formality, db_gender, db_grammar, db_register = (
        _load_sentences_from_db(db_path)
    )

    if not sentences:
        console.print("[yellow]No grammatic sentences found in database.[/yellow]")
        return

    console.print(f"Loaded {len(sentences):,} grammatic sentences.")

    # Load grammar point names and run inference
    gp_names = _load_gp_names(db_path)
    cache_path = os.path.join(data_dir, "corpus-inference.db")
    model_formality, model_gender, _, _ = _run_inference_with_cache(
        sentences, cache_path
    )

    # Compute candidates
    all_candidates = _compute_and_collect_candidates(
        families=families,
        sentences=sentences,
        db_formality=db_formality,
        db_gender=db_gender,
        db_grammar=db_grammar,
        db_register=db_register,
        model_formality=model_formality,
        model_gender=model_gender,
        cache_path=cache_path,
        gp_names=gp_names,
        top_n=top_n,
        seen_sentences=seen_sentences,
    )

    # Output candidates
    console.print()
    console.print("[bold green]Relabel Candidates:[/bold green]")
    console.print()

    for candidate in all_candidates:
        print(candidate.to_command_line())

    # Write files
    _write_candidates_file(candidates_file, all_candidates)
    _append_to_seen_file(seen_file, all_candidates)

    console.print()
    console.print(f"[bold green]Written to {candidates_file}[/bold green]")


def apply_relabel_candidates(db_path: str, candidates_file: str) -> None:
    """Apply relabel suggestions from file to corpus.db.

    Args:
        db_path: Path to corpus.db
        candidates_file: Path to relabel-candidates.txt file

    The function:
    1. Validates all prerequisites before making changes
    2. Creates a relabels tracking table for audit trail
    3. Applies all changes atomically in a transaction
    4. Verifies changes were applied correctly
    """
    from datetime import datetime

    if not os.path.exists(db_path):
        console.print(f"[red]Database not found: {db_path}[/red]")
        return

    if not os.path.exists(candidates_file):
        console.print(f"[red]Candidates file not found: {candidates_file}[/red]")
        return

    # Parse all relabel commands
    commands = _parse_relabel_commands(candidates_file)
    if not commands:
        console.print("[yellow]No relabel commands found in file[/yellow]")
        return

    console.print(f"[bold blue]Parsed {len(commands)} relabel commands[/bold blue]")

    # Connect to database
    conn = sqlite3.connect(db_path)
    try:
        # Step 1: Validate all prerequisites
        console.print("[bold blue]Validating prerequisites...[/bold blue]")
        _validate_relabel_prerequisites(conn, commands)
        console.print("[green]✓ All prerequisites validated[/green]")

        # Step 2: Create relabels tracking table
        _create_relabels_table(conn)

        # Step 3: Get next batch ID
        batch_id = _get_next_batch_id(conn)
        console.print(f"[bold blue]Applying batch {batch_id}...[/bold blue]")

        # Step 4: Apply changes atomically
        timestamp = datetime.now().isoformat()
        _apply_relabel_commands(conn, commands, batch_id, timestamp)

        # Step 5: Verify changes
        console.print("[bold blue]Verifying changes...[/bold blue]")
        _verify_relabel_changes(conn, commands)
        console.print("[green]✓ All changes verified[/green]")

        console.print(
            f"[bold green]Successfully applied {len(commands)} relabels "
            f"(batch {batch_id})[/bold green]"
        )
    finally:
        conn.close()


def _parse_relabel_commands(file_path: str) -> List[Dict[str, Any]]:
    """Parse relabel commands from file."""
    import re

    commands: List[Dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Remove inline comments
            if "#" in line:
                line = line.split("#")[0].strip()

            # Parse: relabel FAMILY "SENTENCE" ...
            match = re.match(r'relabel\s+(\w+)\s+"([^"]*)"(.+)', line)
            if not match:
                console.print(f"[yellow]Warning: Could not parse line: {line}[/yellow]")
                continue

            family = match.group(1)
            sentence = match.group(2)
            params_str = match.group(3)

            # Parse parameters
            params: Dict[str, str] = {}
            for param_match in re.finditer(r'(\w+)=("[^"]*"|[^\s]+)', params_str):
                key = param_match.group(1)
                value = param_match.group(2).strip('"')
                params[key] = value

            commands.append(
                {
                    "family": family,
                    "sentence": sentence,
                    "params": params,
                    "original_line": line,
                }
            )

    return commands


def _validate_relabel_prerequisites(  # pylint: disable=too-many-locals
    conn: sqlite3.Connection, commands: List[Dict[str, Any]]
) -> None:
    """Validate that all sentences exist and have the expected current values."""
    cursor = conn.cursor()

    for cmd in track(commands, description="Validating..."):
        family = cmd["family"]
        sentence = cmd["sentence"]
        params = cmd["params"]

        # Check sentence exists
        cursor.execute("SELECT sentence FROM corpus WHERE sentence = ?", (sentence,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Sentence not found in database: {sentence}")

        # Validate current value matches
        if family == "formality":
            cursor.execute(
                "SELECT formality FROM corpus WHERE sentence = ?", (sentence,)
            )
            row = cursor.fetchone()
            current_db = row[0] if row and row[0] is not None else None
            _validate_label_match(sentence, params["current"], current_db, "formality")
        elif family == "gender":
            cursor.execute("SELECT gender FROM corpus WHERE sentence = ?", (sentence,))
            row = cursor.fetchone()
            current_db = row[0] if row and row[0] is not None else None
            _validate_label_match(sentence, params["current"], current_db, "gender")
        elif family == "grammar_point":
            gp_id = params["gp"]
            gp_num = int(gp_id[2:])  # Remove "gp" prefix
            cursor.execute(
                "SELECT grammar FROM corpus WHERE sentence = ?",
                (sentence,),
            )
            row = cursor.fetchone()
            gp_positive = parse_gp_ids(row[0]) if row and row[0] else []

            cursor.execute(
                "SELECT grammar_negative FROM corpus WHERE sentence = ?",
                (sentence,),
            )
            row = cursor.fetchone()
            gp_negative = parse_gp_ids(row[0]) if row and row[0] else []

            # Check current label status
            if params["current"] == "unlabeled":
                if gp_num in gp_positive or gp_num in gp_negative:
                    raise ValueError(
                        f"Grammar point {gp_id} is already labeled for: {sentence}"
                    )
            elif params["current"] == "positive":
                if gp_num not in gp_positive:
                    raise ValueError(
                        f"Grammar point {gp_id} is not positive for: {sentence}"
                    )
            elif params["current"] == "negative":
                if gp_num not in gp_negative:
                    raise ValueError(
                        f"Grammar point {gp_id} is not negative for: {sentence}"
                    )
        elif family == "register":
            # Parse suggested as complete new label set (e.g., "neutral,sonkeigo")
            suggested = params["suggested"]

            # Check current register_ids
            cursor.execute(
                "SELECT register_ids FROM corpus WHERE sentence = ?",
                (sentence,),
            )
            row = cursor.fetchone()
            current_regs = _parse_register_ids(row[0]) if row and row[0] else []

            # Validate that current field matches actual database state
            actual_labels = []
            for rid in sorted(current_regs):
                rlabel = REGISTER_ID_TO_LABEL.get(rid)
                if rlabel:
                    actual_labels.append(
                        rlabel.value if hasattr(rlabel, "value") else str(rlabel)
                    )
            actual_str = ",".join(actual_labels) if actual_labels else "none"
            expected_current = params["current"]
            if actual_str != expected_current:
                raise ValueError(
                    f"Register mismatch for '{sentence}': "
                    f"expected current='{expected_current}' but found '{actual_str}'"
                )

            # Validate suggested format is valid register names
            suggested_names = [n.strip() for n in suggested.split(",")]
            for name in suggested_names:
                found = False
                for level in REGISTER_LABEL_TO_ID:
                    if level.value == name:
                        found = True
                        break
                if not found:
                    raise ValueError(f"Unknown register name in suggested: {name}")


def _validate_label_match(
    sentence: str, expected: str, actual: Optional[float], column: str
) -> None:
    """Validate that the expected label matches the actual database value."""
    # Map label to numeric value
    label_to_value = {
        # Formality
        "very casual": -1.0,
        "casual": -0.5,
        "neutral": 0.0,
        "formal": 0.5,
        "very formal": 1.0,
        # Gender
        "masculine": -1.0,
        "feminine": 1.0,
        "unpragmatic": None,
    }

    expected_value = label_to_value.get(expected)
    if expected_value is None and expected != "unpragmatic":
        raise ValueError(f"Unknown label: {expected}")

    # Allow small tolerance for floating point comparison
    if expected_value is None:
        if actual is not None:
            raise ValueError(
                f"Mismatch for {sentence}: expected {column}=unpragmatic "
                f"but found {actual}"
            )
    elif actual is None:
        raise ValueError(
            f"Mismatch for {sentence}: expected {column}={expected} but found NULL"
        )
    elif abs(actual - expected_value) > 0.01:
        raise ValueError(
            f"Mismatch for {sentence}: expected {column}={expected} "
            f"({expected_value}) but found {actual}"
        )


def _create_relabels_table(conn: sqlite3.Connection) -> None:
    """Create the relabels tracking table if it doesn't exist."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS relabels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            sentence TEXT NOT NULL,
            command TEXT NOT NULL
        )
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_relabels_batch
        ON relabels(batch_id)
    """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_relabels_sentence
        ON relabels(sentence)
    """
    )
    conn.commit()


def _get_next_batch_id(conn: sqlite3.Connection) -> int:
    """Get the next batch ID for relabeling."""
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(batch_id) FROM relabels")
    row = cursor.fetchone()
    return (row[0] or 0) + 1


def _add_gp_to_list(
    cursor: sqlite3.Cursor, sentence: str, gp_num: int, column: str
) -> None:
    """Add a grammar point to a sentence's positive or negative list."""
    cursor.execute(f"SELECT {column} FROM corpus WHERE sentence = ?", (sentence,))
    row = cursor.fetchone()
    current_list = parse_gp_ids(row[0]) if row and row[0] else []
    if gp_num not in current_list:
        current_list.append(gp_num)
        current_list.sort()
        # Store with "gp" prefix and zero-padded to 4 digits (e.g., "gp0597")
        new_list_str = ",".join(f"gp{gp:04d}" for gp in current_list)
        cursor.execute(
            f"UPDATE corpus SET {column} = ? WHERE sentence = ?",
            (new_list_str, sentence),
        )


def _set_register_ids_from_names(
    cursor: sqlite3.Cursor, sentence: str, register_names: str
) -> None:
    """Set a sentence's register_ids from a comma-separated list of names."""
    # Parse names to IDs
    names = [n.strip() for n in register_names.split(",")]
    reg_ids = []
    for name in names:
        for level, idx in REGISTER_LABEL_TO_ID.items():
            if level.value == name:
                reg_ids.append(idx)
                break
    reg_ids.sort()
    new_list_str = ",".join(str(r) for r in reg_ids)
    cursor.execute(
        "UPDATE corpus SET register_ids = ? WHERE sentence = ?",
        (new_list_str, sentence),
    )


def _apply_relabel_commands(  # pylint: disable=too-many-locals
    conn: sqlite3.Connection,
    commands: List[Dict[str, Any]],
    batch_id: int,
    timestamp: str,
) -> None:
    """Apply all relabel commands atomically within a transaction.

    Note: If any error occurs, SQLite will automatically rollback the transaction.
    The caller is responsible for closing the connection in a finally block.
    """
    cursor = conn.cursor()
    cursor.execute("BEGIN TRANSACTION")

    for cmd in track(commands, description="Applying changes..."):
        family = cmd["family"]
        sentence = cmd["sentence"]
        params = cmd["params"]
        original_line = cmd["original_line"]

        # Apply the change
        if family == "formality":
            new_value = _label_to_value(params["suggested"])
            cursor.execute(
                "UPDATE corpus SET formality = ? WHERE sentence = ?",
                (new_value, sentence),
            )
        elif family == "gender":
            new_value = _label_to_value(params["suggested"])
            cursor.execute(
                "UPDATE corpus SET gender = ? WHERE sentence = ?",
                (new_value, sentence),
            )
        elif family == "grammar_point":
            gp_id = params["gp"]
            gp_num = int(gp_id[2:])
            suggested = params["suggested"]

            if suggested == "positive":
                _add_gp_to_list(cursor, sentence, gp_num, "grammar")
            elif suggested == "negative":
                _add_gp_to_list(cursor, sentence, gp_num, "grammar_negative")
        elif family == "register":
            # suggested is the complete new label set (e.g., "neutral,sonkeigo")
            suggested = params["suggested"]
            _set_register_ids_from_names(cursor, sentence, suggested)

        # Record in relabels table
        cursor.execute(
            "INSERT INTO relabels (batch_id, timestamp, sentence, command) "
            "VALUES (?, ?, ?, ?)",
            (batch_id, timestamp, sentence, original_line),
        )

    conn.commit()


def _label_to_value(label: str) -> Optional[float]:
    """Convert label name to numeric value."""
    label_map = {
        "very casual": -1.0,
        "casual": -0.5,
        "neutral": 0.0,
        "formal": 0.5,
        "very formal": 1.0,
        "masculine": -1.0,
        "feminine": 1.0,
        "unpragmatic": None,
    }
    value = label_map.get(label)
    if value is None and label != "unpragmatic":
        raise ValueError(f"Unknown label: {label}")
    return value


def _verify_relabel_changes(  # pylint: disable=too-many-locals
    conn: sqlite3.Connection, commands: List[Dict[str, Any]]
) -> None:
    """Verify that all changes were applied correctly."""
    cursor = conn.cursor()

    for cmd in track(commands, description="Verifying..."):
        family = cmd["family"]
        sentence = cmd["sentence"]
        params = cmd["params"]

        if family == "formality":
            cursor.execute(
                "SELECT formality FROM corpus WHERE sentence = ?", (sentence,)
            )
            row = cursor.fetchone()
            actual = row[0] if row else None
            expected = _label_to_value(params["suggested"])
            if expected is None:
                if actual is not None:
                    raise ValueError(
                        f"Verification failed for {sentence}: "
                        f"expected formality=NULL but found {actual}"
                    )
            elif actual is None or abs(actual - expected) > 0.01:
                raise ValueError(
                    f"Verification failed for {sentence}: "
                    f"expected formality={expected} but found {actual}"
                )
        elif family == "gender":
            cursor.execute("SELECT gender FROM corpus WHERE sentence = ?", (sentence,))
            row = cursor.fetchone()
            actual = row[0] if row else None
            expected = _label_to_value(params["suggested"])
            if expected is None:
                if actual is not None:
                    raise ValueError(
                        f"Verification failed for {sentence}: "
                        f"expected gender=NULL but found {actual}"
                    )
            elif actual is None or abs(actual - expected) > 0.01:
                raise ValueError(
                    f"Verification failed for {sentence}: "
                    f"expected gender={expected} but found {actual}"
                )
        elif family == "grammar_point":
            gp_id = params["gp"]
            gp_num = int(gp_id[2:])
            suggested = params["suggested"]

            if suggested == "positive":
                cursor.execute(
                    "SELECT grammar FROM corpus WHERE sentence = ?",
                    (sentence,),
                )
                row = cursor.fetchone()
                gp_positive = parse_gp_ids(row[0]) if row and row[0] else []
                if gp_num not in gp_positive:
                    raise ValueError(
                        f"Verification failed for {sentence}: {gp_id} not in grammar"
                    )
            elif suggested == "negative":
                cursor.execute(
                    "SELECT grammar_negative FROM corpus WHERE sentence = ?",
                    (sentence,),
                )
                row = cursor.fetchone()
                gp_negative = parse_gp_ids(row[0]) if row and row[0] else []
                if gp_num not in gp_negative:
                    raise ValueError(
                        f"Verification failed for {sentence}: "
                        f"{gp_id} not in grammar_negative"
                    )
        elif family == "register":
            # suggested is the complete new label set (e.g., "neutral,sonkeigo")
            suggested = params["suggested"]

            # Parse suggested to get expected register IDs
            expected_names = [n.strip() for n in suggested.split(",")]
            expected_ids = set()
            for name in expected_names:
                for level, idx in REGISTER_LABEL_TO_ID.items():
                    if level.value == name:
                        expected_ids.add(idx)
                        break

            cursor.execute(
                "SELECT register_ids FROM corpus WHERE sentence = ?",
                (sentence,),
            )
            row = cursor.fetchone()
            actual_regs = set(_parse_register_ids(row[0]) if row and row[0] else [])
            if actual_regs != expected_ids:
                raise ValueError(
                    f"Verification failed for {sentence}: "
                    f"expected register_ids={sorted(expected_ids)} "
                    f"but found {sorted(actual_regs)}"
                )
