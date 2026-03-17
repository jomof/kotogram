#!/usr/bin/env python3
"""
Grammar-KC Training Dividend Analysis

Measures how richly the KC bottleneck encodes grammar point information
by running four complementary analyses on the 1374 grammar point YAMLs.

Flags:
  (no flags)                Run the four dividend analyses
  --learning-order          Derive a natural KC learning order from the DAG
  --break-cycles            Remove learn_before entries to make the DAG cycle-free
  --discover-gps            Discover novel grammar points via KC clustering
  --find-nuance-divisions   Find GPs whose sentences split into distinct KC sub-clusters
"""

# pylint: disable=too-many-lines

import argparse
import glob
import os
import random
import sqlite3
import sys
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import yaml

if os.path.exists("scripts"):
    sys.path.insert(0, os.getcwd())

# pylint: disable=wrong-import-position
from kotogram.analysis import _ANALYZER
from kotogram.sudachi_japanese_parser import KotogramFormat, SudachiJapaneseParser
from kotogram.tokenizer import ENCODER_FEATURE_FIELDS, FEATURE_FIELDS

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gp_id_map(
    db_path: str = "data/corpus.db",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Map grammar point names -> gpXXXX ids from the corpus database."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, name FROM grammar").fetchall()
    conn.close()
    name_to_id = {name: gid for gid, name in rows}
    id_to_name = dict(rows)
    return name_to_id, id_to_name


def clean_sentence(raw: str) -> str:
    """Strip spaces and {braces} from YAML example sentences."""
    s = raw.replace("{", "").replace("}", "")
    s = s.replace(" ", "")
    return s.strip()


def load_grammar_yamls(grammar_dir: str = "data/grammar") -> List[dict]:
    """Load all grammar YAML files and return parsed dicts."""
    files = sorted(glob.glob(os.path.join(grammar_dir, "*.yaml")))
    results = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
            if d:
                d["_filepath"] = f
                results.append(d)
    return results


def load_gp_sentences_from_db(
    db_path: str = "data/corpus.db",
) -> Dict[str, List[str]]:
    """Load GP-name -> [sentence] mapping from corpus_gp_pos."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT g.name, p.sentence "
        "FROM corpus_gp_pos p JOIN grammar g ON g.id = p.gp_id"
    ).fetchall()
    conn.close()
    gp_sents: Dict[str, List[str]] = defaultdict(list)
    for name, sent in rows:
        gp_sents[name].append(sent)
    return dict(gp_sents)


def extract_competing_pairs(yamls: List[dict]) -> List[Tuple[str, str, str]]:
    """Extract (gp_name, target_sentence, competing_sentence) triples."""
    pairs = []
    for d in yamls:
        gp = d.get("grammar_point", "")
        for ex in d.get("examples", []):
            target_list = ex.get("japanese", [])
            if not target_list:
                continue
            target = clean_sentence(target_list[0])
            for cg in ex.get("competing_grammar", []):
                comp_list = cg.get("competing_japanese", [])
                if comp_list:
                    comp = clean_sentence(comp_list[0])
                    if target and comp and target != comp:
                        pairs.append((gp, target, comp))
    return pairs


def extract_false_friend_edges(
    yamls: List[dict], name_to_id: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Extract (gp_name_a, gp_name_b) false-friend pairs where both exist."""
    edges = []
    for d in yamls:
        gp = d.get("grammar_point", "")
        if gp not in name_to_id:
            continue
        for ff in d.get("false_friends", []):
            ff_gp = ff.get("grammar_point", "")
            if ff_gp and ff_gp in name_to_id and ff_gp != gp:
                edges.append((gp, ff_gp))
    return edges


def extract_prerequisite_edges(
    yamls: List[dict], name_to_id: Dict[str, str]
) -> List[Tuple[str, str]]:
    """Extract (prerequisite_gp, dependent_gp) edges from learn_before.

    If GP-B's learn_before lists GP-A, the edge is (A, B):
    A is a prerequisite of B.
    """
    edges = []
    for d in yamls:
        dependent = d.get("grammar_point", "")
        if dependent not in name_to_id:
            continue
        for prereq_name in d.get("learn_before", []):
            if prereq_name in name_to_id and prereq_name != dependent:
                edges.append((prereq_name, dependent))
    return edges


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def batch_infer_kc_probs(
    sentences: List[str],
    model: Any,
    tokenizer: Any,
    parser: SudachiJapaneseParser,
    batch_size: int = 64,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Run inference on sentences and return binary KC matrix (N x 1024).

    Returns a float32 matrix of KC probabilities (thresholded to binary).
    Threshold defaults to model.config.kc_threshold (adaptive).
    """
    # pylint: disable=too-many-locals,too-many-positional-arguments
    if threshold is None:
        threshold = float(model.config.kc_threshold)
    n = len(sentences)
    kc_dim = model.config.kc_vocab_size
    kc_matrix = np.zeros((n, kc_dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_sents = sentences[start:end]

        kotograms = [
            parser.japanese_to_kotogram(s, KotogramFormat.TRAINING_MASK)
            for s in batch_sents
        ]

        encoded_list = [tokenizer.encode(k) for k in kotograms]
        max_len = max(len(e[FEATURE_FIELDS[0]]) for e in encoded_list)
        bs = len(kotograms)

        field_inputs = {}
        for field in ENCODER_FEATURE_FIELDS:
            batch_ids = torch.zeros((bs, max_len), dtype=torch.long)
            for i, encoded in enumerate(encoded_list):
                ids = encoded[field]
                batch_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            field_inputs[f"input_ids_{field}"] = batch_ids

        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, encoded in enumerate(encoded_list):
            attention_mask[i, : len(encoded[FEATURE_FIELDS[0]])] = 1

        with torch.no_grad():
            pooled = model.pool(field_inputs, attention_mask)
            logits = model.predict_kcs(pooled)
            temp = getattr(model.config, "kc_temperature", 1.0)
            probs = torch.sigmoid(logits / temp)

        probs_np = probs.cpu().numpy()
        kc_matrix[start:end] = (probs_np >= threshold).astype(np.float32)

        if (start // batch_size) % 20 == 0:
            print(f"  Inferred {end}/{n} sentences...", flush=True)

    return kc_matrix  # type: ignore[no-any-return]


def batch_infer_kc_with_probs(
    sentences: List[str],
    model: Any,
    tokenizer: Any,
    parser: SudachiJapaneseParser,
    batch_size: int = 64,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference returning both binary KC matrix and raw probabilities.

    Returns (binary_matrix, prob_matrix), each of shape (N, kc_dim).
    Threshold defaults to model.config.kc_threshold (adaptive).
    """
    # pylint: disable=too-many-locals,too-many-positional-arguments
    if threshold is None:
        threshold = float(model.config.kc_threshold)
    n = len(sentences)
    kc_dim = model.config.kc_vocab_size
    kc_binary = np.zeros((n, kc_dim), dtype=np.float32)
    kc_probs = np.zeros((n, kc_dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_sents = sentences[start:end]

        kotograms = [
            parser.japanese_to_kotogram(s, KotogramFormat.TRAINING_MASK)
            for s in batch_sents
        ]

        encoded_list = [tokenizer.encode(k) for k in kotograms]
        max_len = max(len(e[FEATURE_FIELDS[0]]) for e in encoded_list)
        bs = len(kotograms)

        field_inputs = {}
        for field in ENCODER_FEATURE_FIELDS:
            batch_ids = torch.zeros((bs, max_len), dtype=torch.long)
            for i, encoded in enumerate(encoded_list):
                ids = encoded[field]
                batch_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            field_inputs[f"input_ids_{field}"] = batch_ids

        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)
        for i, encoded in enumerate(encoded_list):
            attention_mask[i, : len(encoded[FEATURE_FIELDS[0]])] = 1

        with torch.no_grad():
            pooled = model.pool(field_inputs, attention_mask)
            logits = model.predict_kcs(pooled)
            temp = getattr(model.config, "kc_temperature", 1.0)
            probs = torch.sigmoid(logits / temp)

        probs_np = probs.cpu().numpy()
        kc_probs[start:end] = probs_np
        kc_binary[start:end] = (probs_np >= threshold).astype(np.float32)

        if (start // batch_size) % 20 == 0:
            print(f"  Inferred {end}/{n} sentences...", flush=True)

    return kc_binary, kc_probs


# ---------------------------------------------------------------------------
# Analysis 1: KC Selectivity
# ---------------------------------------------------------------------------


def analysis_selectivity(
    gp_sents: Dict[str, List[str]],
    _all_sentences: List[str],
    sent_to_idx: Dict[str, int],
    kc_matrix: np.ndarray,
    _id_to_name: Dict[str, str],
    name_to_id: Dict[str, str],
    top_kcs_per_gp: int = 5,
    top_gps_per_kc: int = 5,
) -> Tuple[str, Dict[str, List[int]]]:
    """Compute KC selectivity for each grammar point."""
    # pylint: disable=too-many-locals,too-many-positional-arguments
    _, n_kcs = kc_matrix.shape
    baseline_rate = kc_matrix.mean(axis=0)  # (1024,) marginal firing rate

    gp_selective_kcs: Dict[str, List[int]] = {}
    rows = []

    for gp_name, sents in sorted(gp_sents.items()):
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) < 3:
            continue

        gp_matrix = kc_matrix[indices]
        gp_rate = gp_matrix.mean(axis=0)  # (1024,) conditional firing rate

        lift = np.where(baseline_rate > 0.001, gp_rate / baseline_rate, 0.0)
        top_kc_ids = np.argsort(-lift)[:top_kcs_per_gp]
        selective = [
            (int(kid), float(lift[kid]), float(gp_rate[kid]), float(baseline_rate[kid]))
            for kid in top_kc_ids
            if lift[kid] > 1.5
        ]

        gp_selective_kcs[gp_name] = [s[0] for s in selective]

        if selective:
            gp_id = name_to_id.get(gp_name, "????")
            top_str = ", ".join(
                f"KC{kid} (lift={lift:.1f}, gp={gr:.0%}, base={br:.0%})"
                for kid, lift, gr, br in selective[:3]
            )
            rows.append((gp_id, gp_name[:40], len(indices), top_str))

    # Summary stats
    n_gps_with_selective = sum(1 for v in gp_selective_kcs.values() if v)
    total_gps = len(gp_sents)

    # Reverse map: per-KC, which GPs is it selective for?
    kc_to_gps: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for gp_name, sents in gp_sents.items():
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) < 3:
            continue
        gp_rate = kc_matrix[indices].mean(axis=0)
        lift = np.where(baseline_rate > 0.001, gp_rate / baseline_rate, 0.0)
        for kid in range(n_kcs):
            if lift[kid] > 2.0:
                kc_to_gps[kid].append((gp_name, float(lift[kid])))

    for kid in kc_to_gps:
        kc_to_gps[kid].sort(key=lambda x: -x[1])

    kc_specialist_count = sum(
        1 for kid in range(n_kcs) if len(kc_to_gps.get(kid, [])) >= 1
    )

    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 1: KC SELECTIVITY FOR GRAMMAR POINTS")
    lines.append("=" * 70)
    lines.append(
        f"GPs with at least one selective KC (lift>1.5): {n_gps_with_selective}/{total_gps}"
    )
    lines.append(
        f"KCs selective (lift>2.0) for at least one GP: {kc_specialist_count}/{n_kcs}"
    )
    lines.append("")

    lines.append("Top selective KCs per GP (sample of 30):")
    lines.append(f"{'GP ID':<8} {'Grammar Point':<42} {'#Ex':>4}  Top KCs")
    lines.append("-" * 120)
    for gp_id, gp_short, n_ex, top_str in rows[:30]:
        lines.append(f"{gp_id:<8} {gp_short:<42} {n_ex:>4}  {top_str}")

    lines.append("")
    lines.append("Top GPs per KC (KCs with most grammar specialization):")
    kc_by_breadth = sorted(kc_to_gps.items(), key=lambda x: -len(x[1]))
    for kid, gp_list in kc_by_breadth[:15]:
        top_gps = ", ".join(
            f"{g[:30]}({lift:.1f}x)" for g, lift in gp_list[:top_gps_per_kc]
        )
        lines.append(f"  KC{kid}: {len(gp_list)} GPs — {top_gps}")

    lines.append("")
    return "\n".join(lines), gp_selective_kcs


# ---------------------------------------------------------------------------
# Analysis 2: Fingerprint Distinctiveness
# ---------------------------------------------------------------------------


def analysis_fingerprints(
    gp_sents: Dict[str, List[str]],
    sent_to_idx: Dict[str, int],
    kc_matrix: np.ndarray,
    false_friend_edges: List[Tuple[str, str]],
    _name_to_id: Dict[str, str],
) -> str:
    """Compute GP fingerprints and compare false friends vs random pairs."""
    # pylint: disable=too-many-locals
    # Build GP fingerprints (mean KC activation rate)
    gp_fingerprints: Dict[str, np.ndarray] = {}
    for gp_name, sents in gp_sents.items():
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) < 3:
            continue
        gp_fingerprints[gp_name] = kc_matrix[indices].mean(axis=0)

    if len(gp_fingerprints) < 10:
        return "ANALYSIS 2: Too few GPs with sufficient examples.\n"

    gp_names = list(gp_fingerprints.keys())
    fp_matrix = np.stack([gp_fingerprints[g] for g in gp_names])

    # Normalize for cosine similarity
    norms = np.linalg.norm(fp_matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    fp_normed = fp_matrix / norms

    # Full cosine similarity matrix
    sim_matrix = fp_normed @ fp_normed.T

    # Get upper triangle (exclude diagonal)
    n = len(gp_names)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    all_sims = sim_matrix[mask]

    # False friend similarities
    name_to_fpidx = {g: i for i, g in enumerate(gp_names)}
    ff_sims = []
    for a, b in false_friend_edges:
        if a in name_to_fpidx and b in name_to_fpidx:
            i, j = name_to_fpidx[a], name_to_fpidx[b]
            ff_sims.append(sim_matrix[i, j])
    ff_sims = np.array(ff_sims) if ff_sims else np.array([])

    # Random baseline (same number of random pairs)
    random.seed(42)
    random_sims = []
    for _ in range(max(len(ff_sims), 500)):
        i, j = random.sample(range(n), 2)
        random_sims.append(sim_matrix[i, j])
    random_sims = np.array(random_sims)

    # Find most and least similar GP pairs
    flat_idx = np.argsort(-all_sims)
    most_similar = []
    for idx in flat_idx[:10]:
        row_col = np.argwhere(mask)
        i, j = row_col[idx]
        most_similar.append((gp_names[i], gp_names[j], all_sims[idx]))

    # Active KC counts per GP
    active_counts = (fp_matrix > 0.5).sum(axis=1)

    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 2: KC FINGERPRINT DISTINCTIVENESS")
    lines.append("=" * 70)
    lines.append(f"GPs with fingerprints: {len(gp_fingerprints)}")
    lines.append(
        f"Avg active KCs per GP fingerprint: {active_counts.mean():.1f} (of 1024)"
    )
    lines.append(f"Median active KCs: {np.median(active_counts):.0f}")
    lines.append("")
    lines.append("Pairwise cosine similarity of GP fingerprints:")
    lines.append(
        f"  All pairs:       mean={all_sims.mean():.3f}, std={all_sims.std():.3f}"
    )
    if len(ff_sims) > 0:
        lines.append(
            f"  False friends:   mean={ff_sims.mean():.3f}, std={ff_sims.std():.3f} "
            f"(n={len(ff_sims)})"
        )
    lines.append(
        f"  Random baseline: mean={random_sims.mean():.3f}, std={random_sims.std():.3f} "
        f"(n={len(random_sims)})"
    )
    if len(ff_sims) > 0:
        delta = ff_sims.mean() - random_sims.mean()
        lines.append(
            f"  Delta (FF - random): {delta:+.3f} "
            f"({'higher' if delta > 0 else 'lower'} similarity for false friends)"
        )
    lines.append("")
    lines.append("Most similar GP pairs (by KC fingerprint):")
    for a, b, sim in most_similar[:8]:
        lines.append(f"  {sim:.3f}  {a[:35]} <-> {b[:35]}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 3: Prerequisite DAG Alignment
# ---------------------------------------------------------------------------


def analysis_dag_alignment(
    gp_sents: Dict[str, List[str]],
    sent_to_idx: Dict[str, int],
    kc_matrix: np.ndarray,
    prereq_edges: List[Tuple[str, str]],
) -> str:
    """Check if prerequisite GPs' KCs are contained in dependent GPs' KCs."""
    # pylint: disable=too-many-locals
    gp_fingerprints: Dict[str, np.ndarray] = {}
    for gp_name, sents in gp_sents.items():
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) < 3:
            continue
        gp_fingerprints[gp_name] = (kc_matrix[indices].mean(axis=0) > 0.5).astype(float)

    # Real containment ratios
    real_containments = []
    real_new_kc_counts = []
    for prereq, dependent in prereq_edges:
        if prereq not in gp_fingerprints or dependent not in gp_fingerprints:
            continue
        fp_a = gp_fingerprints[prereq]
        fp_b = gp_fingerprints[dependent]
        a_active = fp_a > 0.5
        b_active = fp_b > 0.5
        n_a = a_active.sum()
        if n_a == 0:
            continue
        contained = (a_active & b_active).sum()
        ratio = contained / n_a
        real_containments.append(float(ratio))
        new_in_b = (b_active & ~a_active).sum()
        real_new_kc_counts.append(int(new_in_b))

    # Shuffled baseline
    random.seed(42)
    gp_list = list(gp_fingerprints.keys())
    shuffled_containments = []
    for _ in range(len(real_containments)):
        a, b = random.sample(gp_list, 2)
        fp_a = gp_fingerprints[a]
        fp_b = gp_fingerprints[b]
        a_active = fp_a > 0.5
        b_active = fp_b > 0.5
        n_a = a_active.sum()
        if n_a == 0:
            continue
        contained = (a_active & b_active).sum()
        shuffled_containments.append(float(contained / n_a))

    real_arr = np.array(real_containments) if real_containments else np.array([0.0])
    shuf_arr = (
        np.array(shuffled_containments) if shuffled_containments else np.array([0.0])
    )

    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 3: PREREQUISITE DAG ALIGNMENT")
    lines.append("=" * 70)
    lines.append(f"Prerequisite edges evaluated: {len(real_containments)}")
    lines.append(
        f"  (from {len(prereq_edges)} total edges, "
        f"after filtering GPs with <3 examples)"
    )
    lines.append("")
    lines.append(
        "KC containment ratio (fraction of prereq's KCs also active in dependent):"
    )
    lines.append(
        f"  Real edges:     mean={real_arr.mean():.3f}, median={np.median(real_arr):.3f}"
    )
    lines.append(
        f"  Shuffled edges: mean={shuf_arr.mean():.3f}, median={np.median(shuf_arr):.3f}"
    )
    delta = real_arr.mean() - shuf_arr.mean()
    lines.append(
        f"  Delta:          {delta:+.3f} ({'ABOVE' if delta > 0 else 'below'} baseline)"
    )
    lines.append("")

    if real_new_kc_counts:
        new_arr = np.array(real_new_kc_counts)
        lines.append(
            f"New KCs added by dependent (beyond prerequisite): "
            f"mean={new_arr.mean():.1f}, median={np.median(new_arr):.0f}"
        )
        lines.append(
            "  (Positive means dependents activate MORE KCs than their prerequisites)"
        )
    lines.append("")

    # Fraction with high containment
    high_thresh = 0.7
    real_high = (real_arr >= high_thresh).mean()
    shuf_high = (shuf_arr >= high_thresh).mean()
    lines.append(
        f"Edges with >=70% containment: "
        f"real={real_high:.1%} vs shuffled={shuf_high:.1%}"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 4: Minimal Pair Discrimination
# ---------------------------------------------------------------------------


def analysis_minimal_pairs(
    competing_pairs: List[Tuple[str, str, str]],
    _all_sentences: List[str],
    sent_to_idx: Dict[str, int],
    kc_matrix: np.ndarray,
    gp_selective_kcs: Dict[str, List[int]],
    name_to_id: Dict[str, str],
) -> str:
    """Check if competing grammar pairs flip the right KCs."""
    # pylint: disable=too-many-locals,too-many-positional-arguments
    threshold = 0.5

    n_evaluated = 0
    total_flipped = 0
    aligned_flips = 0
    total_selective = 0
    examples: list[str] = []

    for gp_name, target, competitor in competing_pairs:
        if target not in sent_to_idx or competitor not in sent_to_idx:
            continue

        t_idx = sent_to_idx[target]
        c_idx = sent_to_idx[competitor]
        t_kcs = kc_matrix[t_idx]
        c_kcs = kc_matrix[c_idx]

        # KCs that flip (ON in target, OFF in competitor)
        flipped = (t_kcs > threshold) & (c_kcs <= threshold)
        n_flipped = int(flipped.sum())
        total_flipped += n_flipped
        n_evaluated += 1

        # Check alignment with selective KCs
        selective = set(gp_selective_kcs.get(gp_name, []))
        if selective and n_flipped > 0:
            flipped_ids = set(np.where(flipped)[0].tolist())
            overlap = flipped_ids & selective
            aligned_flips += len(overlap)
            total_selective += min(len(selective), n_flipped)

            if overlap and len(examples) < 12:
                gp_id = name_to_id.get(gp_name, "????")
                examples.append(
                    f"  {gp_id} {gp_name[:30]}: "
                    f"flipped={sorted(flipped_ids)[:5]}, "
                    f"selective={sorted(selective)[:5]}, "
                    f"overlap={sorted(overlap)}"
                )

    lines = []
    lines.append("=" * 70)
    lines.append("ANALYSIS 4: COMPETING GRAMMAR MINIMAL-PAIR DISCRIMINATION")
    lines.append("=" * 70)
    lines.append(f"Competing pairs evaluated: {n_evaluated}")
    if n_evaluated > 0:
        avg_flipped = total_flipped / n_evaluated
        lines.append(
            f"Average KCs flipped per pair (target ON, competitor OFF): {avg_flipped:.1f}"
        )
    if total_selective > 0:
        align_rate = aligned_flips / total_selective
        lines.append(
            f"Alignment: {aligned_flips}/{total_selective} flipped KCs match "
            f"GP-selective KCs ({align_rate:.1%})"
        )
    lines.append("")

    if examples:
        lines.append("Sample aligned flips:")
        for ex in examples:
            lines.append(ex)
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 5: KC Learning Order (--learning-order)
# ---------------------------------------------------------------------------


def compute_topological_depth(
    all_gps: Set[str],
    prereq_edges: List[Tuple[str, str]],
    jlpt_levels: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    # pylint: disable=too-many-nested-blocks
    """Compute shortest-path depth from roots for each GP.

    Uses BFS from root nodes (no prerequisites) to assign each GP the
    minimum number of prerequisite layers needed to reach it. This
    approach is cycle-resilient: nodes in cycles get depth from
    whichever acyclic path reaches them first.

    For GPs unreachable from roots (isolated cycles), we seed them
    using JLPT level as a proxy (N5=shallowest) and BFS outward.

    Depth 0 = root (no prerequisites or lowest-JLPT cycle seed).
    """
    # pylint: disable=too-many-locals
    children: Dict[str, Set[str]] = defaultdict(set)
    parents: Dict[str, Set[str]] = defaultdict(set)
    for prereq, dependent in prereq_edges:
        if prereq in all_gps and dependent in all_gps:
            children[prereq].add(dependent)
            parents[dependent].add(prereq)

    roots = {gp for gp in all_gps if not parents[gp]}

    depth: Dict[str, int] = {}
    queue: deque = deque()

    for gp in roots:
        depth[gp] = 0
        queue.append(gp)

    while queue:
        node = queue.popleft()
        for child in children[node]:
            if child not in depth:
                depth[child] = depth[node] + 1
                queue.append(child)

    # Handle unreachable nodes (in pure cycles) by seeding with JLPT
    unreachable = all_gps - set(depth.keys())
    if unreachable and jlpt_levels:
        max_reached = max(depth.values()) if depth else 0
        # Sort unreachable by JLPT (N5=5 first = easiest)
        by_jlpt = sorted(
            unreachable,
            key=lambda g: (-jlpt_levels.get(g, 0), g),
        )
        # Seed round by round: easiest JLPT first
        seed_depth = max_reached + 1
        for gp in by_jlpt:
            if gp not in depth:
                depth[gp] = seed_depth
                queue.append(gp)
                # BFS from this seed
                while queue:
                    node = queue.popleft()
                    for child in children[node]:
                        if child not in depth:
                            depth[child] = depth[node] + 1
                            queue.append(child)
                seed_depth = max(depth.values()) + 1

    # Final fallback for anything still unassigned
    max_depth = max(depth.values()) if depth else 0
    for gp in all_gps:
        if gp not in depth:
            depth[gp] = max_depth + 1

    return depth


def extract_jlpt_levels(yamls: List[dict]) -> Dict[str, int]:
    """Map GP name -> JLPT numeric level (5=easiest, 1=hardest)."""
    levels = {}
    for d in yamls:
        gp = d.get("grammar_point", "")
        jlpt = d.get("jlpt", "")
        if gp and jlpt:
            jlpt_str = str(jlpt).replace("N", "").strip()
            if jlpt_str.isdigit():
                levels[gp] = int(jlpt_str)
    return levels


def analysis_learning_order(
    gp_sents: Dict[str, List[str]],
    sent_to_idx: Dict[str, int],
    kc_matrix: np.ndarray,
    prereq_edges: List[Tuple[str, str]],
    yamls: List[dict],
    name_to_id: Dict[str, str],
) -> str:
    """Derive a natural KC learning order from the prerequisite DAG."""
    # pylint: disable=too-many-locals,too-many-positional-arguments
    n_kcs = kc_matrix.shape[1]

    # 1. Build GP fingerprints
    gp_fingerprints: Dict[str, np.ndarray] = {}
    for gp_name, sents in gp_sents.items():
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) < 3:
            continue
        gp_fingerprints[gp_name] = (kc_matrix[indices].mean(axis=0) > 0.5).astype(float)

    gps_with_fp = set(gp_fingerprints.keys())

    # 2. JLPT levels (needed for cycle-breaking)
    jlpt_levels = extract_jlpt_levels(yamls)

    # 3. Topological depth (cycle-resilient BFS)
    filtered_edges = [
        (a, b) for a, b in prereq_edges if a in gps_with_fp and b in gps_with_fp
    ]
    topo_depth = compute_topological_depth(gps_with_fp, filtered_edges, jlpt_levels)
    max_depth = max(topo_depth.values()) if topo_depth else 0

    # 4. For each KC, find its emergence properties
    kc_min_depth = np.full(n_kcs, max_depth + 2, dtype=float)
    kc_min_jlpt = np.full(n_kcs, 0, dtype=float)  # 0 = no JLPT assigned
    kc_gp_count = np.zeros(n_kcs, dtype=int)
    kc_earliest_gp: Dict[int, str] = {}

    # Also track: how many GPs at each depth activate each KC
    kc_depth_histogram = np.zeros((n_kcs, max_depth + 2), dtype=int)

    for gp_name, fp in gp_fingerprints.items():
        active_kcs = np.where(fp > 0.5)[0]
        d = topo_depth.get(gp_name, max_depth + 1)
        jlpt = jlpt_levels.get(gp_name, 0)

        for kid in active_kcs:
            kc_gp_count[kid] += 1
            if d < max_depth + 2:
                kc_depth_histogram[kid, min(d, max_depth + 1)] += 1

            if d < kc_min_depth[kid]:
                kc_min_depth[kid] = d
                kc_earliest_gp[int(kid)] = gp_name

            if jlpt > 0:
                if kc_min_jlpt[kid] == 0 or jlpt > kc_min_jlpt[kid]:
                    kc_min_jlpt[kid] = jlpt  # Higher N = easier

    # 5. Define learning stages by binning emergence depth
    # Adaptive boundaries based on actual depth distribution
    depth_vals = sorted(topo_depth.values())
    p25 = int(np.percentile(depth_vals, 25))
    p50 = int(np.percentile(depth_vals, 50))
    p75 = int(np.percentile(depth_vals, 75))
    p90 = int(np.percentile(depth_vals, 90))

    stage_boundaries = [
        (0, max(1, p25)),
        (max(1, p25) + 1, max(2, p50)),
        (max(2, p50) + 1, max(3, p75)),
        (max(3, p75) + 1, max(4, p90)),
        (max(4, p90) + 1, max_depth + 1),
    ]
    stage_labels = [
        f"Stage 1: Foundations (depth 0-{stage_boundaries[0][1]})",
        f"Stage 2: Early (depth {stage_boundaries[1][0]}-{stage_boundaries[1][1]})",
        f"Stage 3: Intermediate (depth {stage_boundaries[2][0]}-{stage_boundaries[2][1]})",
        f"Stage 4: Upper-Int (depth {stage_boundaries[3][0]}-{stage_boundaries[3][1]})",
        f"Stage 5: Advanced (depth {stage_boundaries[4][0]}-{max_depth})",
    ]

    kc_stage = np.zeros(n_kcs, dtype=int)
    for kid in range(n_kcs):
        d = kc_min_depth[kid]
        for si, (lo, hi) in enumerate(stage_boundaries):
            if lo <= d <= hi:
                kc_stage[kid] = si + 1
                break
        else:
            kc_stage[kid] = 0  # never activated

    # Count roots and cycle nodes
    parents_of: Dict[str, Set[str]] = defaultdict(set)
    for a, b in filtered_edges:
        parents_of[b].add(a)
    roots = {g for g in gps_with_fp if not parents_of[g]}

    # Detect cycle nodes (reachable from roots vs total)
    reachable: Set[str] = set()
    bfs_q: deque = deque(roots)
    visited_bfs: Set[str] = set(roots)
    children_map: Dict[str, Set[str]] = defaultdict(set)
    for a, b in filtered_edges:
        children_map[a].add(b)
    while bfs_q:
        node = bfs_q.popleft()
        reachable.add(node)
        for c in children_map[node]:
            if c not in visited_bfs:
                visited_bfs.add(c)
                bfs_q.append(c)
    cycle_seeded = gps_with_fp - reachable

    # 6. Build report
    lines = []
    lines.append("=" * 70)
    lines.append("KC NATURAL LEARNING ORDER")
    lines.append("=" * 70)
    lines.append(f"GPs with fingerprints: {len(gps_with_fp)}")
    lines.append(f"DAG edges used: {len(filtered_edges)}")
    lines.append(f"Root GPs (no prerequisites): {len(roots)}")
    lines.append(f"GPs reachable from roots: {len(reachable)}")
    lines.append(f"GPs in pure cycles (JLPT-seeded): {len(cycle_seeded)}")
    lines.append(f"Max topological depth: {max_depth}")
    lines.append("")

    # GP depth distribution
    depth_counts: Dict[int, int] = defaultdict(int)
    for d in topo_depth.values():
        depth_counts[d] += 1
    lines.append("GP distribution by topological depth:")
    for d in sorted(depth_counts.keys())[:25]:
        histogram_bar = "#" * min(depth_counts[d], 60)
        lines.append(f"  depth {d:>2}: {depth_counts[d]:>4} GPs  {histogram_bar}")
    if max_depth > 24:
        remaining = sum(v for k, v in depth_counts.items() if k > 24)
        lines.append(f"  depth 25+: {remaining:>4} GPs")
    lines.append("")

    # KC stage summary
    lines.append("KC distribution by learning stage:")
    for si, label in enumerate(stage_labels, 1):
        count = int((kc_stage == si).sum())
        lines.append(f"  {label}: {count} KCs")
    never = int((kc_stage == 0).sum())
    lines.append(f"  (Never activated by any GP): {never} KCs")
    lines.append("")

    # JLPT correlation
    lines.append("JLPT level at which KCs first emerge:")
    for n_level in [5, 4, 3, 2, 1]:
        count = int((kc_min_jlpt == n_level).sum())
        lines.append(f"  N{n_level}: {count} KCs first appear")
    no_jlpt = int((kc_min_jlpt == 0).sum())
    lines.append(f"  (No JLPT GP activates): {no_jlpt} KCs")
    lines.append("")

    # Cross-tabulation: JLPT vs topological stage
    lines.append("Cross-tab: KC emergence stage vs JLPT level:")
    header = f"{'Stage':<35} {'N5':>5} {'N4':>5} {'N3':>5} {'N2':>5} {'N1':>5}"
    lines.append(f"  {header}")
    lines.append(f"  {'-' * len(header)}")
    for si, label in enumerate(stage_labels, 1):
        mask = kc_stage == si
        row_vals = []
        for n_level in [5, 4, 3, 2, 1]:
            count = int(((kc_min_jlpt == n_level) & mask).sum())
            row_vals.append(f"{count:>5}")
        lines.append(f"  {label:<35} {''.join(row_vals)}")
    lines.append("")

    # Detailed stage listings
    for si, label in enumerate(stage_labels, 1):
        stage_kcs = np.where(kc_stage == si)[0]
        if len(stage_kcs) == 0:
            continue
        lines.append("-" * 70)
        lines.append(f"{label}")
        lines.append(f"({len(stage_kcs)} KCs)")
        lines.append("-" * 70)

        # Sort by: min_depth ascending, then gp_count descending
        stage_kcs_sorted = sorted(
            stage_kcs,
            key=lambda k: (kc_min_depth[k], -kc_gp_count[k]),
        )

        for kid in stage_kcs_sorted[:40]:
            earliest = kc_earliest_gp.get(int(kid), "?")
            jlpt = int(kc_min_jlpt[kid]) if kc_min_jlpt[kid] > 0 else 0
            gp_id = name_to_id.get(earliest, "")
            n_gps = kc_gp_count[kid]
            d = int(kc_min_depth[kid])
            lines.append(
                f"  KC{kid:<4} depth={d:<3} JLPT=N{jlpt if jlpt > 0 else '?'}  "
                f"activates for {n_gps:>4} GPs  "
                f"earliest: {earliest[:45]} ({gp_id})"
            )

        if len(stage_kcs_sorted) > 40:
            lines.append(f"  ... and {len(stage_kcs_sorted) - 40} more KCs")
        lines.append("")

    # Validation: do later-stage KCs co-occur with earlier-stage KCs?
    lines.append("=" * 70)
    lines.append("VALIDATION: STAGE CO-OCCURRENCE")
    lines.append("=" * 70)
    lines.append("For each stage, what fraction of its KCs' activations co-occur")
    lines.append("with at least one KC from an earlier stage?")
    lines.append("")

    n_sents = kc_matrix.shape[0]
    for si in range(2, len(stage_labels) + 1):
        cur_kcs = set(np.where(kc_stage == si)[0].tolist())
        earlier_kcs = set(np.where(kc_stage < si)[0].tolist())
        if not cur_kcs or not earlier_kcs:
            continue

        cur_list = sorted(cur_kcs)
        earlier_list = sorted(earlier_kcs)

        # Sample sentences for efficiency
        sample_size = min(n_sents, 5000)
        sample_idx = np.random.default_rng(42).choice(
            n_sents, sample_size, replace=False
        )
        sub_matrix = kc_matrix[sample_idx]

        cur_any = sub_matrix[:, cur_list].sum(axis=1) > 0
        earlier_any = sub_matrix[:, earlier_list].sum(axis=1) > 0
        both = (cur_any & earlier_any).sum()
        cur_total = cur_any.sum()
        rate = both / cur_total if cur_total > 0 else 0

        lines.append(
            f"  {stage_labels[si - 1]}: "
            f"{rate:.1%} of sentences with stage-{si} KCs also have earlier-stage KCs"
        )

    lines.append("")

    # The actual learning order: list all 1024 KCs sorted by emergence
    lines.append("=" * 70)
    lines.append("FULL KC LEARNING ORDER (sorted by emergence depth)")
    lines.append("=" * 70)
    lines.append("(First 200 KCs — the natural curriculum front-loading)")
    lines.append("")

    all_kcs_sorted = sorted(
        range(n_kcs),
        key=lambda k: (kc_min_depth[k], -kc_gp_count[k]),
    )

    for rank, kid in enumerate(all_kcs_sorted[:200], 1):
        earliest = kc_earliest_gp.get(int(kid), "?")
        jlpt = int(kc_min_jlpt[kid]) if kc_min_jlpt[kid] > 0 else 0
        d = int(kc_min_depth[kid])
        n_gps = kc_gp_count[kid]
        lines.append(
            f"  #{rank:<4} KC{kid:<4} depth={d:<3} N{jlpt if jlpt > 0 else '?'}  "
            f"{n_gps:>4} GPs  {earliest[:50]}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cycle Breaking (--break-cycles)
# ---------------------------------------------------------------------------


def _tarjan_sccs(nodes: Set[str], edges: List[Tuple[str, str]]) -> List[List[str]]:
    """Find strongly connected components using iterative Tarjan's."""
    children: Dict[str, List[str]] = defaultdict(list)
    for a, b in edges:
        if a in nodes and b in nodes:
            children[a].append(b)

    index_map: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    sccs: List[List[str]] = []
    counter = [0]

    def strongconnect(v: str) -> None:
        work = [(v, 0, False)]
        while work:
            node, ci, returning = work.pop()
            if not returning:
                index_map[node] = counter[0]
                lowlink[node] = counter[0]
                counter[0] += 1
                stack.append(node)
                on_stack[node] = True

            if returning and ci > 0:
                child = children[node][ci - 1]
                lowlink[node] = min(lowlink[node], lowlink[child])

            pushed = False
            for i in range(ci, len(children[node])):
                w = children[node][i]
                if w not in index_map:
                    work.append((node, i + 1, True))
                    work.append((w, 0, False))
                    pushed = True
                    break
                if on_stack.get(w, False):
                    lowlink[node] = min(lowlink[node], index_map[w])

            if not pushed:
                if lowlink[node] == index_map[node]:
                    scc: List[str] = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.append(w)
                        if w == node:
                            break
                    sccs.append(scc)

    for v in sorted(nodes):
        if v not in index_map:
            strongconnect(v)

    return sccs


def compute_gp_complexity(
    gp_fingerprints: Dict[str, np.ndarray],
    jlpt_levels: Dict[str, int],
    baseline_rate: np.ndarray,
) -> Dict[str, float]:
    """Score each GP's complexity using KC fingerprint properties.

    Complexity = sum of inverse-frequency weights for active KCs.
    This means GPs that activate rare KCs score higher (more complex).
    JLPT is used as a tiebreaker: lower N = higher complexity.
    """
    scores: Dict[str, float] = {}
    idf = np.where(baseline_rate > 0.001, 1.0 / baseline_rate, 0.0)

    for gp, fp in gp_fingerprints.items():
        active = fp > 0.5
        # Primary: rarity-weighted KC count
        primary = float(np.sum(active * idf))
        # Secondary: JLPT (N5=5 → low complexity, N1=1 → high complexity)
        jlpt = jlpt_levels.get(gp, 3)
        secondary = (6 - jlpt) * 0.001  # small tiebreaker
        scores[gp] = primary + secondary

    return scores


def find_feedback_arc_set(
    nodes: Set[str],
    edges: List[Tuple[str, str]],
    complexity: Dict[str, float],
) -> List[Tuple[str, str]]:
    """Find edges to remove to make the graph acyclic.

    Within each SCC, orders nodes by complexity and removes edges
    that point from higher to lower complexity (back-edges).
    Only operates on edges within cyclic SCCs.
    """
    # pylint: disable=too-many-locals
    sccs = _tarjan_sccs(nodes, edges)
    cyclic_sccs = [s for s in sccs if len(s) > 1]

    scc_membership: Dict[str, int] = {}
    for i, scc in enumerate(cyclic_sccs):
        for node in scc:
            scc_membership[node] = i

    # For each SCC, compute complexity rank
    scc_ranks: Dict[str, int] = {}
    for scc in cyclic_sccs:
        ordered = sorted(scc, key=lambda g: complexity.get(g, 0))
        for rank, gp in enumerate(ordered):
            scc_ranks[gp] = rank

    to_remove: List[Tuple[str, str]] = []
    for prereq, dependent in edges:
        if prereq not in scc_membership:
            continue
        if scc_membership.get(prereq) != scc_membership.get(dependent):
            continue
        # Both in same SCC: remove if prereq has higher complexity rank
        if scc_ranks[prereq] >= scc_ranks[dependent]:
            to_remove.append((prereq, dependent))

    return to_remove


def _normalize_yaml_entry(raw: str) -> str:
    """Strip YAML quoting (single/double) from a list entry."""
    s = raw.strip()
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        s = s[1:-1]
    return s


def remove_learn_before_entry(filepath: str, entry_to_remove: str) -> bool:
    """Remove a specific entry from a YAML file's learn_before list."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_learn_before = False
    remove_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("learn_before:"):
            in_learn_before = True
            continue
        if in_learn_before:
            if stripped.startswith("- "):
                entry = _normalize_yaml_entry(stripped[2:])
                if entry == entry_to_remove:
                    remove_idx = i
                    break
            elif stripped and not stripped.startswith("-"):
                in_learn_before = False

    if remove_idx is None:
        return False

    # Check if this is the last entry in learn_before
    remaining_entries = 0
    in_lb = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("learn_before:"):
            in_lb = True
            continue
        if in_lb:
            if stripped.startswith("- ") and i != remove_idx:
                remaining_entries += 1
            elif stripped and not stripped.startswith("-"):
                break

    new_lines = list(lines)
    new_lines.pop(remove_idx)

    # If no entries remain, also remove the learn_before: header
    if remaining_entries == 0:
        for i, line in enumerate(new_lines):
            if line.strip().startswith("learn_before:"):
                new_lines.pop(i)
                break

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return True


def build_gp_to_filepath(yamls: List[dict]) -> Dict[str, str]:
    """Map grammar point names to their YAML file paths."""
    return {d["grammar_point"]: d["_filepath"] for d in yamls if "grammar_point" in d}


def main_break_cycles(ctx: dict) -> None:
    """Use model-derived complexity to minimally break DAG cycles."""
    # pylint: disable=too-many-locals
    print("=" * 70)
    print("CYCLE-FREE DAG: MODEL-GUIDED EDGE REMOVAL")
    print("=" * 70)
    print()

    gp_sents = ctx["gp_sents"]
    kc_matrix = ctx["kc_matrix"]
    sent_to_idx = ctx["sent_to_idx"]
    prereq_edges = ctx["prereq_edges"]
    yamls = ctx["yamls"]
    name_to_id: Dict[str, str] = ctx["name_to_id"]

    # 1. Build GP fingerprints
    gp_fingerprints: Dict[str, np.ndarray] = {}
    for gp_name, sents in gp_sents.items():
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) < 1:
            continue
        gp_fingerprints[gp_name] = (kc_matrix[indices].mean(axis=0) > 0.5).astype(float)

    # Fallback: GPs without enough sentences get zero fingerprint
    all_gps = {d["grammar_point"] for d in yamls if "grammar_point" in d}
    n_kcs = kc_matrix.shape[1]
    for gp in all_gps:
        if gp not in gp_fingerprints:
            gp_fingerprints[gp] = np.zeros(n_kcs, dtype=float)

    # 2. Compute complexity scores
    baseline_rate = kc_matrix.mean(axis=0)
    jlpt_levels = extract_jlpt_levels(yamls)
    complexity = compute_gp_complexity(gp_fingerprints, jlpt_levels, baseline_rate)

    # 3. Find SCCs and feedback arc set
    sccs = _tarjan_sccs(all_gps, prereq_edges)
    cyclic_sccs = [s for s in sccs if len(s) > 1]
    total_cyclic_nodes = sum(len(s) for s in cyclic_sccs)
    scc_sizes = sorted([len(s) for s in cyclic_sccs], reverse=True)

    print(f"Cyclic SCCs: {len(cyclic_sccs)}")
    print(f"Nodes in cycles: {total_cyclic_nodes}")
    print(f"Largest SCC: {scc_sizes[0] if scc_sizes else 0} nodes")
    print(f"Total edges: {len(prereq_edges)}")
    print()

    to_remove = find_feedback_arc_set(all_gps, prereq_edges, complexity)
    print(f"Edges to remove: {len(to_remove)}")
    print(f"Edges retained: {len(prereq_edges) - len(to_remove)}")
    print(f"Removal rate: {len(to_remove) / len(prereq_edges):.1%}")
    print()

    # 4. Verify the remaining graph is acyclic
    remaining = set(prereq_edges) - set(to_remove)
    verify_sccs = _tarjan_sccs(all_gps, list(remaining))
    verify_cyclic = [s for s in verify_sccs if len(s) > 1]
    if verify_cyclic:
        print(f"WARNING: {len(verify_cyclic)} cycles remain after removal!")
        # Additional pass: brute force remove remaining cycle edges
        extra = find_feedback_arc_set(all_gps, list(remaining), complexity)
        to_remove.extend(extra)
        remaining = set(prereq_edges) - set(to_remove)
        print(f"After additional pass: {len(to_remove)} total removals")
    else:
        print("Verified: resulting graph is ACYCLIC")
    print()

    # 5. Sample removed edges
    print("Sample removed edges (prereq → dependent, prereq was MORE complex):")
    sample = sorted(to_remove, key=lambda e: -complexity.get(e[0], 0))[:20]
    for prereq, dependent in sample:
        pc = complexity.get(prereq, 0)
        dc = complexity.get(dependent, 0)
        pj = jlpt_levels.get(prereq, 0)
        dj = jlpt_levels.get(dependent, 0)
        print(
            f"  N{pj} {prereq[:40]:<42} (c={pc:.0f})"
            f"  →  N{dj} {dependent[:40]:<42} (c={dc:.0f})"
        )
    print()

    # 6. Edit YAML files
    gp_to_file = build_gp_to_filepath(yamls)

    # Group removals by dependent (the file to edit)
    removals_by_dep: Dict[str, List[str]] = defaultdict(list)
    for prereq, dependent in to_remove:
        removals_by_dep[dependent].append(prereq)

    edited_files = 0
    total_entries_removed = 0
    failed = []
    for dependent, prereqs in sorted(removals_by_dep.items()):
        filepath = gp_to_file.get(dependent)
        if not filepath:
            failed.append((dependent, "no file found"))
            continue
        for prereq in prereqs:
            ok = remove_learn_before_entry(filepath, prereq)
            if ok:
                total_entries_removed += 1
            else:
                failed.append((dependent, f"entry '{prereq}' not found"))
        edited_files += 1

    print(f"YAML files edited: {edited_files}")
    print(f"learn_before entries removed: {total_entries_removed}")
    if failed:
        print(f"Failed removals: {len(failed)}")
        for dep, reason in failed[:10]:
            print(f"  {dep[:50]}: {reason}")
    print()

    # 7. Final verification
    # Re-parse the edited YAMLs
    edited_yamls = load_grammar_yamls()
    edited_edges = extract_prerequisite_edges(edited_yamls, name_to_id)
    final_sccs = _tarjan_sccs(all_gps, edited_edges)
    final_cyclic = [s for s in final_sccs if len(s) > 1]
    print("Post-edit verification:")
    print(f"  Edges remaining: {len(edited_edges)}")
    print(f"  Cyclic SCCs remaining: {len(final_cyclic)}")
    if final_cyclic:
        print(f"  WARNING: {sum(len(s) for s in final_cyclic)} nodes still in cycles")
    else:
        print("  DAG is now CYCLE-FREE")


def _load_common(db_path: str = "data/corpus.db", need_yaml: bool = False) -> dict:  # pylint: disable=too-many-locals
    """Load data and run inference shared by all analysis modes.

    Sentences come from corpus_gp_pos (cleaner, more complete than YAML examples).
    YAML files are loaded only when structural metadata is needed (prerequisites,
    competing pairs, false friends, JLPT levels, file paths).
    """
    print("Loading grammar point mapping from corpus.db...")
    name_to_id, id_to_name = load_gp_id_map(db_path)
    print(f"  {len(name_to_id)} grammar points in database")

    print("Loading positive-labeled sentences from corpus.db...")
    gp_sents = load_gp_sentences_from_db(db_path)
    total_sents = sum(len(v) for v in gp_sents.values())
    print(f"  {len(gp_sents)} GPs with examples, {total_sents} total sentence-GP pairs")

    yamls = None
    prereq_edges = []
    competing_pairs = []
    if need_yaml:
        print("Loading grammar YAML files (for structural metadata)...")
        yamls = load_grammar_yamls()
        print(f"  {len(yamls)} YAML files loaded")

        print("Extracting prerequisite edges...")
        prereq_edges = extract_prerequisite_edges(yamls, name_to_id)
        print(f"  {len(prereq_edges)} prerequisite edges")

        competing_pairs = extract_competing_pairs(yamls)
        print(f"  {len(competing_pairs)} competing pairs")

    all_sentences_set: Set[str] = set()
    for sents in gp_sents.values():
        all_sentences_set.update(sents)

    for _, target, comp in competing_pairs:
        all_sentences_set.add(target)
        all_sentences_set.add(comp)

    all_sentences = sorted(all_sentences_set)
    sent_to_idx = {s: i for i, s in enumerate(all_sentences)}
    print(f"\nTotal unique sentences for inference: {len(all_sentences)}")

    print("\nLoading model...")
    model, tokenizer = _ANALYZER.load()
    model.eval()
    parser = SudachiJapaneseParser()

    threshold = float(model.config.kc_threshold)
    print(f"Running batch KC inference (threshold={threshold:.3f})...")
    kc_matrix = batch_infer_kc_probs(
        all_sentences, model, tokenizer, parser, batch_size=64, threshold=threshold
    )
    print(f"  KC matrix shape: {kc_matrix.shape}")
    avg_active = kc_matrix.sum(axis=1).mean()
    print(f"  Average active KCs per sentence: {avg_active:.1f}")
    print()

    return {
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "yamls": yamls,
        "gp_sents": gp_sents,
        "total_sents": total_sents,
        "prereq_edges": prereq_edges,
        "competing_pairs": competing_pairs,
        "all_sentences": all_sentences,
        "sent_to_idx": sent_to_idx,
        "kc_matrix": kc_matrix,
        "avg_active": avg_active,
        "db_path": db_path,
        "kc_threshold": threshold,
    }


def main_dividend(ctx: dict) -> None:
    """Run the four grammar-KC dividend analyses."""
    # pylint: disable=too-many-locals
    print("=" * 70)
    print("GRAMMAR-KC TRAINING DIVIDEND ANALYSIS")
    print("=" * 70)
    print()

    name_to_id: Dict[str, str] = ctx["name_to_id"]
    id_to_name = ctx["id_to_name"]
    yamls = ctx["yamls"]
    gp_sents = ctx["gp_sents"]
    kc_matrix = ctx["kc_matrix"]
    sent_to_idx = ctx["sent_to_idx"]
    all_sentences = ctx["all_sentences"]
    prereq_edges = ctx["prereq_edges"]
    avg_active = ctx["avg_active"]
    total_sents = ctx["total_sents"]

    false_friend_edges = extract_false_friend_edges(yamls, name_to_id)
    competing_pairs = ctx["competing_pairs"]

    print("Running Analysis 1: KC Selectivity...")
    report_1, gp_selective_kcs = analysis_selectivity(
        gp_sents, all_sentences, sent_to_idx, kc_matrix, id_to_name, name_to_id
    )
    print(report_1)

    print("Running Analysis 2: Fingerprint Distinctiveness...")
    report_2 = analysis_fingerprints(
        gp_sents, sent_to_idx, kc_matrix, false_friend_edges, name_to_id
    )
    print(report_2)

    print("Running Analysis 3: DAG Alignment...")
    report_3 = analysis_dag_alignment(gp_sents, sent_to_idx, kc_matrix, prereq_edges)
    print(report_3)

    print("Running Analysis 4: Minimal Pair Discrimination...")
    report_4 = analysis_minimal_pairs(
        competing_pairs,
        all_sentences,
        sent_to_idx,
        kc_matrix,
        gp_selective_kcs,
        name_to_id,
    )
    print(report_4)

    full_report = "\n".join(
        [
            "GRAMMAR-KC TRAINING DIVIDEND ANALYSIS",
            "=" * 70,
            f"Model: models/style/ (1024 KCs, threshold {ctx['kc_threshold']:.2f})",
            f"Grammar points: {len(name_to_id)}",
            f"YAML example sentences: {total_sents}",
            f"Unique sentences analyzed: {len(all_sentences)}",
            f"Average active KCs per sentence: {avg_active:.1f}",
            "",
            report_1,
            report_2,
            report_3,
            report_4,
        ]
    )

    output_path = "semantics/grammar-kc-dividend.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"Report written to {output_path}")


def main_learning_order(ctx: dict) -> None:
    """Derive and report a natural KC learning order."""
    print("=" * 70)
    print("KC NATURAL LEARNING ORDER ANALYSIS")
    print("=" * 70)
    print()

    report = analysis_learning_order(
        gp_sents=ctx["gp_sents"],
        sent_to_idx=ctx["sent_to_idx"],
        kc_matrix=ctx["kc_matrix"],
        prereq_edges=ctx["prereq_edges"],
        yamls=ctx["yamls"],
        name_to_id=ctx["name_to_id"],
    )
    print(report)

    output_path = "semantics/kc-learning-order.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")


def main_discover_gps(
    ctx: dict, db_path: str = "data/corpus.db", sample_size: int = 15000
) -> None:
    """Discover novel grammar points by clustering KC patterns."""
    # pylint: disable=too-many-locals
    from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]
    from sklearn.metrics import (  # type: ignore[import-untyped]
        adjusted_rand_score,
        silhouette_score,
    )

    gp_sents = ctx["gp_sents"]
    kc_matrix = ctx["kc_matrix"]
    all_sentences = ctx["all_sentences"]
    sent_to_idx = ctx["sent_to_idx"]

    model, tokenizer = _ANALYZER.load()
    model.eval()
    parser = SudachiJapaneseParser()

    # ------------------------------------------------------------------
    # Step 1: Sample unlabeled corpus sentences and run KC inference
    # ------------------------------------------------------------------
    print("Sampling unlabeled corpus sentences...")
    labeled_sents = set(all_sentences)

    conn = sqlite3.connect(db_path)
    all_corpus = [
        r[0]
        for r in conn.execute(
            "SELECT sentence FROM sentences WHERE grammatic = 1"
        ).fetchall()
    ]
    conn.close()

    unlabeled = [s for s in all_corpus if s not in labeled_sents]
    print(f"  Grammatical corpus sentences: {len(all_corpus)}")
    print(f"  Labeled (already have KC vectors): {len(labeled_sents)}")
    print(f"  Unlabeled candidates: {len(unlabeled)}")

    random.seed(42)
    if len(unlabeled) > sample_size:
        corpus_sample = random.sample(unlabeled, sample_size)
    else:
        corpus_sample = unlabeled
    print(f"  Sampled {len(corpus_sample)} unlabeled sentences")

    print("Running KC inference on corpus sample...")
    corpus_kc = batch_infer_kc_probs(
        corpus_sample, model, tokenizer, parser, batch_size=64
    )
    print(f"  Corpus KC matrix: {corpus_kc.shape}")

    # ------------------------------------------------------------------
    # Step 2: Pool labeled + unlabeled sentences
    # ------------------------------------------------------------------
    print("Pooling labeled and unlabeled sentences...")
    pool_sentences = all_sentences + corpus_sample
    pool_kc = np.vstack([kc_matrix, corpus_kc])
    print(f"  Pool size: {len(pool_sentences)} sentences, KC matrix {pool_kc.shape}")

    # Build GP label map for pooled sentences (sentence -> set of GP names)
    sent_gp_labels: Dict[str, Set[str]] = defaultdict(set)
    for gp_name, sents in gp_sents.items():
        for s in sents:
            sent_gp_labels[s].add(gp_name)

    n_labeled_pool = sum(1 for s in pool_sentences if s in sent_gp_labels)
    print(f"  Sentences with GP labels: {n_labeled_pool}")
    print(f"  Sentences without GP labels: {len(pool_sentences) - n_labeled_pool}")

    # ------------------------------------------------------------------
    # Step 3: Cluster via MiniBatchKMeans - scan k for best silhouette
    # ------------------------------------------------------------------
    print("\nClustering KC vectors (scanning k)...")
    subsample_size = min(20000, len(pool_sentences))
    random.seed(42)
    sub_idx = sorted(random.sample(range(len(pool_sentences)), subsample_size))
    sub_kc = pool_kc[sub_idx]

    k_candidates = [500, 750, 1000, 1250, 1500]
    best_k, best_sil = None, -1.0
    for k in k_candidates:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=3)
        labels = km.fit_predict(sub_kc)
        sil = silhouette_score(
            sub_kc, labels, metric="cosine", sample_size=5000, random_state=42
        )
        print(f"  k={k:5d}  silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"  Best k={best_k} (silhouette={best_sil:.4f})")
    assert best_k is not None

    # Full clustering with best k
    print(f"\nRunning full MiniBatchKMeans with k={best_k}...")
    km_final = MiniBatchKMeans(
        n_clusters=best_k, random_state=42, batch_size=2048, n_init=3
    )
    cluster_labels = km_final.fit_predict(pool_kc)

    # ------------------------------------------------------------------
    # Step 4: Rediscovery evaluation
    # ------------------------------------------------------------------
    print("\nComputing rediscovery metrics...")

    # For each cluster, find which GP's labeled sentences are most concentrated
    cluster_gp_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    cluster_total: Dict[int, int] = defaultdict(int)
    cluster_labeled: Dict[int, int] = defaultdict(int)

    for i, s in enumerate(pool_sentences):
        c = int(cluster_labels[i])
        cluster_total[c] += 1
        if s in sent_gp_labels:
            cluster_labeled[c] += 1
            for gp in sent_gp_labels[s]:
                cluster_gp_counts[c][gp] += 1

    # GP -> total labeled sentence count
    gp_total_sents: Dict[str, int] = {}
    for gp_name, sents in gp_sents.items():
        gp_total_sents[gp_name] = len(sents)

    # Cluster-GP alignment: cluster "rediscovers" a GP if >= 50% of that GP's
    # labeled sentences fall in it
    rediscovered_gps: Set[str] = set()
    cluster_majority_gp: Dict[int, Tuple[str, float]] = {}

    for c, gp_counts in cluster_gp_counts.items():
        if not gp_counts:
            continue
        majority_gp = max(gp_counts, key=lambda k, _gc=gp_counts: _gc[k])  # type: ignore[misc]
        majority_count = gp_counts[majority_gp]
        total_for_gp = gp_total_sents.get(majority_gp, 1)
        concentration = majority_count / total_for_gp
        purity = majority_count / cluster_labeled[c] if cluster_labeled[c] > 0 else 0.0
        cluster_majority_gp[c] = (majority_gp, purity)

        if concentration >= 0.50:
            rediscovered_gps.add(majority_gp)

    # Purity: for labeled sentences, fraction sharing majority GP per cluster
    purities = []
    for c, (_, pur) in cluster_majority_gp.items():
        if cluster_labeled[c] >= 3:
            purities.append(pur)
    avg_purity = np.mean(purities) if purities else 0.0

    # Recovery: fraction of GPs rediscovered
    total_gps_with_sents = len(gp_total_sents)
    recovery = (
        len(rediscovered_gps) / total_gps_with_sents
        if total_gps_with_sents > 0
        else 0.0
    )

    # ARI: build label vectors for labeled sentences only
    true_labels_ari = []
    pred_labels_ari = []
    for i, s in enumerate(pool_sentences):
        if s in sent_gp_labels:
            gps = sent_gp_labels[s]
            true_labels_ari.append(sorted(gps)[0])
            pred_labels_ari.append(int(cluster_labels[i]))

    # Map GP names to integers for ARI
    gp_name_to_int = {g: idx for idx, g in enumerate(sorted(set(true_labels_ari)))}
    true_int = [gp_name_to_int[g] for g in true_labels_ari]
    ari = adjusted_rand_score(true_int, pred_labels_ari)

    print(f"  Avg purity (clusters with >=3 labeled): {avg_purity:.3f}")
    print(
        f"  Recovery: {len(rediscovered_gps)}/{total_gps_with_sents} "
        f"({recovery:.1%}) GPs rediscovered"
    )
    print(f"  Adjusted Rand Index: {ari:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Build GP fingerprints for novelty comparison
    # ------------------------------------------------------------------
    print("\nBuilding GP fingerprints...")
    gp_fingerprints: Dict[str, np.ndarray] = {}
    for gp_name, sents in gp_sents.items():
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) >= 3:
            gp_fingerprints[gp_name] = kc_matrix[indices].mean(axis=0)
    gp_fp_names = list(gp_fingerprints.keys())
    gp_fp_matrix = np.stack([gp_fingerprints[g] for g in gp_fp_names])
    gp_fp_norms = np.linalg.norm(gp_fp_matrix, axis=1, keepdims=True)
    gp_fp_norms = np.where(gp_fp_norms > 1e-8, gp_fp_norms, 1.0)
    gp_fp_normed = gp_fp_matrix / gp_fp_norms

    # Baseline KC firing rate across the whole pool
    baseline_rate = pool_kc.mean(axis=0)

    # ------------------------------------------------------------------
    # Step 6: Characterize novel clusters
    # ------------------------------------------------------------------
    print("\nCharacterizing clusters...")
    novel_candidates: List[Dict[str, Any]] = []
    known_clusters: List[Dict[str, Any]] = []

    for c in range(best_k):
        members = np.where(cluster_labels == c)[0]
        if len(members) < 5:
            continue

        n_labeled_in_cluster = sum(
            1 for i in members if pool_sentences[i] in sent_gp_labels
        )
        n_total_in_cluster = len(members)

        cluster_kc_mean = pool_kc[members].mean(axis=0)
        cluster_norm = np.linalg.norm(cluster_kc_mean)
        if cluster_norm < 1e-8:
            continue
        cluster_normed = cluster_kc_mean / cluster_norm

        # Max cosine similarity to any known GP fingerprint
        sims_to_gps = gp_fp_normed @ cluster_normed
        max_sim = float(sims_to_gps.max())
        nearest_gp_idx = int(sims_to_gps.argmax())
        nearest_gp = gp_fp_names[nearest_gp_idx]

        # Distinctive KCs: highest lift over baseline
        cluster_rate = pool_kc[members].mean(axis=0)
        lift = np.where(
            baseline_rate > 0.001,
            cluster_rate / baseline_rate,
            np.where(cluster_rate > 0, 999.0, 0.0),
        )
        top_kc_indices = np.argsort(-lift)[:5]
        top_kcs = [
            (int(k), float(lift[k]), float(cluster_rate[k]))
            for k in top_kc_indices
            if lift[k] > 1.5
        ]

        # Determine if this is a "known" cluster or novel
        majority_gp_info = cluster_majority_gp.get(c)
        majority_purity = majority_gp_info[1] if majority_gp_info else 0.0

        # Pull example sentences
        example_idx = random.sample(list(members), min(15, len(members)))
        examples = [pool_sentences[i] for i in example_idx]

        # Cross-check: how many distinct GPs appear in this cluster?
        gps_in_cluster = set()
        for i in members:
            s = pool_sentences[i]
            if s in sent_gp_labels:
                gps_in_cluster.update(sent_gp_labels[s])

        info = {
            "cluster_id": c,
            "size": n_total_in_cluster,
            "n_labeled": n_labeled_in_cluster,
            "max_sim_to_gp": max_sim,
            "nearest_gp": nearest_gp,
            "majority_purity": majority_purity,
            "top_kcs": top_kcs,
            "examples": examples,
            "gps_in_cluster": sorted(gps_in_cluster),
            "n_distinct_gps": len(gps_in_cluster),
        }

        if max_sim < 0.65 and majority_purity < 0.4:
            info["confidence"] = "HIGH"
            novel_candidates.append(info)
        elif max_sim < 0.80 and majority_purity < 0.5:
            info["confidence"] = "POSSIBLE_GAP"
            novel_candidates.append(info)
        elif majority_purity >= 0.5 or max_sim >= 0.80:
            known_clusters.append(info)

    # Sort novel candidates by confidence then by size
    confidence_order = {"HIGH": 0, "POSSIBLE_GAP": 1}
    novel_candidates.sort(
        key=lambda x: (confidence_order[str(x["confidence"])], -int(x["size"]))
    )

    # ------------------------------------------------------------------
    # Step 7: Build report
    # ------------------------------------------------------------------
    print(f"\nFound {len(novel_candidates)} novel cluster candidates")
    lines = []
    lines.append("=" * 70)
    lines.append("DISCOVER NEW GRAMMAR POINTS FROM KC PATTERNS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("METHOD")
    lines.append("-" * 70)
    lines.append(
        f"Pooled {len(all_sentences)} labeled (corpus_gp_pos) + "
        f"{len(corpus_sample)} unlabeled corpus sentences "
        f"= {len(pool_sentences)} total"
    )
    lines.append(
        f"Clustering: MiniBatchKMeans, k={best_k} (best silhouette={best_sil:.4f})"
    )
    lines.append(f"Sentences with GP labels: {n_labeled_pool}")
    lines.append(f"Sentences unlabeled (PNU): {len(pool_sentences) - n_labeled_pool}")
    lines.append("")

    lines.append("REDISCOVERY METRICS (validation gate)")
    lines.append("-" * 70)
    lines.append(f"Average cluster purity:  {avg_purity:.3f}")
    lines.append(
        f"GP recovery:             {len(rediscovered_gps)}/{total_gps_with_sents} "
        f"({recovery:.1%})"
    )
    lines.append(f"Adjusted Rand Index:     {ari:.4f}")
    lines.append("")
    lines.append("Interpretation:")
    if recovery >= 0.60:
        lines.append("  Recovery >= 60% — clustering is grammatically meaningful.")
        lines.append("  Novel clusters can be trusted as genuine discoveries.")
    elif recovery >= 0.40:
        lines.append("  Recovery 40-60% — moderate grammatical alignment.")
        lines.append("  Novel clusters should be treated cautiously.")
    else:
        lines.append("  Recovery < 40% — clustering poorly aligns with grammar.")
        lines.append("  Novel clusters may not be reliable.")
    lines.append("")

    # Cross-cutting patterns: clusters where multiple distinct GPs converge
    cross_cutting: List[Dict[str, Any]] = [
        c for c in novel_candidates if c["n_distinct_gps"] >= 3
    ]
    if cross_cutting:
        lines.append("CROSS-CUTTING PATTERNS (multiple GPs converge)")
        lines.append("-" * 70)
        lines.append(
            f"Found {len(cross_cutting)} clusters where 3+ known GPs co-occur,"
        )
        lines.append("suggesting higher-level grammatical patterns:")
        lines.append("")
        for cc_info in cross_cutting[:10]:
            lines.append(
                f"  Cluster {cc_info['cluster_id']} "
                f"(size={cc_info['size']}, {cc_info['n_distinct_gps']} GPs)"
            )
            gps_list: list[str] = cc_info["gps_in_cluster"]
            lines.append(f"    GPs: {', '.join(gps_list[:8])}")
            cc_top_kcs: list[tuple[int, float, float]] = cc_info["top_kcs"]
            if cc_top_kcs:
                kc_str = ", ".join(
                    f"KC{k}(lift={lift:.1f})" for k, lift, _ in cc_top_kcs[:3]
                )
                lines.append(f"    Distinctive KCs: {kc_str}")
            lines.append("    Examples:")
            examples_list: list[str] = cc_info["examples"]
            for ex in examples_list[:5]:
                lines.append(f"      {ex}")
            lines.append("")

    high_conf: List[Dict[str, Any]] = [
        c for c in novel_candidates if c["confidence"] == "HIGH"
    ]
    possible_gap: List[Dict[str, Any]] = [
        c for c in novel_candidates if c["confidence"] == "POSSIBLE_GAP"
    ]

    lines.append(f"NOVEL CLUSTER CANDIDATES: {len(novel_candidates)} total")
    lines.append(f"  High confidence: {len(high_conf)}")
    lines.append(f"  Possible gap:    {len(possible_gap)}")
    lines.append("")

    if high_conf:
        lines.append("HIGH CONFIDENCE CANDIDATES")
        lines.append("-" * 70)
        for hc_info in high_conf[:20]:
            lines.append(
                f"Cluster {hc_info['cluster_id']}  "
                f"size={hc_info['size']}  labeled={hc_info['n_labeled']}  "
                f"nearest_gp={hc_info['nearest_gp']} "
                f"(sim={hc_info['max_sim_to_gp']:.3f})  "
                f"purity={hc_info['majority_purity']:.2f}"
            )
            hc_top_kcs: list[tuple[int, float, float]] = hc_info["top_kcs"]
            if hc_top_kcs:
                kc_str = ", ".join(
                    f"KC{k} (lift={lift:.1f}, rate={r:.2f})"
                    for k, lift, r in hc_top_kcs
                )
                lines.append(f"  Distinctive KCs: {kc_str}")
            hc_gps: list[str] = hc_info["gps_in_cluster"]
            if hc_gps:
                lines.append(f"  GPs present: {', '.join(hc_gps[:8])}")
            lines.append("  Examples:")
            hc_examples: list[str] = hc_info["examples"]
            for ex in hc_examples[:10]:
                lines.append(f"    {ex}")
            lines.append("")

    if possible_gap:
        lines.append("POSSIBLE GAP CANDIDATES")
        lines.append("-" * 70)
        for pg_info in possible_gap[:20]:
            lines.append(
                f"Cluster {pg_info['cluster_id']}  "
                f"size={pg_info['size']}  labeled={pg_info['n_labeled']}  "
                f"nearest_gp={pg_info['nearest_gp']} "
                f"(sim={pg_info['max_sim_to_gp']:.3f})  "
                f"purity={pg_info['majority_purity']:.2f}"
            )
            pg_top_kcs: list[tuple[int, float, float]] = pg_info["top_kcs"]
            if pg_top_kcs:
                kc_str = ", ".join(
                    f"KC{k} (lift={lift:.1f}, rate={r:.2f})"
                    for k, lift, r in pg_top_kcs
                )
                lines.append(f"  Distinctive KCs: {kc_str}")
            pg_gps: list[str] = pg_info["gps_in_cluster"]
            if pg_gps:
                lines.append(f"  GPs present: {', '.join(pg_gps[:8])}")
            lines.append("  Examples:")
            pg_examples: list[str] = pg_info["examples"]
            for ex in pg_examples[:8]:
                lines.append(f"    {ex}")
            lines.append("")

    # Summary of rediscovered clusters (top examples)
    known_clusters.sort(key=lambda x: -x["majority_purity"])
    lines.append("TOP REDISCOVERED CLUSTERS (validation)")
    lines.append("-" * 70)
    for info in known_clusters[:15]:
        lines.append(
            f"Cluster {info['cluster_id']}  "
            f"size={info['size']}  purity={info['majority_purity']:.2f}  "
            f"nearest_gp={info['nearest_gp']} (sim={info['max_sim_to_gp']:.3f})"
        )
    lines.append("")

    report = "\n".join(lines)
    print("\n" + report)

    output_path = "semantics/discovered-grammar-points.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")


def _regress_out_register(
    all_sentences: list,
    kc_matrix: np.ndarray,
    sent_to_idx: dict,
    db_path: str = "data/corpus.db",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Remove formality/gender/register signal from KC vectors via OLS.

    Builds a design matrix of [formality, gender, register one-hots] and
    regresses each KC column on it.  Returns the residual KC matrix (same
    shape as input, full 1024 dims preserved) plus diagnostic lines.
    """
    # pylint: disable=too-many-locals
    from kotogram.constants import REGISTER_ID_TO_LABEL

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id FROM register ORDER BY id")
    register_ids_list = [row[0] for row in cur.fetchall()]
    n_registers = len(register_ids_list)

    cur.execute(
        "SELECT sentence, formality, gender, register_ids "
        "FROM sentences WHERE grammatic = 1"
    )
    label_map = {}
    for sentence, formality, gender, reg_ids_str in cur.fetchall():
        reg_set = set()
        if reg_ids_str:
            for tok in reg_ids_str.split(","):
                tok = tok.strip()
                if tok:
                    reg_set.add(int(tok))
        label_map[sentence] = (formality, gender, reg_set)
    conn.close()

    n_sent = len(all_sentences)

    # Build design matrix: [formality, gender, register_0 .. register_13]
    # Use 0 for missing formality/gender (most sentences have labels)
    n_features = 2 + n_registers
    design_mat = np.zeros((n_sent, n_features), dtype=np.float64)
    has_label = np.zeros(n_sent, dtype=bool)

    for sent in all_sentences:
        if sent not in sent_to_idx or sent not in label_map:
            continue
        idx = sent_to_idx[sent]
        formality, gender, reg_set = label_map[sent]
        has_label[idx] = True
        if formality is not None:
            design_mat[idx, 0] = formality
        if gender is not None:
            design_mat[idx, 1] = gender
        for rid in reg_set:
            if rid in register_ids_list:
                col = register_ids_list.index(rid)
                design_mat[idx, 2 + col] = 1.0

    n_labeled = int(has_label.sum())

    # Drop constant columns (registers with no examples in our sentence set)
    col_std = design_mat[has_label].std(axis=0)
    active_cols = np.where(col_std > 1e-8)[0]
    x_active = design_mat[:, active_cols]

    feature_names = ["formality", "gender"] + [
        REGISTER_ID_TO_LABEL[register_ids_list[i]].name
        if register_ids_list[i] in REGISTER_ID_TO_LABEL
        else f"reg_{register_ids_list[i]}"
        for i in range(n_registers)
    ]
    active_names = [feature_names[c] for c in active_cols]

    # Fit OLS on labeled rows:  KC_col ~ x_active  (with intercept)
    x_fit = x_active[has_label]
    x_fit = np.column_stack([np.ones(n_labeled), x_fit])  # add intercept

    # Solve  (X'X)^-1 X' Y  for all KC columns at once
    y_fit = kc_matrix[has_label].astype(np.float64)
    xt_x = x_fit.T @ x_fit
    xt_y = x_fit.T @ y_fit
    beta = np.linalg.lstsq(xt_x, xt_y, rcond=None)[0]

    # Predict for ALL sentences (including unlabeled, where X is zero → prediction = intercept)
    x_all = np.column_stack([np.ones(n_sent), x_active])
    predicted = x_all @ beta  # (n_sent, n_kc)
    residual = kc_matrix.astype(np.float64) - predicted

    # Compute R² per KC to report how much variance was explained
    ss_tot = ((y_fit - y_fit.mean(axis=0)) ** 2).sum(axis=0)
    ss_res = ((y_fit - x_fit @ beta) ** 2).sum(axis=0)
    r_squared = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, 0.0)

    top_r2_idx = np.argsort(-r_squared)[:20]

    # Compute per-feature beta magnitudes for top KCs
    # beta[0] is intercept, beta[1:] are feature coefficients
    feature_beta = beta[1:]  # (n_active_features, n_kc)

    diag = []
    diag.append(f"Labeled sentences used for regression: {n_labeled}/{n_sent}")
    diag.append(
        f"Design matrix features: {len(active_names)} ({', '.join(active_names)})"
    )
    diag.append(f"Median R² across KCs: {float(np.median(r_squared)):.4f}")
    diag.append(f"Mean R² across KCs: {float(np.mean(r_squared)):.4f}")
    diag.append(f"KCs with R² > 0.10: {int((r_squared > 0.10).sum())}")
    diag.append(f"KCs with R² > 0.25: {int((r_squared > 0.25).sum())}")
    diag.append(f"KCs with R² > 0.50: {int((r_squared > 0.50).sum())}")
    diag.append("Top KCs by register variance explained (R²):")
    for kc_i in top_r2_idx:
        # Find the dominant feature for this KC
        abs_betas = np.abs(feature_beta[:, kc_i])
        top_feat_idx = np.argsort(-abs_betas)[:3]
        feat_str = ", ".join(
            f"{active_names[fi]}(β={float(feature_beta[fi, kc_i]):+.3f})"
            for fi in top_feat_idx
            if abs_betas[fi] > 0.01
        )
        diag.append(f"  KC{kc_i}: R²={float(r_squared[kc_i]):.3f}  [{feat_str}]")

    return residual.astype(np.float32), r_squared, diag


def main_find_nuance_divisions(ctx: dict) -> None:
    """Find GPs whose sentences split into distinct KC sub-clusters."""
    # pylint: disable=too-many-locals
    import math

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    gp_sents = ctx["gp_sents"]
    kc_matrix_raw = ctx["kc_matrix"]
    all_sentences = ctx["all_sentences"]
    sent_to_idx = ctx["sent_to_idx"]
    db_path = ctx.get("db_path", "data/corpus.db")

    min_sentences = 20

    r2_mask_threshold = 0.10

    # ------------------------------------------------------------------
    # Regress out formality/gender/register, then mask high-R² KCs
    # ------------------------------------------------------------------
    print("Regressing out formality/gender/register from KC vectors...")
    kc_residual, r_squared, register_diag = _regress_out_register(
        all_sentences, kc_matrix_raw, sent_to_idx, db_path=db_path
    )
    for line in register_diag:
        print(f"  {line}")

    high_r2_kcs = set(int(i) for i in np.where(r_squared > r2_mask_threshold)[0])
    keep_cols = sorted(set(range(kc_residual.shape[1])) - high_r2_kcs)
    kc_matrix = kc_residual[:, keep_cols]
    print(f"  Additionally masked {len(high_r2_kcs)} KCs with R²>{r2_mask_threshold}")
    print(f"  Final KC dimensions for clustering: {len(keep_cols)}")
    register_diag.append(f"High-R² KC mask threshold: R² > {r2_mask_threshold}")
    register_diag.append(f"KCs masked (R² too high): {len(high_r2_kcs)}")
    register_diag.append(f"Final KC dimensions for clustering: {len(keep_cols)}")
    print()

    # ------------------------------------------------------------------
    # Build GP fingerprints for cross-reference
    # ------------------------------------------------------------------
    gp_fingerprints: Dict[str, np.ndarray] = {}
    for gp_name, sents in gp_sents.items():
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) >= 3:
            gp_fingerprints[gp_name] = kc_matrix[indices].mean(axis=0)
    gp_fp_names = list(gp_fingerprints.keys())
    gp_fp_matrix = np.stack([gp_fingerprints[g] for g in gp_fp_names])
    gp_fp_norms = np.linalg.norm(gp_fp_matrix, axis=1, keepdims=True)
    gp_fp_norms = np.where(gp_fp_norms > 1e-8, gp_fp_norms, 1.0)
    gp_fp_normed = gp_fp_matrix / gp_fp_norms

    # ------------------------------------------------------------------
    # Step 1 & 2: Sub-cluster each GP and score the split
    # ------------------------------------------------------------------
    print("Sub-clustering each GP's sentences...")
    candidates: List[Dict[str, Any]] = []
    n_eligible = 0

    for gp_name, sents in sorted(gp_sents.items()):
        indices = [sent_to_idx[s] for s in sents if s in sent_to_idx]
        if len(indices) < min_sentences:
            continue
        n_eligible += 1

        gp_kc = kc_matrix[indices]
        gp_sentences = [all_sentences[i] for i in indices]

        best_k, best_sil, best_labels = None, -1.0, None
        for k in range(2, min(6, len(indices) // 5 + 1)):
            if k < 2:
                continue
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            labels = km.fit_predict(gp_kc)
            cluster_sizes = [int((labels == c).sum()) for c in range(k)]
            if min(cluster_sizes) < max(3, int(0.15 * len(indices))):
                continue
            if len(set(labels.tolist())) < 2:
                continue
            sil = silhouette_score(gp_kc, labels, metric="cosine")
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels

        if best_labels is None or best_sil <= 0.0:
            continue
        assert best_k is not None

        sub_means = []
        sub_sizes = []
        for c in range(best_k):
            mask = best_labels == c
            sub_means.append(gp_kc[mask].mean(axis=0))
            sub_sizes.append(int(mask.sum()))

        # KC divergence: count KCs where sub-cluster means differ significantly
        # Use pooled std as scale; flag KCs with |diff| > 1.5 * pooled_std
        gp_std = gp_kc.std(axis=0)
        gp_std_safe = np.where(gp_std > 1e-4, gp_std, 1.0)
        n_divergent = 0
        for i in range(best_k):
            for j in range(i + 1, best_k):
                diff = np.abs(sub_means[i] - sub_means[j])
                n_divergent += int((diff > 1.5 * gp_std_safe).sum())

        balance = min(sub_sizes) / max(sub_sizes)
        score = best_sil * math.log(1 + n_divergent) * balance

        candidates.append(
            {
                "gp_name": gp_name,
                "n_sentences": len(indices),
                "best_k": best_k,
                "silhouette": best_sil,
                "n_divergent_kcs": n_divergent,
                "balance": balance,
                "score": score,
                "sub_means": sub_means,
                "sub_sizes": sub_sizes,
                "labels": best_labels,
                "gp_sentences": gp_sentences,
                "gp_kc": gp_kc,
            }
        )

    print(f"  Eligible GPs (>= {min_sentences} sentences): {n_eligible}")
    print(f"  GPs with meaningful splits: {len(candidates)}")

    candidates.sort(key=lambda x: -x["score"])

    # ------------------------------------------------------------------
    # Step 3: Characterize top candidates
    # ------------------------------------------------------------------
    print("\nCharacterizing top candidates...")

    lines = []
    lines.append("=" * 70)
    lines.append("MISSING NUANCE DIVISIONS: INTRA-GP KC CLUSTERING")
    lines.append("=" * 70)
    lines.append("")
    lines.append("METHOD")
    lines.append("-" * 70)
    lines.append(f"Sub-clustered {n_eligible} GPs (>= {min_sentences} sentences each)")
    lines.append("Tried k=2..5, picked best silhouette, required 15% balance")
    lines.append("Scoring: silhouette * log(1 + divergent_KCs) * balance")
    lines.append(f"GPs with meaningful splits (silhouette > 0): {len(candidates)}")
    lines.append("")
    lines.append("REGISTER DECONFOUNDING (OLS REGRESSION)")
    lines.append("-" * 70)
    for dl in register_diag:
        lines.append(dl)
    lines.append("")

    # ------------------------------------------------------------------
    # Step 4: Validation -- check known を splits
    # ------------------------------------------------------------------
    wo_gps: List[Dict[str, Any]] = [
        cand for cand in candidates if cand["gp_name"].startswith("を")
    ]
    if wo_gps:
        lines.append("VALIDATION: を-FAMILY SPLITS")
        lines.append("-" * 70)
        for wo_info in wo_gps[:10]:
            lines.append(
                f"  {wo_info['gp_name']}: k={wo_info['best_k']}, "
                f"sil={wo_info['silhouette']:.3f}, "
                f"divergent_KCs={wo_info['n_divergent_kcs']}, "
                f"score={wo_info['score']:.3f}"
            )
        lines.append("")

    # Detailed characterization of top candidates
    lines.append(f"TOP {min(30, len(candidates))} CANDIDATES")
    lines.append("=" * 70)
    lines.append("")

    for info in candidates[:30]:
        gp_name = info["gp_name"]
        lines.append(f"GP: {gp_name}")
        lines.append(
            f"  Sentences: {info['n_sentences']}  Sub-clusters: {info['best_k']}  "
            f"Silhouette: {info['silhouette']:.3f}  "
            f"Divergent KCs: {info['n_divergent_kcs']}  "
            f"Score: {info['score']:.3f}"
        )
        lines.append(f"  Sub-cluster sizes: {info['sub_sizes']}")

        for c_idx in range(info["best_k"]):
            mask = info["labels"] == c_idx
            sub_kc = info["gp_kc"][mask]
            sub_rate = sub_kc.mean(axis=0)

            # Distinctive KCs for this sub-cluster vs the GP overall
            gp_rate = info["gp_kc"].mean(axis=0)
            safe_gp = np.where(np.abs(gp_rate) > 0.01, np.abs(gp_rate), 0.01)
            diff_vs_gp = np.abs(sub_rate - gp_rate) / safe_gp
            top_kc_idx = np.argsort(-diff_vs_gp)[:5]
            top_kcs = [
                (keep_cols[int(k)], float(diff_vs_gp[k]), float(sub_rate[k]))
                for k in top_kc_idx
                if diff_vs_gp[k] > 0.3
            ]

            # Nearest GP for this sub-cluster
            sub_mean = info["sub_means"][c_idx]
            sub_norm = np.linalg.norm(sub_mean)
            if sub_norm > 1e-8:
                sub_normed = sub_mean / sub_norm
                sims = gp_fp_normed @ sub_normed
                # Exclude self
                self_idx = gp_fp_names.index(gp_name) if gp_name in gp_fp_names else -1
                if self_idx >= 0:
                    sims[self_idx] = -1.0
                nearest_idx = int(sims.argmax())
                nearest_gp = gp_fp_names[nearest_idx]
                nearest_sim = float(sims[nearest_idx])
            else:
                nearest_gp = "N/A"
                nearest_sim = 0.0

            sub_sentences = [
                info["gp_sentences"][i]
                for i in range(len(info["labels"]))
                if info["labels"][i] == c_idx
            ]
            random.seed(42 + c_idx)
            examples = random.sample(sub_sentences, min(8, len(sub_sentences)))

            lines.append(
                f"  --- Sub-cluster {c_idx + 1} ({info['sub_sizes'][c_idx]} sentences) ---"
            )
            lines.append(f"    Nearest other GP: {nearest_gp} (sim={nearest_sim:.3f})")
            if top_kcs:
                kc_str = ", ".join(
                    f"KC{k}(div={d:.1f}, val={v:+.2f})" for k, d, v in top_kcs
                )
                lines.append(f"    Distinctive KCs (vs GP mean): {kc_str}")
            lines.append("    Examples:")
            for ex in examples:
                lines.append(f"      {ex}")

        lines.append("")

    # Summary table
    lines.append("FULL RANKING (score > 0)")
    lines.append("-" * 70)
    lines.append(
        f"{'Rank':>4}  {'Score':>7}  {'Sil':>5}  {'k':>1}  {'DivKC':>5}  {'Bal':>4}  "
        f"{'#Sent':>5}  GP"
    )
    for rank, info in enumerate(candidates, 1):
        lines.append(
            f"{rank:4d}  {info['score']:7.3f}  {info['silhouette']:5.3f}  "
            f"{info['best_k']:1d}  {info['n_divergent_kcs']:5d}  "
            f"{info['balance']:4.2f}  {info['n_sentences']:5d}  {info['gp_name']}"
        )

    report = "\n".join(lines)
    print("\n" + report)

    output_path = "semantics/nuance-divisions.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")


def main_length_bias(db_path: str = "data/corpus.db") -> None:
    """Investigate KC activation patterns vs sentence length."""
    # pylint: disable=too-many-locals
    print("=" * 70)
    print("KC LENGTH BIAS INVESTIGATION")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Load all grammatical sentences from corpus.db
    # ------------------------------------------------------------------
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT sentence FROM sentences WHERE grammatic = 1").fetchall()
    all_sents = [r[0] for r in rows]
    print(f"Loaded {len(all_sents)} grammatical sentences from corpus.db")

    gp_rows = conn.execute(
        "SELECT g.name, p.sentence "
        "FROM corpus_gp_pos p JOIN grammar g ON g.id = p.gp_id"
    ).fetchall()
    conn.close()

    gp_sents: Dict[str, List[str]] = defaultdict(list)
    sent_gps: Dict[str, List[str]] = defaultdict(list)
    for gp_name, sent in gp_rows:
        gp_sents[gp_name].append(sent)
        sent_gps[sent].append(gp_name)

    random.seed(42)
    if len(all_sents) > 50000:
        all_sents = random.sample(all_sents, 50000)
    print(f"Using {len(all_sents)} sentences for analysis")

    # ------------------------------------------------------------------
    # Run inference (binary + raw probabilities)
    # ------------------------------------------------------------------
    print("\nLoading model...")
    model, tokenizer = _ANALYZER.load()
    model.eval()
    jp_parser = SudachiJapaneseParser()
    threshold = float(model.config.kc_threshold)

    print(f"Running batch KC inference (threshold={threshold:.3f})...")
    kc_binary, kc_probs = batch_infer_kc_with_probs(
        all_sents, model, tokenizer, jp_parser, batch_size=64, threshold=threshold
    )
    print(f"  Shape: {kc_binary.shape}")

    lengths = np.array([len(s) for s in all_sents])
    kc_counts = kc_binary.sum(axis=1)

    # ------------------------------------------------------------------
    # Define bins
    # ------------------------------------------------------------------
    bin_defs = [
        ("1-3", 1, 3),
        ("4-7", 4, 7),
        ("8-15", 8, 15),
        ("16-31", 16, 31),
        ("32+", 32, 9999),
    ]

    def bin_mask(lo: int, hi: int) -> np.ndarray:
        return (lengths >= lo) & (lengths <= hi)  # type: ignore[no-any-return]

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("KC LENGTH BIAS INVESTIGATION")
    lines.append("=" * 70)
    lines.append(f"Model threshold (adaptive): {threshold:.3f}")
    lines.append(f"Sentences analyzed: {len(all_sents)}")
    lines.append("")

    # ==================================================================
    # Analysis 1: KC Count vs Length
    # ==================================================================
    print("\nAnalysis 1: KC count vs length bins...")
    lines.append(f"ANALYSIS 1: KC COUNT vs SENTENCE LENGTH (K@{threshold:.2f})")
    lines.append("-" * 70)
    lines.append("")

    header = (
        f"{'Bin':>6}  {'N':>7}  {'MeanLen':>7}  {'MeanKC':>6}  {'P25':>4}  "
        f"{'Med':>4}  {'P75':>4}  {'MeanProb':>8}  {'MedProb':>7}"
    )
    lines.append(header)

    for label, lo, hi in bin_defs:
        mask = bin_mask(lo, hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        bin_lengths = lengths[mask]
        bin_kc_counts = kc_counts[mask]
        bin_probs = kc_probs[mask]

        mean_prob = float(bin_probs.mean())
        median_prob = float(np.median(bin_probs))

        p25 = int(np.percentile(bin_kc_counts, 25))
        med = int(np.median(bin_kc_counts))
        p75 = int(np.percentile(bin_kc_counts, 75))

        lines.append(
            f"{label:>6}  {n_bin:>7}  {bin_lengths.mean():>7.1f}  "
            f"{bin_kc_counts.mean():>6.1f}  {p25:>4}  {med:>4}  {p75:>4}  "
            f"{mean_prob:>8.4f}  {median_prob:>7.4f}"
        )

    total_mask = np.ones(len(lengths), dtype=bool)
    lines.append(
        f"{'Total':>6}  {int(total_mask.sum()):>7}  {lengths.mean():>7.1f}  "
        f"{kc_counts.mean():>6.1f}  "
        f"{int(np.percentile(kc_counts, 25)):>4}  "
        f"{int(np.median(kc_counts)):>4}  "
        f"{int(np.percentile(kc_counts, 75)):>4}  "
        f"{float(kc_probs.mean()):>8.4f}  "
        f"{float(np.median(kc_probs)):>7.4f}"
    )
    lines.append("")
    lines.append(
        f"MeanProb = mean raw sigmoid output (before {threshold:.2f} threshold)"
    )
    lines.append("Higher MeanProb for short sentences suggests the encoder is more")
    lines.append(
        "'confident' on short inputs, not necessarily that more KCs are meaningful."
    )
    lines.append("")

    # ==================================================================
    # Analysis 2: Length-Correlated KCs
    # ==================================================================
    print("Analysis 2: Length-correlated KCs...")
    lines.append("ANALYSIS 2: LENGTH-CORRELATED KCs")
    lines.append("-" * 70)
    lines.append("")

    n_kcs = kc_binary.shape[1]
    correlations = np.zeros(n_kcs)
    for k in range(n_kcs):
        col = kc_binary[:, k]
        if col.std() < 1e-8:
            continue
        correlations[k] = float(np.corrcoef(col, lengths)[0, 1])

    short_biased_idx = np.argsort(correlations)[:20]
    long_biased_idx = np.argsort(-correlations)[:10]

    lines.append("Top 20 KCs NEGATIVELY correlated with length (fire more on SHORT):")
    lines.append(
        f"  {'KC':>5}  {'r':>7}  {'FireRate':>8}  {'ShortRate':>9}  {'LongRate':>8}  Top GPs"
    )
    short_mask = bin_mask(1, 7)
    long_mask = bin_mask(16, 9999)
    for ki in short_biased_idx:
        r_val = correlations[ki]
        fire_rate = float(kc_binary[:, ki].mean())
        short_rate = (
            float(kc_binary[short_mask, ki].mean()) if short_mask.sum() > 0 else 0
        )
        long_rate = float(kc_binary[long_mask, ki].mean()) if long_mask.sum() > 0 else 0

        gp_counts: Dict[str, int] = defaultdict(int)
        active_sents = [
            all_sents[i] for i in range(len(all_sents)) if kc_binary[i, ki] > 0.5
        ]
        for sent in active_sents[:500]:
            for gp in sent_gps.get(sent, []):
                gp_counts[gp] += 1
        top_gp_str = ", ".join(
            f"{g[:25]}({c})"
            for g, c in sorted(gp_counts.items(), key=lambda x: -x[1])[:3]
        )
        lines.append(
            f"  KC{ki:>3}  {r_val:>7.3f}  {fire_rate:>8.3f}  {short_rate:>9.3f}  "
            f"{long_rate:>8.3f}  {top_gp_str}"
        )

    lines.append("")
    lines.append("Top 10 KCs POSITIVELY correlated with length (fire more on LONG):")
    lines.append(
        f"  {'KC':>5}  {'r':>7}  {'FireRate':>8}  {'ShortRate':>9}  {'LongRate':>8}"
    )
    for ki in long_biased_idx:
        r_val = correlations[ki]
        fire_rate = float(kc_binary[:, ki].mean())
        short_rate = (
            float(kc_binary[short_mask, ki].mean()) if short_mask.sum() > 0 else 0
        )
        long_rate = float(kc_binary[long_mask, ki].mean()) if long_mask.sum() > 0 else 0
        lines.append(
            f"  KC{ki:>3}  {r_val:>7.3f}  {fire_rate:>8.3f}  {short_rate:>9.3f}  "
            f"{long_rate:>8.3f}"
        )
    lines.append("")

    # ==================================================================
    # Analysis 3: Short-Sentence KC Exclusivity
    # ==================================================================
    print("Analysis 3: Short-exclusive KCs...")
    lines.append("ANALYSIS 3: SHORT-SENTENCE KC EXCLUSIVITY")
    lines.append("-" * 70)
    lines.append("")
    lines.append("KCs firing >=50% on short (1-7 chars) but <10% on long (16+ chars):")
    lines.append("")

    short_rates = (
        kc_binary[short_mask].mean(axis=0) if short_mask.sum() > 0 else np.zeros(n_kcs)
    )
    long_rates = (
        kc_binary[long_mask].mean(axis=0) if long_mask.sum() > 0 else np.zeros(n_kcs)
    )
    exclusive_kcs = [
        k for k in range(n_kcs) if short_rates[k] >= 0.50 and long_rates[k] < 0.10
    ]

    if exclusive_kcs:
        lines.append(f"Found {len(exclusive_kcs)} short-exclusive KCs:")
        for ki in sorted(exclusive_kcs, key=lambda k: -short_rates[k]):
            active_indices = [
                i for i in range(len(all_sents)) if kc_binary[i, ki] > 0.5
            ]
            active_sents_ki = [all_sents[i] for i in active_indices]
            short_active = [s for s in active_sents_ki if len(s) <= 7]
            long_active = [s for s in active_sents_ki if len(s) >= 16]

            gp_counts_ki: Dict[str, int] = defaultdict(int)
            for sent in active_sents_ki[:1000]:
                for gp in sent_gps.get(sent, []):
                    gp_counts_ki[gp] += 1
            top_gps_ki = sorted(gp_counts_ki.items(), key=lambda x: -x[1])[:5]

            lines.append(
                f"  KC{ki}: short_rate={short_rates[ki]:.2f}, "
                f"long_rate={long_rates[ki]:.3f}, "
                f"total_fires={len(active_indices)}"
            )
            if top_gps_ki:
                gp_str = ", ".join(f"{g}({c})" for g, c in top_gps_ki)
                lines.append(f"    GPs: {gp_str}")
            random.seed(ki)
            short_ex = random.sample(short_active, min(3, len(short_active)))
            lines.append(f"    Short examples: {short_ex}")
            if long_active:
                long_ex = random.sample(long_active, min(3, len(long_active)))
                lines.append(f"    Long examples:  {long_ex}")
            else:
                lines.append("    Long examples:  (none)")
            lines.append("")
    else:
        lines.append("  No KCs found with >=50% short rate and <10% long rate.")
        lines.append("  Trying relaxed threshold: >=30% short, <15% long:")
        relaxed = [
            k for k in range(n_kcs) if short_rates[k] >= 0.30 and long_rates[k] < 0.15
        ]
        lines.append(f"  Found {len(relaxed)} KCs with relaxed criteria.")
        for ki in sorted(relaxed, key=lambda k: -short_rates[k])[:10]:
            lines.append(
                f"    KC{ki}: short_rate={short_rates[ki]:.2f}, "
                f"long_rate={long_rates[ki]:.3f}"
            )
    lines.append("")

    # ==================================================================
    # Analysis 4: Shared KC Substrate Test
    # ==================================================================
    print("Analysis 4: Shared KC substrate / Jaccard overlap...")
    lines.append("ANALYSIS 4: SHARED KC SUBSTRATE (JACCARD OVERLAP)")
    lines.append("-" * 70)
    lines.append("")

    bin_active_kc_sets: Dict[str, set] = {}
    bin_top50_kc_sets: Dict[str, np.ndarray] = {}
    for label, lo, hi in bin_defs:
        mask = bin_mask(lo, hi)
        if mask.sum() == 0:
            continue
        bin_rates = kc_binary[mask].mean(axis=0)
        active_set = set(int(k) for k in range(n_kcs) if bin_rates[k] > 0.05)
        bin_active_kc_sets[label] = active_set
        top50 = np.argsort(-bin_rates)[:50]
        bin_top50_kc_sets[label] = top50

    lines.append("Jaccard overlap of active KC sets (fire rate > 5%) between bins:")
    bin_labels = [label for label, _, _ in bin_defs if label in bin_active_kc_sets]
    header_j = "          " + "  ".join(f"{b:>6}" for b in bin_labels)
    lines.append(header_j)
    for b1 in bin_labels:
        row_vals = []
        for b2 in bin_labels:
            s1 = bin_active_kc_sets[b1]
            s2 = bin_active_kc_sets[b2]
            if len(s1 | s2) == 0:
                row_vals.append("  -.--")
            else:
                jaccard = len(s1 & s2) / len(s1 | s2)
                row_vals.append(f"  {jaccard:.2f}")
        lines.append(f"{b1:>8}  " + "".join(row_vals))
    lines.append("")

    lines.append("Top-50 KC overlap between short (1-7) and other bins:")
    if "1-3" in bin_top50_kc_sets and "4-7" in bin_top50_kc_sets:
        short_top = set(int(k) for k in bin_top50_kc_sets["1-3"]) | set(
            int(k) for k in bin_top50_kc_sets["4-7"]
        )
    elif "4-7" in bin_top50_kc_sets:
        short_top = set(int(k) for k in bin_top50_kc_sets["4-7"])
    else:
        short_top = set()

    for label in bin_labels:
        other_top = set(int(k) for k in bin_top50_kc_sets[label])
        if len(short_top | other_top) > 0:
            overlap = len(short_top & other_top)
            total = len(short_top | other_top)
            lines.append(
                f"  Short(1-7) vs {label}: {overlap}/{total} shared "
                f"(Jaccard={overlap / total:.2f})"
            )
    lines.append("")
    lines.append("Interpretation: Jaccard > 0.6 = strong sharing (generalization).")
    lines.append(
        "Jaccard < 0.3 = largely disjoint KC subspaces (potential memorization)."
    )
    lines.append("")

    # ==================================================================
    # Analysis 5: Grammar Point Length Confound
    # ==================================================================
    print("Analysis 5: Grammar point length confound...")
    lines.append("ANALYSIS 5: GRAMMAR POINT LENGTH CONFOUND")
    lines.append("-" * 70)
    lines.append("")

    gp_length_kc: list[tuple[str, float, float, int]] = []
    sent_to_idx_local = {s: i for i, s in enumerate(all_sents)}
    for gp_name, sents in gp_sents.items():
        indices = [sent_to_idx_local[s] for s in sents if s in sent_to_idx_local]
        if len(indices) < 20:
            continue
        gp_lengths = lengths[indices]
        gp_kc_counts_arr = kc_counts[indices]
        gp_length_kc.append(
            (
                gp_name,
                float(gp_lengths.mean()),
                float(gp_kc_counts_arr.mean()),
                len(indices),
            )
        )

    gp_length_kc.sort(key=lambda x: x[1])

    if gp_length_kc:
        gp_mean_lens = np.array([x[1] for x in gp_length_kc])
        gp_mean_kcs = np.array([x[2] for x in gp_length_kc])
        gp_len_kc_corr = float(np.corrcoef(gp_mean_lens, gp_mean_kcs)[0, 1])

        lines.append(
            f"Correlation between GP mean length and GP mean KC count: "
            f"r = {gp_len_kc_corr:.3f}"
        )
        lines.append(f"({len(gp_length_kc)} GPs with >= 20 sentences)")
        lines.append("")

        if gp_len_kc_corr < -0.15:
            lines.append(
                "NEGATIVE correlation: GPs with shorter sentences do tend to have "
                "more KCs. This could mean the extra KCs on short sentences are "
                "grammar-driven (short GP patterns activate many KCs) rather than "
                "a pure length artifact."
            )
        elif gp_len_kc_corr > 0.15:
            lines.append(
                "POSITIVE correlation: longer GP sentences actually have more KCs. "
                "The short-sentence KC inflation may be independent of grammar."
            )
        else:
            lines.append(
                "WEAK correlation: GP length and KC count are mostly independent. "
                "The short-sentence effect may come from a small number of "
                "length-sensitive KCs, not grammar structure."
            )
        lines.append("")

        lines.append("Shortest-sentence GPs (potential confounders):")
        lines.append(f"  {'GP':<45} {'MeanLen':>7} {'MeanKC':>6} {'N':>5}")
        for gp_name, mean_len, mean_kc, n_s in gp_length_kc[:15]:
            lines.append(f"  {gp_name:<45} {mean_len:>7.1f} {mean_kc:>6.1f} {n_s:>5}")
        lines.append("")

        lines.append("Longest-sentence GPs:")
        lines.append(f"  {'GP':<45} {'MeanLen':>7} {'MeanKC':>6} {'N':>5}")
        for gp_name, mean_len, mean_kc, n_s in gp_length_kc[-15:]:
            lines.append(f"  {gp_name:<45} {mean_len:>7.1f} {mean_kc:>6.1f} {n_s:>5}")
    lines.append("")

    # ==================================================================
    # Conclusion
    # ==================================================================
    lines.append("CONCLUSION")
    lines.append("=" * 70)
    n_short_exclusive = len(exclusive_kcs)
    short_long_jaccard = 0.0
    if "8-15" in bin_active_kc_sets and short_top:
        med_set = bin_active_kc_sets["8-15"]
        if len(short_top | med_set) > 0:
            short_long_jaccard = len(short_top & med_set) / len(short_top | med_set)

    if n_short_exclusive > 10:
        lines.append(
            f"WARNING: {n_short_exclusive} KCs are short-exclusive (>=50% short, "
            f"<10% long). The model may be dedicating KC capacity to encoding "
            f"sentence length."
        )
    elif n_short_exclusive > 0:
        lines.append(
            f"CAUTION: {n_short_exclusive} short-exclusive KCs found. Some KC "
            f"capacity may encode length, but the effect is limited."
        )
    else:
        lines.append(
            "No short-exclusive KCs found at strict thresholds. "
            "Short-sentence KC inflation likely reflects grammatical density, "
            "not memorization."
        )

    if short_long_jaccard > 0.5:
        lines.append(
            f"KC substrate overlap (short vs medium) Jaccard = {short_long_jaccard:.2f} "
            f"-- high overlap suggests generalization."
        )
    elif short_long_jaccard > 0.0:
        lines.append(
            f"KC substrate overlap (short vs medium) Jaccard = {short_long_jaccard:.2f}"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    report = "\n".join(lines)
    print("\n" + report)

    output_path = "semantics/kc-length-bias.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grammar-KC analyses")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--learning-order",
        action="store_true",
        help="Derive a natural KC learning order from the prerequisite DAG",
    )
    group.add_argument(
        "--break-cycles",
        action="store_true",
        help="Remove learn_before entries to make the DAG cycle-free",
    )
    group.add_argument(
        "--discover-gps",
        action="store_true",
        help="Discover novel grammar points via KC clustering",
    )
    group.add_argument(
        "--find-nuance-divisions",
        action="store_true",
        help="Find GPs whose sentences split into distinct KC sub-clusters",
    )
    group.add_argument(
        "--length-bias",
        action="store_true",
        help="Investigate KC activation patterns vs sentence length",
    )
    parser.add_argument(
        "--db-path",
        default="data/corpus.db",
        help="Path to corpus database",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=15000,
        help="Number of unlabeled corpus sentences to sample",
    )
    args = parser.parse_args()

    if args.length_bias:
        main_length_bias(db_path=args.db_path)
        return

    need_yaml = not (args.discover_gps or args.find_nuance_divisions)
    ctx = _load_common(db_path=args.db_path, need_yaml=need_yaml)

    if args.learning_order:
        main_learning_order(ctx)
    elif args.break_cycles:
        main_break_cycles(ctx)
    elif args.discover_gps:
        main_discover_gps(ctx, db_path=args.db_path, sample_size=args.sample_size)
    elif args.find_nuance_divisions:
        main_find_nuance_divisions(ctx)
    else:
        main_dividend(ctx)


if __name__ == "__main__":
    main()
