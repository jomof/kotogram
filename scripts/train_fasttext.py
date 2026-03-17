#!/usr/bin/env python3
"""Train FastText embeddings for surface token representations.

Pre-trains surface token embeddings using the FastText algorithm (skip-gram with
character n-gram features and negative sampling). The resulting embedding matrix
can be loaded to initialize the SurfaceEmbedding layer before style classifier
training, providing morphologically-aware token representations.

Two corpus modes:
  Default   Uses the labeled dataset (feat_surface.bin + offsets.bin from
            labeling phase). Only grammatic sentences are used.
  --corpus  Reads an external text file (one sentence per line), tokenizes
            with Sudachi Mode C, and builds its own vocabulary. The export
            includes a string→ID vocab mapping so the consumer can look up
            vectors by surface string rather than requiring matching IDs.

The exported model always includes n-gram embedding weights so the consumer
can compute vectors for out-of-vocabulary tokens at load time.

Saves:
    .cache/fasttext_checkpoint.pt   Training checkpoint (resumable)
    models/style/fasttext.pt        Final embedding weight matrix + vocab

Usage:
    python -m scripts.train_fasttext
    python -m scripts.train_fasttext --corpus data/wiki_ja.txt --min-count 10
    python -m scripts.train_fasttext --embed-dim 256 --epochs 10
    python -m scripts.train_fasttext --resume
"""

import argparse
import math
import os
import sys
import time
from typing import List, Tuple

import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.nn import functional as F

from kotogram import locations
from kotogram.model import get_inference_device as get_device
from kotogram.tokenizer import (
    CLS_ID,
    MASK_ID,
    PAD_ID,
    UNK_ID,
    Tokenizer,
)
from train import paths as train_paths
from train.display import RichTrainerProgressBar

console = Console()

SPECIAL_IDS = frozenset({PAD_ID, UNK_ID, CLS_ID, MASK_ID})


# ─── Character n-gram utilities ──────────────────────────────────────────────


def char_ngrams(token: str, min_n: int, max_n: int) -> List[str]:
    """Extract character n-grams with boundary markers (FastText convention)."""
    bounded = f"<{token}>"
    grams: List[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(bounded) - n + 1):
            grams.append(bounded[i : i + n])
    return grams


def hash_ngram(ngram: str, num_buckets: int) -> int:
    """FNV-1a hash to a bucket index. Bucket 0 is reserved for padding."""
    h = 2166136261
    for ch in ngram:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h % (num_buckets - 1) + 1


# ─── Corpus loading ──────────────────────────────────────────────────────────


def load_corpus(dataset_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load surface token IDs and sentence offsets from labeling binary files.

    Reads the binary files produced by the labeling phase directly, avoiding
    the expensive re-parse of kotograms.txt.

    Returns:
        token_ids: (N_total,) int32 tensor of all surface token IDs.
        sent_offsets: (N_sentences + 1,) int64 tensor of sentence boundaries.
        word_freqs: (max_id + 1,) int64 tensor of per-token corpus frequency.
    """
    feat_path = os.path.join(dataset_dir, "feat_surface.bin")
    off_path = os.path.join(dataset_dir, "offsets.bin")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Surface features not found at {feat_path}. "
            "Run labeling first: ./train_style --label"
        )

    n_tokens = os.path.getsize(feat_path) // 4
    n_offsets = os.path.getsize(off_path) // 4

    token_ids = torch.from_file(
        feat_path, shared=False, size=n_tokens, dtype=torch.int32
    )
    sent_offsets = torch.from_file(
        off_path, shared=False, size=n_offsets, dtype=torch.int32
    ).long()
    word_freqs = torch.bincount(token_ids.long(), minlength=1)

    return token_ids, sent_offsets, word_freqs


def load_gram_mask(dataset_dir: str, sent_offsets: torch.Tensor) -> torch.Tensor:
    """Load per-sentence grammaticality labels and expand to a per-token boolean mask.

    Grammatic sentences (label == 1) have pragmatic formality, pragmatic gender,
    and were marked grammatic in the source data.  Non-grammatic sentences may
    contain errors that would embed spurious co-occurrence patterns.
    """
    gram_path = os.path.join(dataset_dir, "labels.bin_gram")
    if not os.path.exists(gram_path):
        raise FileNotFoundError(
            f"Gram labels not found at {gram_path}. "
            "Run labeling first: ./train_style --label"
        )
    n_sentences = len(sent_offsets) - 1
    gram_labels = torch.from_file(
        gram_path, shared=False, size=n_sentences, dtype=torch.uint8
    )
    lengths = (sent_offsets[1:] - sent_offsets[:-1]).long()
    return torch.repeat_interleave(gram_labels == 1, lengths)


def load_text_corpus(  # pylint: disable=too-many-locals
    corpus_path: str, min_count: int
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, int], list[str]]:
    """Tokenize an external text file and build a standalone vocabulary.

    Each line of the file is treated as one sentence, tokenized with Sudachi
    Mode C (coarsest).  A vocabulary is built from tokens that appear at least
    ``min_count`` times, with IDs 0-3 reserved for PAD/UNK/CLS/MASK.

    Returns:
        token_ids:    (N_total,) int32 tensor of token IDs.
        sent_offsets: (N_sentences + 1,) int64 tensor of sentence boundaries.
        vocab:        dict mapping surface string -> token ID.
        id_to_str:    list mapping token ID -> surface string.
    """
    from collections import Counter

    from sudachipy import SplitMode, dictionary

    tokenizer_obj = dictionary.Dictionary(dict="full").create(mode=SplitMode.C)

    console.print("Tokenizing corpus (pass 1: counting)...")
    counts: Counter[str] = Counter()
    n_lines = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for m in tokenizer_obj.tokenize(line):
                counts[m.surface()] += 1
            n_lines += 1
    console.print(
        f"  {n_lines:,} sentences, {sum(counts.values()):,} tokens, "
        f"{len(counts):,} unique"
    )

    # Build vocabulary: reserve 0-3 for special tokens
    reserved = ["<pad>", "<unk>", "<cls>", "<mask>"]
    vocab: dict[str, int] = {s: i for i, s in enumerate(reserved)}
    id_to_str: list[str] = list(reserved)
    for token_str, cnt in counts.most_common():
        if cnt < min_count:
            break
        if token_str not in vocab:
            vocab[token_str] = len(id_to_str)
            id_to_str.append(token_str)

    vocab_size = len(id_to_str)
    console.print(f"  Vocabulary: {vocab_size:,} tokens (min_count={min_count})")

    console.print("Tokenizing corpus (pass 2: encoding)...")
    all_ids: list[int] = []
    offsets: list[int] = [0]
    unk_id = vocab["<unk>"]
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for m in tokenizer_obj.tokenize(line):
                all_ids.append(vocab.get(m.surface(), unk_id))
            offsets.append(len(all_ids))

    token_ids = torch.tensor(all_ids, dtype=torch.int32)
    sent_offsets = torch.tensor(offsets, dtype=torch.long)
    return token_ids, sent_offsets, vocab, id_to_str


def build_ngram_lookup(
    surface_vocab: dict[str, int],
    vocab_size: int,
    min_n: int,
    max_n: int,
    num_buckets: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute hashed n-gram bucket IDs as a fixed-width padded tensor.

    Args:
        surface_vocab: mapping from surface string to token ID.

    Returns:
        ngram_table: (vocab_size, MAX_NG) int32 tensor, zero-padded.
        ngram_counts: (vocab_size,) int8 tensor of valid n-gram counts per token.
    """
    raw: dict[int, list[int]] = {}
    max_ng = 0
    for token_str, token_id in surface_vocab.items():
        if token_id in SPECIAL_IDS:
            continue
        ngrams = char_ngrams(token_str, min_n, max_n)
        hashes = [hash_ngram(ng, num_buckets) for ng in ngrams]
        raw[token_id] = hashes
        max_ng = max(max_ng, len(hashes))

    max_ng = max(max_ng, 1)
    ngram_table = torch.zeros(vocab_size, max_ng, dtype=torch.int32)
    ngram_counts = torch.zeros(vocab_size, dtype=torch.int8)

    for token_id, hashes in raw.items():
        n = len(hashes)
        ngram_table[token_id, :n] = torch.tensor(hashes, dtype=torch.int32)
        ngram_counts[token_id] = n

    return ngram_table, ngram_counts


# ─── Pair generation ─────────────────────────────────────────────────────────


def build_training_positions(
    token_ids: torch.Tensor,
    sent_offsets: torch.Tensor,
    gram_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify valid token positions (non-special, grammatic) and their sentence context.

    Returns:
        pos_sent: (N_valid,) int32 -- sentence index for each valid position.
        pos_in_sent: (N_valid,) int32 -- position within its sentence.
        sent_lengths: (N_sentences,) int32 -- length of each sentence.
    """
    n_sentences = len(sent_offsets) - 1
    n_total = len(token_ids)

    lengths = (sent_offsets[1:] - sent_offsets[:-1]).int()
    sent_of_token = torch.repeat_interleave(
        torch.arange(n_sentences, dtype=torch.int32), lengths.long()
    )
    flat_idx = torch.arange(n_total, dtype=torch.long)
    pos_in_sent = (flat_idx - sent_offsets[sent_of_token.long()]).int()

    valid_mask = gram_mask.clone()
    for sid in SPECIAL_IDS:
        valid_mask &= token_ids != sid

    valid = valid_mask.nonzero(as_tuple=True)[0]
    return sent_of_token[valid], pos_in_sent[valid], lengths


def build_keep_prob(word_freqs: torch.Tensor, subsample_thresh: float) -> torch.Tensor:
    """Compute per-token keep probability for frequency subsampling (Mikolov 2013)."""
    freq_f = word_freqs.float()
    for sid in SPECIAL_IDS:
        if sid < len(freq_f):
            freq_f[sid] = 0.0
    total = freq_f.sum().clamp(min=1.0)

    f = freq_f / total
    keep = ((f / subsample_thresh).sqrt() + 1.0) * (
        subsample_thresh / f.clamp(min=1e-15)
    )
    keep = keep.clamp(max=1.0)
    keep[freq_f == 0] = 1.0
    result: torch.Tensor = keep
    return result


def build_neg_table(
    word_freqs: torch.Tensor, table_size: int = 1_000_000
) -> torch.Tensor:
    """Build negative sampling table from unigram^0.75 distribution."""
    powered = word_freqs.float().pow(0.75)
    for sid in SPECIAL_IDS:
        if sid < len(powered):
            powered[sid] = 0.0
    powered[0] = 0.0

    total = powered.sum()
    if total == 0:
        return torch.zeros(table_size, dtype=torch.long)

    probs = powered / total
    counts = (probs * table_size).long().clamp(min=0)

    nonzero_mask = powered > 0
    counts[nonzero_mask] = counts[nonzero_mask].clamp(min=1)

    diff = table_size - counts.sum().item()
    top_id = int(powered.argmax().item())
    counts[top_id] = max(1, counts[top_id].item() + diff)

    neg_table = torch.repeat_interleave(
        torch.arange(len(counts), dtype=torch.long), counts
    )
    if len(neg_table) > table_size:
        neg_table = neg_table[:table_size]
    elif len(neg_table) < table_size:
        pad = neg_table[-1].expand(table_size - len(neg_table))
        neg_table = torch.cat([neg_table, pad])
    return neg_table


def generate_epoch_pairs(  # pylint: disable=too-many-positional-arguments,too-many-locals
    pos_sent: torch.Tensor,
    pos_in_sent: torch.Tensor,
    sent_lengths: torch.Tensor,
    token_ids: torch.Tensor,
    sent_offsets: torch.Tensor,
    keep_prob: torch.Tensor,
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate all (center, context) pairs for one epoch using vectorized ops.

    For each valid position, generates pairs with ALL context words within a
    random window (matching standard FastText), not just one.  N-gram lookup
    is deferred to the training loop to avoid materialising a huge (M, MAX_NG)
    tensor.

    Returns:
        centers: (M,) int64 -- center token IDs
        contexts: (M,) int64 -- context token IDs
    """
    n_pos = len(pos_sent)

    perm = torch.randperm(n_pos)
    p_sent = pos_sent[perm]
    p_pos = pos_in_sent[perm]

    s_starts = sent_offsets[p_sent.long()]
    s_lens = sent_lengths[p_sent.long()]

    center_abs = s_starts + p_pos.long()
    center_ids = token_ids[center_abs].long()

    # Random window per position
    win = torch.randint(1, window + 1, (n_pos,))

    # All possible offsets: [-window, ..., -1, 1, ..., window]
    offsets = torch.cat([torch.arange(-window, 0), torch.arange(1, window + 1)])
    n_offsets = len(offsets)

    # Expand to (n_pos, 2W) -- context positions for every offset
    ctx_pos = p_pos.long().unsqueeze(1) + offsets.unsqueeze(0)

    # Valid: offset within this position's random window AND within sentence
    valid = (
        (offsets.unsqueeze(0).abs() <= win.unsqueeze(1))
        & (ctx_pos >= 0)
        & (ctx_pos < s_lens.long().unsqueeze(1))
    )

    # Safe-clamp for indexing (invalid entries are discarded below)
    ctx_pos_safe = ctx_pos.clamp(min=0)
    ctx_pos_safe = torch.min(
        ctx_pos_safe, (s_lens.long().unsqueeze(1) - 1).clamp(min=0)
    )
    ctx_abs = s_starts.unsqueeze(1) + ctx_pos_safe
    ctx_ids = token_ids[ctx_abs].long()

    # Frequency subsampling on context words
    valid &= torch.rand(n_pos, n_offsets) < keep_prob[ctx_ids]

    # Flatten and keep only valid pairs
    valid_flat = valid.reshape(-1)
    centers_exp = center_ids.unsqueeze(1).expand(n_pos, n_offsets).reshape(-1)

    flat_centers = centers_exp[valid_flat]
    flat_contexts = ctx_ids.reshape(-1)[valid_flat]

    # Shuffle so batches see diverse sentence contexts
    shuf = torch.randperm(len(flat_centers))
    flat_centers = flat_centers[shuf]
    flat_contexts = flat_contexts[shuf]

    if torch.cuda.is_available():
        flat_centers = flat_centers.pin_memory()
        flat_contexts = flat_contexts.pin_memory()

    return flat_centers, flat_contexts


# ─── Model ───────────────────────────────────────────────────────────────────


class FastTextSG(nn.Module):  # type: ignore[misc]
    """FastText skip-gram with character n-gram embeddings and negative sampling.

    Word vector = word_embed[w] + sum(ngram_embed[h] for h in char_ngrams(w))
    Trained via negative sampling: maximize dot product with true context,
    minimize with randomly sampled negatives.
    """

    def __init__(self, vocab_size: int, num_buckets: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.ngram_embed = nn.Embedding(num_buckets, embed_dim, padding_idx=0)
        self.ctx_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        bound = 1.0 / math.sqrt(embed_dim)
        nn.init.uniform_(self.word_embed.weight, -bound, bound)
        nn.init.uniform_(self.ngram_embed.weight, -bound, bound)
        nn.init.uniform_(self.ctx_embed.weight, -bound, bound)
        self.word_embed.weight.data[0].zero_()
        self.ctx_embed.weight.data[0].zero_()

    def forward(  # pylint: disable=too-many-positional-arguments,too-many-locals
        self,
        center_ids: torch.Tensor,
        ngram_ids: torch.Tensor,
        ngram_counts: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        word_vec = self.word_embed(center_ids)  # (B, D)

        ng_embeds = self.ngram_embed(ngram_ids)  # (B, MAX_NG, D)
        mask = torch.arange(ngram_ids.size(1), device=ngram_ids.device)
        mask = (mask.unsqueeze(0) < ngram_counts.unsqueeze(1)).unsqueeze(
            -1
        )  # (B, MAX_NG, 1)
        ng_vec = (ng_embeds * mask).sum(dim=1)  # (B, D)

        center_vec = word_vec + ng_vec  # (B, D)

        pos_vec = self.ctx_embed(pos_ids)  # (B, D)
        pos_score = (center_vec * pos_vec).sum(dim=-1)  # (B,)

        neg_vec = self.ctx_embed(neg_ids)  # (B, K, D)
        neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(-1)).squeeze(-1)  # (B, K)

        pos_loss = -F.logsigmoid(pos_score)  # pylint: disable=not-callable
        neg_loss = -F.logsigmoid(-neg_score).sum(dim=-1)  # pylint: disable=not-callable
        loss = (pos_loss + neg_loss).mean()
        return loss

    @torch.no_grad()
    def export_vectors(
        self, ngram_table: torch.Tensor, ngram_counts: torch.Tensor
    ) -> torch.Tensor:
        """Compute final word vectors: word_embed[w] + sum(ngram_embed[ngrams(w)])."""
        device = self.word_embed.weight.device
        vectors = self.word_embed.weight.clone()

        has_ng = (ngram_counts > 0).nonzero(as_tuple=True)[0]
        chunk = 1024
        for start in range(0, len(has_ng), chunk):
            cpu_ids = has_ng[start : start + chunk]
            ng_ids = ngram_table[cpu_ids].long().to(device)
            ng_cnt = ngram_counts[cpu_ids].long().to(device)

            ng_embeds = self.ngram_embed(ng_ids)  # (C, MAX_NG, D)
            mask = torch.arange(ng_ids.size(1), device=device)
            mask = (mask.unsqueeze(0) < ng_cnt.unsqueeze(1)).unsqueeze(-1)
            ng_vec = (ng_embeds * mask).sum(dim=1)  # (C, D)

            vectors[cpu_ids.to(device)] += ng_vec  # pylint: disable=unsupported-assignment-operation

        return vectors


# ─── Analogy evaluation ──────────────────────────────────────────────────────

# (A, B, C, expected_D, label) -- target = B - A + C ≈ D
# All words verified to be single-morpheme tokens in the surface vocabulary.
ANALOGIES = [
    # -- Semantic relations --
    ("男", "女", "父", "母", "gender"),
    ("男", "女", "兄", "姉", "gender (sibling)"),
    ("王", "王女", "神", "女神", "male → female"),
    ("買う", "売る", "教える", "学ぶ", "action reversal"),
    ("書く", "読む", "話す", "聞く", "I/O pair"),
    ("入る", "出る", "増える", "減る", "converse verb"),
    # -- Antonym patterns --
    ("大きい", "小さい", "長い", "短い", "antonym (size)"),
    ("強い", "弱い", "明るい", "暗い", "antonym (quality)"),
    ("上", "下", "左", "右", "spatial"),
    ("東", "西", "南", "北", "compass"),
    # -- Derivational morphology --
    ("歌", "歌手", "運転", "運転手", "X → X手"),
    ("科学", "科学者", "芸術", "芸術家", "X → X者/家"),
    ("日本語", "日本人", "中国語", "中国人", "X語 → X人"),
    # -- Orthographic / subword --
    ("今日", "きょう", "明日", "あした", "kanji → kana"),
    ("プログラム", "プログラマー", "デザイン", "デザイナー", "stem → agent"),
]


@torch.no_grad()
def run_analogy_report(  # pylint: disable=too-many-positional-arguments,too-many-locals
    model: FastTextSG,
    ngram_table: torch.Tensor,
    ngram_counts: torch.Tensor,
    surface_vocab: dict,
    epoch: int,
    n_epochs: int,
) -> None:
    """Evaluate word analogies (B - A + C ~= ?) and print a per-epoch report."""
    id2word = {v: k for k, v in surface_vocab.items()}

    vectors = model.export_vectors(ngram_table, ngram_counts)
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = vectors / norms

    table = Table(
        title=f"Analogies (Epoch {epoch + 1}/{n_epochs})",
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("", width=16, style="dim")
    table.add_column("A : B :: C : ?")
    table.add_column("Expected")
    table.add_column("Got")
    table.add_column("cos", justify="right", width=5)

    n_correct = 0
    n_run = 0

    for a_w, b_w, c_w, exp_w, label in ANALOGIES:
        a_id = surface_vocab.get(a_w)
        b_id = surface_vocab.get(b_w)
        c_id = surface_vocab.get(c_w)
        exp_id = surface_vocab.get(exp_w)

        oov = [w for w, i in [(a_w, a_id), (b_w, b_id), (c_w, c_id)] if i is None]
        if oov:
            table.add_row(
                label,
                f"{a_w}:{b_w}::{c_w}:?",
                exp_w,
                f"[dim]OOV: {' '.join(oov)}[/dim]",
                "",
            )
            continue

        target = normed[b_id] - normed[a_id] + normed[c_id]
        target = target / target.norm().clamp(min=1e-8)

        sims = normed @ target
        for sid in (a_id, b_id, c_id, *SPECIAL_IDS):
            if sid is not None and sid < len(sims):
                sims[sid] = -2.0

        top_id = int(sims.argmax().item())
        top_word = id2word.get(top_id, f"?{top_id}")
        sim = float(sims[top_id].item())

        n_run += 1
        hit = exp_id is not None and top_id == exp_id
        if hit:
            n_correct += 1

        mark = "[green]✓[/green]" if hit else "[red]✗[/red]"
        exp_col = exp_w if exp_id is not None else f"[dim]{exp_w} (OOV)[/dim]"
        table.add_row(
            label,
            f"{a_w}:{b_w}::{c_w}:?",
            exp_col,
            f"{mark} {top_word}",
            f"{sim:.2f}",
        )

    console.print(table)
    if n_run:
        console.print(f"  [bold]{n_correct}/{n_run}[/bold] correct\n")


# ─── Training ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train FastText embeddings on kotogram surface tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=256,
        help="Embedding dimension (default: 256, matching SurfaceEmbedding)",
    )
    parser.add_argument(
        "--window", type=int, default=5, help="Context window size (default: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4096, help="Batch size (default: 4096)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.025, help="Initial learning rate (default: 0.025)"
    )
    parser.add_argument(
        "--neg-samples",
        type=int,
        default=5,
        help="Negative samples per positive pair (default: 5)",
    )
    parser.add_argument(
        "--min-ngram", type=int, default=2, help="Min char n-gram length (default: 2)"
    )
    parser.add_argument(
        "--max-ngram", type=int, default=5, help="Max char n-gram length (default: 5)"
    )
    parser.add_argument(
        "--num-buckets",
        type=int,
        default=500_000,
        help="Hash buckets for character n-grams (default: 500000)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1e-4,
        help="Frequent-word subsampling threshold (default: 1e-4)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to external text corpus (one sentence per line). "
        "Builds its own vocabulary instead of using labeled data.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum token frequency for vocabulary when using --corpus (default: 5)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser.parse_args()


# pylint: disable=too-many-locals,too-many-statements
def main() -> int:
    """Train FastText embeddings and save to models/style/fasttext.pt."""
    # Allow MPS to exceed its soft memory watermark -- unified memory on
    # Apple Silicon is shared with the OS, so the default limit is too tight.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    args = parse_args()

    dataset_dir = train_paths.get_style_dataset_cache_dir()
    cache_dir = train_paths.get_cache_dir()
    output_dir = locations.get_style_output_dir()
    checkpoint_path = os.path.join(cache_dir, "fasttext_checkpoint.pt")
    final_path = os.path.join(output_dir, "fasttext.pt")

    device = get_device()
    console.print(f"Device: [bold]{device}[/bold]")

    # ── Corpus ──
    t0 = time.perf_counter()
    external_corpus = args.corpus is not None

    if external_corpus:
        # ── External text corpus: build own vocabulary ──
        console.print(f"External corpus: [bold]{args.corpus}[/bold]")
        token_ids, sent_offsets, surface_vocab, id_to_str = load_text_corpus(
            args.corpus, args.min_count
        )
        vocab_size = len(id_to_str)
        n_sentences = len(sent_offsets) - 1
        # All sentences in the external corpus are used (no gram filter)
        gram_mask = torch.ones(len(token_ids), dtype=torch.bool)
        word_freqs = torch.bincount(token_ids.long(), minlength=vocab_size)

        console.print(
            f"Loaded [bold]{n_sentences:,}[/bold] sentences, "
            f"[bold]{len(token_ids):,}[/bold] tokens, "
            f"vocab [bold]{vocab_size:,}[/bold] "
            f"({time.perf_counter() - t0:.1f}s)"
        )
    else:
        # ── Labeled dataset: use existing tokenizer vocabulary ──
        vocab_path = os.path.join(dataset_dir, "vocab.json")
        if not os.path.exists(vocab_path):
            console.print(f"[red]Vocabulary not found at {vocab_path}[/red]")
            console.print("Run labeling first: ./train_style --label")
            return 1

        tokenizer = Tokenizer.load(vocab_path)
        surface_vocab = tokenizer.field_vocabs["surface"]
        id_to_str = [""] * len(surface_vocab)
        for s, i in surface_vocab.items():
            id_to_str[i] = s
        vocab_size = len(surface_vocab)
        console.print(f"Surface vocabulary: [bold]{vocab_size:,}[/bold] tokens")

        console.print("Loading corpus...")
        token_ids, sent_offsets, _ = load_corpus(dataset_dir)
        n_sentences = len(sent_offsets) - 1

        gram_mask = load_gram_mask(dataset_dir, sent_offsets)
        n_gram_sentences = int(
            torch.from_file(
                os.path.join(dataset_dir, "labels.bin_gram"),
                shared=False,
                size=n_sentences,
                dtype=torch.uint8,
            )
            .sum()
            .item()
        )
        n_gram_tokens = int(gram_mask.sum().item())
        word_freqs = torch.bincount(token_ids[gram_mask].long(), minlength=1)

        console.print(
            f"Loaded [bold]{n_sentences:,}[/bold] sentences "
            f"([bold]{n_gram_sentences:,}[/bold] grammatic), "
            f"[bold]{len(token_ids):,}[/bold] tokens "
            f"([bold]{n_gram_tokens:,}[/bold] grammatic) "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    # ── N-gram lookup (padded tensor) ──
    console.print("Building character n-gram lookup...")
    ngram_table, ngram_counts = build_ngram_lookup(
        surface_vocab, vocab_size, args.min_ngram, args.max_ngram, args.num_buckets
    )
    max_ng = ngram_table.size(1)
    n_with_ng = int((ngram_counts > 0).sum().item())
    avg_ng = ngram_counts[ngram_counts > 0].float().mean().item() if n_with_ng else 0.0
    console.print(
        f"N-gram table: {n_with_ng:,} tokens, max_ng={max_ng}, avg={avg_ng:.1f}"
    )

    # ── Training positions ──
    console.print("Building training positions...")
    pos_sent, pos_in_sent, sent_lengths = build_training_positions(
        token_ids, sent_offsets, gram_mask
    )
    n_positions = len(pos_sent)
    console.print(f"Training positions: [bold]{n_positions:,}[/bold]")

    # ── Subsampling & negative sampling table ──
    keep_prob = build_keep_prob(word_freqs, args.subsample)
    neg_table = build_neg_table(word_freqs)
    table_size = len(neg_table)

    # ── Model ──
    model = FastTextSG(vocab_size, args.num_buckets, args.embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"Parameters: [bold]{n_params:,}[/bold]")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Estimate total pairs per epoch for LR schedule (~6 ctx/pos on avg with window=5)
    est_pairs_per_epoch = n_positions * (args.window + 1)
    est_batches_per_epoch = (
        est_pairs_per_epoch + args.batch_size - 1
    ) // args.batch_size
    total_steps = args.epochs * est_batches_per_epoch

    # ── Resume ──
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        console.print(f"Resumed from epoch {start_epoch}")

    # ── Train ──
    console.print(f"\n[bold]Training FastText[/bold] ({args.epochs} epochs)")
    console.print(
        f"  embed_dim={args.embed_dim}  window={args.window}  neg={args.neg_samples}"
    )
    console.print(
        f"  lr={args.lr}  batch={args.batch_size}  buckets={args.num_buckets}"
    )
    console.print(f"  ngram range: {args.min_ngram}-{args.max_ngram}\n")

    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        # Pre-generate all (center, context) pairs for this epoch
        t_gen = time.perf_counter()
        centers, contexts = generate_epoch_pairs(
            pos_sent,
            pos_in_sent,
            sent_lengths,
            token_ids,
            sent_offsets,
            keep_prob,
            args.window,
        )
        n_pairs = len(centers)
        n_batches_epoch = (n_pairs + args.batch_size - 1) // args.batch_size
        console.print(
            f"  Epoch {epoch + 1}: generated {n_pairs:,} pairs "
            f"({n_pairs / n_positions:.1f} ctx/pos, {time.perf_counter() - t_gen:.1f}s)"
        )

        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.perf_counter()

        pbar = RichTrainerProgressBar(
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            total_steps=n_batches_epoch,
            batch_size=args.batch_size,
            total_elements_target=n_pairs,
        )

        for i in range(0, n_pairs, args.batch_size):
            j = min(i + args.batch_size, n_pairs)

            b_center_cpu = centers[i:j]
            b_center = b_center_cpu.to(device, non_blocking=True)
            b_ctx = contexts[i:j].to(device, non_blocking=True)
            b_ng = ngram_table[b_center_cpu].long().to(device)
            b_cnt = ngram_counts[b_center_cpu].long().to(device)
            b_neg = neg_table[
                torch.randint(0, table_size, (j - i, args.neg_samples))
            ].to(device)

            # Linear LR decay (standard FastText schedule)
            progress = global_step / max(total_steps, 1)
            lr = max(args.lr * (1.0 - progress), args.lr * 1e-4)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss = model(b_center, b_ng, b_cnt, b_ctx, b_neg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            avg_loss = epoch_loss / n_batches
            pbar.update(n_batches - 1, loss=avg_loss)

            if device.type == "mps" and n_batches % 200 == 0:
                torch.mps.empty_cache()

        pbar.stop()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.perf_counter() - t_epoch
        console.print(
            f"  Epoch {epoch + 1:3d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  lr={lr:.6f}  ({elapsed:.1f}s)"
        )

        os.makedirs(cache_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
            },
            checkpoint_path,
        )

        run_analogy_report(
            model,
            ngram_table,
            ngram_counts,
            surface_vocab,
            epoch,
            args.epochs,
        )

        del centers, contexts

    # ── Export ──
    console.print("\nExporting word vectors...")
    model.eval()
    vectors = model.export_vectors(ngram_table, ngram_counts)

    os.makedirs(output_dir, exist_ok=True)
    save_dict = {
        "embedding_weight": vectors.cpu(),
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
        "num_buckets": args.num_buckets,
        "min_ngram": args.min_ngram,
        "max_ngram": args.max_ngram,
        # Vocab mapping for string-based lookup (consumer maps its own
        # vocab to these vectors by surface string, not by ID).
        "vocab": surface_vocab,
        # N-gram embedding weights so the consumer can compute vectors
        # for tokens not in this vocabulary (FastText OOV inference).
        "ngram_embed_weight": model.ngram_embed.weight.data.cpu(),
    }
    torch.save(save_dict, final_path)
    console.print(
        f"Saved to [bold]{final_path}[/bold]  shape=({vocab_size}, {args.embed_dim})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
