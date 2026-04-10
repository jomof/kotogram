"""chiVe pretrained Japanese word vector utilities.

Downloads chiVe v1.3 (Sudachi-based word2vec vectors trained on CommonCrawl)
and provides a loader that maps a target vocabulary to chiVe vectors by
normalized surface string.

References:
    https://github.com/WorksApplications/chiVe
"""

from __future__ import annotations

import os
import tarfile
import urllib.request
from typing import TYPE_CHECKING, Optional

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from train import paths as train_paths

if TYPE_CHECKING:
    from kotogram.model import SurfaceEmbedding

console = Console()

CHIVE_VERSION = "1.3"
CHIVE_VARIANT = "mc5"
CHIVE_DIM = 300
CHIVE_URL = (
    f"https://sudachi.s3-ap-northeast-1.amazonaws.com/chive/"
    f"chive-{CHIVE_VERSION}-{CHIVE_VARIANT}.tar.gz"
)
CHIVE_FILENAME = f"chive-{CHIVE_VERSION}-{CHIVE_VARIANT}.tar.gz"


def get_chive_dir() -> str:
    return os.path.join(train_paths.get_cache_dir(), "chive")


def get_chive_txt_path() -> str:
    """Path to the extracted word2vec-format text file."""
    return os.path.join(get_chive_dir(), f"chive-{CHIVE_VERSION}-{CHIVE_VARIANT}.txt")


def load_chive_vocab_set() -> frozenset[str]:
    """Load just the word strings from the chiVe text file.

    Requires the file to already be downloaded (call download_chive() first).
    Returns a frozenset for efficient membership testing.
    """
    txt_path = get_chive_txt_path()
    if not os.path.exists(txt_path):
        raise FileNotFoundError(
            f"chiVe text file not found at {txt_path}. Run download_chive() first."
        )
    words: set[str] = set()
    with open(txt_path, encoding="utf-8") as f:
        f.readline()  # skip header (vocab_size dim)
        for line in f:
            space_idx = line.index(" ")
            words.add(line[:space_idx])
    return frozenset(words)


def parse_chive_vectors(  # pylint: disable=too-many-locals
    txt_path: str,
    surface_vocab: dict[str, int],
    vocab_size: int,
    surface_to_base: Optional[dict[str, str]] = None,
) -> tuple[torch.Tensor, set[str], torch.Tensor]:
    """Parse chiVe text file, mapping surface_vocab entries to vectors and ranks.

    Returns (vectors_tensor, matched_words_set, ranks_tensor).
    """
    base_to_targets: dict[str, list[str]] = {}
    target = set(surface_vocab.keys())

    if surface_to_base:
        for t in target:
            base = surface_to_base.get(t, t)
            if base != t:
                if base not in base_to_targets:
                    base_to_targets[base] = []
                base_to_targets[base].append(t)
    else:
        from sudachipy import SplitMode, dictionary

        tokenizer = dictionary.Dictionary(dict="core").create(mode=SplitMode.C)
        for t in target:
            tokens = tokenizer.tokenize(t)
            base = tokens[0].normalized_form() if len(tokens) > 0 else t
            if base != t:
                if base not in base_to_targets:
                    base_to_targets[base] = []
                base_to_targets[base].append(t)

    vectors = torch.zeros(vocab_size, CHIVE_DIM)
    ranks = torch.full((vocab_size,), 9999999, dtype=torch.int32)
    matched: set[str] = set()
    with open(txt_path, encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            space_idx = line.index(" ")
            word = line[:space_idx]

            targets_to_fill = []
            if word in target and word not in matched:
                targets_to_fill.append(word)

            if word in base_to_targets:
                for t in base_to_targets[word]:
                    if t not in matched:
                        targets_to_fill.append(t)

            if targets_to_fill:
                vals = line[space_idx + 1 :].strip().split()
                vec = torch.tensor([float(v) for v in vals], dtype=torch.float32)
                for t in targets_to_fill:
                    tid = surface_vocab[t]
                    vectors[tid] = vec
                    ranks[tid] = i
                    matched.add(t)

                if len(matched) == len(target):
                    break

    return vectors, matched, ranks


def download_chive() -> str:
    """Download and extract chiVe vectors if not already cached.

    Returns the path to the extracted text file.
    """
    txt_path = get_chive_txt_path()
    if os.path.exists(txt_path):
        console.print(f"chiVe already cached at [bold]{txt_path}[/bold]")
        return txt_path

    if "PYTEST_CURRENT_TEST" in os.environ:
        raise RuntimeError(
            "chiVe download triggered during test run. "
            "Tests must pre-populate the mini chiVe file via training_test_utils."
        )

    chive_dir = get_chive_dir()
    os.makedirs(chive_dir, exist_ok=True)
    tar_path = os.path.join(chive_dir, CHIVE_FILENAME)

    if not os.path.exists(tar_path):
        console.print(f"Downloading chiVe {CHIVE_VERSION}-{CHIVE_VARIANT}...")
        console.print(f"  URL: {CHIVE_URL}")

        req = urllib.request.Request(CHIVE_URL, headers={"User-Agent": "kotogram/1.0"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length", 0))

            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading", total=total)
                with open(tar_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

    console.print("Extracting chiVe archive...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".txt") and member.isfile():
                member.name = os.path.basename(member.name)
                tar.extract(member, chive_dir)
                break

    if not os.path.exists(txt_path):
        extracted = [f for f in os.listdir(chive_dir) if f.endswith(".txt")]
        if extracted:
            os.rename(os.path.join(chive_dir, extracted[0]), txt_path)
        else:
            raise FileNotFoundError(
                f"No .txt file found in chiVe archive. Contents: {os.listdir(chive_dir)}"
            )

    # Clean up tar.gz to save disk space
    if os.path.exists(tar_path):
        os.remove(tar_path)
        console.print("  Removed archive (keeping extracted text)")

    console.print(f"chiVe ready at [bold]{txt_path}[/bold]")
    return txt_path


def get_chive_cache_path() -> str:
    """Path to the pre-extracted vocab-matched vectors (.pt file)."""
    return os.path.join(train_paths.get_style_dataset_cache_dir(), "chive_surface.pt")


def extract_chive_for_vocab(  # pylint: disable=too-many-locals
    surface_vocab: dict[str, int],
    surface_freqs: Optional[dict[str, int]] = None,
    miss_report_limit: int = 30,
    surface_to_base: Optional[dict[str, str]] = None,
) -> str:
    """Scan the full chiVe text file and save only the matching vectors.

    Called once at the end of labeling.  The resulting .pt file (~48 MB for
    40K tokens) is fast to load at training time.

    Args:
        surface_vocab: mapping from normalized surface string to token ID.
        surface_freqs: optional corpus frequency counts per surface string.
            When provided, prints a report of the most frequent unmatched tokens.
        miss_report_limit: how many unmatched tokens to show.

    Returns:
        Path to the saved .pt cache file.
    """
    from rich.table import Table

    chive_txt_path = get_chive_txt_path()
    cache_path = get_chive_cache_path()

    if not os.path.exists(chive_txt_path):
        raise FileNotFoundError(
            f"chiVe text file not found at {chive_txt_path}. "
            "Download failed or was skipped."
        )

    vocab_size = max(surface_vocab.values()) + 1
    target_strings = set(surface_vocab.keys())

    console.print(
        f"Extracting chiVe vectors for {len(target_strings):,} vocab entries..."
    )
    vectors, matched_strings, _ranks = parse_chive_vectors(
        chive_txt_path, surface_vocab, vocab_size, surface_to_base
    )

    matched = len(matched_strings)
    coverage = matched / max(len(target_strings), 1) * 100
    console.print(
        f"  Matched [bold]{matched:,}[/bold] / {len(target_strings):,} "
        f"tokens ({coverage:.1f}% coverage)"
    )

    missed = target_strings - matched_strings
    if missed and surface_freqs is not None:
        missed_with_freq = sorted(
            ((s, surface_freqs.get(s, 0)) for s in missed),
            key=lambda x: -x[1],
        )
        show = missed_with_freq[:miss_report_limit]
        table = Table(title=f"Top {len(show)} unmatched tokens (by corpus frequency)")
        table.add_column("Token", style="bold")
        table.add_column("Frequency", justify="right")
        for token, freq in show:
            table.add_row(token, f"{freq:,}")
        console.print(table)

    # Initialize unmatched tokens with small random vectors so they don't
    # start as degenerate zero points.  Scale matches chiVe's typical norm.
    matched_norms = vectors[vectors.abs().sum(dim=-1) > 0].norm(dim=-1)
    if len(matched_norms) > 0:
        avg_norm = float(matched_norms.mean().item())
    else:
        avg_norm = 1.0

    unmatched_mask = vectors.abs().sum(dim=-1) == 0
    unmatched_mask[0] = False  # keep PAD at zero
    n_unmatched = int(unmatched_mask.sum().item())
    if n_unmatched > 0:
        rand_vecs = torch.randn(n_unmatched, CHIVE_DIM)
        rand_vecs = rand_vecs / rand_vecs.norm(dim=-1, keepdim=True) * avg_norm
        vectors[unmatched_mask] = rand_vecs
        console.print(
            f"  Initialized {n_unmatched:,} unmatched tokens "
            f"with random vectors (norm≈{avg_norm:.2f})"
        )

    torch.save(vectors, cache_path)
    console.print(f"  Saved to [bold]{cache_path}[/bold]")
    return cache_path


def load_chive_for_vocab(surface_vocab: dict[str, int]) -> torch.Tensor:
    """Load pre-extracted chiVe vectors from the labeling-time cache.

    Falls back to scanning the full text file if the cache doesn't exist.

    Returns:
        Tensor of shape (vocab_size, 300).
    """
    cache_path = get_chive_cache_path()

    if os.path.exists(cache_path):
        console.print(f"Loading cached chiVe vectors from [bold]{cache_path}[/bold]")
        vectors: torch.Tensor = torch.load(
            cache_path, map_location="cpu", weights_only=True
        )
        nonzero = int((vectors.abs().sum(dim=-1) > 0).sum().item())
        vocab_size = max(surface_vocab.values()) + 1
        console.print(f"  {nonzero:,} / {vocab_size:,} tokens have chiVe vectors")
        return vectors

    console.print("[yellow]chiVe cache not found; scanning full text file...[/yellow]")
    extract_chive_for_vocab(surface_vocab)
    result: torch.Tensor = torch.load(cache_path, map_location="cpu", weights_only=True)
    return result


def load_pretrained_surface(
    embedding: "SurfaceEmbedding",
    weight: torch.Tensor,
    freeze: bool = True,
) -> int:
    """Load pretrained vectors into a SurfaceEmbedding's field layer.

    Args:
        embedding: the model's SurfaceEmbedding module.
        weight: tensor of shape (vocab_size, embed_dim).
        freeze: if True, the embedding parameters will not be updated.

    Returns:
        Number of non-zero rows loaded.
    """
    emb = embedding.embeddings["surface"]
    with torch.no_grad():
        emb.weight.copy_(weight[: emb.weight.size(0)])
        emb.weight[0].zero_()
    if freeze:
        emb.weight.requires_grad = False
    nonzero = int((weight.abs().sum(dim=-1) > 0).sum().item())
    return nonzero
