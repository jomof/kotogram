"""Shared utilities for Common Crawl scripts."""  # pylint: disable=too-many-lines

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from rich.progress import Progress

console = Console()

STYLE_MODEL_DIR = "models/style"
MAX_SENTENCE_LEN = 100

# ---------------------------------------------------------------------------
# Perf / cache diagnostic log  (.cc/perf-hist.log)
# ---------------------------------------------------------------------------

_perf_entries: list[str] = []
_perf_t0: float = 0.0  # pylint: disable=invalid-name


def perf_start_run(label: str, **kv: Any) -> None:
    """Begin a new perf-log block, clearing any prior entries."""
    global _perf_t0  # noqa: PLW0603  # pylint: disable=global-statement
    _perf_t0 = time.monotonic()
    _perf_entries.clear()
    _perf_entries.append(f"=== {label} {time.strftime('%Y-%m-%dT%H:%M:%S')} ===")
    if kv:
        _perf_entries.append("  ".join(f"{k}={v}" for k, v in kv.items()))


def perf_log(
    section: str, *, indent: int = 0, time_s: float | None = None, **kv: Any
) -> None:
    """Append one diagnostic line."""
    parts: list[str] = []
    for k, v in kv.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    prefix = "  " * indent
    body = f"{prefix}{section}  " + "  ".join(parts)
    if time_s is not None:
        body += f"  {time_s:.1f}s"
    _perf_entries.append(body)


def perf_log_dist(label: str, arr: np.ndarray, *, indent: int = 0) -> None:
    """Log min/p25/median/p75/max/mean/zeros for a 1-D array."""
    if arr.size == 0:
        perf_log(label, indent=indent, n=0)
        return
    n_zero = int((arr == 0).sum())
    perf_log(
        label,
        indent=indent,
        n=arr.size,
        min=float(arr.min()),
        p25=float(np.percentile(arr, 25)),
        median=float(np.median(arr)),
        p75=float(np.percentile(arr, 75)),
        max=float(arr.max()),
        mean=float(arr.mean()),
        zeros=n_zero,
    )


def perf_flush() -> None:
    """Write accumulated entries to .cc/perf-hist.log (append) and clear."""
    if not _perf_entries:
        return
    elapsed = time.monotonic() - _perf_t0
    _perf_entries.append(f"total  {elapsed:.1f}s")
    _perf_entries.append("")
    log_path = CC_CACHE_DIR / "perf-hist.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(_perf_entries) + "\n")
    _perf_entries.clear()


def clean_sentence(sentence: str) -> str:
    """Strip whitespace from a Japanese sentence."""
    return sentence.replace(" ", "").replace("\u3000", "").replace("\t", "")


COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
CC_DATA_BASE = "https://data.commoncrawl.org"
CC_CACHE_DIR = Path(".cc")
CORPUS_DB = Path("data/corpus.db")

_METADATA_TTL = 86400  # cache collinfo/index.html for 24h


def _user_agent() -> dict[str, str]:
    return {"User-Agent": "kotogram-cc/1.0"}


def _cache_path(*parts: str) -> Path:
    """Return a path under .cc/, creating parent dirs as needed."""
    p = CC_CACHE_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _is_fresh(path: Path, ttl: int) -> bool:
    """True if the file exists and was modified within ttl seconds."""
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl


_HTTP_TIMEOUT = 15


def fetch_text(url: str) -> str:
    """Fetch text content from a URL."""
    req = urllib.request.Request(url, headers=_user_agent())
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
        raw: bytes = resp.read()
    return raw.decode("utf-8")


def fetch_bytes(url: str) -> bytes:
    """Fetch raw bytes from a URL."""
    req = urllib.request.Request(url, headers=_user_agent())
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
        raw: bytes = resp.read()
    return raw


def _fetch_text_cached(url: str, *cache_parts: str, ttl: int = _METADATA_TTL) -> str:
    """Fetch text from a URL, caching to .cc/ with a TTL.

    Falls back to stale cache if the network request fails.
    """
    cached = _cache_path(*cache_parts)
    if _is_fresh(cached, ttl):
        return cached.read_text(encoding="utf-8")
    text = fetch_text(url)
    cached.write_text(text, encoding="utf-8")
    return text


def download_progress() -> "Progress":
    """Create a rich Progress bar configured for file downloads."""
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    return Progress(
        "[progress.description]{task.description}",
        BarColumn(bar_width=30),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    )


def download_to_cache(url: str, *cache_parts: str) -> Path:
    """Download a URL to .cc/ if not already cached. Returns local path."""
    dest = _cache_path(*cache_parts)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers=_user_agent())
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0) or None
        with download_progress() as progress:
            task = progress.add_task(f"    {cache_parts[-1]}", total=total)
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.advance(task, len(chunk))
    os.rename(tmp, dest)
    return dest


def _get_collinfo() -> list[dict[str, Any]]:
    """Fetch collinfo.json (cached for 24h)."""
    text = _fetch_text_cached(COLLINFO_URL, "collinfo.json")
    result: list[dict[str, Any]] = json.loads(text)
    return result


def get_latest_crawl_id() -> str:
    """Return the crawl ID of the most recent Common Crawl (e.g. 'CC-MAIN-2026-08')."""
    crawl_id: str = _get_collinfo()[0]["id"]
    return crawl_id


def get_crawl_info(crawl_id: str) -> dict[str, Any]:
    """Return metadata dict for a crawl from collinfo.json."""
    for c in _get_collinfo():
        if c["id"] == crawl_id:
            return c
    raise ValueError(f"Crawl {crawl_id} not found in collinfo.json")


def format_bytes(n: float) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} EB"


def get_crawl_sizes(crawl_id: str) -> dict[str, float]:
    """Parse WARC/WET/WAT total sizes (in TiB) from the crawl's index page.

    Returns dict like {"warc": 79.51, "wet": 5.96, "wat": 14.54}.
    """
    html = _fetch_text_cached(
        f"{CC_DATA_BASE}/crawl-data/{crawl_id}/index.html",
        crawl_id,
        "index.html",
    )
    sizes: dict[str, float] = {}
    for kind in ("WARC", "WET", "WAT"):
        pattern = (
            rf"<td>\s*{kind}\s*</td>"
            r".*?<td[^>]*>\s*([\d.]+)\s*</td>\s*</tr>"
        )
        match = re.search(pattern, html, re.DOTALL)
        if match:
            sizes[kind.lower()] = float(match.group(1))
    return sizes


# ---------------------------------------------------------------------------
# Parallel Sudachi parse + tokeniser encode
# ---------------------------------------------------------------------------


def parse_kotogram_chunk(sentences: list[str]) -> list[str]:
    """Worker: parse raw sentences to kotogram with TRAINING_MASK (exemplar surfaces)."""
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

    parser = SudachiJapaneseParser()
    return [parser.japanese_to_kotogram(s, fmt="TrainingMask") for s in sentences]


_TOKENIZER_PATH: str = ""


def set_tokenizer_path(path: str) -> None:
    """Set the tokenizer JSON path for worker processes."""
    global _TOKENIZER_PATH  # noqa: PLW0603  # pylint: disable=global-statement
    _TOKENIZER_PATH = path


def encode_kotogram_chunk(kotograms: list[str]) -> list[dict[str, list[int]]]:
    """Worker: encode kotogram strings to token-id dicts."""
    from kotogram.tokenizer import Tokenizer

    path = _TOKENIZER_PATH or f"{STYLE_MODEL_DIR}/tokenizer.json"
    tokenizer = Tokenizer.load(path)
    return [tokenizer.encode(k) for k in kotograms]


def parallel_parse_and_encode(
    sentences: list[str],
) -> list[dict[str, list[int]]]:
    """Parse sentences to kotograms then encode, both in parallel."""
    from rich.progress import Progress as RichProgress

    _t0_pe = time.monotonic()
    num_workers = max(1, mp.cpu_count() or 1)
    chunk_size = max(1, len(sentences) // (num_workers * 4))

    chunks = [
        sentences[i : i + chunk_size] for i in range(0, len(sentences), chunk_size)
    ]

    console.print(f"  Parsing {len(sentences):,} sentences ({num_workers} workers)...")
    kotograms: list[str] = []
    with RichProgress(console=console) as progress:
        task = progress.add_task("  Parsing to kotogram...", total=len(sentences))
        with mp.Pool(num_workers) as pool:
            for result in pool.imap(parse_kotogram_chunk, chunks):
                kotograms.extend(result)
                progress.advance(task, len(result))

    enc_chunks = [
        kotograms[i : i + chunk_size] for i in range(0, len(kotograms), chunk_size)
    ]
    encoded: list[dict[str, list[int]]] = []
    with RichProgress(console=console) as progress:
        task = progress.add_task("  Encoding tokens...", total=len(kotograms))
        with mp.Pool(num_workers) as enc_pool:
            for enc_result in enc_pool.imap(encode_kotogram_chunk, enc_chunks):
                encoded.extend(enc_result)
                progress.advance(task, len(enc_result))

    perf_log(
        "parse_and_encode",
        indent=1,
        n=len(sentences),
        workers=num_workers,
        time_s=time.monotonic() - _t0_pe,
    )
    return encoded


# ---------------------------------------------------------------------------
# Content filters
# ---------------------------------------------------------------------------

GRAMMATIC_SOFT_MIN = 0.3

_EXPLICIT_KEYWORDS: tuple[str, ...] = (
    "エロ",
    "セックス",
    "デリヘル",
    "風俗",
    "ソープ",
    "ヘルス",
    "ピンサロ",
    "パイズリ",
    "フェラ",
    "クンニ",
    "中出し",
    "生挿入",
    "潮吹き",
    "クリトリス",
    "チンポ",
    "チ○ポ",
    "チ〇ポ",
    "〇",  # common mask in adult/obscene spellings (ま〇こ, etc.)
    "マンコ",
    "オナニー",
    "アナル",
    "痴女",
    "淫乱",
    "射精",
    "勃起",
    "騎乗位",
    "バイブ",
    "ヌード",
    "全裸",
    "ハメ撮り",
    "乱交",
    "輪姦",
    "レイプ",
    "巨乳",
    "爆乳",
    "おっぱい",
    "パンチラ",
    "下着姿",
    "裏ビデオ",
    "AV女優",
    "アダルト",
    "援交",
    "ヤレる",
    "失禁",
    "調教",
    "緊縛",
    "奴隷",
    "ご奉仕",
    "おまんこ",
    "まんこ",
    "オナ電",
    "オナホ",
    "ちんこ",
    "ちんぽ",
    "おちんちん",
    "性器",
    "陰茎",
    "膣",
    "挿入",
    "絶頂",
    "手コキ",
    "素股",
    "本番",
    "抜き",
    "ぶっかけ",
    "顔射",
    "口内発射",
    "3P",
    "乱痴気",
    "ナンパ",
    "キャバクラ",
    "いちゃキャバ",
    "キャバ嬢",
    "ホストクラブ",
    "ガールズバー",
)

CHAR_FILTER = re.compile(
    r"[a-zA-Z`'()（）『』【】〔〕｛｝\[\]{}<>〈〉《》〖〗〘〙※・_\u00A1"
    r"#$;@\\^«Ø©®°·÷‐−‟′〓§\u3030"
    r"\u0080-\u04FF"
    r"\u0500-\u0E7F"
    r"\u1000-\u1FFF"
    r"\u200B-\u2FFF"
    r"\u3003\u3006\u3012\u3013"
    r"\u3033\u3035\u3097\u3099\u309D"
    r"\u301A-\u301F"
    r"\uA000-\uA63F"
    r"\uAC00-\uD7AF"
    r"\uE000-\uF8FF"
    r"\uFA11"
    r"\uFE00-\uFE0F"
    r"\uFEFF\uFFFC\uFFFD"
    r"\U00010000-\U0001FAFF"
    r"\U000E0100-\U000E01EF"
    r"\U000F0000-\U0010FFFF"
    r"]",
)

_SPAM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"0\d{1,4}[-ー]\d{1,4}[-ー]\d{2,5}"),
    re.compile(r"https?://"),
    re.compile(r"www\."),
    re.compile(r"！{3,}"),
    re.compile(r"!{3,}"),
    re.compile(r"ー{5,}"),
    re.compile(r"高価買取"),
    re.compile(r"無料見積"),
    re.compile(r"お問い合わせ"),
    re.compile(r"今すぐ(?:お?電話|ご連絡|クリック)"),
    re.compile(r"(?:送料|手数料)無料"),
    re.compile(r"キャンペーン(?:中|実施)"),
    re.compile(r"公式サイト"),
)


def content_ok(sentence: str) -> bool:
    """Return False if the sentence is too long, explicit, or spammy."""
    if len(sentence) > MAX_SENTENCE_LEN:
        return False
    if CHAR_FILTER.search(sentence):
        return False
    for kw in _EXPLICIT_KEYWORDS:
        if kw in sentence:
            return False
    for pat in _SPAM_PATTERNS:
        if pat.search(sentence):
            return False
    return True


# ---------------------------------------------------------------------------
# Diversity scoring via GPU nearest-neighbour (matmul decomposition)
# ---------------------------------------------------------------------------

_NN_CHUNK_SIZE = 4096
_NN_CORPUS_TILE = 32_768


def diversity_scores(  # pylint: disable=too-many-locals
    cc_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    device: Any = None,
) -> np.ndarray:
    """L2 distance from each CC embedding to its nearest corpus.db neighbour.

    Two-phase approach to avoid catastrophic cancellation in the matmul
    decomposition (||a||²+||b||²-2a·b collapses to 0 in float32 when all
    embeddings share a large constant norm):

    Phase 1 (fast, approximate): matmul decomposition finds the *index* of
    the approximate nearest corpus neighbour for each CC row.

    Phase 2 (exact): direct ``(a - b)²`` subtraction for each CC row and
    its identified neighbour -- O(N·D), no cancellation.
    """
    import torch
    from rich.progress import Progress as RichProgress

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    _t0_nn = time.monotonic()
    corpus_f32 = np.array(corpus_embeddings, dtype=np.float32)
    corpus_t = torch.from_numpy(corpus_f32).to(device)
    corpus_sq_norms = (corpus_t * corpus_t).sum(dim=1)

    cc_norms = np.linalg.norm(cc_embeddings[: min(1000, len(cc_embeddings))], axis=1)
    corp_norms = corpus_sq_norms[: min(1000, len(corpus_sq_norms))].sqrt().cpu().numpy()
    perf_log(
        "emb_norms",
        indent=2,
        cc_min=float(cc_norms.min()),
        cc_max=float(cc_norms.max()),
        corp_min=float(corp_norms.min()),
        corp_max=float(corp_norms.max()),
    )

    result: np.ndarray = np.zeros(len(cc_embeddings), dtype=np.float32)

    console.print(f"  Nearest-neighbour query on [bold]{device}[/bold]...")
    with RichProgress(console=console) as progress:
        task = progress.add_task("  Querying...", total=len(cc_embeddings))
        for i in range(0, len(cc_embeddings), _NN_CHUNK_SIZE):
            raw = cc_embeddings[i : i + _NN_CHUNK_SIZE]
            chunk_t = torch.from_numpy(np.array(raw, dtype=np.float32)).to(device)
            chunk_sq_norms = (chunk_t * chunk_t).sum(dim=1, keepdim=True)

            # Phase 1: find approximate NN index via matmul decomposition
            min_sq = torch.full((len(chunk_t),), float("inf"), device=device)
            nn_idx = torch.zeros(len(chunk_t), dtype=torch.long, device=device)
            for j in range(0, len(corpus_t), _NN_CORPUS_TILE):
                c_tile = corpus_t[j : j + _NN_CORPUS_TILE]
                c_norms = corpus_sq_norms[j : j + _NN_CORPUS_TILE]
                sq_d = (
                    chunk_sq_norms + c_norms.unsqueeze(0) - 2.0 * (chunk_t @ c_tile.T)
                )
                tile_min, tile_argmin = sq_d.min(dim=1)
                improved = tile_min < min_sq
                min_sq[improved] = tile_min[improved]
                nn_idx[improved] = tile_argmin[improved] + j

            # Phase 2: exact distance via direct subtraction
            nn_emb = corpus_t[nn_idx]
            diff = chunk_t - nn_emb
            exact_sq = (diff * diff).sum(dim=1)
            result[i : i + len(chunk_t)] = exact_sq.clamp(min=0.0).sqrt().cpu().numpy()
            progress.advance(task, len(chunk_t))

    perf_log(
        "nn_query",
        indent=1,
        cc=len(cc_embeddings),
        corpus=len(corpus_embeddings),
        time_s=time.monotonic() - _t0_nn,
    )
    return result


# ---------------------------------------------------------------------------
# Model hashing + inference cache helpers
# ---------------------------------------------------------------------------

_CACHE_META_NAME = "inference-cache-meta.json"


def model_hash(
    *,
    layer_mask: str = "",
    output_rank: int = 0,
) -> str:
    """Cache key combining checkpoint ID with distillation variant.

    Changes when the model is retrained *or* when the distillation
    parameters (layer mask, output rank) change.
    """
    from scripts.checkpoint import read_lock

    lock = read_lock()
    if lock is None:
        raise FileNotFoundError(
            "checkpoint.lock not found. Run: scripts/cc checkpoint pull recon_bpd"
        )
    ckpt_id: str = lock["checkpoint_id"]
    variant = f"{ckpt_id}|mask={layer_mask}|rank={output_rank}"
    return hashlib.md5(variant.encode()).hexdigest()[:12]


def is_cache_valid(cache_dir: Path, model_md5: str) -> bool:
    """Check whether an inference cache directory matches the current model."""
    meta_path = cache_dir / _CACHE_META_NAME
    if not meta_path.exists():
        return False
    meta: dict[str, str] = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("model_hash") == model_md5


def write_cache_meta(
    cache_dir: Path, model_md5: str, *, sentences_fp: str = ""
) -> None:
    """Write/overwrite model hash metadata for an inference cache directory."""
    meta: dict[str, str] = {"model_hash": model_md5}
    if sentences_fp:
        meta["sentences_fp"] = sentences_fp
    (cache_dir / _CACHE_META_NAME).write_text(json.dumps(meta), encoding="utf-8")


def _read_sentences_fp(cache_dir: Path) -> str:
    """Read the stored sentence-list fingerprint, or '' if absent."""
    meta_path = cache_dir / _CACHE_META_NAME
    if not meta_path.exists():
        return ""
    meta: dict[str, str] = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("sentences_fp", "")


def sentences_fingerprint(sentences: list[str]) -> str:
    """Fast SHA-256 fingerprint of a sentence list (O(1) memory)."""
    h = hashlib.sha256()
    for s in sentences:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Batched inference -> embeddings + uncertainty + grammaticality
# ---------------------------------------------------------------------------

BPD_BATCH_SIZE = 64


def embed_and_score(  # pylint: disable=too-many-locals
    encoded_all: list[dict[str, list[int]]],
    model: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run batched BPD inference: embeddings + reconstruction uncertainty.

    Returns (embeddings, uncertainty, gram_probs):
      embeddings:  (N, d_model) float32  -- pooled encoder representations
      uncertainty: (N,) float32  -- per-sentence BPD (bits per token)
      gram_probs:  (N,) float32  -- always 1.0 (no grammaticality head yet)
    """
    import random

    import torch
    from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

    from scripts.recon_bpd.inference import embed_and_bpd

    _t0_es = time.monotonic()
    total = len(encoded_all)
    lengths = np.array([len(e["surface"]) for e in encoded_all], dtype=np.int64)

    # Build batch index ranges, then shuffle so long sentences are spread
    # across the run instead of clustering at the end.
    batch_ranges: list[np.ndarray] = []
    sorted_order = np.argsort(lengths)
    for start in range(0, len(sorted_order), BPD_BATCH_SIZE):
        batch_ranges.append(sorted_order[start : start + BPD_BATCH_SIZE])
    random.shuffle(batch_ranges)

    d_model = model.cfg.d_model
    embeddings = np.zeros((total, d_model), dtype=np.float32)
    uncertainty = np.zeros(total, dtype=np.float32)
    gram_probs_out = np.ones(total, dtype=np.float32)

    is_mps = device.type == "mps"

    console.print(f"  Inference on [bold]{device}[/bold]...")
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("  Embedding + BPD scoring...", total=total)
        scored = 0
        for batch_idx in batch_ranges:
            batch_encoded = [encoded_all[i] for i in batch_idx]
            n_batch = len(batch_encoded)
            max_len = int(lengths[batch_idx].max())

            padded = np.zeros((n_batch, max_len), dtype=np.int64)
            mask_np = np.zeros((n_batch, max_len), dtype=np.float32)
            for i, enc in enumerate(batch_encoded):
                ids = enc["surface"]
                seq_len = len(ids)
                padded[i, :seq_len] = ids
                mask_np[i, :seq_len] = 1.0

            surface_ids = torch.from_numpy(padded).to(device)
            mask = torch.from_numpy(mask_np).to(device)

            with torch.inference_mode():
                use_vec = getattr(model, "_distilled", False)
                pooled, bpd = embed_and_bpd(
                    model, surface_ids, mask, vectorized=use_vec
                )

            embeddings[batch_idx] = pooled.cpu().numpy()
            uncertainty[batch_idx] = bpd.cpu().numpy()

            if is_mps:
                torch.mps.empty_cache()

            scored += n_batch
            elapsed = time.monotonic() - _t0_es
            els = scored / elapsed if elapsed > 0 else 0
            progress.update(
                task, advance=n_batch, description=f"  BPD scoring ({els:.0f} el/s)"
            )

    elapsed_s = time.monotonic() - _t0_es
    mean_bpd = float(uncertainty.mean())
    median_bpd = float(np.median(uncertainty))
    console.print(
        f"  BPD: mean={mean_bpd:.4f}  median={median_bpd:.4f}"
        f"  ({total:,} sentences in {elapsed_s:.1f}s, {total / elapsed_s:.0f} el/s)"
    )
    perf_log(
        "embed_and_score",
        indent=1,
        n=total,
        batches=len(batch_ranges),
        mean_bpd=mean_bpd,
        median_bpd=median_bpd,
        time_s=elapsed_s,
    )
    return embeddings, uncertainty, gram_probs_out


# ---------------------------------------------------------------------------
# Shared sentence-level embedding store
# ---------------------------------------------------------------------------
# Both the CC and corpus paths feed through this store so that the same
# (model, sentence) pair is never embedded twice -- regardless of whether
# it first appeared as a CC candidate or a corpus member.


class EmbedStore:
    """Sentence→embedding cache shared across CC and corpus paths.

    Backed by flat files under ``.cc/embed-store/{model_hash[:12]}/``:
    - ``sents.txt`` – one sentence per line (append-only)
    - ``embed.npy`` – ``[N, d_model]`` float32
    - ``bpd.npy``   – ``[N]`` float32

    Keyed by model hash so a model retrain starts a fresh store.
    """

    def __init__(self, model_md5: str, d_model: int = 512) -> None:  # pylint: disable=unused-argument
        prefix = model_md5[:12]
        self._dir = CC_CACHE_DIR / "embed-store" / prefix
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sents_path = self._dir / "sents.txt"
        self._embed_path = self._dir / "embed.npy"
        self._bpd_path = self._dir / "bpd.npy"

        self._index: dict[str, int] = {}
        self._embed: np.ndarray | None = None
        self._bpd: np.ndarray | None = None
        self._dirty = False
        self._load()

    # -- persistence --

    def _load(self) -> None:
        if not (self._sents_path.exists() and self._embed_path.exists()):
            return
        sents = self._sents_path.read_text(encoding="utf-8").splitlines()
        embed = np.load(str(self._embed_path))
        if embed.shape[0] != len(sents):
            console.print(
                f"  [yellow]EmbedStore: count mismatch "
                f"(embed={embed.shape[0]}, sents={len(sents)}), resetting[/yellow]"
            )
            return
        self._embed = embed
        self._index = {s: i for i, s in enumerate(sents)}
        if self._bpd_path.exists():
            bpd = np.load(str(self._bpd_path))
            if bpd.shape[0] == len(sents):
                self._bpd = bpd

    def flush(self) -> None:
        """Write any pending additions to disk."""
        if not self._dirty or self._embed is None:
            return
        np.save(str(self._embed_path), self._embed)
        if self._bpd is not None:
            np.save(str(self._bpd_path), self._bpd)
        sents = [""] * len(self._index)
        for s, i in self._index.items():
            sents[i] = s
        self._sents_path.write_text("\n".join(sents) + "\n", encoding="utf-8")
        self._dirty = False

    # -- query / update --

    def lookup_embeddings(
        self, sentences: list[str]
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Return ``(hits, misses)`` where hits maps sentence→embedding row."""
        hits: dict[str, np.ndarray] = {}
        misses: list[str] = []
        for s in sentences:
            idx = self._index.get(s)
            if idx is not None and self._embed is not None:
                hits[s] = self._embed[idx]
            else:
                misses.append(s)
        return hits, misses

    def store(
        self,
        sentences: list[str],
        embeddings: np.ndarray,
        bpds: np.ndarray | None = None,
    ) -> None:
        """Add new sentence→embedding mappings (deduplicates internally)."""
        new_s: list[str] = []
        new_rows: list[int] = []
        for i, s in enumerate(sentences):
            if s not in self._index:
                self._index[s] = len(self._index)
                new_s.append(s)
                new_rows.append(i)
        if not new_s:
            return

        new_embed = embeddings[new_rows]
        if self._embed is not None:
            self._embed = np.concatenate([self._embed, new_embed])
        else:
            self._embed = new_embed.copy()

        if bpds is not None:
            new_bpd = bpds[new_rows]
            if self._bpd is not None:
                self._bpd = np.concatenate([self._bpd, new_bpd])
            else:
                self._bpd = new_bpd.copy()

        self._dirty = True

    @property
    def count(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# Corpus embedding cache
# ---------------------------------------------------------------------------


def corpus_embed_path() -> Path:
    """Path to the cached corpus embeddings .npy file."""
    return CC_CACHE_DIR / "corpus-embeddings.npy"


def _load_corpus_sentences() -> list[str]:
    import sqlite3

    conn = sqlite3.connect(str(CORPUS_DB))
    rows = conn.execute("SELECT sentence FROM sentences WHERE grammatic = 1").fetchall()
    conn.close()
    return [r[0] for r in rows]


_CORPUS_EMBED_META = "corpus-embed-meta.json"
_CORPUS_SENTS_CACHE = "corpus-sentences.txt"


def get_corpus_embeddings(  # pylint: disable=too-many-locals
    model: Any,
    device: Any,
    model_md5: str,
    embed_store: EmbedStore | None = None,
) -> np.ndarray:
    """Load or compute corpus.db embeddings (cached to disk).

    Incremental: caches a sentence→embedding mapping so only sentences
    absent from the cache need to be parsed/encoded/embedded.  The output
    array is arranged as [reused | new] so that ``corpus_emb[old_n:]``
    gives exactly the newly-added rows, enabling incremental diversity
    score updates downstream.

    A ``prefix_fp`` (fingerprint of the reused portion) is stored alongside
    a ``full_fp`` (fingerprint of the entire array).  Downstream caches can
    compare their stored ``corpus_fp`` against the new ``prefix_fp`` to
    verify that the old corpus is an exact prefix of the new one; if not
    (e.g. sentences were removed), they fall back to a full recompute.

    If *embed_store* is provided, it is consulted before running inference
    for "new" sentences, avoiding redundant computation for sentences
    already embedded as CC candidates.
    """
    _t0_ce = time.monotonic()
    cache = corpus_embed_path()
    meta_path = CC_CACHE_DIR / _CORPUS_EMBED_META
    sents_path = CC_CACHE_DIR / _CORPUS_SENTS_CACHE

    sentences = _load_corpus_sentences()
    n = len(sentences)
    current_set = set(sentences)

    old_emb: np.ndarray | None = None
    old_sents: list[str] = []

    if cache.exists() and meta_path.exists() and sents_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cached_hash = meta.get("model_hash", "")
        if cached_hash == model_md5:
            old_sents = sents_path.read_text(encoding="utf-8").splitlines()
            existing: np.ndarray = np.load(str(cache))
            if existing.shape[0] == len(old_sents):
                old_emb = existing
            else:
                console.print(
                    f"  Corpus cache invalidated: embedding count"
                    f" ({existing.shape[0]}) != sentence count ({len(old_sents)})"
                )
        else:
            console.print(
                f"  Corpus cache invalidated: model changed"
                f" ({cached_hash} -> {model_md5})"
            )
    elif cache.exists() or meta_path.exists() or sents_path.exists():
        missing = [
            name
            for name, p in [
                ("embeddings", cache),
                ("meta", meta_path),
                ("sentences", sents_path),
            ]
            if not p.exists()
        ]
        console.print(f"  Corpus cache invalidated: missing {', '.join(missing)}")

    cached_map: dict[str, int] = {}
    if old_emb is not None:
        cached_map = {s: i for i, s in enumerate(old_sents) if s in current_set}

    reused_sents = [s for s in old_sents if s in current_set]
    reused_src = [cached_map[s] for s in reused_sents]
    new_sentences = [s for s in sentences if s not in cached_map]

    # -- Full cache hit --
    if not new_sentences:
        console.print(f"  Corpus embeddings loaded from cache ({n:,} sentences)")
        if old_emb is not None and len(reused_sents) == old_emb.shape[0]:
            perf_log(
                "corpus_embeddings",
                cache="hit",
                reused=n,
                new=0,
                total=n,
                time_s=time.monotonic() - _t0_ce,
            )
            return old_emb
        emb: np.ndarray = old_emb[reused_src] if old_emb is not None else np.empty(0)
        np.save(str(cache), emb)
        sents_path.write_text("\n".join(reused_sents) + "\n", encoding="utf-8")
        meta_out = {
            "model_hash": model_md5,
            "count": n,
            "prefix_fp": sentences_fingerprint(reused_sents),
            "full_fp": sentences_fingerprint(reused_sents),
        }
        meta_path.write_text(json.dumps(meta_out), encoding="utf-8")
        perf_log(
            "corpus_embeddings",
            cache="hit_trimmed",
            reused=n,
            new=0,
            total=n,
            time_s=time.monotonic() - _t0_ce,
        )
        return emb

    # -- Partial hit or full miss --
    # Check the shared embedding store for sentences already scored as CC
    # candidates (avoids re-embedding after curate upsert).
    store_hits: dict[str, np.ndarray] = {}
    if embed_store is not None and new_sentences:
        store_hits, new_sentences = embed_store.lookup_embeddings(new_sentences)

    if cached_map or store_hits:
        console.print(
            f"  Corpus embeddings: {len(cached_map):,} file-cached,"
            f" {len(store_hits):,} from embed-store,"
            f" {len(new_sentences):,} new"
        )
    else:
        console.print("  Computing corpus.db embeddings...")

    console.print(f"  Grammatic corpus sentences: {n:,}")

    # Order: [reused from file cache | hits from shared store | freshly computed]
    parts: list[np.ndarray] = []
    if reused_src and old_emb is not None:
        parts.append(old_emb[reused_src])

    store_hit_sents: list[str] = []
    if store_hits:
        store_hit_sents = [s for s in sentences if s in store_hits]
        parts.append(np.stack([store_hits[s] for s in store_hit_sents]))

    if new_sentences:
        encoded = parallel_parse_and_encode(new_sentences)
        new_emb, _unc, _gp = embed_and_score(encoded, model, device)
        if embed_store is not None:
            embed_store.store(new_sentences, new_emb)
            embed_store.flush()
        parts.append(new_emb)
    else:
        new_emb = np.empty((0, 0))
    emb = np.concatenate(parts) if len(parts) > 1 else parts[0]

    all_sents = reused_sents + store_hit_sents + new_sentences
    np.save(str(cache), emb)
    sents_path.write_text("\n".join(all_sents) + "\n", encoding="utf-8")
    meta_out = {
        "model_hash": model_md5,
        "count": n,
        "prefix_fp": sentences_fingerprint(reused_sents),
        "full_fp": sentences_fingerprint(all_sents),
    }
    meta_path.write_text(json.dumps(meta_out), encoding="utf-8")
    _cache_kind = "partial" if cached_map else "miss"
    perf_log(
        "corpus_embeddings",
        cache=_cache_kind,
        reused=len(reused_sents),
        store_hits=len(store_hit_sents),
        new=len(new_sentences),
        total=n,
        time_s=time.monotonic() - _t0_ce,
    )
    console.print(f"  Cached to {cache}")
    return emb


# ---------------------------------------------------------------------------
# CC inference cache
# ---------------------------------------------------------------------------


_INFERENCE_CHUNK = 50_000


def _cc_score_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    return (
        cache_dir / "cc-embeddings.npy",
        cache_dir / "cc-uncertainty.npy",
        cache_dir / "cc-gram-probs.npy",
    )


def get_cc_scores(  # pylint: disable=too-many-locals,too-many-positional-arguments
    crawl_id: str,
    cc_sentences: list[str],
    model: Any,
    device: Any,
    model_md5: str,
    embed_store: EmbedStore | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or compute CC inference results, streamed in chunks.

    Embeddings are returned as a memory-mapped array to avoid loading the
    full (N, d_model) matrix into RAM.

    If *embed_store* is provided, newly computed embeddings are written to
    the shared store so that downstream corpus embedding computation can
    reuse them without re-inference.
    """
    _t0_cc = time.monotonic()
    cache_dir = _cache_path(crawl_id, "inference")
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path, unc_path, gp_path = _cc_score_paths(cache_dir)

    n = len(cc_sentences)
    cached_n = 0
    _cc_cache = "miss_no_file"

    if not emb_path.exists():
        _cc_cache = "miss_no_file"
    elif not is_cache_valid(cache_dir, model_md5):
        _cc_cache = "miss_model"
        _meta_path = cache_dir / _CACHE_META_NAME
        _old_hash = ""
        if _meta_path.exists():
            _old_hash = json.loads(_meta_path.read_text(encoding="utf-8")).get(
                "model_hash", ""
            )
        console.print(
            f"  CC inference cache invalidated: model changed"
            f" ({_old_hash} -> {model_md5})"
        )

    if emb_path.exists() and is_cache_valid(cache_dir, model_md5):
        existing: np.ndarray = np.load(str(emb_path), mmap_mode="r")
        cached_n = existing.shape[0]
        del existing
        if cached_n > n:
            console.print("  Cache stale (sentence count shrank), recomputing...")
            _cc_cache = f"stale_shrunk(cached={cached_n},now={n})"
            cached_n = 0
        elif cached_n > 0:
            stored_fp = _read_sentences_fp(cache_dir)
            current_fp = sentences_fingerprint(cc_sentences[:cached_n])
            if stored_fp != current_fp:
                console.print("  Cache stale (sentence list changed), recomputing...")
                _cc_cache = f"stale_fp(stored={stored_fp[:8]},now={current_fp[:8]})"
                cached_n = 0
            elif cached_n == n:
                console.print(f"  CC inference loaded from cache (model {model_md5})")
                emb_out = np.load(str(emb_path), mmap_mode="r")
                unc_out = np.load(str(unc_path))
                if embed_store is not None:
                    embed_store.store(cc_sentences, np.array(emb_out), unc_out)
                    embed_store.flush()
                perf_log(
                    "cc_scores",
                    cache="hit",
                    cached_n=cached_n,
                    total=n,
                    chunks=0,
                    time_s=time.monotonic() - _t0_cc,
                )
                return (emb_out, unc_out, np.load(str(gp_path)))
            else:
                _cc_cache = "partial"
                console.print(
                    f"  CC inference: {cached_n:,} cached,"
                    f" {n - cached_n:,} new to score"
                )

    if not cached_n:
        console.print("  Running inference (no valid cache)...")

    # Seed arrays from any valid cached prefix
    emb_parts: list[np.ndarray] = []
    unc_parts: list[np.ndarray] = []
    gp_parts: list[np.ndarray] = []

    if cached_n > 0:
        old_emb: np.ndarray = np.load(str(emb_path))
        old_unc: np.ndarray = np.load(str(unc_path))
        emb_parts.append(old_emb[:cached_n])
        unc_parts.append(old_unc[:cached_n])
        gp_parts.append(np.load(str(gp_path))[:cached_n])
        if embed_store is not None:
            embed_store.store(
                cc_sentences[:cached_n], old_emb[:cached_n], old_unc[:cached_n]
            )
            embed_store.flush()
        del old_emb, old_unc

    remaining = n - cached_n
    n_chunks = (remaining + _INFERENCE_CHUNK - 1) // _INFERENCE_CHUNK
    for ci in range(n_chunks):
        start = cached_n + ci * _INFERENCE_CHUNK
        end = min(start + _INFERENCE_CHUNK, n)
        if n_chunks > 1:
            console.print(
                f"\n  [bold]Chunk {ci + 1}/{n_chunks}[/bold]"
                f" ({end - start:,} sentences)"
            )
        chunk_sents = cc_sentences[start:end]
        encoded = parallel_parse_and_encode(chunk_sents)
        c_emb, c_unc, c_gp = embed_and_score(encoded, model, device)
        emb_parts.append(c_emb)
        unc_parts.append(c_unc)
        gp_parts.append(c_gp)
        del encoded

        if embed_store is not None:
            embed_store.store(chunk_sents, c_emb, c_unc)
            embed_store.flush()

        # Flush after every chunk so partial progress survives crashes
        _flush_emb = np.concatenate(emb_parts)
        _flush_unc = np.concatenate(unc_parts)
        _flush_gp = np.concatenate(gp_parts)
        np.save(str(emb_path), _flush_emb)
        np.save(str(unc_path), _flush_unc)
        np.save(str(gp_path), _flush_gp)
        write_cache_meta(
            cache_dir,
            model_md5,
            sentences_fp=sentences_fingerprint(cc_sentences[:end]),
        )
        del _flush_emb, _flush_unc, _flush_gp
        console.print(f"    Cached {end:,}/{n:,} sentences")
    perf_log(
        "cc_scores",
        cache=_cc_cache,
        cached_n=cached_n,
        total=n,
        chunks=n_chunks,
        time_s=time.monotonic() - _t0_cc,
    )
    console.print(f"  Cached to {cache_dir}")

    return (
        np.load(str(emb_path), mmap_mode="r"),
        np.load(str(unc_path)),
        np.load(str(gp_path)),
    )
