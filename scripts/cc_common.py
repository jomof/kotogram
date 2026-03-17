"""Shared utilities for Common Crawl scripts."""

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
    """Worker: parse raw sentences to kotogram format via Sudachi."""
    from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

    parser = SudachiJapaneseParser()
    return [parser.japanese_to_kotogram(s) for s in sentences]


def encode_kotogram_chunk(kotograms: list[str]) -> list[dict[str, list[int]]]:
    """Worker: encode kotogram strings to token-id dicts."""
    from kotogram.tokenizer import Tokenizer

    tokenizer = Tokenizer.load(f"{STYLE_MODEL_DIR}/tokenizer.json")
    return [tokenizer.encode(k) for k in kotograms]


def parallel_parse_and_encode(
    sentences: list[str],
) -> list[dict[str, list[int]]]:
    """Parse sentences to kotograms then encode, both in parallel."""
    from rich.progress import Progress as RichProgress

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
    r"[a-zA-Z`'()（）「」『』【】〔〕｛｝\[\]{}<>〈〉《》〖〗〘〙※・_\u00A1"
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
# Diversity scoring via sklearn (parallel nearest-neighbour)
# ---------------------------------------------------------------------------

_NN_INDEX: Any = None


def _nn_worker_init(corpus: np.ndarray) -> None:
    """Fit the NN index once per worker process."""
    from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

    global _NN_INDEX  # noqa: PLW0603  # pylint: disable=global-statement
    _NN_INDEX = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="brute")
    _NN_INDEX.fit(corpus)


def _nn_query_chunk(chunk: np.ndarray) -> np.ndarray:
    """Worker: query the pre-fitted index for a chunk of vectors."""
    dists, _ = _NN_INDEX.kneighbors(chunk)
    result: np.ndarray = dists[:, 0]
    return result


_NN_CHUNK_SIZE = 4096


def diversity_scores(
    cc_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
) -> np.ndarray:
    """L2 distance from each CC embedding to its nearest corpus.db neighbour."""
    from rich.progress import Progress as RichProgress

    corpus_f32: np.ndarray = corpus_embeddings.astype(np.float32)
    cc_f32: np.ndarray = cc_embeddings.astype(np.float32)

    num_workers = max(1, mp.cpu_count() or 1)
    chunks = [
        cc_f32[i : i + _NN_CHUNK_SIZE] for i in range(0, len(cc_f32), _NN_CHUNK_SIZE)
    ]

    result: np.ndarray = np.zeros(len(cc_f32), dtype=np.float32)

    console.print(f"  Nearest-neighbour query ({num_workers} workers)...")
    with RichProgress(console=console) as progress:
        task = progress.add_task("  Querying...", total=len(cc_f32))
        with mp.Pool(
            num_workers, initializer=_nn_worker_init, initargs=(corpus_f32,)
        ) as pool:
            offset = 0
            for chunk_result in pool.imap(_nn_query_chunk, chunks):
                result[offset : offset + len(chunk_result)] = chunk_result
                offset += len(chunk_result)
                progress.advance(task, len(chunk_result))

    return result


# ---------------------------------------------------------------------------
# Model hashing + inference cache helpers
# ---------------------------------------------------------------------------

_CACHE_META_NAME = "inference-cache-meta.json"


def model_hash() -> str:
    """MD5 of model.pt -- changes when the model is retrained."""
    h = hashlib.md5()  # noqa: S324
    with open(f"{STYLE_MODEL_DIR}/model.pt", "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:12]


def is_cache_valid(cache_dir: Path, model_md5: str) -> bool:
    """Check whether an inference cache directory matches the current model."""
    meta_path = cache_dir / _CACHE_META_NAME
    if not meta_path.exists():
        return False
    meta: dict[str, str] = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("model_hash") == model_md5


def write_cache_meta(cache_dir: Path, model_md5: str) -> None:
    """Write/overwrite model hash metadata for an inference cache directory."""
    (cache_dir / _CACHE_META_NAME).write_text(
        json.dumps({"model_hash": model_md5}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Batched inference -> embeddings + uncertainty + grammaticality
# ---------------------------------------------------------------------------

BATCH_SIZE = 256


def embed_and_score(  # pylint: disable=too-many-locals
    encoded_all: list[dict[str, list[int]]],
    model: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run batched inference in a single encoder pass.

    Returns (embeddings, uncertainty, gram_probs):
      embeddings:  (N, d_model) float32
      uncertainty: (N,) float32  -- mean entropy across classification heads
      gram_probs:  (N,) float32  -- P(grammatic) per sentence
    """
    import torch
    from rich.progress import Progress

    from kotogram.tokenizer import ENCODER_FEATURE_FIELDS, FEATURE_FIELDS

    total = len(encoded_all)
    lengths = [len(e[FEATURE_FIELDS[0]]) for e in encoded_all]
    order = sorted(range(total), key=lambda idx: lengths[idx])

    d_model = model.config.d_model
    embeddings = np.zeros((total, d_model), dtype=np.float32)
    uncertainty = np.zeros(total, dtype=np.float32)
    gram_probs_out = np.zeros(total, dtype=np.float32)

    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        return -(probs * probs.clamp(min=1e-8).log()).sum(-1)

    console.print(f"  Inference on [bold]{device}[/bold]...")
    with Progress(console=console) as progress:
        task = progress.add_task("  Embedding + scoring...", total=total)
        for start in range(0, len(order), BATCH_SIZE):
            batch_idx = order[start : start + BATCH_SIZE]
            batch_encoded = [encoded_all[i] for i in batch_idx]
            n_batch = len(batch_encoded)
            max_len = lengths[batch_idx[-1]]

            field_inputs = {}
            for field in ENCODER_FEATURE_FIELDS:
                ids_t = torch.zeros((n_batch, max_len), dtype=torch.long, device=device)
                for i, enc in enumerate(batch_encoded):
                    ids = enc[field]
                    ids_t[i, : len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=device
                    )
                field_inputs[f"input_ids_{field}"] = ids_t

            mask = torch.zeros((n_batch, max_len), dtype=torch.long, device=device)
            for i, enc in enumerate(batch_encoded):
                mask[i, : len(enc[FEATURE_FIELDS[0]])] = 1

            with torch.no_grad():
                pooled = model.pool(field_inputs, mask)

                gram_logits = model.grammaticality_classifier(pooled)
                form_logits = model.formality_pragmatic_head(pooled)
                gend_logits = model.gender_pragmatic_head(pooled)

                gram_h = _entropy(gram_logits)
                form_h = _entropy(form_logits)
                gend_h = _entropy(gend_logits)
                ent = (gram_h + form_h + gend_h) / 3.0

                gp = torch.softmax(gram_logits, dim=-1)[:, 1]

            pooled_np = pooled.cpu().numpy()
            ent_np = ent.cpu().numpy()
            gp_np = gp.cpu().numpy()

            for i, orig_idx in enumerate(batch_idx):
                embeddings[orig_idx] = pooled_np[i]
                uncertainty[orig_idx] = ent_np[i]
                gram_probs_out[orig_idx] = gp_np[i]

            progress.advance(task, n_batch)

    return embeddings, uncertainty, gram_probs_out


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


def get_corpus_embeddings(model: Any, device: Any, model_md5: str) -> np.ndarray:
    """Load or compute corpus.db embeddings (cached to disk)."""
    cache = corpus_embed_path()
    if cache.exists() and is_cache_valid(CC_CACHE_DIR, model_md5):
        db_mtime = CORPUS_DB.stat().st_mtime
        if cache.stat().st_mtime > db_mtime:
            console.print("  Corpus embeddings loaded from cache")
            loaded: np.ndarray = np.load(str(cache))
            return loaded

    console.print("  Computing corpus.db embeddings...")
    sentences = _load_corpus_sentences()
    console.print(f"  Grammatic corpus sentences: {len(sentences):,}")
    encoded = parallel_parse_and_encode(sentences)
    emb, _unc, _gp = embed_and_score(encoded, model, device)

    np.save(str(cache), emb)
    write_cache_meta(CC_CACHE_DIR, model_md5)
    console.print(f"  Cached to {cache}")
    return emb


# ---------------------------------------------------------------------------
# CC inference cache
# ---------------------------------------------------------------------------


def get_cc_scores(
    crawl_id: str,
    cc_sentences: list[str],
    model: Any,
    device: Any,
    model_md5: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load or compute CC inference results (embeddings, uncertainty, gram_probs)."""
    cache_dir = _cache_path(crawl_id, "inference")
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "cc-scores.npz"

    cached_n = 0
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    if npz_path.exists() and is_cache_valid(cache_dir, model_md5):
        data = np.load(str(npz_path))
        cached_n = data["embeddings"].shape[0]
        if cached_n == len(cc_sentences):
            console.print(f"  CC inference loaded from cache (model {model_md5})")
            return data["embeddings"], data["uncertainty"], data["gram_probs"]
        if cached_n < len(cc_sentences):
            console.print(
                f"  CC inference: {cached_n:,} cached, "
                f"{len(cc_sentences) - cached_n:,} new to score"
            )
            parts.append((data["embeddings"], data["uncertainty"], data["gram_probs"]))
        else:
            console.print("  Cache stale (sentence count shrank), recomputing...")
            cached_n = 0

    if not cached_n:
        console.print("  Running inference (no valid cache)...")
    parts.append(
        embed_and_score(
            parallel_parse_and_encode(cc_sentences[cached_n:]),
            model,
            device,
        )
    )

    emb = np.concatenate([p[0] for p in parts])
    unc = np.concatenate([p[1] for p in parts])
    gp = np.concatenate([p[2] for p in parts])
    np.savez(str(npz_path), embeddings=emb, uncertainty=unc, gram_probs=gp)
    write_cache_meta(cache_dir, model_md5)
    console.print(f"  Cached to {npz_path}")
    return emb, unc, gp
