"""Dataset management: build, upload, download, list, info, resolve."""  # pylint: disable=too-many-lines

import argparse
import glob as glob_mod
import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from kotogram.tokenizer import (
    CLS_ID,
    CLS_TOKEN,
    ENCODER_FEATURE_FIELDS,
    FEATURE_FIELDS,
    MASK_ID,
    MASK_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    Tokenizer,
)
from scripts.dataset_token_histogram import (
    grammatical_token_gram_freq,
    grammatical_token_length_counts,
)
from scripts.gcs import (
    GCS_BUCKET,
    find_repo_root,
    gcs_download_file,
    gcs_exists,
    gcs_list_blobs,
    gcs_read_json,
    gcs_upload_file,
    gcs_write_json,
)
from train.binary_io import (
    EXT_FEAT_PREFIX,
    EXT_KC_PREFIX,
    EXT_LABELS,
    EXT_OFFSETS,
    LABEL_SPECS,
)
from train.dataset import StyleDataset
from train.kc import KcFamilyId
from train.types import Sample

SCHEMA_VERSION = 3
GCS_PREFIX = "kotogram-datasets"
DATASET_LOCK = "dataset.lock"
LOCAL_CACHE = os.path.join(".cache", "datasets")
CHIVE_DIM = 300
CORPUS_DB_PATH = os.path.join("data", "corpus.db")

_SPECIAL_TOKENS = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID,
    CLS_TOKEN: CLS_ID,
    MASK_TOKEN: MASK_ID,
}

_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "vocab",
        "offsets",
        "features",
        "labels",
        "content_mask",
        "sentences",
        "chive_id",
    }
)

# Minimum schema version we can still load (older bundles lack derived
# fields like token_gram_freq but are otherwise structurally compatible).
_MIN_SCHEMA_VERSION = 1

_LABEL_SPECS = LABEL_SPECS

# Backward-compatible aliases for the private GCS helpers.
_gcs_upload_file = gcs_upload_file
_gcs_download_file = gcs_download_file
_gcs_exists = gcs_exists
_gcs_list_blobs = gcs_list_blobs
_gcs_read_json = gcs_read_json
_gcs_write_json = gcs_write_json
_find_repo_root = find_repo_root


def read_lock() -> Optional[Dict[str, Any]]:
    """Read dataset.lock from the repo root. Returns None if missing."""
    from scripts.lock_io import read_lock_file

    return read_lock_file(os.path.join(_find_repo_root(), DATASET_LOCK))


def write_lock(
    dataset_id: str, chive_id: str, corpus_hash: Optional[str] = None
) -> str:
    """Write dataset.lock to the repo root. Returns the path written."""
    from scripts.lock_io import write_lock_file

    data: Dict[str, str] = {"dataset_id": dataset_id, "chive_id": chive_id}
    if corpus_hash is not None:
        data["corpus_hash"] = corpus_hash
    return write_lock_file(
        os.path.join(_find_repo_root(), DATASET_LOCK),
        data,
    )


def _verify_corpus_label_hash(db_path: str) -> str:
    """Check that corpus.db exists, has been labeled, and hasn't changed since.

    Returns the verified content hash.
    Raises ``SystemExit`` on any mismatch.
    """
    from scripts.corpus_hash import corpus_content_hash, read_metadata

    if not os.path.exists(db_path):
        raise SystemExit(
            f"corpus.db not found at {db_path}. "
            "Run 'python -m scripts.dataset corpus-download latest' first."
        )

    content_hash = corpus_content_hash(db_path)
    label_hash = read_metadata(db_path, "label_content_hash")

    if label_hash is None:
        raise SystemExit(
            "No labeling hash found in corpus.db metadata. "
            "Run labeling (scripts/label.py --source-db ...) before building."
        )

    if content_hash != label_hash:
        raise SystemExit(
            f"corpus.db has been modified since last labeling.\n"
            f"  Current content hash:  {content_hash[:16]}...\n"
            f"  Labeling content hash: {label_hash[:16]}...\n"
            "Re-run labeling first."
        )

    return content_hash


def _print_corpus_summary(db_path: str, content_hash: str) -> None:
    """Print corpus statistics so regressions are noticeable at a glance."""
    from scripts.corpus_hash import corpus_summary, read_metadata

    stats = corpus_summary(db_path)
    label_ts = read_metadata(db_path, "label_timestamp") or "unknown"

    print(f"\n  Corpus summary ({db_path}):")
    print(f"    Content hash:   {content_hash[:12]}...")
    print(f"    Labeled at:     {label_ts}")
    gram = stats["grammatic"]
    agram = stats["agrammatic"]
    total = stats["total_sentences"]
    print(f"    Sentences:      {total:,}  (grammatic {gram:,} / agrammatic {agram:,})")
    print(
        f"    Label coverage: formality {stats['formality_labeled']:,}, "
        f"gender {stats['gender_labeled']:,}"
    )
    print(
        f"    Grammar points: {stats['grammar_points']:,} GPs, "
        f"{stats['gp_pos_annotations']:,} pos / {stats['gp_neg_annotations']:,} neg annotations"
    )
    reg = stats.get("register_distribution", {})
    if reg:
        parts = [f"{label} {cnt:,}" for label, cnt in reg.items()]
        print(f"    Registers:      {', '.join(parts)}")
    print()


def _prepare_corpus_gz(db_path: str, content_hash: str) -> str:
    """VACUUM and gzip corpus.db into .cache/datasets/. Returns the .gz path."""
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    gz_path = os.path.join(LOCAL_CACHE, f"corpus-{content_hash}.db.gz")
    if os.path.exists(gz_path):
        print(f"  Corpus .gz already cached: {gz_path}")
        return gz_path

    print(f"  VACUUMing {db_path}...")
    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM")
    conn.close()

    print(f"  Compressing {db_path}...")
    with (
        open(db_path, "rb") as f_in,
        gzip.open(gz_path, "wb", compresslevel=6) as f_out,
    ):
        while True:
            chunk = f_in.read(1 << 20)
            if not chunk:
                break
            f_out.write(chunk)

    raw_mb = os.path.getsize(db_path) / (1024 * 1024)
    gz_mb = os.path.getsize(gz_path) / (1024 * 1024)
    print(f"  Corpus: {raw_mb:.1f} MB -> {gz_mb:.1f} MB gzip")
    return gz_path


def _upload_corpus_gz(gz_path: str, content_hash: str) -> None:
    """Upload a prepared corpus.db.gz to GCS if the blob doesn't already exist."""
    corpus_key = f"{GCS_PREFIX}/corpus/corpus-{content_hash}.db.gz"

    if _gcs_exists(corpus_key):
        print(f"  Corpus {content_hash[:12]}... already in GCS, skipping upload")
    else:
        print(f"  Uploading corpus {content_hash[:12]}...")
        _gcs_upload_file(gz_path, corpus_key)
        print(f"  -> gs://{GCS_BUCKET}/{corpus_key}")

    _gcs_write_json(
        f"{GCS_PREFIX}/corpus-latest.json",
        {
            "corpus_hash": content_hash,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"  Updated corpus-latest.json -> {content_hash[:12]}...")


def download_corpus(corpus_id: str) -> str:
    """Download corpus.db from GCS. Returns the local path."""
    if corpus_id == "latest":
        latest = _gcs_read_json(f"{GCS_PREFIX}/corpus-latest.json")
        corpus_hash = latest["corpus_hash"]
    else:
        corpus_hash = corpus_id

    corpus_key = f"{GCS_PREFIX}/corpus/corpus-{corpus_hash}.db.gz"
    if not _gcs_exists(corpus_key):
        raise FileNotFoundError(
            f"Corpus blob not found: gs://{GCS_BUCKET}/{corpus_key}"
        )

    with tempfile.NamedTemporaryFile(suffix=".db.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        print(f"Downloading corpus {corpus_hash[:12]}...")
        _gcs_download_file(corpus_key, tmp_path)

        os.makedirs(os.path.dirname(CORPUS_DB_PATH) or ".", exist_ok=True)
        print(f"  Decompressing to {CORPUS_DB_PATH}...")
        with gzip.open(tmp_path, "rb") as f_in, open(CORPUS_DB_PATH, "wb") as f_out:
            while True:
                chunk = f_in.read(1 << 20)
                if not chunk:
                    break
                f_out.write(chunk)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    from scripts.corpus_hash import corpus_content_hash

    actual_hash = corpus_content_hash(CORPUS_DB_PATH)
    if actual_hash != corpus_hash:
        raise ValueError(
            f"Corpus hash mismatch after download! "
            f"Expected {corpus_hash[:16]}..., got {actual_hash[:16]}..."
        )

    # Invalidate canonical index caches derived from the old corpus.db
    for stale in glob_mod.glob(".cc/corpus-canonical-*"):
        print(f"  Removing stale cache: {stale}")
        os.unlink(stale)

    size_mb = os.path.getsize(CORPUS_DB_PATH) / (1024 * 1024)
    print(f"  Corpus downloaded: {CORPUS_DB_PATH} ({size_mb:.1f} MB, hash verified)")
    return CORPUS_DB_PATH


def _load_local_vocab(cache_dir: str) -> Dict[str, Dict[str, int]]:
    vocab_path = os.path.join(cache_dir, "vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, Dict[str, int]] = data.get("field_vocabs", data)
    return result


def _load_binary_tensor(path: str, dtype: torch.dtype) -> torch.Tensor:
    itemsize = {torch.int32: 4, torch.float32: 4, torch.uint8: 1}[dtype]
    n_elements = os.path.getsize(path) // itemsize
    return torch.from_file(path, shared=False, size=n_elements, dtype=dtype)


def _load_local_cache(cache_dir: str) -> dict:  # pylint: disable=too-many-locals
    """Load all training data from .cache/style_dataset/ binary files."""
    result: Dict[str, Any] = {}

    result["offsets"] = _load_binary_tensor(
        os.path.join(cache_dir, EXT_OFFSETS), torch.int32
    )

    features: Dict[str, torch.Tensor] = {}
    for field in FEATURE_FIELDS:
        path = os.path.join(cache_dir, f"{EXT_FEAT_PREFIX}{field}.bin")
        if os.path.exists(path):
            features[field] = _load_binary_tensor(path, torch.int32)
    result["features"] = features

    labels: Dict[str, torch.Tensor] = {}
    for name, dtype, _itemsize in _LABEL_SPECS:
        path = os.path.join(cache_dir, f"{EXT_LABELS}_{name}")
        if os.path.exists(path):
            labels[name] = _load_binary_tensor(path, dtype)
    result["labels"] = labels

    reg_path = os.path.join(cache_dir, f"{EXT_LABELS}_reg_ids.bin")
    result["reg_ids"] = (
        _load_binary_tensor(reg_path, torch.int32)
        if os.path.exists(reg_path)
        else torch.zeros(0, dtype=torch.int32)
    )
    reg_off_path = os.path.join(cache_dir, f"{EXT_LABELS}_reg_ids_{EXT_OFFSETS}")
    result["reg_ids_offsets"] = (
        _load_binary_tensor(reg_off_path, torch.int32)
        if os.path.exists(reg_off_path)
        else torch.zeros(0, dtype=torch.int32)
    )

    cm_path = os.path.join(cache_dir, "content_mask.bin")
    result["content_mask"] = (
        _load_binary_tensor(cm_path, torch.uint8).bool()
        if os.path.exists(cm_path)
        else torch.zeros(0, dtype=torch.bool)
    )

    sent_path = os.path.join(cache_dir, "sentences.txt")
    if os.path.exists(sent_path):
        with open(sent_path, encoding="utf-8") as f:
            result["sentences"] = [line.rstrip("\n") for line in f]
    else:
        result["sentences"] = []

    for prefix in ("gp_pos", "gp_neg"):
        ids_path = os.path.join(cache_dir, f"{prefix}_ids.bin")
        off_path = os.path.join(cache_dir, f"{prefix}_offsets.bin")
        if os.path.exists(ids_path) and os.path.exists(off_path):
            result[f"{prefix}_ids"] = _load_binary_tensor(ids_path, torch.int32)
            result[f"{prefix}_offsets"] = _load_binary_tensor(off_path, torch.int32)

    gp_priors_path = os.path.join(cache_dir, "gp_priors.bin")
    result["gp_priors"] = (
        _load_binary_tensor(gp_priors_path, torch.float32)
        if os.path.exists(gp_priors_path)
        else torch.empty(0, dtype=torch.float32)
    )

    kc: Dict[str, Dict[str, torch.Tensor]] = {}
    for family in KcFamilyId:
        name = family.value
        ids_path = os.path.join(cache_dir, f"{EXT_KC_PREFIX}{name}_ids.bin")
        off_path = os.path.join(cache_dir, f"{EXT_KC_PREFIX}{name}_{EXT_OFFSETS}")
        if os.path.exists(ids_path) and os.path.exists(off_path):
            kc[name] = {
                "ids": _load_binary_tensor(ids_path, torch.int32),
                "offsets": _load_binary_tensor(off_path, torch.int32),
            }
    result["kc"] = kc

    return result


def merge_vocabs(
    local_vocab: Dict[str, Dict[str, int]],
    base_vocab: Optional[Dict[str, Dict[str, int]]],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[int, int]]]:
    """Merge local vocab into base (append-only). Returns (merged, remap)."""
    merged: Dict[str, Dict[str, int]] = {}
    remap: Dict[str, Dict[int, int]] = {}

    for field in FEATURE_FIELDS:
        local_fv = local_vocab.get(field, {})
        base_fv = base_vocab.get(field, {}) if base_vocab else {}

        merged_fv = dict(base_fv)
        if not merged_fv:
            merged_fv = dict(_SPECIAL_TOKENS)

        next_id = max(merged_fv.values()) + 1 if merged_fv else len(_SPECIAL_TOKENS)

        for token in local_fv:
            if token not in merged_fv:
                merged_fv[token] = next_id
                next_id += 1

        merged[field] = merged_fv

        field_remap: Dict[int, int] = {}
        for token, local_id in local_fv.items():
            field_remap[local_id] = merged_fv[token]
        remap[field] = field_remap

    return merged, remap


def remap_feature_tensor(
    tensor: torch.Tensor, field_remap: Dict[int, int]
) -> torch.Tensor:
    """Re-index a flat feature tensor using the local-to-merged remap."""
    if not field_remap:
        return tensor

    max_input = max(max(field_remap.keys()) + 1, int(tensor.max().item()) + 1)
    lut = torch.arange(max_input, dtype=torch.int32)
    for local_id, merged_id in field_remap.items():
        if local_id < max_input:
            lut[local_id] = merged_id

    return lut[tensor.long()]


def _remap_all_features(
    features: Dict[str, torch.Tensor],
    remap: Dict[str, Dict[int, int]],
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for field, tensor in features.items():
        field_remap = remap.get(field, {})
        if field_remap:
            result[field] = remap_feature_tensor(tensor, field_remap)
        else:
            result[field] = tensor.clone()
    return result


def merge_content_mask(
    local_mask: torch.Tensor,
    surface_remap: Dict[int, int],
    base_mask: Optional[torch.Tensor],
    merged_surface_vocab_size: int,
) -> torch.Tensor:
    """Three-way content mask merge: base -> overwrite with local -> new tokens get local."""
    merged = torch.zeros(merged_surface_vocab_size, dtype=torch.bool)

    if base_mask is not None:
        n = min(len(base_mask), merged_surface_vocab_size)
        merged[:n] = base_mask[:n]

    for local_id, merged_id in surface_remap.items():
        if local_id < len(local_mask) and merged_id < merged_surface_vocab_size:
            merged[merged_id] = local_mask[local_id]

    return merged


def extract_chive_for_merged_vocab(
    surface_vocab: Dict[str, int],
    surface_to_base: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """Extract chiVe vectors aligned to merged vocab; unmatched tokens are zero vectors."""
    from train.chive import (  # pylint: disable=redefined-outer-name
        download_chive,
        get_chive_txt_path,
        parse_chive_vectors,
    )

    txt_path = get_chive_txt_path()
    if not os.path.exists(txt_path):
        download_chive()

    vocab_size = max(surface_vocab.values()) + 1 if surface_vocab else 0
    vectors, matched = parse_chive_vectors(
        txt_path, surface_vocab, vocab_size, surface_to_base
    )
    print(f"  chiVe: {len(matched):,}/{len(surface_vocab):,} tokens matched")
    return vectors


def compute_chive_hash(vectors: torch.Tensor) -> str:
    """Deterministic content hash of a chiVe tensor."""
    return hashlib.sha256(vectors.numpy().tobytes()).hexdigest()[:16]


def compute_dataset_id(bundle: dict) -> str:
    """Deterministic content hash of the dataset's training-relevant data.

    Includes ``schema_version`` so a new bundle format (e.g. added tensors)
    yields a new ``dataset_id``, fresh GCS keys, and clients pick up changes
    via ``dataset.lock`` instead of reusing a stale local ``ds-*.pt``.
    """
    h = hashlib.sha256()
    h.update(int(bundle.get("schema_version", 0)).to_bytes(4, "little"))
    h.update(json.dumps(bundle["vocab"], sort_keys=True).encode())
    h.update(bundle["offsets"].numpy().tobytes())
    for field in sorted(bundle["features"].keys()):
        h.update(bundle["features"][field].numpy().tobytes())
    for name in sorted(bundle["labels"].keys()):
        h.update(bundle["labels"][name].numpy().tobytes())
    h.update(bundle["content_mask"].numpy().tobytes())
    for prefix in ("gp_pos", "gp_neg"):
        ids_key = f"{prefix}_ids"
        off_key = f"{prefix}_offsets"
        if ids_key in bundle:
            h.update(bundle[ids_key].numpy().tobytes())
            h.update(bundle[off_key].numpy().tobytes())
    for s in bundle["sentences"]:
        h.update(s.encode("utf-8"))
    return h.hexdigest()[:16]


def _resolve_base_dataset(flag: str) -> dict:
    """Download + load a base dataset for vocab inheritance."""
    if flag == "latest":
        latest = _gcs_read_json(f"{GCS_PREFIX}/latest.json")
        dataset_id = latest["dataset_id"]
    else:
        dataset_id = flag
    local_path = _ensure_dataset_local(dataset_id)
    return load_dataset(local_path)


def _check_dataset_integrity(bundle: Dict[str, Any]) -> None:  # pylint: disable=too-many-locals
    """Defense in depth: verify no canonical duplicates or banned vocab surfaces."""
    from collections import defaultdict

    from kotogram.masking import SURFACE_EXEMPLARS
    from scripts.integrity import DataIntegrityException

    # 1. Vocab check: no banned surfaces in the surface vocabulary
    surface_vocab = bundle.get("vocab", {}).get("surface", {})
    exemplar_surfaces = set(SURFACE_EXEMPLARS.values())
    for surface_str in surface_vocab:
        if surface_str in (PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, MASK_TOKEN):
            continue
        # Parse the surface as a standalone token to check if it would be masked.
        # We can't use get_surface_mask_for_features without full POS info, so
        # instead check if the surface is a digit string (indicating a number
        # that should have been replaced by the "1" exemplar).
        if surface_str.isdigit() and surface_str not in exemplar_surfaces:
            raise DataIntegrityException(
                f"[Dataset Build] Banned surface in vocab: {surface_str!r} "
                f"(digit string not in exemplars). "
                f"Check that label.py uses TRAINING_MASK format."
            )

    # 2. Sentence check: no canonical duplicates
    sentences = bundle.get("sentences", [])
    if sentences:
        import time as _time

        from scripts.canonical_index import parallel_canonicalize

        print(f"  Checking {len(sentences):,} sentences for canonical duplicates...")
        _t0 = _time.monotonic()
        canonical = parallel_canonicalize(sentences)
        _elapsed = _time.monotonic() - _t0
        print(f"  Canonicalized in {_elapsed:.1f}s, checking for duplicates...")
        groups: dict[str, list[str]] = defaultdict(list)
        for orig, canon in zip(sentences, canonical):
            groups[canon].append(orig)

        dupes = {k: v for k, v in groups.items() if len(v) > 1}
        if dupes:
            sample_key = next(iter(dupes))
            sample = dupes[sample_key][:3]
            raise DataIntegrityException(
                f"[Dataset Build] {len(dupes)} canonical duplicate group(s) found. "
                f"Sample: key={sample_key!r}, count={len(dupes[sample_key])}, "
                f"sentences={sample!r}. "
                f"Run corpus.db migration or check curate upsert canonical gating."
            )
        print(f"  Integrity OK: {len(sentences):,} sentences, 0 canonical duplicates")

    print("  Integrity checks passed (vocab + canonical dedup)")


def build_dataset(  # pylint: disable=too-many-locals
    cache_dir: Optional[str] = None,
    base_dataset_flag: str = "latest",
) -> Tuple[str, str]:
    """Build a .pt dataset bundle from .cache/style_dataset/. Returns (dataset_id, path)."""
    from train import paths as train_paths

    if cache_dir is None:
        cache_dir = train_paths.get_style_dataset_cache_dir()

    content_hash = _verify_corpus_label_hash(CORPUS_DB_PATH)
    _print_corpus_summary(CORPUS_DB_PATH, content_hash)

    print(f"Building dataset from {cache_dir}")

    local_vocab = _load_local_vocab(cache_dir)
    local_data = _load_local_cache(cache_dir)

    base_dataset: Optional[dict] = None
    base_dataset_id: Optional[str] = None
    if base_dataset_flag != "none":
        if base_dataset_flag == "latest" and not _gcs_exists(
            f"{GCS_PREFIX}/latest.json"
        ):
            print("  No base dataset available (no latest.json in GCS), starting fresh")
        else:
            base_dataset = _resolve_base_dataset(base_dataset_flag)
            base_dataset_id = base_dataset.get("dataset_id")
            print(f"  Inheriting vocab from base dataset: {base_dataset_id}")

    base_vocab = base_dataset["vocab"] if base_dataset else None
    merged_vocab, remap = merge_vocabs(local_vocab, base_vocab)

    for field in ENCODER_FEATURE_FIELDS:
        base_count = len(base_vocab.get(field, {})) if base_vocab else 0
        merged_count = len(merged_vocab[field])
        new_count = merged_count - base_count
        if new_count > 0:
            print(
                f"  {field}: {base_count} -> {merged_count} (+{new_count} new tokens)"
            )

    features = _remap_all_features(
        {
            f: local_data["features"][f]
            for f in ENCODER_FEATURE_FIELDS
            if f in local_data["features"]
        },
        remap,
    )

    surface_vocab_size = len(merged_vocab.get("surface", {}))
    content_mask = merge_content_mask(
        local_data["content_mask"],
        remap.get("surface", {}),
        base_dataset["content_mask"] if base_dataset else None,
        surface_vocab_size,
    )

    chive_surface: Optional[torch.Tensor] = None
    chive_id: Optional[str] = None

    # Fast path: if base dataset has same surface vocab, reuse its chiVe directly.
    if base_dataset is not None and merged_vocab.get("surface") == base_dataset.get(
        "vocab", {}
    ).get("surface"):
        chive_id = base_dataset.get("chive_id")
        if chive_id:
            cached = os.path.join(LOCAL_CACHE, f"chive-{chive_id}.pt")
            if os.path.exists(cached):
                chive_surface = load_chive(cached)
                print(f"  chiVe unchanged from base dataset ({chive_id})")

    if chive_surface is None:
        s2b_path = os.path.join(cache_dir, "surface_to_base.json")
        surface_to_base = {}
        if os.path.exists(s2b_path):
            with open(s2b_path, encoding="utf-8") as f:
                surface_to_base = json.load(f)

        print("  Extracting chiVe vectors for merged vocabulary...")
        chive_surface = extract_chive_for_merged_vocab(
            merged_vocab["surface"], surface_to_base
        )
        chive_id = compute_chive_hash(chive_surface)
        # If this hash already exists locally, the content is identical -- skip saving.
        cached = os.path.join(LOCAL_CACHE, f"chive-{chive_id}.pt")
        if os.path.exists(cached):
            print(f"  chiVe ID: {chive_id} (already cached)")
            chive_surface = load_chive(cached)
        else:
            print(f"  chiVe ID: {chive_id}")

    git_commit = ""
    import shutil

    if shutil.which("git"):
        proc = subprocess.run(
            ["git", "log", "-1", "--format=%h"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if proc.returncode == 0:
            git_commit = proc.stdout.strip()

    n_sentences = len(local_data["offsets"]) - 1
    n_tokens = int(local_data["offsets"][-1].item())

    # Only include data needed for recon_bpd training:
    # surface features, gram labels, content mask, sentences.
    # Full vocab is kept for base-dataset inheritance.
    slim_features = {}
    for field in ENCODER_FEATURE_FIELDS:
        if field in features:
            slim_features[field] = features[field]

    slim_labels: Dict[str, torch.Tensor] = {}
    if "gram" in local_data["labels"]:
        slim_labels["gram"] = local_data["labels"]["gram"]

    bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "base_dataset_id": base_dataset_id,
        "chive_id": chive_id,
        "sentence_count": n_sentences,
        "token_count": n_tokens,
        "vocab": merged_vocab,
        "offsets": local_data["offsets"],
        "features": slim_features,
        "labels": slim_labels,
        "content_mask": content_mask,
        "sentences": local_data["sentences"],
    }

    for prefix in ("gp_pos", "gp_neg"):
        ids_key = f"{prefix}_ids"
        off_key = f"{prefix}_offsets"
        if ids_key in local_data and local_data[ids_key].numel() > 0:
            bundle[ids_key] = local_data[ids_key]
            bundle[off_key] = local_data[off_key]

    # Defense in depth: check for canonical duplicates and banned vocab surfaces
    _check_dataset_integrity(bundle)

    bundle["dataset_id"] = compute_dataset_id(bundle)
    dataset_id = bundle["dataset_id"]

    counts = grammatical_token_length_counts(bundle)
    bundle["token_length_counts"] = torch.from_numpy(counts)
    print(
        f"  Token histogram (gram=1): len={counts.size} "
        f"(embedded in bundle as token_length_counts)"
    )

    tgf = grammatical_token_gram_freq(bundle)
    bundle["token_gram_freq"] = torch.from_numpy(tgf)
    nonzero = int((tgf > 0).sum())
    print(
        f"  Token gram freq: {nonzero:,}/{len(tgf):,} active "
        f"(embedded in bundle as token_gram_freq)"
    )

    os.makedirs(LOCAL_CACHE, exist_ok=True)
    output_path = os.path.join(LOCAL_CACHE, f"ds-{dataset_id}.pt")
    torch.save(bundle, output_path)

    chive_path = os.path.join(LOCAL_CACHE, f"chive-{chive_id}.pt")
    if not os.path.exists(chive_path):
        torch.save(chive_surface, chive_path)

    corpus_gz_path = _prepare_corpus_gz(CORPUS_DB_PATH, content_hash)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    chive_mb = os.path.getsize(chive_path) / (1024 * 1024)
    corpus_gz_mb = os.path.getsize(corpus_gz_path) / (1024 * 1024)
    lbl_w = 28
    mb_w = 8

    def _tmb(t: torch.Tensor) -> float:
        return t.nelement() * t.element_size() / (1024 * 1024)

    def _row(label: str, mb: float, desc: str = "") -> None:
        base = f"  {label:<{lbl_w}} {mb:>{mb_w}.1f} MB"
        if desc:
            base += f"  {desc}"
        print(base)

    print(f"\nDataset built: {dataset_id}")
    print(f"  {n_sentences:,} sentences, {n_tokens:,} tokens\n")

    print("  Tensor storage (zip data/):")
    for fname, ftensor in sorted(bundle["features"].items()):
        _row(f"  features/{fname}", _tmb(ftensor), "int32 token IDs per position")
    _row("  offsets", _tmb(bundle["offsets"]), "int32 sentence→token boundaries")
    for lname, ltensor in sorted(bundle["labels"].items()):
        _row(f"  labels/{lname}", _tmb(ltensor), "per-sentence classification label")
    _row(
        "  content_mask",
        _tmb(bundle["content_mask"]),
        "bool[V] content vs function tokens",
    )
    _row(
        "  token_length_counts",
        _tmb(bundle["token_length_counts"]),
        "uint64 gram sentence-length histogram",
    )
    _row(
        "  token_gram_freq",
        _tmb(bundle["token_gram_freq"]),
        "int64[V] per-token gram position freq",
    )
    for prefix in ("gp_pos", "gp_neg"):
        ids_key, off_key = f"{prefix}_ids", f"{prefix}_offsets"
        if ids_key in bundle:
            _row(
                f"  {prefix} (ids+offsets)",
                _tmb(bundle[ids_key]) + _tmb(bundle[off_key]),
                "int32 ragged grammar-point labels",
            )

    print("  Pickled in data.pkl:")
    _row(
        "  sentences",
        sum(len(s.encode("utf-8")) for s in bundle["sentences"]) / (1024 * 1024),
        "list[str] original sentence text",
    )
    _row(
        "  vocab",
        len(json.dumps(bundle["vocab"]).encode()) / (1024 * 1024),
        "dict field→token→id maps",
    )
    print(
        "    + scalars: schema_version, dataset_id, created_at, "
        "git_commit, base_dataset_id, chive_id, "
        "sentence_count, token_count"
    )

    print(f"  {'-' * (lbl_w + mb_w + 8)}")
    _row("total on disk", size_mb)
    _row("chiVe (separate file)", chive_mb, "float32[V,300] pretrained embeddings")
    _row(
        "corpus.db.gz",
        corpus_gz_mb,
        f"vacuumed+gzipped SQLite, hash {content_hash[:12]}...",
    )
    print(f"\n  Dataset: {output_path}")
    print(f"  chiVe:   {chive_path}")
    return dataset_id, output_path


def upload_dataset(pt_path: Optional[str] = None, *, force: bool = False) -> str:
    """Upload dataset + chiVe + corpus.db to GCS, update latest.json and dataset.lock.

    Corpus.db is VACUUMed, gzipped, and uploaded if the content-addressed blob
    doesn't already exist.  By default, skips dataset upload when the object
    already exists (same dataset_id).  Use ``force=True`` to replace blobs.
    """
    content_hash = _verify_corpus_label_hash(CORPUS_DB_PATH)

    corpus_gz = os.path.join(LOCAL_CACHE, f"corpus-{content_hash}.db.gz")
    if not os.path.exists(corpus_gz):
        corpus_gz = _prepare_corpus_gz(CORPUS_DB_PATH, content_hash)
    _upload_corpus_gz(corpus_gz, content_hash)

    if pt_path is None:
        if not os.path.isdir(LOCAL_CACHE):
            raise FileNotFoundError(f"No local datasets in {LOCAL_CACHE}")
        candidates = sorted(
            (
                f
                for f in os.listdir(LOCAL_CACHE)
                if f.startswith("ds-") and f.endswith(".pt")
            ),
            key=lambda f: os.path.getmtime(os.path.join(LOCAL_CACHE, f)),
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No dataset .pt files in {LOCAL_CACHE}")
        pt_path = os.path.join(LOCAL_CACHE, candidates[0])

    bundle = load_dataset(pt_path)
    dataset_id: str = bundle["dataset_id"]
    chive_id: str = bundle["chive_id"]

    ds_key = f"{GCS_PREFIX}/datasets/ds-{dataset_id}.pt"
    uri = f"gs://{GCS_BUCKET}/{ds_key}"
    if not _gcs_exists(ds_key):
        print(f"Uploading dataset {dataset_id}...")
        _gcs_upload_file(pt_path, ds_key)
        print(f"  -> {uri}")
    elif force:
        print(f"  Replacing dataset {dataset_id} on GCS (--force)...")
        _gcs_upload_file(pt_path, ds_key)
        print(f"  -> {uri}")
    else:
        print(
            f"  Dataset {dataset_id} already in GCS, skipping upload "
            f"(use --force if local .pt was rebuilt with the same id)"
        )

    chive_key = f"{GCS_PREFIX}/chive/chive-{chive_id}.pt"
    chive_local = os.path.join(LOCAL_CACHE, f"chive-{chive_id}.pt")
    if not _gcs_exists(chive_key):
        if not os.path.exists(chive_local):
            raise FileNotFoundError(
                f"chiVe not found at {chive_local}. "
                "It should have been created during 'build'."
            )
        print(f"  Uploading chiVe {chive_id}...")
        _gcs_upload_file(chive_local, chive_key)
    elif force:
        if not os.path.exists(chive_local):
            raise FileNotFoundError(
                f"chiVe not found at {chive_local}. "
                "It should have been created during 'build'."
            )
        print(f"  Replacing chiVe {chive_id} on GCS (--force)...")
        _gcs_upload_file(chive_local, chive_key)
    else:
        print(f"  chiVe {chive_id} already in GCS, skipping upload")

    _gcs_write_json(
        f"{GCS_PREFIX}/latest.json",
        {
            "dataset_id": dataset_id,
            "chive_id": chive_id,
            "created_at": bundle["created_at"],
        },
    )
    print(f"  Updated latest.json -> {dataset_id}")

    from scripts.corpus_hash import corpus_content_hash

    corpus_hash = (
        corpus_content_hash(CORPUS_DB_PATH) if os.path.exists(CORPUS_DB_PATH) else None
    )
    lock_path = write_lock(dataset_id, chive_id, corpus_hash=corpus_hash)
    print(f"  Updated {lock_path}")

    return uri


def _ensure_dataset_local(dataset_id: str) -> str:
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    local_path = os.path.join(LOCAL_CACHE, f"ds-{dataset_id}.pt")
    if os.path.exists(local_path):
        return local_path
    gcs_key = f"{GCS_PREFIX}/datasets/ds-{dataset_id}.pt"
    print(f"Downloading dataset {dataset_id}...")
    _gcs_download_file(gcs_key, local_path)
    return local_path


def _ensure_chive_local(chive_id: str) -> str:
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    local_path = os.path.join(LOCAL_CACHE, f"chive-{chive_id}.pt")
    if os.path.exists(local_path):
        return local_path
    gcs_key = f"{GCS_PREFIX}/chive/chive-{chive_id}.pt"
    print(f"Downloading chiVe {chive_id}...")
    _gcs_download_file(gcs_key, local_path)
    return local_path


def download_dataset(dataset_id: str) -> str:
    """Download a dataset .pt by ID. Returns local path."""
    if dataset_id == "latest":
        latest = _gcs_read_json(f"{GCS_PREFIX}/latest.json")
        dataset_id = latest["dataset_id"]
    return _ensure_dataset_local(dataset_id)


def download_chive(chive_id: str) -> str:
    """Download a chiVe .pt by ID. Returns local path."""
    return _ensure_chive_local(chive_id)


def load_dataset(path: str) -> dict:
    """Load and validate a .pt dataset bundle (memory-mapped).

    Accepts any schema version >= ``_MIN_SCHEMA_VERSION``.  Older bundles
    may lack derived fields (e.g. ``token_gram_freq``); callers that need
    those fields should check and raise with a clear message.
    """
    bundle: dict = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    version = bundle.get("schema_version", 0)
    if version < _MIN_SCHEMA_VERSION:
        raise ValueError(
            f"Dataset schema version {version} < minimum {_MIN_SCHEMA_VERSION}. "
            "Rebuild the dataset."
        )

    missing = _REQUIRED_KEYS - set(bundle.keys())
    if missing:
        raise ValueError(f"Dataset missing required keys: {missing}")

    return bundle


def load_chive(path: str) -> torch.Tensor:
    """Load a chiVe .pt tensor (memory-mapped)."""
    result: torch.Tensor = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True
    )
    return result


def resolve_dataset_by_id(dataset_id: str) -> dict:
    """Resolve a dataset ID to its bundle (without chiVe weights)."""
    ds_path = _ensure_dataset_local(dataset_id)
    return load_dataset(ds_path)


def resolve_dataset(flag: Optional[str] = None) -> Tuple[dict, torch.Tensor]:
    """Resolve a --dataset flag (None/latest/<id>) to (bundle, chive_tensor)."""
    if flag is None:
        lock = read_lock()
        if lock is None:
            raise FileNotFoundError(
                f"No {DATASET_LOCK} found. Run 'python -m scripts.dataset build' "
                "and 'python -m scripts.dataset upload' first, or pass --dataset latest."
            )
        dataset_id = lock["dataset_id"]
        chive_id = lock["chive_id"]
    elif flag == "latest":
        latest = _gcs_read_json(f"{GCS_PREFIX}/latest.json")
        dataset_id = latest["dataset_id"]
        chive_id = latest["chive_id"]
    else:
        dataset_id = flag
        ds_path = _ensure_dataset_local(dataset_id)
        tmp = load_dataset(ds_path)
        chive_id = tmp["chive_id"]

    ds_path = _ensure_dataset_local(dataset_id)
    chive_path = _ensure_chive_local(chive_id)

    bundle = load_dataset(ds_path)
    chive = load_chive(chive_path)

    return bundle, chive


class BundledStyleDataset(StyleDataset):
    """StyleDataset backed by an in-memory .pt bundle instead of mmap'd files."""

    _sentences: List[str]
    content_mask: Optional[Any]
    content_drop_ratio: float
    pristine_static_mapping: Optional[Any]  # torch.Tensor or None
    pristine_vocab: Optional[Dict[str, int]]

    @classmethod
    def from_bundle(
        cls,
        bundle: dict,
        sample_ratio: float = 1.0,
        feature_fields: Optional[Sequence[str]] = None,
        verbose: bool = True,
    ) -> "BundledStyleDataset":
        """Create from a loaded .pt bundle dict."""
        tokenizer = Tokenizer()
        tokenizer.load_state({"field_vocabs": bundle["vocab"], "frozen": True})

        ds = cls.__new__(cls)
        ds.data_dir = "<bundle>"
        ds.tokenizer = tokenizer
        ds.verbose = verbose
        ds._feature_fields = list(feature_fields or ENCODER_FEATURE_FIELDS)
        ds._sentences = bundle.get("sentences", [])

        ds.offsets = bundle["offsets"]
        total_samples = len(ds.offsets) - 1
        ds.indices = torch.arange(total_samples, dtype=torch.long)

        ds.features = {
            f: bundle["features"][f]
            for f in ds._feature_fields
            if f in bundle["features"]
        }

        ds.labels = dict(bundle["labels"])
        n = total_samples
        _defaults: Dict[str, torch.Tensor] = {
            "f_val": torch.full((n,), 0.5, dtype=torch.float32),
            "f_prag": torch.ones(n, dtype=torch.uint8),
            "g_val": torch.full((n,), 0.5, dtype=torch.float32),
            "g_prag": torch.ones(n, dtype=torch.uint8),
            "reg_ids": bundle.get("reg_ids", torch.zeros(n, dtype=torch.int32)),
            "reg_offsets": bundle.get(
                "reg_ids_offsets", torch.arange(n + 1, dtype=torch.int32)
            ),
        }
        for key, default in _defaults.items():
            if key not in ds.labels:
                ds.labels[key] = default

        kc_maps: Dict[str, Dict[str, torch.Tensor]] = {}
        ds.kc_maps = kc_maps
        for family_name, family_data in bundle.get("kc", {}).items():
            ds.kc_maps[family_name] = dict(family_data)
        if "gp_pos_ids" in bundle:
            ds.kc_maps["grammar_point_pos"] = {
                "ids": bundle["gp_pos_ids"],
                "offsets": bundle["gp_pos_offsets"],
            }
        if "gp_neg_ids" in bundle:
            ds.kc_maps["grammar_point_neg"] = {
                "ids": bundle["gp_neg_ids"],
                "offsets": bundle["gp_neg_offsets"],
            }

        ds.gp_priors = bundle.get("gp_priors", torch.empty(0, dtype=torch.float32))

        ds.content_mask = bundle.get("content_mask")
        ds.content_drop_ratio = 0.0
        ds.pristine_static_mapping = None
        ds.pristine_vocab = None

        ds._full_indices = ds.indices.clone()
        ds._sample_ratio = sample_ratio
        ds._apply_balanced_sampling(sample_ratio, seed=42)
        ds._len = len(ds.indices)

        return ds

    def _get_kc_targets(self, real_idx: int) -> Dict:
        """Return KC targets, gracefully handling empty kc_maps."""
        if not self.kc_maps:
            return {fam: [] for fam in KcFamilyId}
        return super()._get_kc_targets(real_idx)

    def __getitem__(self, idx: int) -> Sample:
        sample = super().__getitem__(idx)
        surface = sample.feature_ids.get("surface")

        # Step 1: Pristine on full raw sequence (before content_drop)
        # Context-dependent rules (quote parity, sentence-final period)
        # need the unmodified token sequence.
        pristine_result = None
        if self.pristine_static_mapping is not None and self.pristine_vocab is not None:
            if surface is not None:
                from scripts.recon_bpd.token_remap import apply_pristine

                pristine_result = apply_pristine(
                    surface,
                    self.pristine_vocab,
                    static_mapping=self.pristine_static_mapping,
                )

        # Step 2+3: Content drop (computed on dirty IDs)
        # Apply same mask to both dirty and pristine to maintain alignment.
        if self.content_drop_ratio > 0 and self.content_mask is not None:
            if surface is not None:
                is_content = self.content_mask[surface]
                is_special = surface < 4
                droppable = (~is_content) & (~is_special)
                drop = droppable & (torch.rand(len(surface)) < self.content_drop_ratio)
                keep = ~drop
                if not keep.all():
                    sample.feature_ids = {
                        k: v[keep] for k, v in sample.feature_ids.items()
                    }
                    if pristine_result is not None:
                        pristine_result = pristine_result[keep]

        # Step 4: In pristine target, map remaining non-content to PAD
        if pristine_result is not None and self.content_mask is not None:
            post_pristine = pristine_result
            is_content = self.content_mask[post_pristine.clamp(min=0)]
            is_special = post_pristine < 4  # PAD, UNK, CLS, MASK
            still_noncontent = (~is_content) & (~is_special) & (post_pristine > 0)
            if still_noncontent.any():
                pristine_result = pristine_result.clone()
                pristine_result[still_noncontent] = 0  # PAD

        if pristine_result is not None:
            sample.pristine_ids = pristine_result

        return sample

    def get_sentence_by_idx(self, real_idx: int) -> str:
        """Return sentence text from the in-memory list."""
        if 0 <= real_idx < len(self._sentences):
            return self._sentences[real_idx]
        return ""

    def filter_by_grammaticality(  # pylint: disable=protected-access
        self, label: int = 1
    ) -> "BundledStyleDataset":
        """Return a subset filtered by grammaticality, sharing underlying data."""
        if "gram" not in self.labels:
            if self.verbose:
                print("Warning: No grammaticality labels found, returning self.")
            return self

        mask = self.labels["gram"][self.indices] == label
        new_indices = self.indices[mask]

        child = BundledStyleDataset.__new__(BundledStyleDataset)
        child.data_dir = self.data_dir
        child.tokenizer = self.tokenizer
        child.verbose = self.verbose
        child._feature_fields = self._feature_fields
        child._sentences = self._sentences
        child.offsets = self.offsets
        child.features = self.features
        child.labels = self.labels
        child.kc_maps = self.kc_maps
        child.gp_priors = self.gp_priors
        child.content_mask = self.content_mask
        child.content_drop_ratio = self.content_drop_ratio
        child.pristine_static_mapping = self.pristine_static_mapping
        child.pristine_vocab = self.pristine_vocab
        child.indices = new_indices
        child._full_indices = new_indices.clone()
        child._sample_ratio = 1.0
        child._len = len(new_indices)
        return child


def log_mlflow_dataset(bundle: dict) -> None:
    """Log dataset provenance and metadata to the active MLflow run."""
    import mlflow  # type: ignore[import-untyped]
    import mlflow.data  # type: ignore[import-untyped]
    import pandas as pd  # type: ignore[import-untyped]

    if not mlflow.active_run():
        return

    dataset_id: str = bundle["dataset_id"]
    chive_id: str = bundle["chive_id"]

    mlflow.set_tag("dataset_id", dataset_id)
    mlflow.set_tag("chive_id", chive_id)
    mlflow.set_tag(
        "dataset_uri",
        f"gs://{GCS_BUCKET}/{GCS_PREFIX}/datasets/ds-{dataset_id}.pt",
    )

    gram = bundle["labels"].get("gram")
    gram_count = int((gram == 1).sum().item()) if gram is not None else 0
    ungram_count = int((gram == 0).sum().item()) if gram is not None else 0

    summary = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "chive_id": chive_id,
                "base_dataset_id": bundle.get("base_dataset_id") or "none",
                "git_commit": bundle.get("git_commit", ""),
                "schema_version": bundle.get("schema_version", 0),
                "created_at": bundle.get("created_at", ""),
                "sentences": bundle.get("sentence_count", 0),
                "tokens": bundle.get("token_count", 0),
                "surface_vocab_size": len(bundle.get("vocab", {}).get("surface", {})),
                "grammatical": gram_count,
                "agrammatical": ungram_count,
            }
        ]
    )

    gcs_uri = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/datasets/ds-{dataset_id}.pt"
    ds = mlflow.data.from_pandas(  # pylint: disable=no-member
        summary,
        source=gcs_uri,
        name="kotogram-bpd",
        digest=dataset_id,
    )
    mlflow.log_input(ds, context="training")


def _cmd_build(args: argparse.Namespace) -> None:
    build_dataset(base_dataset_flag=args.base_dataset)


def _cmd_upload(args: argparse.Namespace) -> None:
    upload_dataset(args.path, force=args.force)


def _cmd_download(args: argparse.Namespace) -> None:
    path = download_dataset(args.id)
    print(f"Downloaded to {path}")


def _cmd_corpus_download(args: argparse.Namespace) -> None:
    download_corpus(args.id)


def _cmd_list(_args: argparse.Namespace) -> None:
    blobs = _gcs_list_blobs(f"{GCS_PREFIX}/datasets/")
    if not blobs:
        print("No datasets in GCS")
        return

    latest_id = ""
    if _gcs_exists(f"{GCS_PREFIX}/latest.json"):
        latest = _gcs_read_json(f"{GCS_PREFIX}/latest.json")
        latest_id = latest.get("dataset_id", "")

    for name in sorted(blobs):
        ds_name = os.path.basename(name)
        marker = " <- latest" if latest_id and latest_id in ds_name else ""
        print(f"  {ds_name}{marker}")


def _cmd_info(args: argparse.Namespace) -> None:
    bundle, _ = resolve_dataset(args.id)

    print(f"Dataset: {bundle['dataset_id']}")
    print(f"  Schema version: {bundle['schema_version']}")
    print(f"  Created: {bundle['created_at']}")
    print(f"  Git commit: {bundle.get('git_commit', 'N/A')}")
    print(f"  Base dataset: {bundle.get('base_dataset_id') or 'none'}")
    print(f"  chiVe ID: {bundle['chive_id']}")
    print(f"  Sentences: {bundle['sentence_count']:,}")
    print(f"  Tokens: {bundle['token_count']:,}")
    print("  Vocab sizes:")
    for field, vocab in bundle["vocab"].items():
        print(f"    {field}: {len(vocab):,}")
    gram = bundle["labels"].get("gram")
    if gram is not None:
        print(f"  Grammatical: {int((gram == 1).sum().item()):,}")
        print(f"  Agrammatical: {int((gram == 0).sum().item()):,}")


def _cmd_resolve(args: argparse.Namespace) -> None:
    bundle, _chive = resolve_dataset(args.dataset)
    ds = BundledStyleDataset.from_bundle(bundle)
    print(f"Resolved dataset: {bundle['dataset_id']}")
    print(f"  Sentences: {len(ds)}")
    log_mlflow_dataset(bundle)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scripts.dataset",
        description="Dataset management: build, upload, download, list, info",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build .pt from .cache/style_dataset/")
    p_build.add_argument(
        "--base-dataset",
        default="latest",
        help="Base dataset for vocab inheritance: 'latest', '<id>', or 'none' "
        "(default: latest)",
    )

    p_upload = sub.add_parser("upload", help="Upload dataset to GCS")
    p_upload.add_argument(
        "path", nargs="?", default=None, help="Path to .pt file (default: most recent)"
    )
    p_upload.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Replace dataset/chiVe blobs even if keys already exist (same dataset_id)",
    )

    p_download = sub.add_parser("download", help="Download dataset from GCS")
    p_download.add_argument("id", help="Dataset ID or 'latest'")

    p_corpus_dl = sub.add_parser("corpus-download", help="Download corpus.db from GCS")
    p_corpus_dl.add_argument("id", help="Corpus hash or 'latest'")

    sub.add_parser("list", help="List datasets in GCS")

    p_info = sub.add_parser("info", help="Show dataset metadata")
    p_info.add_argument(
        "id",
        nargs="?",
        default=None,
        help="Dataset ID or 'latest' (default: local dataset.lock)",
    )

    p_resolve = sub.add_parser("resolve", help="Resolve and validate a dataset")
    p_resolve.add_argument(
        "--dataset",
        default=None,
        help="Dataset: 'latest', '<id>', or omit for dataset.lock",
    )

    args = parser.parse_args()

    cmds = {
        "build": _cmd_build,
        "upload": _cmd_upload,
        "download": _cmd_download,
        "corpus-download": _cmd_corpus_download,
        "list": _cmd_list,
        "info": _cmd_info,
        "resolve": _cmd_resolve,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
