"""Dataset management: build, upload, download, list, info, resolve."""

import argparse
import hashlib
import json
import os
import subprocess
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
    TOKEN_LEN_HIST_PREFIX,
    save_token_length_histogram,
    token_length_histogram_path,
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

SCHEMA_VERSION = 1
GCS_PREFIX = "kotogram-datasets"
DATASET_LOCK = "dataset.lock"
LOCAL_CACHE = os.path.join(".cache", "datasets")
CHIVE_DIM = 300

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
    lock_path = os.path.join(_find_repo_root(), DATASET_LOCK)
    if not os.path.exists(lock_path):
        return None
    with open(lock_path, encoding="utf-8") as f:
        result: Dict[str, Any] = json.load(f)
    return result


def write_lock(dataset_id: str, chive_id: str) -> str:
    """Write dataset.lock to the repo root. Returns the path written."""
    lock_path = os.path.join(_find_repo_root(), DATASET_LOCK)
    data = {
        "dataset_id": dataset_id,
        "chive_id": chive_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return lock_path


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
    """Deterministic content hash of the dataset's training-relevant data."""
    h = hashlib.sha256()
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


def build_dataset(  # pylint: disable=too-many-locals
    cache_dir: Optional[str] = None,
    base_dataset_flag: str = "latest",
) -> Tuple[str, str]:
    """Build a .pt dataset bundle from .cache/style_dataset/. Returns (dataset_id, path)."""
    from train import paths as train_paths

    if cache_dir is None:
        cache_dir = train_paths.get_style_dataset_cache_dir()

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

    bundle["dataset_id"] = compute_dataset_id(bundle)
    dataset_id = bundle["dataset_id"]

    os.makedirs(LOCAL_CACHE, exist_ok=True)
    output_path = os.path.join(LOCAL_CACHE, f"ds-{dataset_id}.pt")
    torch.save(bundle, output_path)

    hist_path = save_token_length_histogram(bundle, dataset_id)
    print(f"  Token histogram (gram=1): {hist_path}")

    chive_path = os.path.join(LOCAL_CACHE, f"chive-{chive_id}.pt")
    if not os.path.exists(chive_path):
        torch.save(chive_surface, chive_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    chive_mb = os.path.getsize(chive_path) / (1024 * 1024)
    col_w = 28

    def _tmb(t: torch.Tensor) -> float:
        return t.nelement() * t.element_size() / (1024 * 1024)

    def _row(label: str, mb: float) -> None:
        print(f"  {label:<{col_w}} {mb:>8.1f} MB")

    print(f"\nDataset built: {dataset_id}")
    print(f"  {n_sentences:,} sentences, {n_tokens:,} tokens\n")
    for fname, ftensor in sorted(bundle["features"].items()):
        _row(f"features/{fname}", _tmb(ftensor))
    _row("offsets", _tmb(bundle["offsets"]))
    for lname, ltensor in sorted(bundle["labels"].items()):
        _row(f"labels/{lname}", _tmb(ltensor))
    _row("content_mask", _tmb(bundle["content_mask"]))
    for prefix in ("gp_pos", "gp_neg"):
        ids_key, off_key = f"{prefix}_ids", f"{prefix}_offsets"
        if ids_key in bundle:
            _row(
                f"{prefix} (ids+offsets)", _tmb(bundle[ids_key]) + _tmb(bundle[off_key])
            )
    _row(
        "sentences",
        sum(len(s.encode("utf-8")) for s in bundle["sentences"]) / (1024 * 1024),
    )
    _row("vocab", len(json.dumps(bundle["vocab"]).encode()) / (1024 * 1024))
    print(f"  {'-' * (col_w + 12)}")
    _row("total on disk", size_mb)
    _row("chiVe (separate file)", chive_mb)
    print(f"\n  Dataset: {output_path}")
    print(f"  chiVe:   {chive_path}")
    return dataset_id, output_path


def upload_dataset(pt_path: Optional[str] = None) -> str:
    """Upload dataset + chiVe to GCS, update latest.json and dataset.lock."""
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
    if _gcs_exists(ds_key):
        print(f"  Dataset {dataset_id} already in GCS, skipping upload")
    else:
        print(f"Uploading dataset {dataset_id}...")
        _gcs_upload_file(pt_path, ds_key)
        print(f"  -> {uri}")

    chive_key = f"{GCS_PREFIX}/chive/chive-{chive_id}.pt"
    if _gcs_exists(chive_key):
        print(f"  chiVe {chive_id} already in GCS, skipping upload")
    else:
        chive_local = os.path.join(LOCAL_CACHE, f"chive-{chive_id}.pt")
        if not os.path.exists(chive_local):
            raise FileNotFoundError(
                f"chiVe not found at {chive_local}. "
                "It should have been created during 'build'."
            )
        print(f"  Uploading chiVe {chive_id}...")
        _gcs_upload_file(chive_local, chive_key)

    hist_local = token_length_histogram_path(dataset_id)
    hist_key = f"{GCS_PREFIX}/datasets/{TOKEN_LEN_HIST_PREFIX}{dataset_id}.npy"
    if os.path.exists(hist_local):
        if _gcs_exists(hist_key):
            print(f"  Token histogram {dataset_id} already in GCS, skipping")
        else:
            print(f"  Uploading token histogram {dataset_id}...")
            _gcs_upload_file(hist_local, hist_key)
    else:
        print(f"  No local token histogram at {hist_local} (skipping)")

    _gcs_write_json(
        f"{GCS_PREFIX}/latest.json",
        {
            "dataset_id": dataset_id,
            "chive_id": chive_id,
            "created_at": bundle["created_at"],
        },
    )
    print(f"  Updated latest.json -> {dataset_id}")

    lock_path = write_lock(dataset_id, chive_id)
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
    """Load and validate a .pt dataset bundle (memory-mapped)."""
    bundle: dict = torch.load(path, map_location="cpu", weights_only=False, mmap=True)

    version = bundle.get("schema_version", 0)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Dataset schema version {version} != expected {SCHEMA_VERSION}. "
            "Rebuild the dataset or implement migration."
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
    upload_dataset(args.path)


def _cmd_download(args: argparse.Namespace) -> None:
    path = download_dataset(args.id)
    print(f"Downloaded to {path}")


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

    p_download = sub.add_parser("download", help="Download dataset from GCS")
    p_download.add_argument("id", help="Dataset ID or 'latest'")

    sub.add_parser("list", help="List datasets in GCS")

    p_info = sub.add_parser("info", help="Show dataset metadata")
    p_info.add_argument("id", help="Dataset ID or 'latest'")

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
        "list": _cmd_list,
        "info": _cmd_info,
        "resolve": _cmd_resolve,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
