"""Checkpoint management: upload, download, pull, list, info.

Mirrors the dataset publishing pattern in scripts/dataset.py.
Checkpoints are stored on GCS under kotogram-checkpoints/{model_type}/
with a best.json pointer and append-only history.jsonl.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from scripts.gcs import (
    GCS_BUCKET,
    find_repo_root,
    gcs_append_jsonl,
    gcs_download_file,
    gcs_exists,
    gcs_list_blobs,
    gcs_read_json,
    gcs_upload_file,
    gcs_write_json,
)

GCS_PREFIX = "kotogram-checkpoints"
CHECKPOINT_LOCK = "checkpoint.lock"
LOCAL_CACHE = os.path.join(".cache", "checkpoints")


def compute_checkpoint_id(
    config_dict: dict,
    dataset_id: str,
    epoch: int,
    run_timestamp: str,
) -> str:
    """Deterministic 16-hex-char ID from training context."""
    canonical = json.dumps(
        {
            "config": sorted(config_dict.items()),
            "dataset_id": dataset_id,
            "epoch": epoch,
            "ts": run_timestamp,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gcs_model_prefix(model_type: str) -> str:
    return f"{GCS_PREFIX}/{model_type}"


def upload_best(
    model_type: str,
    checkpoint_path: str,
    metadata: Dict[str, Any],
) -> str:
    """Upload a checkpoint to GCS as the new best, returning the GCS URI.

    1. Uploads the .pt file (idempotent -- skips if blob already exists).
    2. Uploads the metadata sidecar .meta.json.
    3. Overwrites best.json with the new pointer.
    4. Appends to history.jsonl.
    """
    checkpoint_id = metadata["checkpoint_id"]
    prefix = _gcs_model_prefix(model_type)

    sha = _file_sha256(checkpoint_path)
    metadata["sha256"] = sha
    metadata["file_size_bytes"] = os.path.getsize(checkpoint_path)

    pt_key = f"{prefix}/checkpoints/ckpt-{checkpoint_id}.pt"
    if not gcs_exists(pt_key):
        print(f"  Uploading checkpoint {checkpoint_id}...")
        gcs_upload_file(checkpoint_path, pt_key)
    else:
        print(f"  Checkpoint {checkpoint_id} already in GCS, skipping upload")

    meta_key = f"{prefix}/checkpoints/ckpt-{checkpoint_id}.meta.json"
    gcs_write_json(meta_key, metadata)

    best_data = {
        "checkpoint_id": checkpoint_id,
        "model_type": model_type,
        "criteria": metadata["criteria"],
        "epoch": metadata["epoch"],
        "dataset_id": metadata["dataset_id"],
        "chive_id": metadata["chive_id"],
        "mlflow_run_id": metadata.get("mlflow_run_id"),
        "created_at": metadata["created_at"],
    }
    gcs_write_json(f"{prefix}/best.json", best_data)

    history_record = {
        **best_data,
        "previous_criteria": metadata.get("previous_criteria"),
    }
    gcs_append_jsonl(f"{prefix}/history.jsonl", history_record)

    uri = f"gs://{GCS_BUCKET}/{pt_key}"
    print(f"  Updated best.json -> {checkpoint_id}")
    return uri


def read_best(model_type: str) -> Optional[Dict[str, Any]]:
    """Read best.json from GCS for a model type. Returns None if missing."""
    key = f"{_gcs_model_prefix(model_type)}/best.json"
    if not gcs_exists(key):
        return None
    return gcs_read_json(key)


def download_checkpoint(model_type: str, checkpoint_id: str) -> str:
    """Download a specific checkpoint .pt from GCS, returning local path."""
    os.makedirs(LOCAL_CACHE, exist_ok=True)
    local_path = os.path.join(LOCAL_CACHE, f"ckpt-{checkpoint_id}.pt")
    if os.path.exists(local_path):
        return local_path
    gcs_key = f"{_gcs_model_prefix(model_type)}/checkpoints/ckpt-{checkpoint_id}.pt"
    print(f"Downloading checkpoint {checkpoint_id}...")
    gcs_download_file(gcs_key, local_path)
    return local_path


def download_metadata(model_type: str, checkpoint_id: str) -> Dict[str, Any]:
    """Download the metadata sidecar for a checkpoint."""
    gcs_key = (
        f"{_gcs_model_prefix(model_type)}/checkpoints/ckpt-{checkpoint_id}.meta.json"
    )
    return gcs_read_json(gcs_key)


def verify_checkpoint(local_path: str, expected_sha256: str) -> bool:
    """Verify checkpoint integrity against its expected SHA256."""
    return _file_sha256(local_path) == expected_sha256


def pull_best(model_type: str) -> Optional[str]:
    """Pull the current best checkpoint: download from GCS, write checkpoint.lock.

    Returns the local path to the downloaded .pt, or None if no best exists.
    """
    best = read_best(model_type)
    if best is None:
        print(f"No best checkpoint found for {model_type}")
        return None

    checkpoint_id = best["checkpoint_id"]
    local_path = download_checkpoint(model_type, checkpoint_id)

    meta = download_metadata(model_type, checkpoint_id)
    expected_sha = meta.get("sha256")
    if expected_sha:
        if verify_checkpoint(local_path, expected_sha):
            print("  Integrity verified (SHA256 match)")
        else:
            print("  WARNING: SHA256 mismatch! Checkpoint may be corrupted.")

    write_lock(
        model_type=model_type,
        checkpoint_id=checkpoint_id,
        criteria=best.get("criteria", {}),
        epoch=best.get("epoch", 0),
        dataset_id=best.get("dataset_id", ""),
        chive_id=best.get("chive_id", ""),
        mlflow_run_id=best.get("mlflow_run_id"),
    )

    print(f"  checkpoint.lock -> {checkpoint_id}")
    print(f"  Criteria: {json.dumps(best.get('criteria', {}))}")
    return local_path


def pull_specific(model_type: str, checkpoint_id: str) -> Optional[str]:
    """Pull a specific checkpoint by ID: download from GCS, write checkpoint.lock."""
    meta = download_metadata(model_type, checkpoint_id)
    local_path = download_checkpoint(model_type, checkpoint_id)

    expected_sha = meta.get("sha256")
    if expected_sha:
        if verify_checkpoint(local_path, expected_sha):
            print("  Integrity verified (SHA256 match)")
        else:
            print("  WARNING: SHA256 mismatch! Checkpoint may be corrupted.")

    write_lock(
        model_type=model_type,
        checkpoint_id=checkpoint_id,
        criteria=meta.get("criteria", {}),
        epoch=meta.get("epoch", 0),
        dataset_id=meta.get("dataset_id", ""),
        chive_id=meta.get("chive_id", ""),
        mlflow_run_id=meta.get("mlflow_run_id"),
    )

    print(f"  checkpoint.lock -> {checkpoint_id}")
    return local_path


def list_checkpoints(model_type: str) -> List[str]:
    """List checkpoint IDs available on GCS for a model type."""
    prefix = f"{_gcs_model_prefix(model_type)}/checkpoints/ckpt-"
    blobs = gcs_list_blobs(prefix)
    ids = set()
    for name in blobs:
        basename = name.rsplit("/", 1)[-1]
        if basename.startswith("ckpt-") and basename.endswith(".pt"):
            ids.add(basename[5:-3])
    return sorted(ids)


# ── Lock file I/O ──────────────────────────────────────────────────────


def read_lock() -> Optional[Dict[str, Any]]:
    """Read checkpoint.lock from the repo root. Returns None if missing."""
    lock_path = os.path.join(find_repo_root(), CHECKPOINT_LOCK)
    if not os.path.exists(lock_path):
        return None
    with open(lock_path, encoding="utf-8") as f:
        result: Dict[str, Any] = json.load(f)
    return result


def write_lock(
    model_type: str,
    checkpoint_id: str,
    *,
    criteria: Dict[str, float],
    epoch: int,
    dataset_id: str,
    chive_id: str,
    mlflow_run_id: Optional[str] = None,
) -> str:
    """Write checkpoint.lock to the repo root. Returns the path written."""
    lock_path = os.path.join(find_repo_root(), CHECKPOINT_LOCK)
    data: Dict[str, Any] = {
        "model_type": model_type,
        "checkpoint_id": checkpoint_id,
        "criteria": criteria,
        "epoch": epoch,
        "dataset_id": dataset_id,
        "chive_id": chive_id,
        "mlflow_run_id": mlflow_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return lock_path


# ── CLI ────────────────────────────────────────────────────────────────


def _cmd_pull(args: argparse.Namespace) -> None:
    if args.id:
        path = pull_specific(args.model_type, args.id)
    else:
        path = pull_best(args.model_type)
    if path:
        print(f"  Local: {path}")


def _cmd_list(args: argparse.Namespace) -> None:
    best = read_best(args.model_type)
    best_id = best["checkpoint_id"] if best else None

    ids = list_checkpoints(args.model_type)
    if not ids:
        print(f"No checkpoints found for {args.model_type}")
        return
    for cid in ids:
        marker = " <- best" if cid == best_id else ""
        print(f"  {cid}{marker}")
    print(f"\n{len(ids)} checkpoint(s)")


def _cmd_info(args: argparse.Namespace) -> None:
    checkpoint_id = args.checkpoint_id
    if checkpoint_id == "best":
        best = read_best(args.model_type)
        if not best:
            print(f"No best checkpoint for {args.model_type}")
            return
        checkpoint_id = best["checkpoint_id"]

    meta = download_metadata(args.model_type, checkpoint_id)
    print(json.dumps(meta, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkpoint management for kotogram models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help="Download best (or specific) checkpoint")
    p_pull.add_argument("model_type", help="Model type (e.g. recon_bpd)")
    p_pull.add_argument(
        "--id", default=None, help="Specific checkpoint ID (default: best)"
    )
    p_pull.set_defaults(func=_cmd_pull)

    p_list = sub.add_parser("list", help="List available checkpoints on GCS")
    p_list.add_argument("model_type", help="Model type (e.g. recon_bpd)")
    p_list.set_defaults(func=_cmd_list)

    p_info = sub.add_parser("info", help="Show metadata for a checkpoint")
    p_info.add_argument("model_type", help="Model type (e.g. recon_bpd)")
    p_info.add_argument("checkpoint_id", help="Checkpoint ID or 'best'")
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
