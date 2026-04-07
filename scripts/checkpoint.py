"""Checkpoint management: upload, download, pull, list, info.

Mirrors the dataset publishing pattern in scripts/dataset.py.
Checkpoints are stored on GCS under kotogram-checkpoints/{model_type}/
with a best.json pointer and append-only history.jsonl.
"""

import argparse
import hashlib
import json
import os
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
    from scripts.lock_io import read_lock_file

    return read_lock_file(os.path.join(find_repo_root(), CHECKPOINT_LOCK))


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
    from scripts.lock_io import write_lock_file

    return write_lock_file(
        os.path.join(find_repo_root(), CHECKPOINT_LOCK),
        {
            "model_type": model_type,
            "checkpoint_id": checkpoint_id,
            "criteria": criteria,
            "epoch": epoch,
            "dataset_id": dataset_id,
            "chive_id": chive_id,
            "mlflow_run_id": mlflow_run_id,
        },
    )


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


def _format_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / (1024**3):.1f} GB"
    return f"{n / (1024**2):.1f} MB"


def _print_info_compact(meta: dict) -> None:
    cfg = meta.get("config", {})
    criteria = meta.get("criteria", {})
    prev = meta.get("previous_criteria", {})

    print(f"Checkpoint: {meta['checkpoint_id']}")
    print(f"  Model:       {meta.get('model_type', 'N/A')}")
    print(f"  Created:     {meta.get('created_at', 'N/A')}")
    print(f"  Epoch:       {meta.get('epoch', 'N/A')}/{cfg.get('epochs', '?')}")
    print(f"  Git commit:  {meta.get('git_commit', 'N/A')}")
    print(f"  Machine:     {meta.get('machine', 'N/A')}")
    print(f"  PyTorch:     {meta.get('pytorch_version', 'N/A')}")
    size = meta.get("file_size_bytes")
    if size:
        print(f"  Size:        {_format_bytes(size)}")

    if criteria:
        print("  Criteria:")
        for k, v in criteria.items():
            delta = ""
            if k in prev:
                d = v - prev[k]
                delta = f"  ({d:+.4f})"
            print(f"    {k}: {v:.4f}{delta}")

    arch_keys = ["d_model", "ffn_dim", "num_layers", "num_heads", "kc_vocab_size"]
    arch_parts = [f"{k}={cfg[k]}" for k in arch_keys if k in cfg]
    if arch_parts:
        print(f"  Architecture: {', '.join(arch_parts)}")

    print(f"  Dataset:     {meta.get('dataset_id', 'N/A')}")
    print(f"  chiVe:       {meta.get('chive_id', 'N/A')}")
    if meta.get("mlflow_run_id"):
        print(f"  MLflow:      {meta['mlflow_experiment']}/{meta['mlflow_run_id']}")


# ---------------------------------------------------------------------------
# Reflective model introspection (--decoders)
# ---------------------------------------------------------------------------


def _classify_param(name: str, shape: tuple) -> str:  # pylint: disable=too-many-return-statements
    """Classify a parameter by its role based on name suffix and shape."""
    ndim = len(shape)
    base = name.rsplit(".", 1)[-1] if "." in name else name
    if base == "weight" and ndim == 2:
        # Parent module: for "a.b.weight" -> "b", for "a.weight" -> "a"
        parts = name.split(".")
        parent = parts[-2] if len(parts) >= 2 else ""
        if "in_proj" in name:
            return "MultiheadAttention (QKV)"
        # Embedding: parent name ends with _embed or contains pos_embed,
        # but NOT proj/linear layers that happen to contain "embed" in the path.
        if parent.endswith("_embed") or parent == "embed":
            return "Embedding"
        if "pos_embed" in parent:
            return "Embedding"
        return "Linear"
    if base == "bias" and ndim == 1:
        return "bias"
    if base in ("weight", "bias") and ndim == 1:
        parent = name.rsplit(".", 2)[-2] if name.count(".") >= 2 else ""
        if "norm" in parent.lower():
            return "LayerNorm"
        return "bias" if base == "bias" else "param"
    if base in ("query", "pe"):
        return "buffer"
    return "param"


def _param_bytes(shape: tuple, dtype_str: str) -> int:
    from functools import reduce

    numel = reduce(lambda a, b: a * b, shape, 1)
    bits = {
        "torch.float32": 32,
        "torch.float16": 16,
        "torch.bfloat16": 16,
        "torch.int64": 64,
        "torch.int32": 32,
        "torch.int16": 16,
        "torch.int8": 8,
        "torch.uint8": 8,
        "torch.bool": 8,
    }
    return numel * bits.get(dtype_str, 32) // 8


def _print_decoders(model_type: str, checkpoint_id: str) -> None:  # pylint: disable=too-many-locals
    """Load the checkpoint and list decoder heads with input/output signatures."""
    import torch

    local_path = os.path.join(LOCAL_CACHE, f"ckpt-{checkpoint_id}.pt")
    if not os.path.exists(local_path):
        local_path = download_checkpoint(model_type, checkpoint_id)

    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    # Group params by top-level module
    modules: dict[str, list[tuple[str, tuple, str]]] = {}
    for key, tensor in cleaned.items():
        top = key.split(".")[0]
        if top not in modules:
            modules[top] = []
        modules[top].append((key, tuple(tensor.shape), str(tensor.dtype)))

    # Discover d_model from the encoder's feedforward layer input dim
    d_model = 0
    for key, tensor in cleaned.items():
        if "encoder" in key and "linear2" in key and key.endswith(".weight"):
            d_model = tensor.shape[0]
            break

    # Discover kc_vocab_size: the output of the module whose last linear
    # produces a dim that is then consumed as the first linear input of
    # another module.  In practice: look for a head that takes d_model as
    # input and outputs something != d_model.
    kc_dim = 0
    for mod, params in modules.items():
        linears = [(k, s) for k, s, _ in params if _classify_param(k, s) == "Linear"]
        if linears and linears[0][1][1] == d_model:
            last_out = linears[-1][1][0]
            if last_out != d_model and last_out > kc_dim:
                kc_dim = last_out

    # A "decoder head" is a top-level module whose first linear input dim
    # equals d_model (consumes pooled encoder) or kc_dim (consumes KC logits),
    # and which does NOT contain self-attention (those are encoder/pooler trunk).
    heads: list[tuple[str, list]] = []
    for mod, params in sorted(modules.items()):
        linears = [(k, s) for k, s, _ in params if _classify_param(k, s) == "Linear"]
        if not linears:
            continue
        has_attn = any("in_proj" in k for k, _, _ in params)
        if has_attn:
            continue
        first_in = linears[0][1][1]
        embeds = [(k, s) for k, s, _ in params if _classify_param(k, s) == "Embedding"]
        embed_total = sum(s[1] for _, s in embeds)
        core_in = first_in - embed_total if embed_total else first_in
        if core_in in (d_model, kc_dim):
            heads.append((mod, params))

    if not heads:
        print("\n  No decoder heads found.")
        return

    print(f"\n  Decoder heads (d_model={d_model}, kc={kc_dim}):")
    for mod_name, params in heads:
        total_b = sum(_param_bytes(s, d) for _, s, d in params)
        linears = [(k, s) for k, s, _ in params if _classify_param(k, s) == "Linear"]
        embeds = [(k, s) for k, s, _ in params if _classify_param(k, s) == "Embedding"]

        # Input description
        first_in = linears[0][1][1]
        embed_total = sum(s[1] for _, s in embeds)
        core_in = first_in - embed_total if embed_total else first_in

        in_parts: list[str] = []
        if core_in == d_model:
            in_parts.append("pooled")
        elif core_in == kc_dim:
            in_parts.append("kc")
        else:
            in_parts.append(str(core_in))
        for ek, es in embeds:
            short = ek.split(".")[-2] if "." in ek else ek
            in_parts.append(f"{short}[{es[1]}]")

        # Output description: terminal linears (whose output dim isn't the
        # input dim of a later linear in the same module)
        consumed_dims: set[int] = {ls[1] for _, ls in linears}
        out_parts: list[str] = []
        for lk, ls in linears:
            out_dim = ls[0]
            if out_dim not in consumed_dims:
                # Use the non-digit parent as the label
                segments = [
                    p for p in lk.rsplit(".", 1)[0].split(".") if not p.isdigit()
                ]
                short = segments[-1] if segments else lk
                out_parts.append(f"{short}[{out_dim:,}]")
        if not out_parts:
            lk, ls = linears[-1]
            segments = [p for p in lk.rsplit(".", 1)[0].split(".") if not p.isdigit()]
            short = segments[-1] if segments else lk
            out_parts.append(f"{short}[{ls[0]:,}]")

        in_str = " + ".join(in_parts)
        out_str = ", ".join(out_parts)
        print(
            f"    {mod_name:<16s} {in_str} -> {out_str}    ({_format_bytes(total_b)})"
        )


def _cmd_info(args: argparse.Namespace) -> None:
    model_type = args.model_type
    checkpoint_id = args.checkpoint_id

    if model_type is None or checkpoint_id is None:
        lock = read_lock()
        if lock is None:
            print("No checkpoint.lock found. Provide model_type and checkpoint_id.")
            return
        model_type = model_type or lock["model_type"]
        checkpoint_id = checkpoint_id or lock["checkpoint_id"]

    if checkpoint_id == "best":
        best = read_best(model_type)
        if not best:
            print(f"No best checkpoint for {model_type}")
            return
        checkpoint_id = best["checkpoint_id"]

    meta = download_metadata(model_type, checkpoint_id)
    if args.detailed:
        print(json.dumps(meta, indent=2))
    else:
        _print_info_compact(meta)

    if args.decoders:
        _print_decoders(model_type, checkpoint_id)


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
    p_info.add_argument(
        "model_type",
        nargs="?",
        default=None,
        help="Model type (default: from checkpoint.lock)",
    )
    p_info.add_argument(
        "checkpoint_id",
        nargs="?",
        default=None,
        help="Checkpoint ID or 'best' (default: from checkpoint.lock)",
    )
    p_info.add_argument(
        "--detailed",
        action="store_true",
        help="Show full metadata as JSON",
    )
    p_info.add_argument(
        "--decoders",
        action="store_true",
        help="Show model structure introspected from weights",
    )
    p_info.set_defaults(func=_cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
