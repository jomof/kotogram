"""Shared Google Cloud Storage helpers for dataset and checkpoint modules."""

import json
import os
import subprocess
from typing import Any, Dict, List

GCS_BUCKET = "jomof-public-files"


def _gcs_bucket() -> Any:
    from google.cloud import storage  # type: ignore[import-untyped]

    return storage.Client().bucket(GCS_BUCKET)


def gcs_upload_file(local_path: str, gcs_key: str) -> str:
    blob = _gcs_bucket().blob(gcs_key)
    blob.upload_from_filename(local_path)
    return f"gs://{GCS_BUCKET}/{gcs_key}"


def gcs_download_file(gcs_key: str, local_path: str) -> str:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    blob = _gcs_bucket().blob(gcs_key)
    blob.download_to_filename(local_path)
    return local_path


def gcs_exists(gcs_key: str) -> bool:
    return bool(_gcs_bucket().blob(gcs_key).exists())


def gcs_list_blobs(prefix: str) -> List[str]:
    return [str(b.name) for b in _gcs_bucket().list_blobs(prefix=prefix)]


def gcs_read_json(gcs_key: str) -> Dict[str, Any]:
    blob = _gcs_bucket().blob(gcs_key)
    result: Dict[str, Any] = json.loads(blob.download_as_text())
    return result


def gcs_write_json(gcs_key: str, data: dict) -> None:
    blob = _gcs_bucket().blob(gcs_key)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")


def gcs_append_jsonl(gcs_key: str, record: dict) -> None:
    """Append a JSON record to a JSONL file on GCS (read-append-write)."""
    blob = _gcs_bucket().blob(gcs_key)
    existing = ""
    if blob.exists():
        existing = blob.download_as_text()
    line = json.dumps(record, separators=(",", ":")) + "\n"
    blob.upload_from_string(existing + line, content_type="application/jsonl")


def find_repo_root() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return os.getcwd()
