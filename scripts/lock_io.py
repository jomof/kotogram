"""Generic lock-file read/write helpers shared by checkpoint.py and dataset.py."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def read_lock_file(lock_path: str) -> Optional[Dict[str, Any]]:
    """Read a JSON lock file. Returns None if missing."""
    if not os.path.exists(lock_path):
        return None
    with open(lock_path, encoding="utf-8") as f:
        result: Dict[str, Any] = json.load(f)
    return result


def write_lock_file(lock_path: str, data: Dict[str, Any]) -> str:
    """Write a JSON lock file with a ``created_at`` timestamp. Returns the path written."""
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return lock_path
