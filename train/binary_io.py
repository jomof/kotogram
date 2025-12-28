"""Binary I/O utilities for the columnar dataset format."""

import array
import os
from typing import List

# File extensions and types
EXT_OFFSETS = "offsets.bin"
EXT_LABELS = "labels.bin"
EXT_FEAT_PREFIX = "feat_"
EXT_KC_PREFIX = "kc_"
EXT_SENTENCES = "sentences.txt"
EXT_KOTOGRAMS = "kotograms.txt"


def write_int_array(path: str, data: List[int], typecode: str = "i") -> None:
    """Write a list of integers to a raw binary file."""
    arr = array.array(typecode, data)
    with open(path, "wb") as f:
        arr.tofile(f)


def write_float_array(path: str, data: List[float], typecode: str = "f") -> None:
    """Write a list of floats to a raw binary file."""
    arr = array.array(typecode, data)
    with open(path, "wb") as f:
        arr.tofile(f)


def merge_shards(
    shard_dir: str,
    output_file: str,
    num_workers: int,
    shard_template: str,
    dtype_size: int = 4,
) -> int:
    """Concatenate binary shards into a single file. Returns total elements."""
    total_elements = 0
    with open(output_file, "wb") as out_f:
        for i in range(num_workers):
            # shard_template should accept worker_id formatted, e.g., "shard_{}_feat_surface.bin"
            fname = shard_template.format(i)
            path = os.path.join(shard_dir, fname)
            if os.path.exists(path):
                with open(path, "rb") as in_f:
                    # Python's shutil.copyfileobj is optimized
                    import shutil

                    shutil.copyfileobj(in_f, out_f)

                size = os.path.getsize(path)
                total_elements += size // dtype_size
    return total_elements


def merge_offset_shards(
    shard_dir: str,
    output_file: str,
    num_workers: int,
    shard_template: str,
) -> int:
    """
    Merge offset shards into a single global offset file.
    Assumes each shard starts with 0.
    Writes global offsets: [0, s1_1, s1_2, ..., s1_end, s1_end+s2_1, ...]
    """
    current_total = 0
    total_elements = 0

    with open(output_file, "wb") as out_f:
        # Write initial 0
        array.array("i", [0]).tofile(out_f)
        total_elements += 1

        for i in range(num_workers):
            path = os.path.join(shard_dir, shard_template.format(i))
            if not os.path.exists(path):
                continue

            with open(path, "rb") as in_f:
                shard_data = array.array("i")
                shard_data.fromfile(in_f, os.path.getsize(path) // 4)

                if len(shard_data) <= 1:
                    continue

                # Offsets start from 0 in each shard. We add current_total to implementation global offsets.
                # Skip first 0 (it is redundant with end of previous shard)
                updates = array.array("i", [x + current_total for x in shard_data[1:]])
                updates.tofile(out_f)

                current_total += shard_data[-1]
                total_elements += len(updates)

    return total_elements
