"""Shared recon_bpd model architecture and inference utilities."""

from typing import Iterable


def count_encoder_layers(keys: Iterable[str]) -> int:
    """Count the number of unique encoder layer indices in a state dict's keys."""
    layer_nums: set[int] = set()
    for k in keys:
        if "encoder.layers." in k:
            parts = k.split(".")
            idx = parts.index("layers") + 1
            if idx < len(parts) and parts[idx].isdigit():
                layer_nums.add(int(parts[idx]))
    return max(layer_nums) + 1 if layer_nums else 1
