"""Distributed training utilities."""

import os
import sys
from datetime import timedelta
from typing import Tuple

import torch
import torch.distributed as dist


def is_main_process() -> bool:
    """Check if we are on the main process (rank 0)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed training if available.

    Supports NCCL for CUDA and Gloo for CPU/MPS.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
            if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in os.environ:
                os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=60),
        )
        print(
            f"Distributed init: Rank {rank}/{world_size} (Local {local_rank}) using {backend} backend"
        )
        return rank, world_size, local_rank

    return 0, 1, 0
