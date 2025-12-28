"""Worker logic for style training data encoding."""

import torch


def _worker_init_fn(_: int) -> None:
    """Worker initialization function to limit per-worker threads."""
    torch.set_num_threads(1)
    try:
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
    except RuntimeError as e:  # worker-init=special-carveout
        # This can happen if the runtime is already initialized with >1 thread.
        # It's safe to ignore if we simply cannot change it now.
        if "cannot set number of interop threads" not in str(e):
            raise e
