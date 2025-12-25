import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TrainerConfig:
    """Configuration for model training."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    patience: int = 5  # Early stopping patience
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    gradient_clip: float = 1.0
    use_class_weights: bool = True
    formality_loss_weight: float = 1.0  # Weight for formality loss in multi-task
    gender_loss_weight: float = 1.0  # Weight for gender loss in multi-task
    grammaticality_loss_weight: float = (
        1.0  # Weight for grammaticality loss in multi-task
    )
    register_loss_weight: float = 1.0  # Weight for register loss in multi-task
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    use_amp: bool = False  # Mixed precision training
    grad_accum_steps: int = 1  # Gradient accumulation steps
    local_rank: int = 0  # Local rank for distributed training
    world_size: int = 1  # World size for distributed training

    # Round 16: Resource Safety Guardrails
    cpu_threads: int = 2
    interop_threads: int = 1
    set_env_thread_limits: bool = True
    dataloader_num_workers_style: Optional[int] = None
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    dataloader_pin_memory: Optional[bool] = None
    dataloader_show_config: bool = True

    # Round 17: Resumable Pretraining and Enhanced Resource Safety
    checkpoint_dir: Optional[str] = None
    checkpoint_every_n_steps: Optional[int] = None
    resume_from: Optional[str] = None

    # Round 18: Interactive Mode and Performance Tuning
    interactive_mode: bool = False
    dataloader_num_workers: Optional[int] = None  # If None, auto-tune
    torch_num_threads: Optional[int] = None  # intra-op threads
    torch_num_interop_threads: Optional[int] = None
    cpu_reserve_cores: int = 2  # Cores to reserve for OS/SSH
    progress_update_every: int = 50  # Batches between progress bar updates
    log_flush_every: int = 200  # Batches between stdout flushes

    # If True, forces num_workers=0 and pin_memory=False to keep machine responsive
    interactive_dataloader: Optional[bool] = None

    def __post_init__(self) -> None:
        # Auto-detect interactive environment (GCP VM, SSH, tmux)
        if self.interactive_dataloader is None:
            is_ssh = "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ
            is_tmux = "TMUX" in os.environ
            self.interactive_dataloader = is_ssh or is_tmux
