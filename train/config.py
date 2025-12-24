from dataclasses import dataclass

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
