import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from kotogram.model import ModelConfig


@dataclass(frozen=True)
class DataLoaderConfig:
    """Resolved DataLoader settings."""

    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: Optional[int] = None


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware and thread configuration."""

    # Resolved values (filled by TrainerConfig.__post_init__)
    cpu_threads: int = 2
    interop_threads: int = 1

    # User overrides
    torch_num_threads: Optional[int] = None
    torch_num_interop_threads: Optional[int] = None
    cpu_reserve_cores: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_threads": self.cpu_threads,
            "interop_threads": self.interop_threads,
            "torch_num_threads": self.torch_num_threads,
            "torch_num_interop_threads": self.torch_num_interop_threads,
            "cpu_reserve_cores": self.cpu_reserve_cores,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardwareConfig":
        return cls(**d)


@dataclass(frozen=True)
class DataLoaderSettings:
    """Input settings for DataLoader auto-tuning."""

    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataLoaderSettings":
        return cls(**d)


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint and resumption settings."""

    dir: Optional[str] = None
    every_n_steps: Optional[int] = None
    resume_from: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dir": self.dir,
            "every_n_steps": self.every_n_steps,
            "resume_from": self.resume_from,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointConfig":
        return cls(**d)


# Loss Weights
_FORMALITY_LOSS_WEIGHT_DEFAULT = 1.0
_GRAMMATICALITY_LOSS_WEIGHT_DEFAULT = 1.0
_REGISTER_LOSS_WEIGHT_DEFAULT = 1.0
_GENDER_LOSS_WEIGHT_DEFAULT = 1.0
_GENDER_MSE_SCALING_FACTOR_DEFAULT = 10.0


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for model training."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 20  # Fine-tuning epochs
    kc_epochs: int = 3
    patience: int = 5  # Early stopping patience
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    gradient_clip: float = 1.0
    formality_loss_weight: float = (
        _FORMALITY_LOSS_WEIGHT_DEFAULT  # Weight for formality loss in multi-task
    )
    gender_loss_weight: float = (
        _GENDER_LOSS_WEIGHT_DEFAULT  # Weight for gender loss in multi-task
    )
    grammaticality_loss_weight: float = _GRAMMATICALITY_LOSS_WEIGHT_DEFAULT  # Weight for grammaticality loss in multi-task
    register_loss_weight: float = (
        _REGISTER_LOSS_WEIGHT_DEFAULT  # Weight for register loss in multi-task
    )
    gender_mse_scaling_factor: float = (
        _GENDER_MSE_SCALING_FACTOR_DEFAULT  # Scaling factor for gender MSE component
    )
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    grad_accum_steps: int = 1  # Gradient accumulation steps
    encoder_lr_factor: float = (
        1.0  # LR multiplier for encoder during fine-tuning (< 1.0 after pretraining)
    )

    # Orthogonal components
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    dataloader: DataLoaderSettings = field(default_factory=DataLoaderSettings)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Process-specific Configuration
    # (Removed ProcessSettings as unused)

    # Global Toggles and Intervals
    progress_update_every: int = 50  # Batches between progress bar updates
    log_flush_every: int = 200  # Batches between stdout flushes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "kc_epochs": self.kc_epochs,
            "patience": self.patience,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "gradient_clip": self.gradient_clip,
            "grad_accum_steps": self.grad_accum_steps,
            "formality_loss_weight": self.formality_loss_weight,
            "gender_loss_weight": self.gender_loss_weight,
            "grammaticality_loss_weight": self.grammaticality_loss_weight,
            "register_loss_weight": self.register_loss_weight,
            "gender_mse_scaling_factor": self.gender_mse_scaling_factor,
            "encoder_lr_factor": self.encoder_lr_factor,
            "hardware": self.hardware.to_dict(),
            "dataloader": self.dataloader.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "progress_update_every": self.progress_update_every,
            "log_flush_every": self.log_flush_every,
        }

    @staticmethod
    def load_config(path: str) -> Tuple["ModelConfig", "TrainerConfig"]:
        """Load unified configuration from a JSON file.

        Returns:
            (ModelConfig, TrainerConfig)
        """

        from kotogram.model import ModelConfig

        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        model_config = ModelConfig.from_dict(d)
        trainer_config = TrainerConfig.from_dict(d.get("trainer", {}))
        return model_config, trainer_config

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        from dataclasses import fields

        d = dict(d)
        if "hardware" in d:
            d["hardware"] = HardwareConfig.from_dict(d["hardware"])
        if "dataloader" in d:
            d["dataloader"] = DataLoaderSettings.from_dict(d["dataloader"])
        if "checkpoint" in d:
            d["checkpoint"] = CheckpointConfig.from_dict(d["checkpoint"])

        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    def resolve_dataloader_config(
        self, device: torch.device, mode: str = "train"
    ) -> DataLoaderConfig:
        """Resolve a safe DataLoader configuration for the current process and environment."""
        return _get_safe_dataloader_config(self, device, mode)

    def __post_init__(self) -> None:
        # Resolve thread counts
        cpu_t, interop_t = _choose_torch_threads(self)
        object.__setattr__(self.hardware, "cpu_threads", cpu_t)
        object.__setattr__(self.hardware, "interop_threads", interop_t)


def detect_cpu_cores() -> int:
    """Returns number of CPU cores available."""
    # os.cpu_count() can return None
    count = os.cpu_count()
    return count if count is not None else 4


def _choose_torch_threads(config: "TrainerConfig") -> Tuple[int, int]:
    """Auto-tunes PyTorch threads based on config and system state.

    Returns:
        (intra_op_threads, inter_op_threads)
    """
    cores = detect_cpu_cores()
    usable = max(1, cores)

    if config.hardware.torch_num_threads is not None:
        t = config.hardware.torch_num_threads
    else:
        # Intra-op: primary computation threads
        t = max(1, usable)

    if config.hardware.torch_num_interop_threads is not None:
        it = config.hardware.torch_num_interop_threads
    else:
        # Inter-op: parallelism between independent ops
        it = max(1, min(4, usable // 4))

    return t, it


def _get_safe_dataloader_config(
    config: TrainerConfig,
    device: torch.device,
    mode: str = "train",
) -> DataLoaderConfig:
    """Determine safe and performant DataLoader settings based on environment and load."""
    cpu_count = os.cpu_count() or 1
    is_cuda = "cuda" in str(device)

    # 1. Base Policy
    if is_cuda:
        # Conservative defaults for CUDA
        num_workers = min(4, max(2, cpu_count // 8))
        if config.dataloader.num_workers is not None:
            num_workers = config.dataloader.num_workers

        pin_memory = (
            config.dataloader.pin_memory
            if config.dataloader.pin_memory is not None
            else True
        )
        prefetch_factor = config.dataloader.prefetch_factor
        persistent_workers = config.dataloader.persistent_workers and num_workers > 0
    else:
        # Avoid workers on non-CUDA to save overhead
        num_workers = 0
        pin_memory = False
        prefetch_factor = None
        persistent_workers = False

    # 2. Evaluation adjustments
    if mode == "val" and num_workers > 0:
        num_workers = max(1, num_workers // 2)
        prefetch_factor = 1

    return DataLoaderConfig(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def configure_runtime_thread_limits(config: TrainerConfig) -> None:
    """Set torch and environment thread limits to prevent oversubscription."""
    torch.set_num_threads(config.hardware.cpu_threads)
    if torch.get_num_interop_threads() != config.hardware.interop_threads:
        torch.set_num_interop_threads(config.hardware.interop_threads)

    for env_var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[env_var] = str(config.hardware.cpu_threads)

    # Tokenizers parallelism is usually harmful when we manage threads ourselves
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _safe_configure_threads(config: TrainerConfig) -> None:
    """Configures PyTorch threads safely, ignoring errors if already set."""
    torch.set_num_threads(config.hardware.cpu_threads)
    if torch.get_num_interop_threads() != config.hardware.interop_threads:
        torch.set_num_interop_threads(config.hardware.interop_threads)
