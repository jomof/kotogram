import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from kotogram.model import ModelConfig


@dataclass(frozen=True)
class ProcessSettings:
    """Settings that vary between main and worker processes."""

    show_dataloader_config: bool = True
    show_safety_logs: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "show_dataloader_config": self.show_dataloader_config,
            "show_safety_logs": self.show_safety_logs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcessSettings":
        return cls(**d)


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
    set_env_thread_limits: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_threads": self.cpu_threads,
            "interop_threads": self.interop_threads,
            "torch_num_threads": self.torch_num_threads,
            "torch_num_interop_threads": self.torch_num_interop_threads,
            "cpu_reserve_cores": self.cpu_reserve_cores,
            "set_env_thread_limits": self.set_env_thread_limits,
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
    # If True, forces num_workers=0 and pin_memory=False
    interactive_dataloader: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "interactive_dataloader": self.interactive_dataloader,
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


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for model training."""

    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 20  # Fine-tuning epochs
    mlm_epochs: int = 5
    kc_epochs: int = 3
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
    encoder_lr_factor: float = (
        1.0  # LR multiplier for encoder during fine-tuning (< 1.0 after pretraining)
    )

    # Orthogonal components
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    dataloader: DataLoaderSettings = field(default_factory=DataLoaderSettings)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Process-specific Configuration
    main: ProcessSettings = field(
        default_factory=lambda: ProcessSettings(
            show_dataloader_config=True, show_safety_logs=True
        )
    )
    worker: ProcessSettings = field(
        default_factory=lambda: ProcessSettings(
            show_dataloader_config=False, show_safety_logs=False
        )
    )

    # Global Toggles and Intervals
    interactive_mode: bool = False
    progress_update_every: int = 50  # Batches between progress bar updates
    log_flush_every: int = 200  # Batches between stdout flushes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "mlm_epochs": self.mlm_epochs,
            "kc_epochs": self.kc_epochs,
            "patience": self.patience,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "use_amp": self.use_amp,
            "grad_accum_steps": self.grad_accum_steps,
            "formality_loss_weight": self.formality_loss_weight,
            "gender_loss_weight": self.gender_loss_weight,
            "grammaticality_loss_weight": self.grammaticality_loss_weight,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "encoder_lr_factor": self.encoder_lr_factor,
            "hardware": self.hardware.to_dict(),
            "dataloader": self.dataloader.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "main": self.main.to_dict(),
            "worker": self.worker.to_dict(),
            "interactive_mode": self.interactive_mode,
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

        with open(path, "r") as f:
            d = json.load(f)
        model_config = ModelConfig.from_dict(d)
        trainer_config = TrainerConfig.from_dict(d.get("trainer", {}))
        return model_config, trainer_config

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        d = dict(d)
        if "hardware" in d:
            d["hardware"] = HardwareConfig.from_dict(d["hardware"])
        if "dataloader" in d:
            d["dataloader"] = DataLoaderSettings.from_dict(d["dataloader"])
        if "checkpoint" in d:
            d["checkpoint"] = CheckpointConfig.from_dict(d["checkpoint"])
        if "main" in d:
            d["main"] = ProcessSettings.from_dict(d["main"])
        if "worker" in d:
            d["worker"] = ProcessSettings.from_dict(d["worker"])
        return cls(**d)

    def resolve_dataloader_config(
        self, device: torch.device, is_main: bool, mode: str = "train"
    ) -> DataLoaderConfig:
        """Resolve a safe DataLoader configuration for the current process and environment."""
        process = self.main if is_main else self.worker
        return _get_safe_dataloader_config(self, device, process, mode)

    def __post_init__(self) -> None:
        # Auto-detect interactive environment (GCP VM, SSH, tmux)
        if self.dataloader.interactive_dataloader is None:
            is_ssh = "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ
            is_tmux = "TMUX" in os.environ
            object.__setattr__(
                self.dataloader, "interactive_dataloader", is_ssh or is_tmux
            )

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
    reserve = config.hardware.cpu_reserve_cores if config.interactive_mode else 0
    usable = max(1, cores - reserve)

    if config.hardware.torch_num_threads is not None:
        t = config.hardware.torch_num_threads
    else:
        # Intra-op: primary computation threads
        # If interactive, be more conservative (share with system)
        t = max(1, usable // (2 if config.interactive_mode else 1))

    if config.hardware.torch_num_interop_threads is not None:
        it = config.hardware.torch_num_interop_threads
    else:
        # Inter-op: parallelism between independent ops
        it = max(1, min(4, usable // 4))

    return t, it


def _get_safe_dataloader_config(
    config: TrainerConfig,
    device: torch.device,
    process: ProcessSettings,
    mode: str = "train",
) -> DataLoaderConfig:
    """Determine safe and performant DataLoader settings based on environment and load."""
    cpu_count = os.cpu_count() or 1
    is_cuda = "cuda" in str(device)

    # 1. Base Policy
    if config.dataloader.interactive_dataloader:
        # Keep machine responsive in interactive sessions
        if process.show_safety_logs and mode == "train":
            print(
                "  [Safety] Interactive dataloader mode detected (SSH/tmux). Forces num_workers=0, pin_memory=False."
            )
        num_workers = 0
        pin_memory = False
        prefetch_factor = None
        persistent_workers = False
    elif is_cuda:
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

    # 3. Safety Valve: Check system stress
    stressed = False
    reasons = []

    # Load average check
    try:
        load1, _, _ = os.getloadavg()
        if load1 > cpu_count * 1.5:
            stressed = True
            reasons.append(f"high load ({load1:.1f} > {cpu_count * 1.5:.1f})")
    except (AttributeError, OSError):
        pass

    # Memory check (Linux only)
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {
                    line.split(":")[0]: int(line.split(":")[1].split()[0])
                    for line in f
                    if ":" in line
                }
            mem_available_kb = meminfo.get("MemAvailable", 0)
            if mem_available_kb < 1024 * 1024:  # Less than 1GB available
                stressed = True
                reasons.append(f"low memory ({mem_available_kb // 1024}MB available)")
        except Exception:
            pass

    # Downgrade if stressed
    if stressed:
        if num_workers > 1:
            num_workers = max(1, num_workers // 2)
        pin_memory = False
        if prefetch_factor is not None:
            prefetch_factor = 1

        if process.show_safety_logs and mode == "train":
            reason_str = ", ".join(reasons)
            print(
                f"  [Safety] System stressed ({reason_str}). Downgraded DataLoader settings: "
                f"workers={num_workers}, pin={pin_memory}, prefetch={prefetch_factor}"
            )

            print(
                f"  [Runtime] DataLoader ({mode}): workers={num_workers}, "
                f"pin={pin_memory}, persistent={persistent_workers}, "
                f"prefetch={prefetch_factor or 'default'}, threads={config.hardware.cpu_threads}"
            )

    return DataLoaderConfig(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def configure_runtime_thread_limits(config: TrainerConfig) -> None:
    """Set torch and environment thread limits to prevent oversubscription."""
    try:
        torch.set_num_threads(config.hardware.cpu_threads)
        torch.set_num_interop_threads(config.hardware.interop_threads)
    except RuntimeError:
        # Already set or parallel work started, ignore
        pass

    if config.hardware.set_env_thread_limits:
        for env_var in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            if env_var not in os.environ:
                os.environ[env_var] = str(config.hardware.cpu_threads)

        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _safe_configure_threads(config: TrainerConfig) -> None:
    """Configures PyTorch threads safely, ignoring errors if already set."""
    torch.set_num_threads(config.hardware.cpu_threads)
    try:
        torch.set_num_interop_threads(config.hardware.interop_threads)
    except RuntimeError:
        pass  # Already set or parallel work started
