import contextlib
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

from train.kc import KcFamilyId

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


# Loss Weights
_FORMALITY_LOSS_WEIGHT_DEFAULT = 1.0
_GRAMMATICALITY_LOSS_WEIGHT_DEFAULT = 1.0
_REGISTER_LOSS_WEIGHT_DEFAULT = 1.0
_GENDER_LOSS_WEIGHT_DEFAULT = 1.0
_GENDER_MSE_SCALING_FACTOR_DEFAULT = 10.0


@dataclass(frozen=True)
class KCConfig:
    """Pretraining (KC) hyperparameter configuration."""

    sparsity_weight: float = 0.1
    target_spill_rate: float = 0.5  # Target probability for (k+1)th KC (0.0 = disabled)
    freeze_encoder_epochs: int = 0

    # Diversity / Coverage
    diversity_weight: float = 1e-3
    diversity_weight_thawed: float = 0.4
    diversity_eps: float = 1e-9
    diversity_warmup_epochs: int = 0

    # Load Balancing
    lb_weight: float = 0.0
    lb_weight_thawed: float = 0.1

    # Coverage Loss (encourage all KC logits to be used)
    coverage_weight: float = 0.0  # Start at 0, can enable after diversity is working
    coverage_weight_thawed: float = (
        0.1  # Weight when encoder is thawed (comparable to load_bal)
    )
    # Minimum probability threshold for a KC logit to be considered "used".
    # For each KC logit, we compute its maximum probability across all samples in the batch.
    # If that max probability is below this threshold, the logit is penalized.
    # A value of 0.05 means each KC should reach at least 5% probability for at least one sample.
    # Lower values (e.g., 0.01) are more lenient; higher values (e.g., 0.1) require stronger activations.
    coverage_min_prob: float = 0.5  # Minimum max probability per KC logit

    # Collapse Prevention
    collapse_weight_thawed: float = 10.0

    # Prior KC Exclusivity removed (prior losses now handled by style classifier)

    # Temperature
    temperature_thawed: float = 1.8

    # Optimization
    kc_grad_cap: float = 5.0

    # Dynamic Training Constraints
    entropy_floor: float = 0.95
    # Dynamic Training Constraints
    kl_cap: float = 0.05

    # Saturation Penalty
    sat_weight: float = 1.0

    # Performance: Skip diagnostic metrics until epoch N
    skip_first_metrics: int = 0

    # Grammar Point (Multi-Label PNU) Loss
    gp_unlabeled_weight: float = (
        0.001  # Weight for unlabeled positions (weak negative assumption)
    )
    gp_pos_weight: float = 1.0  # Weight for labeled positives
    gp_neg_weight: float = 250.0  # Weight for labeled negatives

    # Style Oversampling (for addressing class imbalance in gender/formality)
    style_oversample: bool = True  # Enable oversampling of non-neutral examples
    formality_boost: float = 5.0  # Multiplier for |formality| > 0.25
    gender_boost: float = 50.0  # Multiplier for |gender| > 0.25 (was 15.0, increased for 96% neutral problem)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KCConfig":
        from dataclasses import fields

        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in valid}
        # Handle list -> tuple conversion for first_batch_debug_epochs if needed

        return cls(**kwargs)


@dataclass(frozen=True)
class TrainerConfig:
    """Configuration for model training."""

    learning_rate: float = 5e-5
    batch_size: int = 128
    epochs: int = 20  # Fine-tuning epochs
    kc_epochs: int = 11
    freeze_encoder_epochs: int = 0
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
    # Encoder LR is hardcoded to 0.0 (frozen) during style training

    # Orthogonal components
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    dataloader: DataLoaderSettings = field(default_factory=DataLoaderSettings)

    # Process-specific Configuration
    # (Removed ProcessSettings as unused)

    # Global Toggles and Intervals
    progress_update_every: int = 50  # Batches between progress bar updates
    log_flush_every: int = 200  # Batches between stdout flushes

    # Knowledge Component Targets (Training only)
    kc_target_specs: Dict[KcFamilyId, int] = field(default_factory=dict)

    # KC Configuration
    kc_config: KCConfig = field(default_factory=KCConfig)

    # Runtime flags (from wrapper, not persisted to model)
    retrain: bool = False  # Start from scratch, ignore checkpoints
    sample_ratio: float = 1.0  # Data sampling ratio (1.0 = 100%)
    label_only: bool = False  # Run only preprocessing/labeling phase
    report_only: bool = False  # Generate performance report and exit

    # Evaluation frequency: run full validation every N epochs (1 = every epoch)
    # Set higher to skip expensive accuracy computation on intermediate epochs
    eval_every_n_epochs: int = 5

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
            "hardware": self.hardware.to_dict(),
            "dataloader": self.dataloader.to_dict(),
            "kc_config": self.kc_config.to_dict(),
            "progress_update_every": self.progress_update_every,
            "log_flush_every": self.log_flush_every,
            "kc_target_specs": {k.value: v for k, v in self.kc_target_specs.items()},
            "retrain": self.retrain,
            "sample_ratio": self.sample_ratio,
            "label_only": self.label_only,
            "report_only": self.report_only,
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

        # Handle kc_target_specs
        if "kc_target_specs" in d:
            raw_specs = d["kc_target_specs"]
            new_specs = {}
            # We assume keys are integers (or strings of integers) mapping to vocab sizes

            for k, v in raw_specs.items():
                if isinstance(k, (int, str)):
                    fid = KcFamilyId(int(k))
                    new_specs[fid] = v
            d["kc_target_specs"] = new_specs

        if "kc_config" in d:
            d["kc_config"] = KCConfig.from_dict(d["kc_config"])

        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})

    def resolve_dataloader_config(
        self,
        device: torch.device,
        mode: str = "train",
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
    mode: str,
) -> DataLoaderConfig:
    """Determine safe and performant DataLoader settings based on environment and load."""
    cpu_count = os.cpu_count() or 1
    is_cuda = "cuda" in str(device)

    # 1. Base Policy
    if is_cuda or device.type == "mps":
        # Conservative defaults for CUDA/MPS
        num_workers = min(4, max(2, cpu_count // 4))  # Slightly relaxed from //8
        if config.dataloader.num_workers is not None:
            num_workers = config.dataloader.num_workers

        # Force disable pin_memory on MPS to avoid warnings/hangs,
        # even if config requests it (unless we are sure it's safe)
        if device.type == "mps":
            pin_memory = False
        else:
            pin_memory = (
                config.dataloader.pin_memory
                if config.dataloader.pin_memory is not None
                else True
            )

        if num_workers == 0:
            prefetch_factor = None
        else:
            prefetch_factor = config.dataloader.prefetch_factor

        persistent_workers = config.dataloader.persistent_workers and num_workers > 0
    else:
        # Avoid workers on CPU-only to save overhead unless explicitly requested
        if config.dataloader.num_workers is not None:
            num_workers = config.dataloader.num_workers
            pin_memory = config.dataloader.pin_memory or False
            prefetch_factor = config.dataloader.prefetch_factor
            persistent_workers = (
                config.dataloader.persistent_workers and num_workers > 0
            )
        else:
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
    with contextlib.suppress(RuntimeError):
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
    with contextlib.suppress(RuntimeError):
        if torch.get_num_interop_threads() != config.hardware.interop_threads:
            torch.set_num_interop_threads(config.hardware.interop_threads)
