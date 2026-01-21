"""PyTorch utility functions."""

import importlib.util
import math
from typing import TYPE_CHECKING, Optional

import torch

from kotogram.model import NUM_REGISTER_CLASSES, ModelConfig

if TYPE_CHECKING:
    from kotogram.model import InferenceClassifier

# 50KB tolerance for header overhead
SIZE_VERIFICATION_TOLERANCE = 50 * 1024


def initialize_dataset_indices(
    offsets_len: int, indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Initialize dataset indices.

    If indices is None, returns a tensor of all indices [0, offsets_len - 1].
    Otherwise, returns the provided indices.
    """
    total_samples = offsets_len - 1
    if indices is not None:
        return indices
    return torch.arange(total_samples, dtype=torch.long)


def _is_layer_saved_in_slim(layer_name: str) -> bool:
    """Check if a layer is saved in the slim model.

    Uses KC family registry to determine which KC decoder layers are included.
    """
    from train.kc import KC_FAMILIES, KcMseFamily

    if not layer_name.startswith("kc_decoders."):
        return True  # Non-KC decoder layers are always saved

    parts = layer_name.split(".")
    sublayer = parts[1] if len(parts) >= 2 else ""

    # Label pathway layers: keep if any non-MSE family decoder is needed
    if sublayer in ("label_hidden1", "label_hidden2", "activation"):
        return any(
            not isinstance(fam, KcMseFamily) and not fam.is_slim_decoder
            for fam in KC_FAMILIES.values()
        )

    # MSE pathway layers: keep if any MSE family decoder is needed
    if sublayer in ("mse_hidden1", "mse_hidden2", "tanh"):
        return any(
            isinstance(fam, KcMseFamily) and not fam.is_slim_decoder
            for fam in KC_FAMILIES.values()
        )

    # Per-family decoders: check is_slim_decoder
    if sublayer in ("decoders", "mse_decoders") and len(parts) >= 3:
        family_name = parts[2]
        family = next(
            (
                fam
                for fid, fam in KC_FAMILIES.items()
                if fid.name.lower() == family_name
            ),
            None,
        )
        return family is not None and not family.is_slim_decoder

    return False


def calculate_model_static_memory(config: ModelConfig) -> int:
    """Calculate static memory (bytes) for model parameters, gradients, and optimizer."""
    # Approximate parameter count
    # Embeddings
    vocab_sum = sum(config.vocab_sizes.values())
    embed_params = vocab_sum * max(
        config.field_embed_dims.values()
    )  # Rough upper bound

    # Transformer
    # Each layer:
    # - Self Attn: 4 * d_model^2 (q,k,v,o)
    # - FeedFwd: 2 * d_model * dim_feedforward(=2*d_model usually but config says hidden_dim)
    # - LayerNorms: 2 * d_model
    layer_params = (
        4 * config.d_model * config.d_model + 2 * config.d_model * config.hidden_dim
    )
    transformer_params = config.num_layers * layer_params

    # Heads
    head_params = config.d_model * (
        1
        + config.num_formality_pragmatic_classes
        + 1
        + config.num_gender_pragmatic_classes
        + config.num_grammaticality_classes
        + NUM_REGISTER_CLASSES
    )
    # KC Head params (always enabled)
    head_params += config.d_model * config.kc_vocab_size

    total_params = embed_params + transformer_params + head_params

    # Bytes: 4 (weights) + 4 (grads) + 8 (optimizer states) = 16 bytes/param
    return total_params * 16


def calculate_element_size_bytes(config: ModelConfig, is_kc: bool) -> int:
    """Calculate memory (bytes) required per training sample (batch size 1)."""
    # 1. Inputs (int64)
    # Assume generic number of fields ~ 10
    input_size = 10 * config.max_seq_len * 8

    # 2. Activations (Forward) stored for Backward
    # Size ~ num_layers * seq_len * d_model * FACTOR
    # FACTOR depends on implementation (Flash Attention reduces this, but standard logic is higher)
    # Using a safe heuristic factor of 16 (4 bytes * 4 intermediate tensors per layer)
    activation_size = config.num_layers * config.max_seq_len * config.d_model * 16

    # 3. Targets & Head Activations
    # KC targets can be sparse but we might materialize things
    # If KC: The KC head logits are (1, kc_vocab_size) per sample (after pooling)
    head_size = 0
    if is_kc:
        # Logits + Gradients for KC Head
        head_size += config.kc_vocab_size * 4 * 2  # Value + Grad

    return input_size + activation_size + head_size


def estimate_optimal_batch_size(
    device: torch.device, config: ModelConfig, is_kc: bool
) -> int:
    """Estimate optimal batch size based on available memory.

    Args:
        device: The PyTorch device to estimate for.
        config: The model configuration.
        is_kc: Whether training is for Knowledge Components (affects memory usage).

    Returns:
        Estimated optimal batch size.

    Raises:
        RuntimeError: If device type is not supported or memory cannot be queried.
        ImportError: If required dependencies are missing for the device type.
    """
    total_mem_bytes = 0

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_mem_bytes = props.total_memory
    elif device.type == "mps":
        # Heuristic for Apple Silicon (unified memory)
        if importlib.util.find_spec("psutil"):
            import psutil  # type: ignore[import-untyped]

            # Use 11% of System RAM as "safe" limit for training on MPS
            # User request: Target 512 (sweet spot).
            # Previous tuning: 0.15 -> ~730. 0.11 -> ~535 (close to 512).
            total_mem_bytes = int(psutil.virtual_memory().total * 0.11)
        else:
            raise ImportError(
                "psutil is required for auto-batch-size on MPS devices. "
                "Please install it or set a specific batch size."
            )
    elif device.type == "cpu":
        return 32
    else:
        raise RuntimeError(
            f"Auto-batch size estimation not supported for device type: {device.type}"
        )

    # Calculate static model memory
    static_mem = calculate_model_static_memory(config)

    # Calculate per-sample memory
    sample_mem = calculate_element_size_bytes(config, is_kc)

    # Available memory for batches
    available_mem = total_mem_bytes - static_mem

    if available_mem <= 0:
        # Edge case: Model too big for device
        return 1

    # Raw count
    raw_batch = available_mem // sample_mem

    # Safety factor (0.8) to account for fragmentation, overhead, miscellaneous
    safe_batch = int(raw_batch * 0.8)

    # Round down to nearest power of 2
    # User request: restore power-of-2 rounding (e.g. 532 -> 512)
    # Ensure safe_batch is at least 1 for log2
    safe_batch = max(1, safe_batch)
    target = 2 ** int(math.log2(safe_batch))

    # Clamp min (32)
    target = max(32, target)
    return int(target)


def verify_model_size(
    model: "InferenceClassifier",
    actual_file_size: int,
) -> None:
    """Verify model file size matches ArchitectureReport expectations.

    Args:
        model: The InferenceClassifier model to verify.
        actual_file_size: The actual file size in bytes from disk.

    Raises:
        RuntimeError: If the file size differs from expected by more than tolerance.
    """
    # Import here to avoid circular imports
    from train.architecture_report import generate_architecture_report

    report = generate_architecture_report(model)

    # The report gives us actual model sizes in memory (FP32 = 4 bytes/param).
    # The saved file uses FP8 (1 byte/param), so divide by 4.
    # Filter out KC decoder layers that are stripped from slim models.
    saved_layers = [
        layer for layer in report.layers if _is_layer_saved_in_slim(layer.name)
    ]
    expected_size = sum(layer.size_bytes for layer in saved_layers) // 4

    if not math.isclose(
        actual_file_size, expected_size, abs_tol=SIZE_VERIFICATION_TOLERANCE
    ):
        # Generate detailed breakdown for error message
        breakdown_lines = ["", "Size Breakdown (FP8 bytes):"]
        breakdown_lines.append(f"{'Component':<30} {'Size':<15}")
        breakdown_lines.append("-" * 50)

        # Group by top-level component (using saved_layers, not all layers)
        component_sizes: dict[str, int] = {}
        for layer in saved_layers:
            top_component = layer.name.split(".")[0]
            fp8_size = layer.size_bytes // 4  # Convert to FP8
            component_sizes[top_component] = (
                component_sizes.get(top_component, 0) + fp8_size
            )

        for component, size in sorted(component_sizes.items()):
            breakdown_lines.append(f"{component:<30} {size:<15,}")

        breakdown_lines.append("-" * 50)
        breakdown_lines.append(f"{'Total (expected)':<30} {expected_size:<15,}")
        breakdown_lines.append(f"{'Actual file size':<30} {actual_file_size:<15,}")

        raise RuntimeError(
            f"Model size verification failed. Expected ~{expected_size:,} bytes, "
            f"found {actual_file_size:,} bytes. (Tolerance: {SIZE_VERIFICATION_TOLERANCE:,} bytes)\n"
            + "\n".join(breakdown_lines)
        )
