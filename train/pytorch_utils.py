"""PyTorch utility functions."""

import importlib.util
import math

import torch

from kotogram.model import NUM_REGISTER_CLASSES, ModelConfig


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
    if config.kc_enabled:
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
    if is_kc and config.kc_enabled:
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
