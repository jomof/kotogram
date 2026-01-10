"""PyTorch utility functions."""

import importlib.util
import math
from typing import Any, Callable, Dict, Mapping

import torch

from kotogram.model import NUM_REGISTER_CLASSES, ModelConfig

# 50KB tolerance for header overhead
SIZE_VERIFICATION_TOLERANCE = 50 * 1024


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


def calculate_detailed_size(config_dict: Mapping[str, Any]) -> Dict[str, int]:
    """Calculate the expected size of model components in bytes (FP8)."""
    # pylint: disable=too-many-locals

    # FEATURE_FIELDS defaults logic matches MultiFieldEmbedding
    from kotogram.tokenizer import FEATURE_FIELDS

    d_model = int(config_dict.get("d_model", 256))
    hidden_dim = int(config_dict.get("hidden_dim", 512))
    num_layers = int(config_dict.get("num_layers", 3))

    vocab_sizes = config_dict.get("vocab_sizes", {})
    field_embed_dims = config_dict.get("field_embed_dims", {})

    embed_params = 0
    for field in FEATURE_FIELDS:
        size = int(vocab_sizes.get(field, 100))
        dim = int(field_embed_dims.get(field, 32))
        embed_params += size * dim

    total_embed_dim = sum(int(field_embed_dims.get(f, 32)) for f in FEATURE_FIELDS)
    embed_params += total_embed_dim * d_model + d_model  # Project weights/bias
    embed_params += 2 * d_model  # LayerNorm

    layer_params = (
        4 * (d_model * d_model + d_model)
        + 4 * d_model
        + 2 * (d_model * hidden_dim + hidden_dim)
    )
    transformer_params = num_layers * layer_params

    final_norm_params = 2 * d_model

    kc_vocab_size = int(config_dict.get("kc_vocab_size", 0))
    classifier_input_dim = d_model + kc_vocab_size
    head_mlp_hidden = hidden_dim

    def mlp_params(out_dim: int) -> int:
        l1 = classifier_input_dim * head_mlp_hidden + head_mlp_hidden
        l2 = head_mlp_hidden * out_dim + out_dim
        return int(l1 + l2)

    num_formality = int(config_dict.get("num_formality_pragmatic_classes", 2))
    num_gender = int(config_dict.get("num_gender_pragmatic_classes", 2))
    num_gram = int(config_dict.get("num_grammaticality_classes", 2))
    num_reg = NUM_REGISTER_CLASSES

    head_params = 0
    head_params += mlp_params(1)
    head_params += mlp_params(num_formality)
    head_params += mlp_params(1)
    head_params += mlp_params(num_gender)
    head_params += mlp_params(num_gram)
    head_params += mlp_params(num_reg)

    # KCHead: Single linear projection (d_model → kc_vocab_size) + LayerNorm
    # Architecture matches kotogram/model.py KCHead:
    #   self.output = nn.Linear(config.d_model, config.kc_vocab_size)
    #   self.layer_norm = nn.LayerNorm(config.kc_vocab_size)
    kc_head_params = (
        (d_model * kc_vocab_size + kc_vocab_size)  # output layer: weights + bias
        + (2 * kc_vocab_size)  # layer norm: weight + bias
    )
    pos_encoding_buffer = 512 * d_model

    return {
        "embeddings": int(embed_params),
        "transformer": int(transformer_params),
        "final_norm": int(final_norm_params),
        "heads": int(head_params),
        "kc_head": int(kc_head_params),
        "pos_encoding": int(pos_encoding_buffer),
    }


def generate_size_breakdown_report(
    state_dict: Mapping[str, torch.Tensor], expected_breakdown: Dict[str, int]
) -> str:
    """Generate a detailed report comparing actual vs expected component sizes."""
    # Calculate approximate active breakdown for diagnostics
    actual_breakdown = {
        "embeddings": 0,
        "transformer": 0,
        "final_norm": 0,
        "heads": 0,
        "kc_head": 0,
        "pos_encoding": 0,
        "other": 0,
    }

    for k, v in state_dict.items():
        # Estimate size: numel * element_size
        size = v.numel() * v.element_size()

        if k.startswith("embedding."):
            actual_breakdown["embeddings"] += size
        elif k.startswith("encoder."):
            actual_breakdown["transformer"] += size
        elif k == "pos_encoding.pe":
            actual_breakdown["pos_encoding"] += size
        elif (
            k.startswith("formality_")
            or k.startswith("gender_")
            or k.startswith("grammaticality_")
            or k.startswith("register_")
        ):
            actual_breakdown["heads"] += size
        elif k.startswith("kc_head."):
            actual_breakdown["kc_head"] += size
        else:
            # Catch-all for other params (norms, biases not captured above if naming differs)
            actual_breakdown["other"] += size

    # Breakdown message
    breakdown_msg = "\nSize Breakdown (Bytes):\n"
    breakdown_msg += (
        f"{'Component':<20} {'Expected':<15} {'Approx. Tensor Payload':<20}\n"
    )
    breakdown_msg += "-" * 60 + "\n"

    all_keys = set(expected_breakdown.keys()) | set(actual_breakdown.keys())
    for key in sorted(all_keys):
        exp = expected_breakdown.get(key, 0)
        act = actual_breakdown.get(key, 0)
        breakdown_msg += f"{key:<20} {exp:<15,} {act:<20,}\n"

    return breakdown_msg


def verify_model_size_policy(
    actual_size: int,
    expected_size: int,
    expected_breakdown: Dict[str, int],
    state_dict_provider: Callable[[], Mapping[str, torch.Tensor]],
) -> None:
    """
    Verify that the actual model size on disk matches expectation within tolerance.
    Raises RuntimeError with a detailed breakdown report if verification fails.
    """
    if not math.isclose(
        actual_size, expected_size, abs_tol=SIZE_VERIFICATION_TOLERANCE
    ):
        state_dict = state_dict_provider()
        report = generate_size_breakdown_report(state_dict, expected_breakdown)
        raise RuntimeError(
            f"Model size verification failed. Expected ~{expected_size:,} bytes, "
            f"found {actual_size:,} bytes. (Tolerance: {SIZE_VERIFICATION_TOLERANCE:,} bytes)\n"
            f"{report}"
        )
