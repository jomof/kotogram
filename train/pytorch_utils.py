"""PyTorch utility functions."""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kotogram.model import InferenceClassifier

# 100KB tolerance for safetensors header/metadata overhead
SIZE_VERIFICATION_TOLERANCE = 100 * 1024


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

    # Grammar point pathway: always saved
    if sublayer in ("gp_hidden", "gp_decoder"):
        return True

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

    # The report gives FP32 sizes (4 bytes/param). Saved file uses FP8 (1 byte)
    # for most layers, but FP16 (2 bytes) for classification heads.
    _fp16_prefixes = (
        "formality_pragmatic_head.",
        "gender_pragmatic_head.",
        "grammaticality_classifier.",
        "kc_head.",
    )
    saved_layers = [
        layer for layer in report.layers if _is_layer_saved_in_slim(layer.name)
    ]
    expected_size = sum(
        layer.size_bytes // 2
        if any(layer.name.startswith(p) for p in _fp16_prefixes)
        else layer.size_bytes // 4
        for layer in saved_layers
    )

    if not math.isclose(
        actual_file_size, expected_size, abs_tol=SIZE_VERIFICATION_TOLERANCE
    ):
        breakdown_lines = ["", "Size Breakdown (quantized bytes):"]
        breakdown_lines.append(f"{'Component':<30} {'Size':<15}")
        breakdown_lines.append("-" * 50)

        component_sizes: dict[str, int] = {}
        for layer in saved_layers:
            top_component = layer.name.split(".")[0]
            is_fp16 = any(layer.name.startswith(p) for p in _fp16_prefixes)
            q_size = layer.size_bytes // 2 if is_fp16 else layer.size_bytes // 4
            component_sizes[top_component] = (
                component_sizes.get(top_component, 0) + q_size
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
