"""Model architecture introspection and reporting.

This module provides data structures and utilities to introspect a constructed
InferenceClassifier model and generate a structured architecture report.
"""

from dataclasses import dataclass
from typing import List, Tuple

from torch import nn

from kotogram.model import InferenceClassifier


@dataclass
class LayerInfo:
    """Information about a single layer in the model architecture."""

    name: str  # Fully qualified name, e.g. "embedding.embeddings.pos"
    module_type: str  # Short type name, e.g. "Embedding"
    input_dim: int  # Input dimension (or -1 if not applicable)
    output_dim: int  # Output dimension
    param_count: int  # Number of parameters
    size_bytes: int  # Size in bytes (FP8)
    depth: int  # Nesting depth for display indentation
    is_container: bool = False  # True if this is a 0-param container module
    trainer_role: str = ""  # "kc", "style", "shared", or "" (container/unknown)


@dataclass
class ArchitectureReport:
    """Complete model architecture report."""

    model_name: str
    layers: List[LayerInfo]
    total_params: int
    total_size_bytes: int


def _get_kc_trainer_modules() -> set[str]:
    """Return top-level module names trained by KCTrainer.

    Based on KCTrainer._create_optimizer parameter groups:
    - kc_head: KC prediction head
    - kc_decoders: KC-to-target decoders (if present)
    - embedding: shared encoder (trained at 10% LR)
    - encoder: shared transformer encoder (trained at 10% LR)
    """
    return {"kc_head", "kc_decoders", "embedding", "encoder", "position_encoding"}


def _get_style_trainer_modules() -> set[str]:
    """Return top-level module names trained by StyleTrainer.

    Based on Trainer.__init__ optimizer parameter groups:
    - embedding + encoder: frozen (LR=0), but still in param groups
    - All classifier heads: trained at full LR
    Note: formality/gender value predictions come from KC decoder MSE pathway
    Note: register is now handled by KC decoder, not a separate head
    """
    return {
        "formality_pragmatic_head",
        "gender_pragmatic_head",
        "grammaticality_classifier",
        "embedding",
        "encoder",
        "position_encoding",
    }


def _determine_trainer_role(module_name: str) -> str:
    """Determine which trainer(s) use a module based on its name.

    Returns:
        "kc" - Only used by KC trainer
        "style" - Only used by Style trainer
        "shared" - Used by both trainers
        "" - Container or unknown
    """
    # Get the top-level module name (first part before any dots)
    top_level = module_name.split(".")[0]

    kc_modules = _get_kc_trainer_modules()
    style_modules = _get_style_trainer_modules()

    in_kc = top_level in kc_modules
    in_style = top_level in style_modules

    if in_kc and in_style:
        return "shared"
    if in_kc:
        return "kc"
    if in_style:
        return "style"
    return ""


def _get_param_count(module: nn.Module) -> int:
    """Count parameters in a module (excluding children to avoid double-counting)."""
    count = 0
    for param in module.parameters(recurse=False):
        count += param.numel()
    for buf in module.buffers(recurse=False):
        count += buf.numel()
    return count


def _get_size_bytes(module: nn.Module) -> int:
    """Get actual size in bytes for a module's DIRECT parameters and buffers.

    Uses numel() * element_size() for accurate size calculation matching
    the pattern in pytorch_utils.py.
    """
    size = 0
    for param in module.parameters(recurse=False):
        size += param.numel() * param.element_size()
    for buf in module.buffers(recurse=False):
        size += buf.numel() * buf.element_size()
    return size


def _get_module_type_name(module: nn.Module) -> str:
    """Get a short, readable type name for a module."""
    type_name = type(module).__name__
    # Shorten common PyTorch names
    replacements = {
        "TransformerEncoder": "TrfEncoder",
        "TransformerEncoderLayer": "TrfEncLayer",
        "SurfaceEmbedding": "SurfEmbed",
        "PositionalEncoding": "PosEnc",
        "LayerNorm": "LN",
        "Embedding": "Emb",
        "ModuleDict": "Dict",
    }
    return replacements.get(type_name, type_name)


def _get_layer_dims(module: nn.Module) -> Tuple[int, int]:
    """Extract input/output dimensions from a module."""
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if isinstance(module, nn.Embedding):
        return module.num_embeddings, module.embedding_dim
    if isinstance(module, nn.LayerNorm):
        size = module.normalized_shape[0] if module.normalized_shape else -1
        return size, size
    # For container modules, return -1
    return -1, -1


def generate_architecture_report(  # pylint: disable=too-many-locals
    model: InferenceClassifier,
    model_name: str = "InferenceClassifier",
) -> ArchitectureReport:
    """Generate an architecture report by introspecting a model.

    Walks the entire module tree using named_modules() to show ALL layers.

    Args:
        model: The constructed PyTorch model to introspect
        model_name: Display name for the model

    Returns:
        ArchitectureReport with layer information and connections
    """
    layers: List[LayerInfo] = []
    total_params = 0
    total_size = 0

    # Walk all modules in the model
    for name, module in model.named_modules():
        # Skip the root module (empty name)
        if not name:
            continue

        # Calculate depth from dots in name
        depth = name.count(".")

        param_count = _get_param_count(module)
        size_bytes = _get_size_bytes(module)
        in_dim, out_dim = _get_layer_dims(module)

        # Check if this is a container (has children but no direct params)
        has_children = len(list(module.children())) > 0
        is_container = param_count == 0 and has_children

        # Determine trainer role (for all modules, including containers)
        trainer_role = _determine_trainer_role(name)

        # Get module type name for detection
        module_type = _get_module_type_name(module)

        layers.append(
            LayerInfo(
                name=name,
                module_type=module_type,
                input_dim=in_dim,
                output_dim=out_dim,
                param_count=param_count,
                size_bytes=size_bytes,
                depth=depth,
                is_container=is_container,
                trainer_role=trainer_role,
            )
        )

        total_params += param_count
        total_size += size_bytes

    # Sort by training execution order (data-driven from trainer roles):
    # 1. Shared backbone first (used by both trainers)
    # 2. KC modules second (KCTrainer runs before StyleTrainer)
    # 3. Style modules third (StyleTrainer runs after KCTrainer)
    # Within each group, preserve hierarchical order (top-level before children)
    role_order = {"shared": 0, "kc": 1, "style": 2, "": 3}

    def sort_key(layer: LayerInfo) -> tuple[int, str]:
        """Sort by role priority, then by name to keep hierarchy intact."""
        return (role_order.get(layer.trainer_role, 3), layer.name)

    layers.sort(key=sort_key)

    return ArchitectureReport(
        model_name=model_name,
        layers=layers,
        total_params=total_params,
        total_size_bytes=total_size,
    )


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def format_count(count: int) -> str:
    """Format parameter count as human-readable."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)
