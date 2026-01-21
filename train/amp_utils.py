"""Automatic Mixed Precision (AMP) utilities for training.

Provides device-aware autocast context managers and gradient scalers that handle
differences between CUDA, MPS, and CPU devices.
"""

from contextlib import contextmanager
from typing import Generator, Optional

import torch


@contextmanager
def device_autocast(
    device: torch.device, enabled: bool = True
) -> Generator[None, None, None]:
    """Device-aware autocast context manager.

    Handles platform-specific differences in mixed precision support:
    - CUDA: Full fp16 support with autocast
    - MPS: fp16 autocast supported (PyTorch 2.5.0+), but uses "cpu" device string
    - CPU: No mixed precision (falls back to fp32)

    Args:
        device: The device being used for training
        enabled: Whether to enable autocast. If False, this is a no-op.

    Yields:
        Context with autocast enabled or disabled based on device and enabled flag.

    Example:
        >>> device = torch.device("cuda")
        >>> with device_autocast(device, enabled=True):
        ...     output = model(input)  # Runs in fp16 where beneficial
    """
    if not enabled:
        yield
        return

    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            yield
    elif device.type == "mps":
        # MPS supports autocast as of PyTorch 2.5.0
        # Note: The device string is "cpu" even though we're using MPS
        # This is a PyTorch implementation detail for MPS autocast
        with torch.amp.autocast("cpu", dtype=torch.float16):
            yield
    else:
        # CPU: no mixed precision benefit, run in fp32
        yield


class ConditionalGradScaler:
    """Gradient scaler that only enables on CUDA.

    GradScaler is not supported on MPS (causes float64 errors), so this wrapper
    only enables scaling on CUDA devices and becomes a pass-through on other devices.

    Attributes:
        enabled: Whether gradient scaling is active
        scaler: The underlying GradScaler (CUDA only) or None
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        """Initialize conditional gradient scaler.

        Args:
            device: The device being used for training
            enabled: Whether to enable gradient scaling (only effective on CUDA)
        """
        self.enabled = enabled and device.type == "cuda"
        self.scaler: Optional[torch.amp.GradScaler] = None
        if self.enabled:
            self.scaler = torch.amp.GradScaler("cuda")

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss for backward pass.

        Args:
            loss: The loss tensor to scale

        Returns:
            Scaled loss if scaler is active, otherwise unmodified loss
        """
        if self.scaler:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with optional unscaling.

        Args:
            optimizer: The optimizer to step
        """
        if self.scaler:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update the scale factor (CUDA only).

        For non-CUDA devices, this is a no-op.
        """
        if self.scaler:
            self.scaler.update()

    def get_scale(self) -> float:
        """Get the current scale factor.

        Returns:
            Current scale factor if scaler is active, otherwise 1.0
        """
        if self.scaler:
            return self.scaler.get_scale()
        return 1.0
