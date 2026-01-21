"""Tests for automatic mixed precision (AMP) training support."""

import pytest
import torch

from train.amp_utils import ConditionalGradScaler, device_autocast


class TestDeviceAutocast:
    """Tests for device_autocast context manager."""

    def test_autocast_disabled(self):
        """Test that autocast is a no-op when disabled."""
        device = torch.device("cpu")
        x = torch.tensor([1.0, 2.0, 3.0])

        with device_autocast(device, enabled=False):
            y = x * 2.0

        assert y.dtype == torch.float32

    def test_autocast_cpu_noop(self):
        """Test that autocast is a no-op on CPU even when enabled."""
        device = torch.device("cpu")
        x = torch.tensor([1.0, 2.0, 3.0])

        with device_autocast(device, enabled=True):
            y = x * 2.0

        # CPU doesn't support fp16, should remain fp32
        assert y.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autocast_cuda(self):
        """Test that autocast works on CUDA."""
        device = torch.device("cuda")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)

        with device_autocast(device, enabled=True):
            # Operations inside autocast may use fp16
            y = x * 2.0
            # Matmul is one operation that typically uses fp16 in autocast
            z = torch.matmul(x.unsqueeze(0), x.unsqueeze(1))

        # Results are converted back to fp32 when exiting autocast
        assert y.device.type == "cuda"
        assert z.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_autocast_mps(self):
        """Test that autocast works on MPS."""
        device = torch.device("mps")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)

        with device_autocast(device, enabled=True):
            y = x * 2.0

        assert y.device.type == "mps"


class TestConditionalGradScaler:
    """Tests for ConditionalGradScaler."""

    def test_scaler_disabled_on_cpu(self):
        """Test that scaler is disabled on CPU."""
        device = torch.device("cpu")
        scaler = ConditionalGradScaler(device, enabled=True)

        assert not scaler.enabled
        assert scaler.scaler is None
        assert scaler.get_scale() == 1.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_scaler_enabled_on_cuda(self):
        """Test that scaler is enabled on CUDA."""
        device = torch.device("cuda")
        scaler = ConditionalGradScaler(device, enabled=True)

        assert scaler.enabled
        assert scaler.scaler is not None
        assert scaler.get_scale() > 0

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_scaler_disabled_on_mps(self):
        """Test that scaler is disabled on MPS (not supported)."""
        device = torch.device("mps")
        scaler = ConditionalGradScaler(device, enabled=True)

        # MPS doesn't support GradScaler
        assert not scaler.enabled
        assert scaler.scaler is None

    def test_scaler_scale_passthrough(self):
        """Test that scale() is a pass-through when disabled."""
        device = torch.device("cpu")
        scaler = ConditionalGradScaler(device, enabled=True)

        loss = torch.tensor(1.5)
        scaled_loss = scaler.scale(loss)

        # Should return unmodified loss when disabled
        assert scaled_loss is loss

    def test_scaler_step_calls_optimizer(self):
        """Test that step() calls optimizer when disabled."""
        device = torch.device("cpu")
        scaler = ConditionalGradScaler(device, enabled=True)

        # Create a simple model and optimizer
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Set some gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        # Step should work even without scaler
        scaler.step(optimizer)
        scaler.update()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_scaler_integration_cuda(self):
        """Integration test: full training step with CUDA scaler."""
        device = torch.device("cuda")
        scaler = ConditionalGradScaler(device, enabled=True)

        # Simple model
        model = torch.nn.Linear(10, 1).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Forward pass with autocast
        x = torch.randn(4, 10, device=device)
        target = torch.randn(4, 1, device=device)

        with device_autocast(device, enabled=True):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)

        # Backward with scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Step and update
        scaler.step(optimizer)
        scaler.update()

        # Verify gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestAMPNumericalStability:
    """Tests for numerical stability with mixed precision."""

    def test_loss_computation_stable(self):
        """Test that loss computation is stable in fp16."""
        device = torch.device("cpu")  # Use CPU for consistent testing

        # Simulate a scenario with small and large loss components
        structural_loss = torch.tensor(0.5)
        diversity_loss = torch.tensor(0.001)  # Small component
        sparsity_loss = torch.tensor(0.1)

        with device_autocast(device, enabled=True):
            combined_loss = structural_loss + diversity_loss + sparsity_loss

        # Should be able to represent this accurately
        expected = 0.5 + 0.001 + 0.1
        assert abs(combined_loss.item() - expected) < 1e-6

    def test_gradient_accumulation(self):
        """Test that gradient accumulation works with mixed precision."""
        device = torch.device("cpu")

        model = torch.nn.Linear(5, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = ConditionalGradScaler(device, enabled=True)

        # Accumulate gradients over 2 steps
        grad_accum_steps = 2
        x = torch.randn(2, 5)
        target = torch.randn(2, 1)

        for _ in range(grad_accum_steps):
            with device_autocast(device, enabled=True):
                output = model(x)
                loss = torch.nn.functional.mse_loss(output, target)
                # Scale by accumulation steps
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

        # Check gradients accumulated
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

        scaler.step(optimizer)
        scaler.update()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
