"""Unit tests for architecture report generation."""

import torch
from torch import nn

from train.architecture_report import (
    ArchitectureReport,
    LayerInfo,
    format_count,
    format_size,
    generate_architecture_report,
)


class TestFormatHelpers:
    """Tests for formatting helper functions."""

    def test_format_count_millions(self) -> None:
        assert format_count(1_500_000) == "1.5M"

    def test_format_count_thousands(self) -> None:
        assert format_count(1_500) == "1.5K"

    def test_format_count_small(self) -> None:
        assert format_count(100) == "100"

    def test_format_size_megabytes(self) -> None:
        assert format_size(1_572_864) == "1.5 MB"

    def test_format_size_kilobytes(self) -> None:
        assert format_size(1_536) == "1.5 KB"

    def test_format_size_bytes(self) -> None:
        assert format_size(100) == "100 B"


class TestLayerInfo:
    """Tests for LayerInfo dataclass."""

    def test_layer_info_fields(self) -> None:
        layer = LayerInfo(
            name="test_layer",
            module_type="Linear",
            input_dim=256,
            output_dim=512,
            param_count=131584,
            size_bytes=131584,
            depth=0,
        )
        assert layer.name == "test_layer"
        assert layer.module_type == "Linear"
        assert layer.input_dim == 256
        assert layer.output_dim == 512
        assert layer.param_count == 131584

    def test_layer_info_default_is_container(self) -> None:
        layer = LayerInfo(
            name="terminal",
            module_type="Linear",
            input_dim=512,
            output_dim=1,
            param_count=513,
            size_bytes=513,
            depth=1,
        )
        assert layer.is_container is False


class TestArchitectureReport:
    """Tests for ArchitectureReport dataclass."""

    def test_report_structure(self) -> None:
        layers = [
            LayerInfo("embedding", "MFEmbed", -1, 256, 1000, 1000, 0),
            LayerInfo("encoder", "TrfEncoder", 256, 256, 2000, 2000, 0),
        ]
        report = ArchitectureReport(
            model_name="TestModel",
            layers=layers,
            total_params=3000,
            total_size_bytes=3000,
        )
        assert report.model_name == "TestModel"
        assert len(report.layers) == 2
        assert report.total_params == 3000

    def test_layer_access_by_iteration(self) -> None:
        """Verify layers can be accessed via iteration."""
        layers = [
            LayerInfo("layer1", "Linear", 10, 20, 100, 100, 0),
            LayerInfo("layer2", "Linear", 20, 30, 200, 200, 0),
        ]
        report = ArchitectureReport(
            model_name="TestModel",
            layers=layers,
            total_params=300,
            total_size_bytes=300,
        )
        # Find layer by name via iteration (the proper way)
        found = next((layer for layer in report.layers if layer.name == "layer2"), None)
        assert found is not None
        assert found.param_count == 200


class TestGenerateArchitectureReport:
    """Tests for generate_architecture_report function."""

    def test_simple_model(self) -> None:
        """Test report generation on a simple model."""

        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(1000, 64)
                self.encoder = nn.Linear(64, 32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.encoder(self.embedding(x))

        model = SimpleModel()
        report = generate_architecture_report(model, "SimpleModel")

        assert report.model_name == "SimpleModel"
        # Should have some layers
        assert len(report.layers) >= 1
        # Total params should be positive
        assert report.total_params > 0

    def test_report_param_counts_sum(self) -> None:
        """Verify that individual layer param counts relate to total."""

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(100, 16)  # 1600 params

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.embedding(x)

        model = TinyModel()
        report = generate_architecture_report(model, "TinyModel")

        # The embedding should be in the report
        layer_param_sum = sum(layer.param_count for layer in report.layers)
        assert layer_param_sum == report.total_params

    def test_is_container_flag(self) -> None:
        """Verify is_container is set correctly."""

        class ContainerModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sequential = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.ReLU(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.sequential(x)

        model = ContainerModel()
        report = generate_architecture_report(model)

        # Sequential should be marked as container
        sequential_layer = next(
            layer for layer in report.layers if layer.name == "sequential"
        )
        assert sequential_layer.is_container is True
        # Linear inside sequential should not be container
        linear_layer = next(layer for layer in report.layers if "0" in layer.name)
        assert linear_layer.is_container is False
