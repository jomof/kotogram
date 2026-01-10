"""Unit tests for learning rate scaling based on sample ratio."""

import dataclasses
import unittest

from train.config import TrainerConfig


class TestLearningRateScaling(unittest.TestCase):
    """Test learning rate scaling logic that will be applied in train_style.py."""

    def test_lr_scaling_at_100_percent(self) -> None:
        """100% sample should use base learning rate (no scaling)."""
        base_config = TrainerConfig(learning_rate=5e-5)
        sample_ratio = 1.0  # 100%

        # At 100%, no scaling should occur
        if sample_ratio < 1.0:
            lr_scale = 1.0 / sample_ratio
            scaled_lr = base_config.learning_rate * lr_scale
            result_config = dataclasses.replace(base_config, learning_rate=scaled_lr)
        else:
            result_config = base_config

        self.assertAlmostEqual(result_config.learning_rate, 5e-5)

    def test_lr_scaling_at_50_percent(self) -> None:
        """50% sample should double the learning rate."""
        base_config = TrainerConfig(learning_rate=5e-5)
        sample_ratio = 0.5  # 50%

        if sample_ratio < 1.0:
            lr_scale = 1.0 / sample_ratio
            scaled_lr = base_config.learning_rate * lr_scale
            result_config = dataclasses.replace(base_config, learning_rate=scaled_lr)
        else:
            result_config = base_config

        # 5e-5 * 2 = 1e-4
        self.assertAlmostEqual(result_config.learning_rate, 1e-4)

    def test_lr_scaling_at_10_percent(self) -> None:
        """10% sample should multiply learning rate by 10."""
        base_config = TrainerConfig(learning_rate=5e-5)
        sample_ratio = 0.1  # 10%

        if sample_ratio < 1.0:
            lr_scale = 1.0 / sample_ratio
            scaled_lr = base_config.learning_rate * lr_scale
            result_config = dataclasses.replace(base_config, learning_rate=scaled_lr)
        else:
            result_config = base_config

        # 5e-5 * 10 = 5e-4
        self.assertAlmostEqual(result_config.learning_rate, 5e-4)

    def test_lr_scaling_at_1_percent(self) -> None:
        """1% sample should multiply learning rate by 100."""
        base_config = TrainerConfig(learning_rate=5e-5)
        sample_ratio = 0.01  # 1%

        if sample_ratio < 1.0:
            lr_scale = 1.0 / sample_ratio
            scaled_lr = base_config.learning_rate * lr_scale
            result_config = dataclasses.replace(base_config, learning_rate=scaled_lr)
        else:
            result_config = base_config

        # 5e-5 * 100 = 5e-3
        self.assertAlmostEqual(result_config.learning_rate, 5e-3)

    def test_lr_scale_preserves_other_config_fields(self) -> None:
        """Scaling LR should not affect other TrainerConfig fields."""
        base_config = TrainerConfig(
            learning_rate=5e-5,
            batch_size=32,
            epochs=10,
            patience=5,
        )
        sample_ratio = 0.5

        if sample_ratio < 1.0:
            lr_scale = 1.0 / sample_ratio
            scaled_lr = base_config.learning_rate * lr_scale
            result_config = dataclasses.replace(base_config, learning_rate=scaled_lr)
        else:
            result_config = base_config

        # LR should be scaled
        self.assertAlmostEqual(result_config.learning_rate, 1e-4)
        # Other fields preserved
        self.assertEqual(result_config.batch_size, 32)
        self.assertEqual(result_config.epochs, 10)
        self.assertEqual(result_config.patience, 5)


if __name__ == "__main__":
    unittest.main()
