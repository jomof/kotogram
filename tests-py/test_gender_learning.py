import unittest

import torch
from torch import nn

from kotogram.model import ModelConfig
from train.trainer import StyleClassifier


# pylint: disable=too-many-locals
class TestGenderLearning(unittest.TestCase):
    def test_gradient_flow(self):
        """Verify that gradients flow to gender heads."""
        config = ModelConfig(
            vocab_sizes={"surface": 100},
            hidden_dim=32,
            # num_formality_classes deprecated/removed from init
            num_formality_pragmatic_classes=2,
            num_grammaticality_classes=2,
            num_register_classes=7,
        )
        model = StyleClassifier(config)

        # Dummy input
        batch_size = 4
        # Default model uses all features: surface, pos, pos_detail_1, pos_detail_2, conjugated_type, conjugated_form, lemma
        field_inputs = {}
        from kotogram.tokenizer import FEATURE_FIELDS

        field_inputs = {}
        for field in FEATURE_FIELDS:
            field_inputs[f"input_ids_{field}"] = torch.randint(0, 100, (batch_size, 10))
        attention_mask = torch.ones((batch_size, 10))

        # Forward pass
        _, _, gender_val, gender_prag, _, _ = model(field_inputs, attention_mask)

        # Dummy targets
        gender_val_targets = torch.randn(batch_size)
        gender_prag_targets = torch.randint(0, 2, (batch_size,))

        # Compute loss
        loss_val = nn.functional.mse_loss(gender_val.squeeze(-1), gender_val_targets)
        loss_prag = nn.functional.cross_entropy(gender_prag, gender_prag_targets)
        total_loss = loss_val + loss_prag

        # Backward
        total_loss.backward()

        # Check gradients on the first Linear layer of the Sequential head
        self.assertIsNotNone(model.gender_value_head[0].weight.grad)
        self.assertNotEqual(
            model.gender_value_head[0].weight.grad.abs().sum().item(), 0.0
        )

        self.assertIsNotNone(model.gender_pragmatic_head[0].weight.grad)
        self.assertNotEqual(
            model.gender_pragmatic_head[0].weight.grad.abs().sum().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
