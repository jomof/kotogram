"""Tests for gradient flow through gender prediction paths."""

import unittest

import torch
from torch import nn

from kotogram.model import InferenceClassifier, KCDecoderInference, ModelConfig


class TestGenderLearning(unittest.TestCase):
    def test_gradient_flow_pragmatic(self):
        """Verify that gradients flow to gender pragmatic head."""
        config = ModelConfig(
            vocab_sizes={"surface": 100},
            hidden_dim=32,
            num_formality_pragmatic_classes=2,
            num_grammaticality_classes=2,
            num_register_classes=7,
        )
        model = InferenceClassifier(config)

        # Dummy input
        batch_size = 4
        from kotogram.tokenizer import FEATURE_FIELDS

        field_inputs = {}
        for field in FEATURE_FIELDS:
            field_inputs[f"input_ids_{field}"] = torch.randint(0, 100, (batch_size, 10))
        attention_mask = torch.ones((batch_size, 10))

        # Forward pass - now returns 3 outputs (pragmatic heads only)
        # Register is handled by KC decoder
        _formality_prag, gender_prag, _gram = model(field_inputs, attention_mask)

        # Dummy targets
        gender_prag_targets = torch.randint(0, 2, (batch_size,))

        # Compute loss for pragmatic head
        loss_prag = nn.functional.cross_entropy(gender_prag, gender_prag_targets)

        # Backward
        loss_prag.backward()

        # Check gradients on gender pragmatic head
        self.assertIsNotNone(model.gender_pragmatic_head[0].weight.grad)
        self.assertNotEqual(
            model.gender_pragmatic_head[0].weight.grad.abs().sum().item(), 0.0
        )

    def test_gradient_flow_kc_decoder_mse(self):
        """Verify that gradients flow through KC decoder MSE pathway for gender values."""
        config = ModelConfig(
            vocab_sizes={"surface": 100},
            hidden_dim=32,
            kc_vocab_size=64,
        )
        model = InferenceClassifier(config)

        # Initialize KC decoder with gender/formality support
        model.kc_decoders = KCDecoderInference(
            config, num_grammar_points=10, has_formality=True, has_gender=True
        )

        # Dummy input
        batch_size = 4
        from kotogram.tokenizer import FEATURE_FIELDS

        field_inputs = {}
        for field in FEATURE_FIELDS:
            field_inputs[f"input_ids_{field}"] = torch.randint(0, 100, (batch_size, 10))
        attention_mask = torch.ones((batch_size, 10))

        # Get KC probs and predict style values
        model.train()
        kc_logits = model.predict_kcs(field_inputs, attention_mask)
        cur_temp = getattr(model.config, "kc_temperature", 1.0)
        kc_probs = torch.sigmoid(kc_logits / cur_temp)

        _formality_val, gender_val = model.kc_decoders.predict_style_values(kc_probs)

        # Dummy targets
        gender_val_targets = torch.randn(batch_size, 1)

        # Compute loss for MSE pathway
        loss_val = nn.functional.mse_loss(gender_val, gender_val_targets)

        # Backward
        loss_val.backward()

        # Check gradients on KC decoder MSE hidden layers
        self.assertIsNotNone(model.kc_decoders.mse_hidden1.weight.grad)
        self.assertNotEqual(
            model.kc_decoders.mse_hidden1.weight.grad.abs().sum().item(), 0.0
        )

        # Check gradients on gender MSE decoder
        self.assertIsNotNone(model.kc_decoders.mse_decoders["gender"].weight.grad)
        self.assertNotEqual(
            model.kc_decoders.mse_decoders["gender"].weight.grad.abs().sum().item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
