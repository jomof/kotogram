"""Tests for continuous gender prediction via KC decoder MSE pathway."""

import unittest

import torch

from kotogram.model import ModelConfig


class TestContinuousGender(unittest.TestCase):
    def test_model_architecture(self):
        """Test that the model architecture uses KC decoder for gender values."""
        from kotogram.model import InferenceClassifier as RealInferenceClassifier

        config = ModelConfig(vocab_sizes={"surface": 100})
        model = RealInferenceClassifier(config)

        # Gender values now come from KC decoder MSE pathway, not standalone head
        self.assertFalse(hasattr(model, "gender_value_head"))
        self.assertTrue(hasattr(model, "gender_pragmatic_head"))
        self.assertFalse(hasattr(model, "gender_classifier"))  # Legacy, should be gone
        self.assertTrue(hasattr(model, "kc_decoders"))  # KC decoder handles values

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes (pragmatic heads only)."""
        from kotogram.model import InferenceClassifier as RealInferenceClassifier

        config = ModelConfig(vocab_sizes={"surface": 100})
        model = RealInferenceClassifier(config)

        bs = 2
        from kotogram.tokenizer import FEATURE_FIELDS

        inputs = {}
        for field in FEATURE_FIELDS:
            inputs[f"input_ids_{field}"] = torch.randint(0, 10, (bs, 10))

        mask = torch.ones(bs, 10)

        out = model(inputs, mask)
        # forward() returns 4 outputs: formality_prag, gender_prag, gram, register
        self.assertEqual(len(out), 4)

        formality_prag, gender_prag, _gram, _reg = out

        self.assertEqual(gender_prag.shape, (bs, 2))
        self.assertEqual(formality_prag.shape, (bs, 2))

    def test_predict_method(self):
        """Test predict() returns StylePrediction with gender_value."""
        from kotogram.model import (
            InferenceClassifier as RealInferenceClassifier,
        )
        from kotogram.model import (
            KCDecoderInference,
        )

        config = ModelConfig(vocab_sizes={"surface": 100})
        model = RealInferenceClassifier(config)

        # Initialize KC decoder with gender support
        model.kc_decoders = KCDecoderInference(
            config, num_grammar_points=10, has_formality=True, has_gender=True
        )

        bs = 2
        from kotogram.tokenizer import FEATURE_FIELDS

        inputs = {}
        for field in FEATURE_FIELDS:
            inputs[f"input_ids_{field}"] = torch.randint(0, 10, (bs, 10))

        mask = torch.ones(bs, 10)

        prediction = model.predict(inputs, mask)

        # Check gender_value comes through
        self.assertEqual(prediction.gender_value.shape, (bs, 1))
        self.assertEqual(prediction.gender_pragmatic_probs.shape, (bs, 2))

        # Check Tanh range for values
        self.assertTrue(torch.all(prediction.gender_value >= -1.0))
        self.assertTrue(torch.all(prediction.gender_value <= 1.0))

    def test_analysis_functions(self):
        """Test that analysis.gender() handles the new return type."""
        # Covered by test_gender.py which uses the real model/pipeline


if __name__ == "__main__":
    unittest.main()
