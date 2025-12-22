"""Tests for model-based gender analysis of Japanese sentences."""

import unittest
from kotogram import SudachiJapaneseParser, grammar, GenderLevel
from unittest.mock import patch
import torch
from kotogram.model import StylePrediction


class TestGenderModel(unittest.TestCase):
    """Test gender analysis using the neural model."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = SudachiJapaneseParser(dict_type='full')

        # Mock the model loader for tests to avoid needing a real model file
        from unittest.mock import patch
        from kotogram.model import Tokenizer, ModelConfig, StyleClassifier

        # Create dummy tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer._frozen = True
        
        # Create dummy model
        config = ModelConfig(vocab_sizes=self.tokenizer.get_vocab_sizes())
        self.model = StyleClassifier(config)
        self.model.eval()

        # Patch the internal loader in analysis module
        patcher = patch('kotogram.analysis._load_style_model', return_value=(self.model, self.tokenizer))
        patcher.start()
        self.addCleanup(patcher.stop)


    def test_masculine_basic(self):
        """Test basic masculine sentence."""
        text = "俺が行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        
        # Mock predict output: (formality, gender_val, gender_prag, gram, register)
        # gender_val = -0.9 (masculine), gender_prag = [0.1, 0.9] (pragmatic)
        with patch.object(self.model, 'predict') as mock_predict:
            mock_predict.return_value = StylePrediction(
                formality_value=torch.tensor([-0.9]),    # formality_value
                formality_pragmatic_probs=torch.tensor([[0.1, 0.9]]), # formality_prag (pragmatic)
                gender_value=torch.tensor([-0.9]),    # gender_val
                gender_pragmatic_probs=torch.tensor([[0.1, 0.9]]), # gender_prag (pragmatic)
                grammaticality_probs=torch.tensor([[0.1, 0.9]]), # grammatic
                register_probs=torch.tensor([[0.0]*14])  # register
            )
            
            result = grammar(kotogram)
            self.assertEqual(result.gender, GenderLevel.MASCULINE)
            self.assertAlmostEqual(result.gender_score, -0.9)
            self.assertTrue(result.gender_is_pragmatic)

    def test_feminine_basic(self):
        """Test basic feminine sentence."""
        text = "あたしが行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        
        with patch.object(self.model, 'predict') as mock_predict:
            mock_predict.return_value = StylePrediction(
                formality_value=torch.tensor([0.9]),      # formality_value
                formality_pragmatic_probs=torch.tensor([[0.1, 0.9]]), # formality_prag
                gender_value=torch.tensor([0.9]),     # gender_val (feminine)
                gender_pragmatic_probs=torch.tensor([[0.1, 0.9]]), 
                grammaticality_probs=torch.tensor([[0.1, 0.9]]), 
                register_probs=torch.tensor([[0.0]*14])
            )
             
            result = grammar(kotogram)
            self.assertEqual(result.gender, GenderLevel.FEMININE)
            self.assertAlmostEqual(result.gender_score, 0.9)
            self.assertTrue(result.gender_is_pragmatic)

    def test_neutral_basic(self):
        """Test basic neutral sentence."""
        text = "私は行きます"
        kotogram = self.parser.japanese_to_kotogram(text)
        
        with patch.object(self.model, 'predict') as mock_predict:
            mock_predict.return_value = StylePrediction(
                formality_value=torch.tensor([0.0]),      # formality_value
                formality_pragmatic_probs=torch.tensor([[0.1, 0.9]]), # formality_prag
                gender_value=torch.tensor([0.0]),     # gender_val (neutral)
                gender_pragmatic_probs=torch.tensor([[0.1, 0.9]]), 
                grammaticality_probs=torch.tensor([[0.1, 0.9]]), 
                register_probs=torch.tensor([[0.0]*14])
            )

            result = grammar(kotogram)
            self.assertEqual(result.gender, GenderLevel.NEUTRAL)
            self.assertAlmostEqual(result.gender_score, 0.0)
            self.assertTrue(result.gender_is_pragmatic)

    def test_unpragmatic(self):
        """Test unpragmatic sentence (might return None)."""
        text = "xxxx" 
        kotogram = self.parser.japanese_to_kotogram(text)
        
        with patch.object(self.model, 'predict') as mock_predict:
            mock_predict.return_value = StylePrediction(
                formality_value=torch.tensor([0.0]), 
                formality_pragmatic_probs=torch.tensor([[0.9, 0.1]]), # formality unpragmatic 
                gender_value=torch.tensor([0.0]), 
                gender_pragmatic_probs=torch.tensor([[0.9, 0.1]]), # gender_prag (UNPRAGMATIC)
                grammaticality_probs=torch.tensor([[0.1, 0.9]]), 
                register_probs=torch.tensor([[0.0]*14])
            )

            result = grammar(kotogram)
            self.assertEqual(result.gender, GenderLevel.UNPRAGMATIC_GENDER)
            self.assertFalse(result.gender_is_pragmatic)

if __name__ == "__main__":
    unittest.main()
