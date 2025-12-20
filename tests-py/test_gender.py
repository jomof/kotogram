"""Tests for model-based gender analysis of Japanese sentences."""

import unittest
from kotogram import SudachiJapaneseParser, gender
from unittest.mock import patch
import torch


class TestGenderModel(unittest.TestCase):
    """Test gender analysis using the neural model."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

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
        self.mock_loader = patcher.start()
        self.addCleanup(patcher.stop)


    def test_masculine_basic(self):
        """Test basic masculine sentence."""
        text = "俺が行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        
        # Mock predict output: (formality, gender_val, gender_prag, gram, register)
        # gender_val = -0.9 (masculine), gender_prag = [0.1, 0.9] (pragmatic)
        with patch.object(self.model, 'predict') as mock_predict:
            mock_predict.return_value = (
                torch.tensor([[0.0]*6]), # formality
                torch.tensor([-0.9]),    # gender_val
                torch.tensor([[0.1, 0.9]]), # gender_prag (pragmatic)
                torch.tensor([[0.1, 0.9]]), # grammatic
                torch.tensor([[0.0]*9])  # register
            )
            
            result = gender(kotogram)
            if result is not None:
                self.assertIsInstance(result, float)
                self.assertLess(result, -0.5)

    def test_feminine_basic(self):
        """Test basic feminine sentence."""
        text = "あたしが行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        
        with patch.object(self.model, 'predict') as mock_predict:
             mock_predict.return_value = (
                torch.tensor([[0.0]*6]), 
                torch.tensor([0.9]),     # gender_val (feminine)
                torch.tensor([[0.1, 0.9]]), 
                torch.tensor([[0.1, 0.9]]), 
                torch.tensor([[0.0]*9])
            )
             
             result = gender(kotogram)
             if result is not None:
                self.assertIsInstance(result, float)
                self.assertGreater(result, 0.5)

    def test_neutral_basic(self):
        """Test basic neutral sentence."""
        text = "私は行きます"
        kotogram = self.parser.japanese_to_kotogram(text)
        
        with patch.object(self.model, 'predict') as mock_predict:
             mock_predict.return_value = (
                torch.tensor([[0.0]*6]), 
                torch.tensor([0.0]),     # gender_val (neutral)
                torch.tensor([[0.1, 0.9]]), 
                torch.tensor([[0.1, 0.9]]), 
                torch.tensor([[0.0]*9])
            )

             result = gender(kotogram)
             if result is not None:
                self.assertIsInstance(result, float)
                self.assertTrue(-0.5 <= result <= 0.5)

    def test_unpragmatic(self):
        """Test unpragmatic sentence (might return None)."""
        text = "xxxx" 
        kotogram = self.parser.japanese_to_kotogram(text)
        
        with patch.object(self.model, 'predict') as mock_predict:
             mock_predict.return_value = (
                torch.tensor([[0.0]*6]), 
                torch.tensor([0.0]), 
                torch.tensor([[0.9, 0.1]]), # gender_prag (UNPRAGMATIC)
                torch.tensor([[0.1, 0.9]]), 
                torch.tensor([[0.0]*9])
            )

             result = gender(kotogram)
             self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
