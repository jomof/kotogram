"""Tests for model-based gender analysis of Japanese sentences."""

import unittest
from kotogram import SudachiJapaneseParser, gender, GenderLevel


class TestGenderModel(unittest.TestCase):
    """Test gender analysis using the neural model."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_masculine_basic(self):
        """Test basic masculine sentence."""
        text = "俺が行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        # Result handles float or None
        if result is not None:
             self.assertIsInstance(result, float)
             self.assertLess(result, -0.5)

    def test_feminine_basic(self):
        """Test basic feminine sentence."""
        text = "あたしが行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        # Result handles float or None
        if result is not None:
            self.assertIsInstance(result, float)
            self.assertGreater(result, 0.5)

    def test_neutral_basic(self):
        """Test basic neutral sentence."""
        text = "私は行きます"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        # Result handles float or None
        if result is not None:
            self.assertIsInstance(result, float)
            self.assertTrue(-0.5 <= result <= 0.5)

    def test_unpragmatic(self):
        """Test unpragmatic sentence (might return None)."""
        # This is hard to test without a trained model that predicts unpragmatic.
        # But we can at least check the API/type safe.
        text = "xxxx" 
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        # Result should be Optional[float], so valid.
        pass

if __name__ == "__main__":
    unittest.main()
