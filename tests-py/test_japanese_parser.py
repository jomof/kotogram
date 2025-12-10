"""Tests for Japanese parser implementations."""

import unittest
from kotogram import JapaneseParser, kotogram_to_japanese, split_kotogram


class TestJapaneseParserInterface(unittest.TestCase):
    """Test cases for the JapaneseParser abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """JapaneseParser is abstract and cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            JapaneseParser()

    def test_subclass_must_implement_japanese_to_kotogram(self):
        """Subclasses must implement japanese_to_kotogram method."""

        class IncompleteParser(JapaneseParser):
            pass

        with self.assertRaises(TypeError):
            IncompleteParser()


class TestKotogramToJapanese(unittest.TestCase):
    """Test cases for kotogram_to_japanese conversion."""

    def test_simple_kotogram_to_japanese(self):
        """Convert simple kotogram back to Japanese."""
        kotogram = "⌈ˢ猫ᵖn:common_noun⌉"
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, "猫")

    def test_multiple_tokens_without_spaces(self):
        """Convert multiple tokens without spaces."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt⌉⌈ˢ食べるᵖv⌉"
        result = kotogram_to_japanese(kotogram, spaces=False)
        self.assertEqual(result, "猫を食べる")

    def test_multiple_tokens_with_spaces(self):
        """Convert multiple tokens with spaces."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt⌉⌈ˢ食べるᵖv⌉"
        result = kotogram_to_japanese(kotogram, spaces=True)
        self.assertEqual(result, "猫 を 食べる")

    def test_punctuation_collapse(self):
        """Punctuation should not have spaces around it when collapse_punctuation=True."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢ。ᵖauxs⌉"
        result = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        self.assertEqual(result, "猫。")

    def test_punctuation_no_collapse(self):
        """Punctuation can have spaces when collapse_punctuation=False."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢ。ᵖauxs⌉"
        result = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=False)
        self.assertEqual(result, "猫 。")


class TestSplitKotogram(unittest.TestCase):
    """Test cases for split_kotogram function."""

    def test_split_single_token(self):
        """Split kotogram with single token."""
        kotogram = "⌈ˢ猫ᵖn:common_noun⌉"
        result = split_kotogram(kotogram)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "⌈ˢ猫ᵖn:common_noun⌉")

    def test_split_multiple_tokens(self):
        """Split kotogram with multiple tokens."""
        kotogram = "⌈ˢ猫ᵖn⌉⌈ˢをᵖprt⌉⌈ˢ食べるᵖv⌉"
        result = split_kotogram(kotogram)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "⌈ˢ猫ᵖn⌉")
        self.assertEqual(result[1], "⌈ˢをᵖprt⌉")
        self.assertEqual(result[2], "⌈ˢ食べるᵖv⌉")

    def test_split_empty_kotogram(self):
        """Split empty kotogram returns empty list."""
        result = split_kotogram("")
        self.assertEqual(result, [])

    def test_split_complex_tokens(self):
        """Split kotogram with complex token annotations."""
        kotogram = "⌈ˢ食べるᵖv:general:e-ichidan-ba:terminalᵇ食べるᵈ食べるʳタベル⌉"
        result = split_kotogram(kotogram)
        self.assertEqual(len(result), 1)
        self.assertIn("ᵇ", result[0])  # base form marker
        self.assertIn("ᵈ", result[0])  # lemma marker
        self.assertIn("ʳ", result[0])  # pronunciation marker


if __name__ == "__main__":
    unittest.main()
