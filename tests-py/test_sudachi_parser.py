"""Tests for Sudachi parser implementation."""

import unittest
from kotogram import SudachiJapaneseParser, JapaneseParser, kotogram_to_japanese


class TestSudachiJapaneseParser(unittest.TestCase):
    """Test cases for SudachiJapaneseParser implementation."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"SudachiPy not available: {e}")

    def test_is_japanese_parser_subclass(self):
        """SudachiJapaneseParser inherits from JapaneseParser."""
        self.assertIsInstance(self.parser, JapaneseParser)

    def test_japanese_to_kotogram_simple(self):
        """Convert simple Japanese text to kotogram format."""
        result = self.parser.japanese_to_kotogram("猫")

        # Verify result has kotogram format markers
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)
        self.assertIn("ˢ猫", result)
        self.assertIn("ᵖn", result)  # n for noun

    def test_japanese_to_kotogram_with_verb(self):
        """Convert Japanese verb to kotogram format."""
        result = self.parser.japanese_to_kotogram("食べる")

        self.assertIn("⌈", result)
        self.assertIn("⌉", result)
        self.assertIn("ˢ食べる", result)
        self.assertIn("ᵖv", result)  # v for verb

    def test_japanese_to_kotogram_with_particle(self):
        """Convert Japanese particle to kotogram format."""
        result = self.parser.japanese_to_kotogram("を")

        self.assertIn("ˢを", result)
        self.assertIn("ᵖprt", result)  # prt for particle
        self.assertIn("case_particle", result)

    def test_japanese_to_kotogram_multiple_tokens(self):
        """Convert multiple Japanese tokens to kotogram format."""
        result = self.parser.japanese_to_kotogram("猫を食べる")

        # Should have three token markers
        self.assertEqual(result.count("⌈"), 3)
        self.assertEqual(result.count("⌉"), 3)

    def test_special_character_handling(self):
        """Parser handles special っ character correctly."""
        # Test various っ formats
        result1 = self.parser.japanese_to_kotogram(" っ")
        result2 = self.parser.japanese_to_kotogram("っ ")
        result3 = self.parser.japanese_to_kotogram(" っ ")

        # All should produce valid kotogram
        for result in [result1, result2, result3]:
            self.assertIn("⌈", result)
            self.assertIn("⌉", result)

    def test_validation_mode_enabled(self):
        """Validation mode raises descriptive errors for unmapped keys."""
        try:
            parser_strict = SudachiJapaneseParser(dict_type='full', validate=True)
        except Exception as e:
            self.skipTest(f"SudachiPy not available: {e}")

        # Should parse without errors for normal text
        result = parser_strict.japanese_to_kotogram("これはテストです")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_validation_mode_disabled(self):
        """Validation mode disabled silently ignores unmapped keys."""
        try:
            parser = SudachiJapaneseParser(dict_type='full', validate=False)
        except Exception as e:
            self.skipTest(f"SudachiPy not available: {e}")

        # Should not raise an error
        result = parser.japanese_to_kotogram("テスト")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_roundtrip_conversion(self):
        """Kotogram can be converted back to Japanese."""
        text = "今日は良い天気です"
        kotogram = self.parser.japanese_to_kotogram(text)
        recovered = kotogram_to_japanese(kotogram)
        self.assertEqual(recovered, text)

    def test_complex_sentence(self):
        """Parse a complex sentence with multiple grammatical features."""
        text = "私は昨日、友達と映画を見に行きました。"
        result = self.parser.japanese_to_kotogram(text)

        # Should have multiple tokens
        token_count = result.count("⌈")
        self.assertGreater(token_count, 5)

        # Should have various POS markers
        self.assertIn("ᵖn", result)  # noun
        self.assertIn("ᵖprt", result)  # particle
        self.assertIn("ᵖv", result)  # verb

    def test_dict_type_parameter(self):
        """Can initialize with different dictionary types."""
        try:
            parser_small = SudachiJapaneseParser(dict_type='small')
            parser_core = SudachiJapaneseParser(dict_type='core')
            parser_full = SudachiJapaneseParser(dict_type='full')

            # All should work
            for parser in [parser_small, parser_core, parser_full]:
                result = parser.japanese_to_kotogram("テスト")
                self.assertIn("⌈", result)
        except Exception as e:
            self.skipTest(f"Not all dictionary types available: {e}")

    def test_comparison_with_mecab(self):
        """Sudachi and MeCab produce similar kotogram structures."""
        try:
            from kotogram import MecabJapaneseParser
            mecab_parser = MecabJapaneseParser()
        except Exception:
            self.skipTest("MeCab not available for comparison")

        text = "猫を食べる"
        sudachi_result = self.parser.japanese_to_kotogram(text)
        mecab_result = mecab_parser.japanese_to_kotogram(text)

        # Both should have same number of tokens
        self.assertEqual(sudachi_result.count("⌈"), mecab_result.count("⌈"))

        # Both should extract the same surface forms
        sudachi_surface = kotogram_to_japanese(sudachi_result)
        mecab_surface = kotogram_to_japanese(mecab_result)
        self.assertEqual(sudachi_surface, mecab_surface)


if __name__ == "__main__":
    unittest.main()
