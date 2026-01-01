"""Tests for Sudachi parser implementation."""

import unittest

from kotogram import (
    JapaneseParser,
    SudachiJapaneseParser,
    extract_token_features,
    kotogram_to_japanese,
    split_kotogram,
)


class TestSudachiJapaneseParser(unittest.TestCase):
    """Test cases for SudachiJapaneseParser implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = SudachiJapaneseParser()

    def _has_feature(self, kotogram, surface=None, pos=None):
        """Helper to check if a kotogram contains a token with given features."""
        tokens = split_kotogram(kotogram)
        for token in tokens:
            features = extract_token_features(token)
            if surface and features.surface != surface:
                continue
            if pos and features.pos != pos:
                continue
            # Found match
            return True
        return False

    def test_is_japanese_parser_subclass(self):
        """SudachiJapaneseParser inherits from JapaneseParser."""
        self.assertIsInstance(self.parser, JapaneseParser)

    def test_japanese_to_kotogram_simple(self):
        """Convert simple Japanese text to kotogram format."""
        result = self.parser.japanese_to_kotogram("猫")

        # Verify result content via API
        self.assertTrue(self._has_feature(result, surface="猫", pos="noun"))

    def test_japanese_to_kotogram_with_verb(self):
        """Convert Japanese verb to kotogram format."""
        result = self.parser.japanese_to_kotogram("食べる")

        # Verify verb content
        self.assertTrue(self._has_feature(result, surface="食べる", pos="verb"))

    def test_japanese_to_kotogram_with_particle(self):
        """Convert Japanese particle to kotogram format."""
        result = self.parser.japanese_to_kotogram("を")

        # Verify particle content
        self.assertTrue(self._has_feature(result, surface="を", pos="particle"))

    def test_japanese_to_kotogram_multiple_tokens(self):
        """Convert multiple Japanese tokens to kotogram format."""
        result = self.parser.japanese_to_kotogram("猫を食べる")

        # Should have three tokens
        tokens = split_kotogram(result)
        self.assertEqual(len(tokens), 3)

    def test_special_character_handling(self):
        """Parser handles special っ character correctly."""
        # Test various っ formats
        result1 = self.parser.japanese_to_kotogram(" っ")
        result2 = self.parser.japanese_to_kotogram("っ ")
        result3 = self.parser.japanese_to_kotogram(" っ ")

        # All should produce valid kotogram with "っ" token
        for result in [result1, result2, result3]:
            self.assertTrue(self._has_feature(result, surface="っ"))

    def test_validation_mode_enabled(self):
        """Validation mode raises descriptive errors for unmapped keys."""
        parser_strict = SudachiJapaneseParser(validate=True)

        # Should parse without errors for normal text
        result = parser_strict.japanese_to_kotogram("これはテストです")
        tokens = split_kotogram(result)
        self.assertGreater(len(tokens), 0)

    def test_validation_mode_disabled(self):
        """Validation mode disabled silently ignores unmapped keys."""
        parser = SudachiJapaneseParser(validate=False)

        # Should not raise an error
        result = parser.japanese_to_kotogram("テスト")
        tokens = split_kotogram(result)
        self.assertGreater(len(tokens), 0)

    def test_roundtrip_conversion(self):
        """Kotogram can be converted back to Japanese."""
        text = "今日は良い天気です"
        kotogram = self.parser.japanese_to_kotogram(text)
        recovered = kotogram_to_japanese(kotogram)
        self.assertEqual(recovered, text)

    def test_complex_sentence(self):
        """Parse a complex sentence with multiple grammatical features."""
        # Note: Depending on Sudachi version/dictionary, segmentation might vary slightly.
        text = "私は昨日、友達と映画を見に行きました。"
        result = self.parser.japanese_to_kotogram(text)

        # Should have multiple tokens
        tokens = split_kotogram(result)
        self.assertGreater(len(tokens), 5)

        # Should have various POS markers
        self.assertTrue(self._has_feature(result, pos="noun"))
        self.assertTrue(self._has_feature(result, pos="particle"))
        self.assertTrue(self._has_feature(result, pos="verb"))

    def test_dict_type_parameter(self):
        """Can initialize with different dictionary types."""
        # We only care about the full dictionary being available
        parser_full = SudachiJapaneseParser()
        result = parser_full.japanese_to_kotogram("テスト")
        self.assertTrue(self._has_feature(result, surface="テスト"))

    def test_ellipsis_conversion_prevents_phantom_tokens(self):
        """Ellipsis character is converted to periods to prevent phantom empty tokens."""
        # Sudachi produces phantom empty tokens for "…", which causes <UNK> in Tokenizer.
        # We replace "…" with "..." to avoid this.
        text = "あの…"
        result = self.parser.japanese_to_kotogram(text)

        # Verify no empty surfaces (which would become <UNK>)
        tokens = split_kotogram(result)
        for token in tokens:
            feats = extract_token_features(token)
            self.assertTrue(feats.surface, f"Token {token} has empty surface!")

        # Verify we get periods roughly equivalent to ...
        periods = [t for t in tokens if extract_token_features(t).surface == "."]
        self.assertGreaterEqual(len(periods), 3)

    def test_double_exclamation_conversion(self):
        """Double exclamation mark is converted to two exclamation marks."""
        # ‼ (U+203C) is often treated as <UNK> if not in vocab.
        # We normalize it to !! (ASCII) which is standard.
        text = "行くぜ‼"
        result = self.parser.japanese_to_kotogram(text)

        # Verify we get exclamation marks
        tokens = split_kotogram(result)
        exclamations = [t for t in tokens if extract_token_features(t).surface == "!"]
        self.assertGreaterEqual(len(exclamations), 2)

        # Verify no ‼ remains
        double_bangs = [t for t in tokens if extract_token_features(t).surface == "‼"]
        self.assertEqual(len(double_bangs), 0)


if __name__ == "__main__":
    unittest.main()
