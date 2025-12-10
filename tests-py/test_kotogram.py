"""Tests for kotogram utility functions."""

import unittest
from kotogram import (
    SudachiJapaneseParser,
    kotogram_to_japanese,
    split_kotogram,
)


class TestKotogramToJapanese(unittest.TestCase):
    """Test cases for kotogram_to_japanese function."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_basic_conversion(self):
        """Convert basic kotogram to Japanese."""
        text = "猫を食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_with_spaces(self):
        """Add spaces between tokens."""
        text = "猫を食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, spaces=True)
        self.assertIn(' ', result)
        # Removing spaces should give original text
        self.assertEqual(result.replace(' ', ''), text)

    def test_punctuation_collapse(self):
        """Punctuation should not have spaces around it."""
        text = "こんにちは。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        # Should not have space before punctuation
        self.assertNotIn(' 。', result)
        self.assertEqual(result.replace(' ', ''), text)


class TestFurigana(unittest.TestCase):
    """Test cases for furigana feature."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_kanji_gets_furigana(self):
        """Kanji should get hiragana furigana."""
        text = "漢字"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        self.assertIn('[', result)
        self.assertIn(']', result)
        # Should contain hiragana reading
        self.assertIn('かんじ', result)

    def test_hiragana_no_furigana(self):
        """Pure hiragana should not get furigana."""
        text = "ひらがな"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should not have furigana markers
        self.assertNotIn('[', result)
        self.assertEqual(result, text)

    def test_katakana_no_furigana(self):
        """Pure katakana should not get furigana."""
        text = "カタカナ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should not have furigana markers
        self.assertNotIn('[', result)
        self.assertEqual(result, text)

    def test_particles_no_pronunciation_furigana(self):
        """Particles should NOT get pronunciation furigana."""
        text = "猫を見る"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should NOT have [お] for を - particles show IME input
        self.assertNotIn('[お]', result)
        # Should have を as-is (IME input)
        self.assertIn('を', result)

    def test_particle_wa_no_pronunciation(self):
        """Particle は should not get [わ]."""
        text = "私は学生です"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # は is the IME input, not わ
        self.assertNotIn('[わ]', result)

    def test_particle_he_no_pronunciation(self):
        """Particle へ should not get [え]."""
        text = "東京へ行く"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # へ is the IME input, not え
        self.assertNotIn('[え]', result)

    def test_furigana_is_hiragana(self):
        """Furigana should be in hiragana, not katakana."""
        text = "漢字"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)

        # Extract furigana from brackets
        import re
        furigana_parts = re.findall(r'\[(.*?)\]', result)
        self.assertTrue(len(furigana_parts) > 0)

        # Check that furigana is hiragana, not katakana
        for furi in furigana_parts:
            has_katakana = any(
                0x30A1 <= ord(c) <= 0x30F6 for c in furi if c != 'ー'
            )
            self.assertFalse(has_katakana, f"Furigana contains katakana: {furi}")

    def test_small_kana_preserved(self):
        """Small kana like っ should be preserved in furigana."""
        text = "学校"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should preserve small っ (different IME input than large つ)
        if 'がっこ' in result:
            self.assertIn('っ', result)

    def test_furigana_with_spaces(self):
        """Furigana should work with spaces option."""
        text = "猫を食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True, spaces=True)
        # Should have both furigana and spaces
        self.assertIn('[', result)
        # Should be able to handle both features

    def test_default_no_furigana(self):
        """Default behavior should not include furigana."""
        text = "漢字"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        # Default should not have furigana
        self.assertNotIn('[', result)
        self.assertEqual(result, text)


class TestSplitKotogram(unittest.TestCase):
    """Test cases for split_kotogram function."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_split_single_token(self):
        """Split kotogram with single token."""
        text = "猫"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        self.assertEqual(len(tokens), 1)
        self.assertTrue(tokens[0].startswith('⌈'))
        self.assertTrue(tokens[0].endswith('⌉'))

    def test_split_multiple_tokens(self):
        """Split kotogram with multiple tokens."""
        text = "猫を食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        # Should have multiple tokens
        self.assertGreater(len(tokens), 1)
        # Each token should be properly formatted
        for token in tokens:
            self.assertTrue(token.startswith('⌈'))
            self.assertTrue(token.endswith('⌉'))

    def test_split_empty_kotogram(self):
        """Split empty kotogram returns empty list."""
        tokens = split_kotogram("")
        self.assertEqual(len(tokens), 0)

    def test_split_preserves_annotations(self):
        """Split tokens should preserve all annotations."""
        text = "猫"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        # Token should contain surface marker
        self.assertIn('ˢ', tokens[0])
        # Token should contain POS marker
        self.assertIn('ᵖ', tokens[0])

    def test_roundtrip_with_split(self):
        """Splitting and rejoining should preserve kotogram."""
        text = "猫を食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        rejoined = ''.join(tokens)
        self.assertEqual(rejoined, kotogram)


if __name__ == "__main__":
    unittest.main()
