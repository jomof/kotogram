"""Tests for kotogram utility functions."""

import unittest
from kotogram import (
    MecabJapaneseParser,
    SudachiJapaneseParser,
    kotogram_to_japanese,
    split_kotogram,
)


class TestKotogramToJapanese(unittest.TestCase):
    """Test cases for kotogram_to_japanese function."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.skipTest(f"MeCab not available: {e}")

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_basic_conversion_mecab(self):
        """Convert basic kotogram to Japanese with MeCab."""
        text = "猫を食べる"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_basic_conversion_sudachi(self):
        """Convert basic kotogram to Japanese with Sudachi."""
        text = "猫を食べる"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_with_spaces_mecab(self):
        """Add spaces between tokens with MeCab."""
        text = "猫を食べる"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, spaces=True)
        self.assertIn(' ', result)
        # Removing spaces should give original text
        self.assertEqual(result.replace(' ', ''), text)

    def test_with_spaces_sudachi(self):
        """Add spaces between tokens with Sudachi."""
        text = "猫を食べる"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, spaces=True)
        self.assertIn(' ', result)
        # Removing spaces should give original text
        self.assertEqual(result.replace(' ', ''), text)

    def test_punctuation_collapse_mecab(self):
        """Punctuation should not have spaces around it with MeCab."""
        text = "こんにちは。"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        # Should not have space before punctuation
        self.assertNotIn(' 。', result)
        self.assertEqual(result.replace(' ', ''), text)

    def test_punctuation_collapse_sudachi(self):
        """Punctuation should not have spaces around it with Sudachi."""
        text = "こんにちは。"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        # Should not have space before punctuation
        self.assertNotIn(' 。', result)
        self.assertEqual(result.replace(' ', ''), text)


class TestFurigana(unittest.TestCase):
    """Test cases for furigana feature."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.skipTest(f"MeCab not available: {e}")

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_kanji_gets_furigana_mecab(self):
        """Kanji should get hiragana furigana with MeCab."""
        text = "漢字"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        self.assertIn('[', result)
        self.assertIn(']', result)
        # Should contain hiragana reading
        self.assertIn('かんじ', result)

    def test_kanji_gets_furigana_sudachi(self):
        """Kanji should get hiragana furigana with Sudachi."""
        text = "漢字"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        self.assertIn('[', result)
        self.assertIn(']', result)
        # Should contain hiragana reading
        self.assertIn('かんじ', result)

    def test_hiragana_no_furigana_mecab(self):
        """Pure hiragana should not get furigana with MeCab."""
        text = "ひらがな"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should not have furigana markers
        self.assertNotIn('[', result)
        self.assertEqual(result, text)

    def test_hiragana_no_furigana_sudachi(self):
        """Pure hiragana should not get furigana with Sudachi."""
        text = "ひらがな"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should not have furigana markers
        self.assertNotIn('[', result)
        self.assertEqual(result, text)

    def test_katakana_no_furigana_mecab(self):
        """Pure katakana should not get furigana with MeCab."""
        text = "カタカナ"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should not have furigana markers
        self.assertNotIn('[', result)
        self.assertEqual(result, text)

    def test_katakana_no_furigana_sudachi(self):
        """Pure katakana should not get furigana with Sudachi."""
        text = "カタカナ"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should not have furigana markers
        self.assertNotIn('[', result)
        self.assertEqual(result, text)

    def test_particles_no_pronunciation_furigana_mecab(self):
        """Particles should NOT get pronunciation furigana with MeCab."""
        text = "猫を見る"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should NOT have [お] for を - particles show IME input
        self.assertNotIn('[お]', result)
        # Should have を as-is (IME input)
        self.assertIn('を', result)

    def test_particles_no_pronunciation_furigana_sudachi(self):
        """Particles should NOT get pronunciation furigana with Sudachi."""
        text = "猫を見る"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should NOT have [お] for を - particles show IME input
        self.assertNotIn('[お]', result)
        # Should have を as-is (IME input)
        self.assertIn('を', result)

    def test_particle_wa_no_pronunciation_mecab(self):
        """Particle は should not get [わ] with MeCab."""
        text = "私は学生です"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # は is the IME input, not わ
        self.assertNotIn('[わ]', result)

    def test_particle_wa_no_pronunciation_sudachi(self):
        """Particle は should not get [わ] with Sudachi."""
        text = "私は学生です"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # は is the IME input, not わ
        self.assertNotIn('[わ]', result)

    def test_particle_he_no_pronunciation_mecab(self):
        """Particle へ should not get [え] with MeCab."""
        text = "東京へ行く"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # へ is the IME input, not え
        self.assertNotIn('[え]', result)

    def test_particle_he_no_pronunciation_sudachi(self):
        """Particle へ should not get [え] with Sudachi."""
        text = "東京へ行く"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # へ is the IME input, not え
        self.assertNotIn('[え]', result)

    def test_furigana_is_hiragana_mecab(self):
        """Furigana should be in hiragana, not katakana with MeCab."""
        text = "漢字"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
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

    def test_furigana_is_hiragana_sudachi(self):
        """Furigana should be in hiragana, not katakana with Sudachi."""
        text = "漢字"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
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

    def test_small_kana_preserved_mecab(self):
        """Small kana like っ should be preserved in furigana with MeCab."""
        text = "学校"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should preserve small っ (different IME input than large つ)
        if 'がっこ' in result:
            self.assertIn('っ', result)

    def test_small_kana_preserved_sudachi(self):
        """Small kana like っ should be preserved in furigana with Sudachi."""
        text = "学校"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should preserve small っ (different IME input than large つ)
        if 'がっこ' in result:
            self.assertIn('っ', result)

    def test_furigana_with_spaces_mecab(self):
        """Furigana should work with spaces option with MeCab."""
        text = "猫を食べる"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True, spaces=True)
        # Should have both furigana and spaces
        self.assertIn('[', result)
        # Should be able to handle both features

    def test_furigana_with_spaces_sudachi(self):
        """Furigana should work with spaces option with Sudachi."""
        text = "猫を食べる"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True, spaces=True)
        # Should have both furigana and spaces
        self.assertIn('[', result)
        # Should be able to handle both features

    def test_default_no_furigana_mecab(self):
        """Default behavior should not include furigana with MeCab."""
        text = "漢字"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        # Default should not have furigana
        self.assertNotIn('[', result)
        self.assertEqual(result, text)

    def test_default_no_furigana_sudachi(self):
        """Default behavior should not include furigana with Sudachi."""
        text = "漢字"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        # Default should not have furigana
        self.assertNotIn('[', result)
        self.assertEqual(result, text)


class TestSplitKotogram(unittest.TestCase):
    """Test cases for split_kotogram function."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.skipTest(f"MeCab not available: {e}")

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_split_single_token_mecab(self):
        """Split kotogram with single token using MeCab."""
        text = "猫"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        self.assertEqual(len(tokens), 1)
        self.assertTrue(tokens[0].startswith('⌈'))
        self.assertTrue(tokens[0].endswith('⌉'))

    def test_split_single_token_sudachi(self):
        """Split kotogram with single token using Sudachi."""
        text = "猫"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        self.assertEqual(len(tokens), 1)
        self.assertTrue(tokens[0].startswith('⌈'))
        self.assertTrue(tokens[0].endswith('⌉'))

    def test_split_multiple_tokens_mecab(self):
        """Split kotogram with multiple tokens using MeCab."""
        text = "猫を食べる"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        # Should have multiple tokens
        self.assertGreater(len(tokens), 1)
        # Each token should be properly formatted
        for token in tokens:
            self.assertTrue(token.startswith('⌈'))
            self.assertTrue(token.endswith('⌉'))

    def test_split_multiple_tokens_sudachi(self):
        """Split kotogram with multiple tokens using Sudachi."""
        text = "猫を食べる"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
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

    def test_split_preserves_annotations_mecab(self):
        """Split tokens should preserve all annotations with MeCab."""
        text = "猫"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        # Token should contain surface marker
        self.assertIn('ˢ', tokens[0])
        # Token should contain POS marker
        self.assertIn('ᵖ', tokens[0])

    def test_split_preserves_annotations_sudachi(self):
        """Split tokens should preserve all annotations with Sudachi."""
        text = "猫"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        # Token should contain surface marker
        self.assertIn('ˢ', tokens[0])
        # Token should contain POS marker
        self.assertIn('ᵖ', tokens[0])

    def test_roundtrip_with_split_mecab(self):
        """Splitting and rejoining should preserve kotogram with MeCab."""
        text = "猫を食べる"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        rejoined = ''.join(tokens)
        self.assertEqual(rejoined, kotogram)

    def test_roundtrip_with_split_sudachi(self):
        """Splitting and rejoining should preserve kotogram with Sudachi."""
        text = "猫を食べる"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        rejoined = ''.join(tokens)
        self.assertEqual(rejoined, kotogram)


class TestCrossParserCompatibility(unittest.TestCase):
    """Test that kotogram utilities work with both parser outputs."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.skipTest(f"MeCab not available: {e}")

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_both_parsers_produce_valid_kotogram(self):
        """Both parsers should produce kotogram that converts back correctly."""
        text = "猫を食べる"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        # Both should convert back to original text
        self.assertEqual(kotogram_to_japanese(mecab_kotogram), text)
        self.assertEqual(kotogram_to_japanese(sudachi_kotogram), text)

    def test_both_parsers_furigana_works(self):
        """Furigana should work with both parser outputs."""
        text = "漢字"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = kotogram_to_japanese(mecab_kotogram, furigana=True)
        sudachi_result = kotogram_to_japanese(sudachi_kotogram, furigana=True)

        # Both should have furigana
        self.assertIn('[', mecab_result)
        self.assertIn('[', sudachi_result)

    def test_both_parsers_split_works(self):
        """Split should work with both parser outputs."""
        text = "猫を食べる"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_tokens = split_kotogram(mecab_kotogram)
        sudachi_tokens = split_kotogram(sudachi_kotogram)

        # Both should produce valid tokens
        self.assertGreater(len(mecab_tokens), 0)
        self.assertGreater(len(sudachi_tokens), 0)

        for token in mecab_tokens:
            self.assertTrue(token.startswith('⌈') and token.endswith('⌉'))
        for token in sudachi_tokens:
            self.assertTrue(token.startswith('⌈') and token.endswith('⌉'))


if __name__ == "__main__":
    unittest.main()
