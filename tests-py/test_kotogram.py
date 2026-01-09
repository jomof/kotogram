"""Tests for kotogram utility functions."""

import unittest

from kotogram import (
    SudachiJapaneseParser,
    extract_token_features,
    kotogram_to_japanese,
    split_kotogram,
)


class TestKotogramToJapanese(unittest.TestCase):
    """Test cases for kotogram_to_japanese function."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = SudachiJapaneseParser()

    def test_basic_conversion(self):
        """Convert basic kotogram to Japanese."""
        text = "猫を食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_kanji_gets_furigana(self):
        """Kanji should get hiragana furigana."""
        text = "漢字"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        self.assertIn("[", result)
        self.assertIn("]", result)
        # Should contain hiragana reading
        self.assertIn("かんじ", result)

    def test_hiragana_no_furigana(self):
        """Pure hiragana should not get furigana."""
        text = "ひらがな"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should not have furigana markers
        self.assertNotIn("[", result)
        self.assertEqual(result, text)

    def test_katakana_no_furigana(self):
        """Pure katakana should not get furigana."""
        text = "カタカナ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should not have furigana markers
        self.assertNotIn("[", result)
        self.assertEqual(result, text)

    def test_particles_no_pronunciation_furigana(self):
        """Particles should NOT get pronunciation furigana."""
        text = "猫を見る"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should NOT have [お] for を - particles show IME input
        self.assertNotIn("[お]", result)
        # Should have を as-is (IME input)
        self.assertIn("を", result)

    def test_particle_wa_no_pronunciation(self):
        """Particle は should not get [わ]."""
        text = "私は学生です"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # は is the IME input, not わ
        self.assertNotIn("[わ]", result)

    def test_particle_he_no_pronunciation(self):
        """Particle へ should not get [え]."""
        text = "東京へ行く"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # へ is the IME input, not え
        self.assertNotIn("[え]", result)

    def test_furigana_is_hiragana(self):
        """Furigana should be in hiragana, not katakana."""
        text = "漢字"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)

        # Extract furigana from brackets
        import re

        furigana_parts = re.findall(r"\[(.*?)\]", result)
        self.assertTrue(len(furigana_parts) > 0)

        # Check that furigana is hiragana, not katakana
        for furi in furigana_parts:
            has_katakana = any(0x30A1 <= ord(c) <= 0x30F6 for c in furi if c != "ー")
            self.assertFalse(has_katakana, f"Furigana contains katakana: {furi}")

    def test_small_kana_preserved(self):
        """Small kana like っ should be preserved in furigana."""
        text = "学校"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Should preserve small っ (different IME input than large つ)
        if "がっこ" in result:
            self.assertIn("っ", result)

    def test_default_no_furigana(self):
        """Default behavior should not include furigana."""
        text = "漢字"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        # Default should not have furigana
        self.assertNotIn("[", result)
        self.assertEqual(result, text)


class TestSplitKotogram(unittest.TestCase):
    """Test cases for split_kotogram function."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = SudachiJapaneseParser()

    def test_split_single_token(self):
        """Split kotogram with single token."""
        text = "猫"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        self.assertEqual(len(tokens), 1)
        # Verify content via API
        features = extract_token_features(tokens[0])
        self.assertEqual(features.surface, "猫")

    def test_split_multiple_tokens(self):
        """Split kotogram with multiple tokens."""
        text = "猫を食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        # Should have multiple tokens
        self.assertGreater(len(tokens), 1)
        # Each token should be parseable
        for token in tokens:
            features = extract_token_features(token)
            self.assertTrue(features.surface, f"Token failed to parse surface: {token}")

    def test_split_empty_kotogram(self):
        """Split empty kotogram returns empty list."""
        tokens = split_kotogram("")
        self.assertEqual(len(tokens), 0)

    def test_split_preserves_annotations(self):
        """Split tokens should preserve all annotations."""
        text = "猫"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        # Token should contain annotations accessible via API
        features = extract_token_features(tokens[0])
        self.assertTrue(features.surface)
        self.assertTrue(features.pos)

    def test_roundtrip_with_split(self):
        """Splitting and rejoining should preserve kotogram."""
        text = "猫を食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)
        rejoined = "".join(tokens)
        self.assertEqual(rejoined, kotogram)


class TestExtractTokenFeatures(unittest.TestCase):
    """Test cases for extract_token_features function."""

    def test_reading_gram_uses_surface_when_no_reading_for_grammar_pos(self):
        """Grammar POS tokens without reading should use surface for reading_gram."""
        # Craft a particle token without reading - should use surface
        token = "⌈ˢをᵖparticle:case-particle⌉"
        features = extract_token_features(token)
        self.assertEqual(features.surface, "を")
        # Particle is on grammar whitelist, so reading_gram should be surface
        self.assertEqual(features.reading_gram, "を")

    def test_reading_gram_uses_mask_when_no_reading_for_content_word(self):
        """Content words without reading should use READING_MASK for reading_gram."""
        from kotogram.masking import READING_MASK

        # Craft a noun token without reading - should use mask
        token = "⌈ˢテストᵖnoun⌉"
        features = extract_token_features(token)
        self.assertEqual(features.surface, "テスト")
        # Noun is NOT on grammar whitelist, so reading_gram should be masked
        self.assertEqual(features.reading_gram, READING_MASK)

    def test_reading_gram_uses_reading_when_available(self):
        """Tokens with reading should use reading-based logic."""
        from kotogram.masking import READING_MASK

        # Token with reading - content word (noun) gets masked
        token = "⌈ˢ猫ᵖnounʳネコ⌉"
        features = extract_token_features(token)
        self.assertEqual(features.surface, "猫")
        self.assertEqual(features.reading, "ネコ")
        # Noun with reading should be masked
        self.assertEqual(features.reading_gram, READING_MASK)

    def test_verb_reading_gram_preserved(self):
        """Verbs should preserve their reading for reading_gram (grammatically important)."""
        # Verb with reading - should keep reading since verb conjugations carry grammar
        token = "⌈ˢ行きᵖverbʳイキ⌉"
        features = extract_token_features(token)
        self.assertEqual(features.surface, "行き")
        self.assertEqual(features.reading, "イキ")
        # Verb is on grammar whitelist, reading should be preserved
        self.assertEqual(features.reading_gram, "イキ")

    def test_punctuation_uses_surface_for_reading_gram(self):
        """Punctuation (aux-symbol) without reading should use surface for reading_gram."""
        # Period without reading - aux-symbol is on grammar whitelist
        token = "⌈ˢ。ᵖaux-symbol:period⌉"
        features = extract_token_features(token)
        self.assertEqual(features.surface, "。")
        self.assertEqual(features.pos, "aux-symbol")
        # aux-symbol is on grammar whitelist, so reading_gram should be surface
        self.assertEqual(features.reading_gram, "。")

    def test_comma_uses_surface_for_reading_gram(self):
        """Comma (aux-symbol) without reading should use surface for reading_gram."""
        token = "⌈ˢ、ᵖaux-symbol:comma⌉"
        features = extract_token_features(token)
        self.assertEqual(features.surface, "、")
        self.assertEqual(features.reading_gram, "、")

    def test_question_mark_uses_surface_for_reading_gram(self):
        """Question mark (aux-symbol) without reading should use surface for reading_gram."""
        token = "⌈ˢ？ᵖaux-symbol⌉"
        features = extract_token_features(token)
        self.assertEqual(features.surface, "？")
        self.assertEqual(features.reading_gram, "？")

    def test_punctuation_reading_gram_in_real_sentence(self):
        """Punctuation in real parsed sentence should have surface as reading_gram."""
        parser = SudachiJapaneseParser()
        # Parse a sentence ending with period
        kotogram = parser.japanese_to_kotogram("行きました。")
        tokens = split_kotogram(kotogram)

        # Last token should be the period
        last_token = tokens[-1]
        features = extract_token_features(last_token)
        self.assertEqual(features.surface, "。")
        self.assertEqual(features.pos, "aux-symbol")
        # Period should use surface as reading_gram
        self.assertEqual(features.reading_gram, "。")

    def test_reading_gram_never_empty(self):
        """reading_gram must never be empty string - should fall back to READING_MASK."""
        from kotogram.masking import READING_MASK

        # Token with no reading and no valid surface replacement
        # (simulating edge case where both are empty or missing)
        token = "⌈ˢᵖnoun⌉"  # No surface, no reading
        features = extract_token_features(token)
        # Should fall back to READING_MASK, never empty string
        self.assertEqual(features.reading_gram, READING_MASK)
        self.assertTrue(features.reading_gram)  # Must be truthy

    def test_reading_gram_explicit_empty_marker_gets_mask(self):
        """Explicit empty reading_gram marker should still result in READING_MASK."""
        from kotogram.masking import READING_MASK

        # Token with explicit empty reading_gram marker
        token = "⌈ˢテストᵖnounᵍ⌉"  # Empty ᵍ marker
        features = extract_token_features(token)
        # Should fall back to READING_MASK
        self.assertEqual(features.reading_gram, READING_MASK)


if __name__ == "__main__":
    unittest.main()
