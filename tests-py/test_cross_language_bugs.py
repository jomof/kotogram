"""
Tests for cross-language validation bugs that were discovered and fixed.

These tests document specific issues found during cross-language validation
to prevent regression.
"""

import unittest

from kotogram import SudachiJapaneseParser, kotogram_to_japanese


class TestCrossLanguageBugs(unittest.TestCase):
    """Test cases for bugs discovered during cross-language validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Initialize Sudachi parser
        self.parser = SudachiJapaneseParser(dict_type="full")

    def test_bug1_small_tsu_compound_verb(self):
        """
        Bug 1 Regression Test: "もって" (motte)
        """
        text = "もって"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_bug1_period(self):
        """Period 。 regression test."""
        text = "こんにちは。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_bug1_question_mark(self):
        """Question mark ？ regression test."""
        text = "何？"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_integration_full_sentence(self):
        """
        Integration test: Full sentence from validation failure.
        Sentence: "きみにちょっとしたものをもってきたよ。"
        """
        text = "きみにちょっとしたものをもってきたよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_integration_sentence_with_furigana(self):
        """Same sentence with furigana mode."""
        text = "きみにちょっとしたものをもってきたよ。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram, furigana=True)
        # Just ensure it runs and contains key parts (exact string depends on reading resolution)
        self.assertIn("もって", result)
        self.assertIn("。", result)

    def test_real_world_sentence_tatoeba(self):
        """Test with a real sentence from Tatoeba corpus."""
        text = "何かしてみましょう。"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)

    def test_compound_verb_parsing(self):
        """Test that compound verbs like もって are parsed correctly."""
        text = "もって"
        kotogram = self.parser.japanese_to_kotogram(text)

        # Should produce two tokens: もっ (verb) + て (particle)
        from kotogram import split_kotogram

        tokens = split_kotogram(kotogram)
        self.assertEqual(len(tokens), 2)

        # Round trip
        result = kotogram_to_japanese(kotogram)
        self.assertEqual(result, text)


if __name__ == "__main__":
    unittest.main()
