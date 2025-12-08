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
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_bug1_small_tsu_collapses_with_particle(self):
        """
        Bug 1: Missing POS_TO_CHARS.auxs entries

        Issue: TypeScript was missing many punctuation characters in POS_TO_CHARS.auxs,
        most critically the small tsu 'っ' which is needed to properly collapse
        compound verb forms.

        Example: "もっ" + "て" should collapse to "もって" when collapse_punctuation=True
        because 'っ' at the end of a token should attach to the following particle.
        """
        # Kotogram for "もって" (motte) - verb stem + te particle
        kotogram = "⌈ˢもっᵖv:general:godan-ta:conjunctive-geminateᵇもつᵈもつʳモッ⌉⌈ˢてᵖprt:conjunctive_particleʳテ⌉"

        # With collapse_punctuation=True, small tsu should attach to following て
        collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        self.assertEqual(collapsed, "もって")

        # With collapse_punctuation=False, should keep space
        not_collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=False)
        self.assertEqual(not_collapsed, "もっ て")

    def test_bug1_period_collapses(self):
        """Period 。 should collapse when collapse_punctuation=True."""
        kotogram = "⌈ˢこんにちはᵖint⌉⌈ˢ。ᵖauxs:period⌉"

        collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        self.assertEqual(collapsed, "こんにちは。")

        not_collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=False)
        self.assertEqual(not_collapsed, "こんにちは 。")

    def test_bug1_question_mark_collapses(self):
        """Question mark ？ should collapse when collapse_punctuation=True."""
        kotogram = "⌈ˢ何ᵖpronʳナン⌉⌈ˢ？ᵖauxs:period⌉"

        collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        self.assertEqual(collapsed, "何？")

        not_collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=False)
        self.assertEqual(not_collapsed, "何 ？")

    def test_integration_full_sentence_with_compound_verbs(self):
        """
        Integration test: Full sentence from validation failure.

        This was one of the actual failing test cases that revealed the bugs.
        Sentence: "きみにちょっとしたものをもってきたよ。"
        Meaning: "I brought you a little something."
        """
        kotogram = (
            "⌈ˢきみᵖpronʳキミ⌉⌈ˢにᵖprt:case_particleʳニ⌉⌈ˢちょっとᵖadvʳチョット⌉"
            "⌈ˢしᵖv:non_self_reliant:sa-irregular:conjunctiveᵇするᵈするʳシ⌉"
            "⌈ˢたᵖauxv:auxv-ta:attributiveʳタ⌉⌈ˢものᵖn:common_noun:suru-possibleʳモノ⌉"
            "⌈ˢをᵖprt:case_particleʳヲ⌉"
            "⌈ˢもっᵖv:general:godan-ta:conjunctive-geminateᵇもつᵈもつʳモッ⌉"
            "⌈ˢてᵖprt:conjunctive_particleʳテ⌉"
            "⌈ˢきᵖv:non_self_reliant:ka-irregular:conjunctiveᵇくるᵈくるʳキ⌉"
            "⌈ˢたᵖauxv:auxv-ta:terminalʳタ⌉⌈ˢよᵖprt:sentence_final_particleʳヨ⌉"
            "⌈ˢ。ᵖauxs:period⌉"
        )

        # Default: no spaces, punctuation naturally attached
        default_result = kotogram_to_japanese(kotogram)
        self.assertEqual(default_result, "きみにちょっとしたものをもってきたよ。")

        # With spaces + collapse: should collapse もっ+て and attach period
        collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        self.assertEqual(collapsed, "きみ に ちょっと し た もの を もって き た よ。")
        self.assertIn("もって", collapsed)  # Should be collapsed
        self.assertNotIn(" 。", collapsed)  # Period should be attached

        # With spaces but no collapse: should keep もっ て separated and space before period
        not_collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=False)
        self.assertEqual(not_collapsed, "きみ に ちょっと し た もの を もっ て き た よ 。")
        self.assertIn("もっ て", not_collapsed)  # Should be separated
        self.assertIn(" 。", not_collapsed)  # Period should have space before it

    def test_integration_sentence_with_furigana_and_compound_verbs(self):
        """Same sentence with furigana mode."""
        kotogram = (
            "⌈ˢきみᵖpronʳキミ⌉⌈ˢにᵖprt:case_particleʳニ⌉⌈ˢちょっとᵖadvʳチョット⌉"
            "⌈ˢしᵖv:non_self_reliant:sa-irregular:conjunctiveᵇするᵈするʳシ⌉"
            "⌈ˢたᵖauxv:auxv-ta:attributiveʳタ⌉⌈ˢものᵖn:common_noun:suru-possibleʳモノ⌉"
            "⌈ˢをᵖprt:case_particleʳヲ⌉"
            "⌈ˢもっᵖv:general:godan-ta:conjunctive-geminateᵇもつᵈもつʳモッ⌉"
            "⌈ˢてᵖprt:conjunctive_particleʳテ⌉"
            "⌈ˢきᵖv:non_self_reliant:ka-irregular:conjunctiveᵇくるᵈくるʳキ⌉"
            "⌈ˢたᵖauxv:auxv-ta:terminalʳタ⌉⌈ˢよᵖprt:sentence_final_particleʳヨ⌉"
            "⌈ˢ。ᵖauxs:period⌉"
        )

        result = kotogram_to_japanese(
            kotogram, spaces=True, furigana=True, collapse_punctuation=True
        )

        # Should have proper collapsing
        self.assertIn("もって", result)  # Collapsed
        self.assertNotIn(" 。", result)  # Period attached
        # Pure kana tokens shouldn't get furigana, but check it doesn't break
        self.assertTrue(len(result) > 0)

    def test_real_world_sentence_from_tatoeba(self):
        """Test with a real sentence from Tatoeba corpus."""
        text = "何かしてみましょう。"
        kotogram = self.parser.japanese_to_kotogram(text)

        # Default
        default_result = kotogram_to_japanese(kotogram)
        self.assertEqual(default_result, text)

        # With spaces + collapse
        collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        # Period should be attached
        self.assertFalse(collapsed.endswith(" 。"))
        self.assertTrue(collapsed.endswith("。"))

        # With spaces but no collapse
        not_collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=False)
        # Period should have space before it
        self.assertTrue(not_collapsed.endswith(" 。"))

    def test_compound_verb_parsing(self):
        """Test that compound verbs like もって are parsed correctly."""
        text = "もって"
        kotogram = self.parser.japanese_to_kotogram(text)

        # Should produce two tokens: もっ (verb) + て (particle)
        from kotogram import split_kotogram
        tokens = split_kotogram(kotogram)
        self.assertEqual(len(tokens), 2)

        # With collapse, should join back to もって
        result_collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=True)
        self.assertEqual(result_collapsed, "もって")

        # Without collapse, should have space
        result_not_collapsed = kotogram_to_japanese(kotogram, spaces=True, collapse_punctuation=False)
        self.assertEqual(result_not_collapsed, "もっ て")


if __name__ == '__main__':
    unittest.main()
