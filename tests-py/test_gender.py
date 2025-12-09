"""Tests for gender-associated speech analysis of Japanese sentences."""

import unittest
from kotogram import MecabJapaneseParser, SudachiJapaneseParser, gender, GenderLevel


class TestGenderMecab(unittest.TestCase):
    """Test gender analysis with MeCab parser."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = MecabJapaneseParser()
        except Exception as e:
            self.skipTest(f"MeCab not available: {e}")

    def test_masculine_ore_pronoun(self):
        """俺 (ore) pronoun should be MASCULINE."""
        text = "俺は行く"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_masculine_boku_pronoun(self):
        """僕 (boku) pronoun should be MASCULINE."""
        text = "僕は学生だ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_masculine_ze_particle(self):
        """Sentence with ぜ particle should be MASCULINE."""
        text = "行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_masculine_zo_particle(self):
        """Sentence with ぞ particle should be MASCULINE."""
        text = "食べるぞ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_feminine_atashi_pronoun(self):
        """あたし (atashi) pronoun should be FEMININE."""
        text = "あたしは行く"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_feminine_wa_particle(self):
        """Sentence with わ particle should be FEMININE."""
        text = "行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_feminine_kashira(self):
        """Sentence with かしら should be FEMININE."""
        text = "何かしら"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_neutral_watashi_pronoun(self):
        """私 (watashi) pronoun should be NEUTRAL."""
        text = "私は行く"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.NEUTRAL)

    def test_neutral_formal_sentence(self):
        """Formal sentence without gender markers should be NEUTRAL."""
        text = "私は学生です"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.NEUTRAL)

    def test_neutral_plain_verb(self):
        """Plain verb without gender markers should be NEUTRAL."""
        text = "食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.NEUTRAL)

    def test_unpragmatic_ore_with_wa(self):
        """俺 with わ particle should be UNPRAGMATIC_GENDER."""
        text = "俺が行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.UNPRAGMATIC_GENDER)

    def test_unpragmatic_atashi_with_ze(self):
        """あたし with ぜ particle should be UNPRAGMATIC_GENDER."""
        text = "あたしが行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.UNPRAGMATIC_GENDER)

    def test_masculine_ore_with_ze(self):
        """俺 with ぜ particle (consistent masculine) should be MASCULINE."""
        text = "俺が行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_feminine_atashi_with_wa(self):
        """あたし with わ particle (consistent feminine) should be FEMININE."""
        text = "あたしが行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_empty_kotogram(self):
        """Empty kotogram should return NEUTRAL."""
        result = gender("")
        self.assertEqual(result, GenderLevel.NEUTRAL)


class TestGenderSudachi(unittest.TestCase):
    """Test gender analysis with Sudachi parser."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_masculine_ore_pronoun(self):
        """俺 (ore) pronoun should be MASCULINE."""
        text = "俺は行く"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_masculine_boku_pronoun(self):
        """僕 (boku) pronoun should be MASCULINE."""
        text = "僕は学生だ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_masculine_ze_particle(self):
        """Sentence with ぜ particle should be MASCULINE."""
        text = "行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_masculine_zo_particle(self):
        """Sentence with ぞ particle should be MASCULINE."""
        text = "食べるぞ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_feminine_atashi_pronoun(self):
        """あたし (atashi) pronoun should be FEMININE."""
        text = "あたしは行く"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_feminine_wa_particle(self):
        """Sentence with わ particle should be FEMININE."""
        text = "行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_feminine_kashira(self):
        """Sentence with かしら should be FEMININE."""
        text = "何かしら"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_neutral_watashi_pronoun(self):
        """私 (watashi) pronoun should be NEUTRAL."""
        text = "私は行く"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.NEUTRAL)

    def test_neutral_formal_sentence(self):
        """Formal sentence without gender markers should be NEUTRAL."""
        text = "私は学生です"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.NEUTRAL)

    def test_neutral_plain_verb(self):
        """Plain verb without gender markers should be NEUTRAL."""
        text = "食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.NEUTRAL)

    def test_unpragmatic_ore_with_wa(self):
        """俺 with わ particle should be UNPRAGMATIC_GENDER."""
        text = "俺が行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.UNPRAGMATIC_GENDER)

    def test_unpragmatic_atashi_with_ze(self):
        """あたし with ぜ particle should be UNPRAGMATIC_GENDER."""
        text = "あたしが行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.UNPRAGMATIC_GENDER)

    def test_masculine_ore_with_ze(self):
        """俺 with ぜ particle (consistent masculine) should be MASCULINE."""
        text = "俺が行くぜ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_feminine_atashi_with_wa(self):
        """あたし with わ particle (consistent feminine) should be FEMININE."""
        text = "あたしが行くわ"
        kotogram = self.parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_empty_kotogram(self):
        """Empty kotogram should return NEUTRAL."""
        result = gender("")
        self.assertEqual(result, GenderLevel.NEUTRAL)


class TestGenderEdgeCases(unittest.TestCase):
    """Test edge cases for gender-associated speech analysis."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.mecab_parser = None

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.sudachi_parser = None

        if not self.mecab_parser and not self.sudachi_parser:
            self.skipTest("No parsers available")

    def test_formal_with_ore_is_masculine_mecab(self):
        """Formal sentence with 俺 should still be MASCULINE."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "俺は行きます"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_formal_with_ore_is_masculine_sudachi(self):
        """Formal sentence with 俺 should still be MASCULINE."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "俺は行きます"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_formal_with_atashi_is_feminine_mecab(self):
        """Formal sentence with あたし should still be FEMININE."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "あたしは行きます"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_formal_with_atashi_is_feminine_sudachi(self):
        """Formal sentence with あたし should still be FEMININE."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "あたしは行きます"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_katakana_ore_mecab(self):
        """Katakana オレ should be MASCULINE."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "オレは行く"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_katakana_ore_sudachi(self):
        """Katakana オレ should be MASCULINE."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "オレは行く"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.MASCULINE)

    def test_katakana_atashi_mecab(self):
        """Katakana アタシ should be FEMININE."""
        if not self.mecab_parser:
            self.skipTest("MeCab not available")

        text = "アタシは行く"
        kotogram = self.mecab_parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)

    def test_katakana_atashi_sudachi(self):
        """Katakana アタシ should be FEMININE."""
        if not self.sudachi_parser:
            self.skipTest("Sudachi not available")

        text = "アタシは行く"
        kotogram = self.sudachi_parser.japanese_to_kotogram(text)
        result = gender(kotogram)
        self.assertEqual(result, GenderLevel.FEMININE)


class TestGenderAdditionalPatterns(unittest.TestCase):
    """Additional tests for gender-associated speech patterns."""

    def setUp(self):
        """Set up MeCab/Sudachi parsers if available."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception:
            self.mecab_parser = None

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception:
            self.sudachi_parser = None

        if not self.mecab_parser and not self.sudachi_parser:
            self.skipTest("No parsers available")

    def _assert_gender_for_all_parsers(self, text: str, expected: GenderLevel):
        """Helper: run the same assertion for all available parsers."""
        for parser_name, parser in (("MeCab", self.mecab_parser), ("Sudachi", self.sudachi_parser)):
            if not parser:
                continue
            kotogram = parser.japanese_to_kotogram(text)
            result = gender(kotogram)
            self.assertEqual(
                result,
                expected,
                msg=f"{parser_name} failed for {text!r}: expected {expected}, got {result}",
            )

    # --- More masculine markers ---

    def test_masculine_ore_sama(self):
        """俺様 (oresama) is strongly MASCULINE."""
        text = "俺様が行く"
        self._assert_gender_for_all_parsers(text, GenderLevel.MASCULINE)

    def test_masculine_plural_ore_tachi(self):
        """俺たち plural pronoun should be MASCULINE."""
        text = "俺たちは行く"
        self._assert_gender_for_all_parsers(text, GenderLevel.MASCULINE)

    def test_masculine_second_person_omae(self):
        """お前 as second person pronoun typically MASCULINE."""
        text = "お前、行けよ"
        self._assert_gender_for_all_parsers(text, GenderLevel.MASCULINE)

    def test_masculine_rude_negation_shiraneeyo(self):
        """～ねえよ rough male speech."""
        text = "そんなの知らねえよ"
        self._assert_gender_for_all_parsers(text, GenderLevel.MASCULINE)

    def test_masculine_sentence_final_daro(self):
        """～だろ sentence-final rough assertive, often masculine."""
        text = "結局僕がやるんだろ"
        self._assert_gender_for_all_parsers(text, GenderLevel.MASCULINE)

    def test_masculine_multiple_markers(self):
        """Combination of 俺 + ぞ + rough style remains MASCULINE."""
        text = "俺さ、絶対行くぞ"
        self._assert_gender_for_all_parsers(text, GenderLevel.MASCULINE)

    # --- More feminine markers ---

    def test_feminine_atakushi_desu_wa(self):
        """あたくし + ですわ is stereotypically FEMININE / refined."""
        text = "あたくしは行きますわ"
        self._assert_gender_for_all_parsers(text, GenderLevel.FEMININE)

    def test_feminine_no_yo(self):
        """～のよ sentence-final feminine marker."""
        text = "行くのよ"
        self._assert_gender_for_all_parsers(text, GenderLevel.FEMININE)

    def test_feminine_no_ne(self):
        """～のね sentence-final feminine marker."""
        text = "疲れたのね"
        self._assert_gender_for_all_parsers(text, GenderLevel.FEMININE)

    def test_feminine_desu_wa_without_pronoun(self):
        """ですわ feminine politeness even without explicit pronoun."""
        text = "今日は暑いですわ"
        self._assert_gender_for_all_parsers(text, GenderLevel.FEMININE)

    def test_feminine_multiple_markers(self):
        """あたし + のよ combination stays FEMININE."""
        text = "あたしね、先に行くのよ"
        self._assert_gender_for_all_parsers(text, GenderLevel.FEMININE)

    # --- Conflicting/unpragmatic mixes beyond わ/ぜ ---

    def test_unpragmatic_ore_no_yo(self):
        """俺 + のよ (feminine ending) should be UNPRAGMATIC_GENDER."""
        text = "俺は先に行くのよ"
        self._assert_gender_for_all_parsers(text, GenderLevel.UNPRAGMATIC_GENDER)

    def test_unpragmatic_ore_desu_wa(self):
        """俺 + ですわ (feminine polite) should be UNPRAGMATIC_GENDER."""
        text = "俺は大丈夫ですわ"
        self._assert_gender_for_all_parsers(text, GenderLevel.UNPRAGMATIC_GENDER)

    def test_unpragmatic_atashi_zo(self):
        """あたし + ぞ (masculine) should be UNPRAGMATIC_GENDER."""
        text = "あたしがやるぞ"
        self._assert_gender_for_all_parsers(text, GenderLevel.UNPRAGMATIC_GENDER)

    def test_unpragmatic_atashi_daro(self):
        """あたし + だろ (masculine-ish) should be UNPRAGMATIC_GENDER."""
        text = "あたしがやるだろ"
        self._assert_gender_for_all_parsers(text, GenderLevel.UNPRAGMATIC_GENDER)

    # --- Neutral / false-positive guards ---

    def test_neutral_wa_inside_word_not_sentence_final(self):
        """和食 / 今日は etc. should not trigger feminine 'わ' heuristics."""
        text = "今日は和食を食べる"
        self._assert_gender_for_all_parsers(text, GenderLevel.NEUTRAL)

    def test_neutral_kawaii_not_feminine(self):
        """かわいい alone should be NEUTRAL (no explicit gender marker)."""
        text = "この猫かわいい"
        self._assert_gender_for_all_parsers(text, GenderLevel.NEUTRAL)

    def test_neutral_past_formal_without_markers(self):
        """Polite past without gendered markers should be NEUTRAL."""
        text = "昨日は早く寝ました"
        self._assert_gender_for_all_parsers(text, GenderLevel.NEUTRAL)

    def test_neutral_watakushi_formal(self):
        """わたくし + formal verb often business-like, treat as NEUTRAL."""
        text = "わたくしは後ほど参ります"
        self._assert_gender_for_all_parsers(text, GenderLevel.NEUTRAL)

    def test_neutral_only_emoji(self):
        """Sentence with emoji but no markers should be NEUTRAL."""
        text = "今日は忙しかった😂"
        self._assert_gender_for_all_parsers(text, GenderLevel.NEUTRAL)

    # --- Quoted speech (direct quote masculine) ---

    def test_masculine_inside_quotes(self):
        """
        Direct speech '俺は行く' inside quotes should still be MASCULINE.

        This asserts that the classifier looks into quoted segments
        rather than only the narrative frame.
        """
        text = "彼は『俺は行く』と言った"
        self._assert_gender_for_all_parsers(text, GenderLevel.MASCULINE)

    # --- Robustness / whitespace-only kotogram ---

    def test_whitespace_kotogram_is_neutral(self):
        """Whitespace-only kotogram should be NEUTRAL."""
        result = gender("   ")
        self.assertEqual(result, GenderLevel.NEUTRAL)


class TestCrossParserGender(unittest.TestCase):
    """Test that gender analysis works consistently across parsers."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.mecab_parser = MecabJapaneseParser()
        except Exception as e:
            self.mecab_parser = None

        try:
            self.sudachi_parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.sudachi_parser = None

        if not self.mecab_parser or not self.sudachi_parser:
            self.skipTest("Both parsers required for cross-parser tests")

    def test_both_detect_masculine(self):
        """Both parsers should detect masculine sentences."""
        text = "俺は行く"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = gender(mecab_kotogram)
        sudachi_result = gender(sudachi_kotogram)

        self.assertEqual(mecab_result, GenderLevel.MASCULINE)
        self.assertEqual(sudachi_result, GenderLevel.MASCULINE)

    def test_both_detect_feminine(self):
        """Both parsers should detect feminine sentences."""
        text = "あたしは行く"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = gender(mecab_kotogram)
        sudachi_result = gender(sudachi_kotogram)

        self.assertEqual(mecab_result, GenderLevel.FEMININE)
        self.assertEqual(sudachi_result, GenderLevel.FEMININE)

    def test_both_detect_neutral(self):
        """Both parsers should detect neutral sentences."""
        text = "私は学生です"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = gender(mecab_kotogram)
        sudachi_result = gender(sudachi_kotogram)

        self.assertEqual(mecab_result, GenderLevel.NEUTRAL)
        self.assertEqual(sudachi_result, GenderLevel.NEUTRAL)

    def test_both_detect_unpragmatic(self):
        """Both parsers should detect unpragmatic gender mixing."""
        text = "俺が行くわ"

        mecab_kotogram = self.mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.sudachi_parser.japanese_to_kotogram(text)

        mecab_result = gender(mecab_kotogram)
        sudachi_result = gender(sudachi_kotogram)

        self.assertEqual(mecab_result, GenderLevel.UNPRAGMATIC_GENDER)
        self.assertEqual(sudachi_result, GenderLevel.UNPRAGMATIC_GENDER)


if __name__ == "__main__":
    unittest.main()
