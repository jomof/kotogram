"""Tests for Japanese parser implementations."""

import unittest
from unittest.mock import Mock, MagicMock
from kotogram import JapaneseParser, MecabJapaneseParser, kotogram_to_japanese, split_kotogram


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


class TestMecabJapaneseParser(unittest.TestCase):
    """Test cases for MecabJapaneseParser implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock MeCab tagger
        self.mock_tagger = Mock()
        self.parser = MecabJapaneseParser(mecab_tagger=self.mock_tagger)

    def test_initialization_with_tagger(self):
        """MecabJapaneseParser can be initialized with a tagger."""
        parser = MecabJapaneseParser(mecab_tagger=self.mock_tagger)
        self.assertIsInstance(parser, JapaneseParser)
        self.assertEqual(parser.tagger, self.mock_tagger)

    def test_japanese_to_kotogram_simple(self):
        """Convert simple Japanese text to kotogram format."""
        # Mock MeCab output for "猫" (cat)
        self.mock_tagger.parse.return_value = (
            "猫\t名詞,普通名詞,一般,,,,ネコ,猫,猫,ネコ,猫,ネコ,和,,,,,,,\n"
            "EOS\n"
        )

        result = self.parser.japanese_to_kotogram("猫")

        # Verify the tagger was called
        self.mock_tagger.parse.assert_called_once_with("猫")

        # Verify result has kotogram format markers
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)
        self.assertIn("ˢ猫", result)
        self.assertIn("ᵖn", result)  # n for noun

    def test_japanese_to_kotogram_with_verb(self):
        """Convert Japanese verb to kotogram format."""
        # Mock MeCab output for "食べる" (to eat)
        self.mock_tagger.parse.return_value = (
            "食べる\t動詞,一般,,下一段-バ行,終止形-一般,タベル,食べる,食べる,タベル,食べる,タベル,和,,,,,,,\n"
            "EOS\n"
        )

        result = self.parser.japanese_to_kotogram("食べる")

        self.assertIn("⌈", result)
        self.assertIn("⌉", result)
        self.assertIn("ˢ食べる", result)
        self.assertIn("ᵖv", result)  # v for verb

    def test_japanese_to_kotogram_with_particle(self):
        """Convert Japanese particle to kotogram format."""
        # Mock MeCab output for "を" (object particle)
        self.mock_tagger.parse.return_value = (
            "を\t助詞,格助詞,,,,ヲ,を,を,オ,を,オ,和,,,,,,,\n"
            "EOS\n"
        )

        result = self.parser.japanese_to_kotogram("を")

        self.assertIn("ˢを", result)
        self.assertIn("ᵖprt", result)  # prt for particle
        self.assertIn("case_particle", result)

    def test_japanese_to_kotogram_multiple_tokens(self):
        """Convert multiple Japanese tokens to kotogram format."""
        # Mock MeCab output for "猫を食べる" (eat a cat)
        self.mock_tagger.parse.return_value = (
            "猫\t名詞,普通名詞,一般,,,,ネコ,猫,猫,ネコ,猫,ネコ,和,,,,,,,\n"
            "を\t助詞,格助詞,,,,ヲ,を,を,オ,を,オ,和,,,,,,,\n"
            "食べる\t動詞,一般,,下一段-バ行,終止形-一般,タベル,食べる,食べる,タベル,食べる,タベル,和,,,,,,,\n"
            "EOS\n"
        )

        result = self.parser.japanese_to_kotogram("猫を食べる")

        # Should have three token markers
        self.assertEqual(result.count("⌈"), 3)
        self.assertEqual(result.count("⌉"), 3)

    def test_special_character_handling(self):
        """Parser handles special っ character correctly."""
        self.mock_tagger.parse.return_value = "EOS\n"

        # Test various っ formats
        self.parser.japanese_to_kotogram(" っ")
        self.mock_tagger.parse.assert_called_with("っ")

        self.parser.japanese_to_kotogram("っ ")
        self.mock_tagger.parse.assert_called_with("っ")

        self.parser.japanese_to_kotogram(" っ ")
        self.mock_tagger.parse.assert_called_with("っ")

    def test_validation_mode_enabled(self):
        """Validation mode raises descriptive errors for unmapped keys."""
        # Create parser with validation enabled
        parser = MecabJapaneseParser(mecab_tagger=self.mock_tagger, validate=True)

        # Mock MeCab output with an unmapped POS value
        self.mock_tagger.parse.return_value = (
            "test\tUNMAPPED_POS,一般,,,,テスト,test,test,テスト,test,テスト,和,,,,,,,\n"
            "EOS\n"
        )

        # Should raise KeyError with descriptive message
        with self.assertRaises(KeyError) as context:
            parser.japanese_to_kotogram("test")

        # Verify error message contains useful information
        error_msg = str(context.exception)
        self.assertIn("POS_MAP", error_msg)
        self.assertIn("UNMAPPED_POS", error_msg)
        self.assertIn("Raw MeCab token:", error_msg)

    def test_validation_mode_disabled(self):
        """Validation mode disabled silently ignores unmapped keys."""
        # Parser with validation disabled (default)
        parser = MecabJapaneseParser(mecab_tagger=self.mock_tagger, validate=False)

        # Mock MeCab output with an unmapped POS value
        self.mock_tagger.parse.return_value = (
            "test\tUNMAPPED_POS,一般,,,,テスト,test,test,テスト,test,テスト,和,,,,,,,\n"
            "EOS\n"
        )

        # Should not raise an error
        result = parser.japanese_to_kotogram("test")

        # Result should still have markers
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)


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


class TestMecabParserInternals(unittest.TestCase):
    """Test internal methods of MecabJapaneseParser."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tagger = Mock()
        self.parser = MecabJapaneseParser(mecab_tagger=self.mock_tagger)

    def test_parse_raw_mecab_output(self):
        """Parse raw MeCab output into token dictionaries."""
        raw = (
            "猫\t名詞,普通名詞,一般,,,,ネコ,猫,猫,ネコ,猫,ネコ,和,,,,,,,\n"
            "EOS\n"
        )
        tokens = self.parser._parse_raw_mecab_output(raw)

        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["surface"], "猫")
        self.assertEqual(tokens[0]["pos"], "n")
        self.assertEqual(tokens[0]["pos_detail_1"], "common_noun")

    def test_raw_token_to_kotogram(self):
        """Convert raw token dictionary to kotogram format."""
        token = {
            "surface": "猫",
            "pos": "n",
            "pos_detail_1": "common_noun",
            "lemma": "猫",
        }
        result = self.parser._raw_token_to_kotogram(token)

        self.assertIn("⌈", result)
        self.assertIn("⌉", result)
        self.assertIn("ˢ猫", result)
        self.assertIn("ᵖn", result)
        self.assertIn("common_noun", result)

    def test_raw_token_with_conjugation(self):
        """Convert token with conjugation info to kotogram."""
        token = {
            "surface": "食べる",
            "pos": "v",
            "conjugated_type": "e-ichidan-ba",
            "conjugated_form": "terminal",
            "base_orthography": "食べる",
        }
        result = self.parser._raw_token_to_kotogram(token)

        self.assertIn("e-ichidan-ba", result)
        self.assertIn("terminal", result)
        self.assertIn("ᵇ食べる", result)  # base form


class TestTatoebaValidation(unittest.TestCase):
    """Test cases for previously failing Tatoeba sentences."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tagger = Mock()
        self.parser = MecabJapaneseParser(mecab_tagger=self.mock_tagger, validate=False)

    def test_classical_negative_auxiliary_zu(self):
        """Test classical negative auxiliary verb ず (zu)."""
        # Sentence: 会社に入ると、自分が望むと望まざるとにかかわらず、会社のために働かなくてはいけない。
        # This sentence uses 望まざる (nozoma-zaru) with classical negative ず
        self.mock_tagger.parse.return_value = (
            "ざる\t助動詞,*,*,文語助動詞-ズ,連体形-一般,ザル,ざる,ざる,ザル,ざる,ザル,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("ざる")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_greek_letter_character_pos(self):
        """Test Greek letter character POS mapping."""
        # Sentence contains Greek letters like α (alpha), β (beta), γ (gamma)
        # The 文字 (character) is in POS2 field
        self.mock_tagger.parse.return_value = (
            "α\t名詞,普通名詞,文字,,,アルファ,α,α,アルファ,α,アルファ,外,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("α")
        self.assertIn("⌈", result)
        # Just check it parses successfully - character is POS2 detail
        self.assertIn("ˢα", result)

    def test_classical_past_tense_ki(self):
        """Test classical past tense auxiliary き (ki)."""
        # Used in classical sentences with past tense
        self.mock_tagger.parse.return_value = (
            "き\t助動詞,*,*,文語助動詞-キ,連体形-一般,キ,き,き,キ,き,キ,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("き")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_stem_sa_form(self):
        """Test stem-sa conjugation form."""
        # Sentence: 彼女は申し訳なさそうに「食べられなかった」と言った。
        # This uses さそう with stem-sa form
        self.mock_tagger.parse.return_value = (
            "さ\t接尾辞,形状詞的,*,*,語幹-サ,サ,さ,さ,サ,さ,サ,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("さ")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_colloquial_nanda(self):
        """Test colloquial auxiliary なんだ (nanda)."""
        # Sentence: あらまあ、ホント、全く知らなんだ。
        self.mock_tagger.parse.return_value = (
            "なんだ\t助動詞,*,*,助動詞-ナンダ,連体形-一般,ナンダ,なんだ,なんだ,ナンダ,なんだ,ナンダ,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("なんだ")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_attributive_abbreviated(self):
        """Test attributive-abbreviated form (連体形-省略)."""
        # Sentence: もう～、なにやってんだよ。
        # Uses てんだ (shortened form of ているんだ)
        self.mock_tagger.parse.return_value = (
            "てん\t助動詞,*,*,助動詞-タ,連体形-省略,テン,てん,てん,テン,てん,テン,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("てん")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_realis_auxiliary(self):
        """Test realis-auxiliary form (已然形-補助)."""
        # Classical sentence: 心ここに在らざれば聞けども聴こえず。
        self.mock_tagger.parse.return_value = (
            "ざれ\t助動詞,*,*,文語助動詞-ズ,已然形-補助,ザレ,ざれ,ざれ,ザレ,ざれ,ザレ,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("ざれ")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_ichidan_ha_row(self):
        """Test i-ichidan ha-row verb conjugation."""
        # Rare verb form
        self.mock_tagger.parse.return_value = (
            "干る\t動詞,一般,*,上一段-ハ行,終止形-一般,ヒル,干る,干る,ヒル,干る,ヒル,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("干る")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_kansai_dialect_hen(self):
        """Test Kansai dialect negative へん (hen)."""
        self.mock_tagger.parse.return_value = (
            "へん\t助動詞,*,*,助動詞-ヘン,終止形-一般,ヘン,へん,へん,ヘン,へん,ヘン,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("へん")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)

    def test_classical_presumptive_mu(self):
        """Test classical presumptive auxiliary む (mu)."""
        self.mock_tagger.parse.return_value = (
            "む\t助動詞,*,*,文語助動詞-ム,終止形-一般,ム,む,む,ム,む,ム,和,,,,,,,\n"
            "EOS\n"
        )
        result = self.parser.japanese_to_kotogram("む")
        self.assertIn("⌈", result)
        self.assertIn("⌉", result)


if __name__ == "__main__":
    unittest.main()
