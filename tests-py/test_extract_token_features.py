"""Tests for extract_token_features function."""

import unittest
from kotogram import MecabJapaneseParser, SudachiJapaneseParser, extract_token_features
from kotogram.kotogram import split_kotogram


class TestExtractTokenFeaturesMecab(unittest.TestCase):
    """Test extract_token_features with MeCab parser."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = MecabJapaneseParser()
        except Exception as e:
            self.skipTest(f"MeCab not available: {e}")

    def test_verb_with_all_fields(self):
        """Test extracting features from a verb with all POS fields."""
        text = "食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Should have one token: 食べる
        self.assertEqual(len(tokens), 1)

        features = extract_token_features(tokens[0])
        self.assertEqual(features['surface'], '食べる')
        self.assertEqual(features['pos'], 'v')
        self.assertEqual(features['pos_detail1'], 'general')
        # pos_detail2 may be omitted if it's "general"
        self.assertEqual(features['conjugated_type'], 'e-ichidan-ba')
        self.assertIn(features['conjugated_form'], ['terminal', 'attributive'])

    def test_auxv_masu_semantic_parsing(self):
        """Test that auxv-masu is correctly identified in conjugated_type, not pos_detail1."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be ます
        masu_token = tokens[1]
        features = extract_token_features(masu_token)

        self.assertEqual(features['surface'], 'ます')
        self.assertEqual(features['pos'], 'auxv')
        # auxv-masu should be in conjugated_type, not pos_detail1
        self.assertEqual(features['conjugated_type'], 'auxv-masu')
        self.assertEqual(features['conjugated_form'], 'terminal')
        # pos_detail1 should be empty because parser omits it
        self.assertEqual(features['pos_detail1'], '')

    def test_auxv_desu(self):
        """Test extracting features from です."""
        text = "学生です"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be です
        desu_token = tokens[1]
        features = extract_token_features(desu_token)

        self.assertEqual(features['surface'], 'です')
        self.assertEqual(features['pos'], 'auxv')
        self.assertEqual(features['conjugated_type'], 'auxv-desu')
        self.assertEqual(features['conjugated_form'], 'terminal')

    def test_auxv_da_plain_copula(self):
        """Test extracting features from plain copula だ."""
        text = "学生だ"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be だ
        da_token = tokens[1]
        features = extract_token_features(da_token)

        self.assertEqual(features['surface'], 'だ')
        self.assertEqual(features['pos'], 'auxv')
        self.assertEqual(features['conjugated_type'], 'auxv-da')
        self.assertEqual(features['conjugated_form'], 'terminal')

    def test_particle_extraction(self):
        """Test extracting features from particles."""
        text = "私は"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be は
        wa_token = tokens[1]
        features = extract_token_features(wa_token)

        self.assertEqual(features['surface'], 'は')
        self.assertEqual(features['pos'], 'prt')
        self.assertIn('particle', features['pos_detail1'])

    def test_noun_extraction(self):
        """Test extracting features from a noun."""
        text = "学生"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        self.assertEqual(len(tokens), 1)
        features = extract_token_features(tokens[0])

        self.assertEqual(features['surface'], '学生')
        self.assertEqual(features['pos'], 'n')
        self.assertEqual(features['pos_detail1'], 'common_noun')
        # Nouns don't have conjugation
        self.assertEqual(features['conjugated_type'], '')
        self.assertEqual(features['conjugated_form'], '')

    def test_imperative_verb_kudasai(self):
        """Test extracting features from polite imperative ください."""
        text = "ください"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        self.assertEqual(len(tokens), 1)
        features = extract_token_features(tokens[0])

        self.assertEqual(features['surface'], 'ください')
        self.assertEqual(features['pos'], 'v')
        self.assertEqual(features['conjugated_type'], 'godan-ra')
        self.assertEqual(features['conjugated_form'], 'imperative')

    def test_adjective_extraction(self):
        """Test extracting features from an adjective."""
        text = "高い"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        self.assertEqual(len(tokens), 1)
        features = extract_token_features(tokens[0])

        self.assertEqual(features['surface'], '高い')
        self.assertEqual(features['pos'], 'adj')
        self.assertEqual(features['pos_detail1'], 'general')
        self.assertEqual(features['conjugated_type'], 'adjective')

    def test_base_orth_extraction(self):
        """Test extraction of base orthography."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # First token: 食べ
        features = extract_token_features(tokens[0])
        self.assertEqual(features['base_orth'], '食べる')

    def test_lemma_extraction(self):
        """Test extraction of lemma/dictionary form."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # First token: 食べ
        features = extract_token_features(tokens[0])
        self.assertEqual(features['lemma'], '食べる')

    def test_reading_extraction(self):
        """Test extraction of reading."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # First token: 食べ
        features = extract_token_features(tokens[0])
        self.assertIn('タベ', features['reading'])

    def test_empty_fields_default_to_empty_string(self):
        """Test that missing fields default to empty string."""
        # Create a minimal token with only surface and pos
        minimal_token = "⌈ˢテストᵖn⌉"
        features = extract_token_features(minimal_token)

        self.assertEqual(features['surface'], 'テスト')
        self.assertEqual(features['pos'], 'n')
        # All other fields should be empty strings
        self.assertEqual(features['pos_detail1'], '')
        self.assertEqual(features['pos_detail2'], '')
        self.assertEqual(features['conjugated_type'], '')
        self.assertEqual(features['conjugated_form'], '')
        self.assertEqual(features['base_orth'], '')
        self.assertEqual(features['lemma'], '')
        self.assertEqual(features['reading'], '')


class TestExtractTokenFeaturesSudachi(unittest.TestCase):
    """Test extract_token_features with Sudachi parser."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            self.parser = SudachiJapaneseParser(dict_type='full')
        except Exception as e:
            self.skipTest(f"Sudachi not available: {e}")

    def test_verb_extraction_sudachi(self):
        """Test extracting verb features with Sudachi."""
        text = "食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        self.assertEqual(len(tokens), 1)
        features = extract_token_features(tokens[0])

        self.assertEqual(features['surface'], '食べる')
        self.assertEqual(features['pos'], 'v')
        self.assertIn(features['conjugated_type'], ['e-ichidan-ba', 'ichidan'])

    def test_auxv_masu_sudachi(self):
        """Test auxv-masu extraction with Sudachi."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be ます
        masu_token = tokens[1]
        features = extract_token_features(masu_token)

        self.assertEqual(features['surface'], 'ます')
        self.assertEqual(features['pos'], 'auxv')
        self.assertEqual(features['conjugated_type'], 'auxv-masu')
        self.assertEqual(features['conjugated_form'], 'terminal')

    def test_cross_parser_consistency(self):
        """Test that both parsers produce features that can be extracted consistently."""
        mecab_parser = None
        try:
            mecab_parser = MecabJapaneseParser()
        except Exception:
            self.skipTest("MeCab not available for cross-parser test")

        text = "食べます"

        # Parse with both
        mecab_kotogram = mecab_parser.japanese_to_kotogram(text)
        sudachi_kotogram = self.parser.japanese_to_kotogram(text)

        mecab_tokens = split_kotogram(mecab_kotogram)
        sudachi_tokens = split_kotogram(sudachi_kotogram)

        # Extract features from ます token
        mecab_masu = extract_token_features(mecab_tokens[1])
        sudachi_masu = extract_token_features(sudachi_tokens[1])

        # Both should identify ます correctly
        self.assertEqual(mecab_masu['surface'], 'ます')
        self.assertEqual(sudachi_masu['surface'], 'ます')
        self.assertEqual(mecab_masu['pos'], 'auxv')
        self.assertEqual(sudachi_masu['pos'], 'auxv')
        self.assertEqual(mecab_masu['conjugated_type'], 'auxv-masu')
        self.assertEqual(sudachi_masu['conjugated_type'], 'auxv-masu')


class TestExtractTokenFeaturesEdgeCases(unittest.TestCase):
    """Test edge cases for extract_token_features."""

    def test_empty_token(self):
        """Test handling of empty token."""
        features = extract_token_features("")
        # Should return dictionary with all empty values
        self.assertEqual(features['surface'], '')
        self.assertEqual(features['pos'], '')

    def test_malformed_token_no_markers(self):
        """Test handling of token without markers."""
        features = extract_token_features("テスト")
        # Should return dictionary with all empty values
        self.assertEqual(features['surface'], '')
        self.assertEqual(features['pos'], '')

    def test_token_with_only_surface(self):
        """Test token with only surface marker.

        Note: The regex for surface requires ᵖ marker, so this will be empty.
        This is expected behavior as kotogram format always includes POS.
        """
        token = "⌈ˢテスト⌉"
        features = extract_token_features(token)
        # Surface extraction requires ᵖ marker to terminate, so this will be empty
        self.assertEqual(features['surface'], '')
        self.assertEqual(features['pos'], '')

    def test_token_with_surface_and_pos(self):
        """Test token with surface and POS only."""
        token = "⌈ˢテストᵖn:common_noun⌉"
        features = extract_token_features(token)
        self.assertEqual(features['surface'], 'テスト')
        self.assertEqual(features['pos'], 'n')
        self.assertEqual(features['pos_detail1'], 'common_noun')

    def test_complex_conjugated_verb(self):
        """Test verb with multiple conjugation details."""
        # This is a real kotogram token structure
        token = "⌈ˢ食べᵖv:general:e-ichidan-ba:conjunctiveᵇ食べるᵈ食べるʳタベ⌉"
        features = extract_token_features(token)

        self.assertEqual(features['surface'], '食べ')
        self.assertEqual(features['pos'], 'v')
        self.assertEqual(features['pos_detail1'], 'general')
        # pos_detail2 might be omitted or "general"
        self.assertEqual(features['conjugated_type'], 'e-ichidan-ba')
        self.assertEqual(features['conjugated_form'], 'conjunctive')
        self.assertEqual(features['base_orth'], '食べる')
        self.assertEqual(features['lemma'], '食べる')
        self.assertEqual(features['reading'], 'タベ')


if __name__ == '__main__':
    unittest.main()
