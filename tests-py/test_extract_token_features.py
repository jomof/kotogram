"""Tests for extract_token_features function."""

import unittest
from kotogram import SudachiJapaneseParser, extract_token_features
from kotogram.kotogram import split_kotogram


class TestExtractTokenFeaturesSudachi(unittest.TestCase):
    """Test extract_token_features with Sudachi parser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = SudachiJapaneseParser(dict_type='full')

    def test_verb_extraction_sudachi(self):

        """Test extracting verb features with Sudachi."""
        text = "食べる"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        self.assertEqual(len(tokens), 1)
        features = extract_token_features(tokens[0])

        self.assertEqual(features.surface, '食べる')
        self.assertEqual(features.pos, 'verb')
        self.assertIn(features.conjugated_type, ['lower-ichidan-ba', 'ichidan'])

    def test_auxv_masu_sudachi(self):
        """Test auxv-masu extraction with Sudachi."""
        text = "食べます"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be ます
        masu_token = tokens[1]
        features = extract_token_features(masu_token)

        self.assertEqual(features.surface, 'ます')
        self.assertEqual(features.pos, 'aux-verb')
        self.assertEqual(features.conjugated_type, 'aux-masu')
        self.assertEqual(features.conjugated_form, 'terminal')

    def test_auxv_desu(self):
        """Test extracting features from です."""
        text = "学生です"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be です
        desu_token = tokens[1]
        features = extract_token_features(desu_token)

        self.assertEqual(features.surface, 'です')
        self.assertEqual(features.pos, 'aux-verb')
        self.assertEqual(features.conjugated_type, 'aux-desu')
        self.assertEqual(features.conjugated_form, 'terminal')

    def test_auxv_da_plain_copula(self):
        """Test extracting features from plain copula だ."""
        text = "学生だ"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be だ
        da_token = tokens[1]
        features = extract_token_features(da_token)

        self.assertEqual(features.surface, 'だ')
        self.assertEqual(features.pos, 'aux-verb')
        self.assertEqual(features.conjugated_type, 'aux-da')
        self.assertEqual(features.conjugated_form, 'terminal')

    def test_particle_extraction(self):
        """Test extracting features from particles."""
        text = "私は"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        # Second token should be は
        wa_token = tokens[1]
        features = extract_token_features(wa_token)

        self.assertEqual(features.surface, 'は')
        self.assertEqual(features.pos, 'particle')
        self.assertIn('particle', features.pos_detail1)

    def test_noun_extraction(self):
        """Test extracting features from a noun."""
        text = "学生"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        self.assertEqual(len(tokens), 1)
        features = extract_token_features(tokens[0])

        self.assertEqual(features.surface, '学生')
        self.assertEqual(features.pos, 'noun')
        self.assertEqual(features.pos_detail1, 'common-noun')
        # Nouns don't have conjugation
        self.assertEqual(features.conjugated_type, '')
        self.assertEqual(features.conjugated_form, '')

    def test_adjective_extraction(self):
        """Test extracting features from an adjective."""
        text = "高い"
        kotogram = self.parser.japanese_to_kotogram(text)
        tokens = split_kotogram(kotogram)

        self.assertEqual(len(tokens), 1)
        features = extract_token_features(tokens[0])

        self.assertEqual(features.surface, '高い')
        self.assertEqual(features.pos, 'adj')
        self.assertEqual(features.pos_detail1, 'general')
        self.assertEqual(features.conjugated_type, 'i-adjective')

    def test_empty_fields_default_to_empty_string(self):
        """Test that irrelevant fields are empty strings for a noun."""
        # Use a real noun from parser
        kotogram = self.parser.japanese_to_kotogram("テスト")
        tokens = split_kotogram(kotogram)
        features = extract_token_features(tokens[0])

        self.assertEqual(features.surface, 'テスト')
        self.assertEqual(features.pos, 'noun')
        
        # Nouns should have empty conjugation fields
        self.assertEqual(features.conjugated_type, '')
        self.assertEqual(features.conjugated_form, '')
        
        # Base orth and lemma might be populated by parser (likely identical to surface for simple noun)
        # We just verify they are not None.
        self.assertIsNotNone(features.base_orth)
        self.assertIsNotNone(features.lemma)



class TestExtractTokenFeaturesEdgeCases(unittest.TestCase):
    """Test edge cases for extract_token_features."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = SudachiJapaneseParser(dict_type='full')


    def test_empty_token(self):
        """Test handling of empty token."""
        features = extract_token_features("")
        # Should return dictionary with all empty values
        self.assertEqual(features.surface, '')
        self.assertEqual(features.pos, '')

    def test_malformed_token_no_markers(self):
        """Test handling of token without markers."""
        features = extract_token_features("テスト")
        # Should return dictionary with all empty values
        self.assertEqual(features.surface, '')
        self.assertEqual(features.pos, '')

    def test_complex_conjugated_verb_parsed(self):
        """Test verb with multiple conjugation details using parser."""
        # "食べ" in "食べます" is a conjunctive form
        kotogram = self.parser.japanese_to_kotogram("食べます")
        tokens = split_kotogram(kotogram)
        # First token is 食べ
        token = tokens[0]
        features = extract_token_features(token)

        self.assertEqual(features.surface, '食べ')
        self.assertEqual(features.pos, 'verb')
        # Check expected values for distinct fields
        self.assertIn('ichidan', features.conjugated_type)  # e.g., ichidan or lower-ichidan-ba
        self.assertEqual(features.conjugated_form, 'continuative')
        self.assertEqual(features.lemma, '食べる')





if __name__ == '__main__':
    unittest.main()
