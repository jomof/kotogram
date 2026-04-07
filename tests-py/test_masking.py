import unittest

from kotogram.japanese_parser import KotogramFormat
from kotogram.kotogram import (
    extract_token_features,
    kotogram_to_japanese,
    split_kotogram,
)
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.tokenizer import get_vocab_strings


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.parser = SudachiJapaneseParser()

    def test_basic_masking(self):
        """Test that a given name surface is replaced with the exemplar."""
        # 花子 (Hanako) is a given name -> exemplar is 太郎
        text = "花子が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        # Surface replaced with exemplar
        self.assertIn("ˢ太郎", kotogram)
        self.assertNotIn("ˢ花子", kotogram)
        # reading_gram carries the mask tag
        self.assertIn("ᵍ<given-name>", kotogram)

    def test_round_trip_masked(self):
        """Test round trip reconstruction uses exemplar surface."""
        text = "花子が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        reconstructed = kotogram_to_japanese(kotogram)

        # Reconstructed text uses the exemplar surface, not the original
        self.assertEqual(reconstructed, "太郎が走る")

    def test_pos_consistency(self):
        """Test that the masked token retains POS and uses exemplar surface."""
        original_text = "花子が走る"
        original_kotogram = self.parser.japanese_to_kotogram(original_text)
        orig_tokens = split_kotogram(original_kotogram)
        orig_features = extract_token_features(orig_tokens[0])
        self.assertEqual(orig_features.surface, "花子")
        self.assertEqual(orig_features.pos_detail_3, "given-name")

        masked_kotogram = self.parser.japanese_to_kotogram(
            original_text, fmt=KotogramFormat.TRAINING_MASK
        )
        masked_tokens = split_kotogram(masked_kotogram)
        masked_features = extract_token_features(masked_tokens[0])

        # Surface replaced with exemplar
        self.assertEqual(masked_features.surface, "太郎")

        # POS tags preserved from original
        self.assertEqual(masked_features.pos, orig_features.pos)
        self.assertEqual(masked_features.pos_detail_1, orig_features.pos_detail_1)
        self.assertEqual(masked_features.pos_detail_2, orig_features.pos_detail_2)
        self.assertEqual(masked_features.pos_detail_3, orig_features.pos_detail_3)

        self.assertEqual(masked_features.reading, "")
        self.assertEqual(masked_features.reading_gram, "<given-name>")
        self.assertEqual(masked_features.lemma, "")
        self.assertNotIn("ᵈ", masked_tokens[0])

    def test_common_noun_not_masked(self):
        """Test that common nouns are NOT masked."""
        # 猫 (Neko/Cat) is a common noun
        text = "猫が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        # Should remain unchanged
        self.assertIn("ˢ猫", kotogram)
        self.assertNotIn("ˢ<given-name>", kotogram)
        self.assertNotIn("ˢ<proper-noun>", kotogram)

        reconstructed = kotogram_to_japanese(kotogram)
        self.assertEqual(reconstructed, "猫が走る")

    def test_surface_vocab_uses_exemplar(self):
        """Surface vocab uses the exemplar surface, not the original or mask tag."""
        text = "花子が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        tokens = split_kotogram(kotogram)
        features = extract_token_features(tokens[0])

        vocab_strings = get_vocab_strings(features)
        # Surface vocab now gets the exemplar (太郎), not the original (花子)
        self.assertEqual(vocab_strings["surface"], "太郎")
        self.assertEqual(features.reading_gram, "<given-name>")

    def test_multiple_names(self):
        """Test masking multiple given names in one sentence."""
        text = "花子と次郎が遊ぶ"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        reconstructed = kotogram_to_japanese(kotogram)
        # Both given names replaced with the exemplar 太郎
        self.assertEqual(reconstructed, "太郎と太郎が遊ぶ")
        self.assertIn("ᵍ<given-name>", kotogram)

    def test_merge_prevention(self):
        """Regression test for surnames merging -- exemplars must stay separate tokens."""
        # "渡辺五郎" -> 2 tokens: 渡辺(Surname) + 五郎(Given)
        text = "こちらは渡辺五郎です。"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        # Surfaces replaced with exemplars (田中 for surname, 太郎 for given-name)
        self.assertIn("ˢ田中", kotogram)
        self.assertIn("ˢ太郎", kotogram)
        self.assertIn("ᵍ<surname>", kotogram)
        self.assertIn("ᵍ<given-name>", kotogram)
        reconstructed = kotogram_to_japanese(kotogram)
        self.assertEqual(reconstructed, "こちらは田中太郎です")

    def test_pos_stability(self):
        """Regression test for adjacent particle POS stability ('ka')."""
        text = "「いらっしゃ～い」「よぉ」「なんだ、啓太か・・・」"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        # Should not raise RuntimeError; given-name replaced with exemplar
        self.assertIn("ˢ太郎", kotogram)
        self.assertIn("ᵍ<given-name>", kotogram)

    def test_generic_person_masking(self):
        """Test masking of generic person names uses person-name exemplar."""
        # "ジョン" (John) is parsed as noun:proper-noun:person-name (generic)
        text = "ジョン"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        tokens = split_kotogram(kotogram)
        t0 = extract_token_features(tokens[0])

        self.assertEqual(t0.surface, "田中")  # person-name exemplar
        self.assertEqual(t0.reading, "")
        self.assertEqual(t0.reading_gram, "<person-name>")
        self.assertEqual(t0.pos_detail_1, "proper-noun")
        self.assertEqual(t0.pos_detail_2, "person-name")

    def test_generic_place_masking(self):
        """Test masking of generic place names uses place-name exemplar."""
        # "東京" (Tokyo) is noun:proper-noun:place-name (generic/general)
        # 東京 is both the original AND the exemplar, so surface stays 東京
        text = "東京"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        tokens = split_kotogram(kotogram)
        t0 = extract_token_features(tokens[0])

        self.assertEqual(t0.surface, "東京")  # exemplar == original here
        self.assertEqual(t0.reading, "")
        self.assertEqual(t0.reading_gram, "<place-name>")
        self.assertEqual(t0.pos_detail_1, "proper-noun")
        self.assertEqual(t0.pos_detail_2, "place-name")

    def test_generic_proper_noun_masking(self):
        """Test masking of generic proper nouns uses proper-noun exemplar."""
        # "トヨタ" (Toyota) is noun:proper-noun (generic)
        text = "トヨタ"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        tokens = split_kotogram(kotogram)
        t0 = extract_token_features(tokens[0])

        self.assertEqual(t0.surface, "東京")  # proper-noun exemplar
        self.assertEqual(t0.reading, "")
        self.assertEqual(t0.reading_gram, "<proper-noun>")
        self.assertEqual(t0.pos_detail_1, "proper-noun")

    def test_strict_hierarchy_assertions(self):
        """Test strict assertions verify hierarchy logic."""
        from kotogram.kotogram import Token, TokenFeatures
        from kotogram.masking import apply_training_mask

        # Case 1: Claims given-name but hierarchy logic fails
        # (e.g. wrong pos_detail_2)
        bad_given = Token(
            "Bad",
            features=TokenFeatures(
                pos="noun",
                pos_detail_1="proper-noun",
                pos_detail_2="place-name",  # Wrong category
                pos_detail_3="given-name",
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "failed hierarchy check"):
            apply_training_mask([bad_given])

        # Retry Case 2: Noun:Proper-Noun but wrong sub-cat (e.g. place-name vs surname)
        bad_surname_2 = Token(
            "Bad",
            features=TokenFeatures(
                pos="noun",
                pos_detail_1="proper-noun",
                pos_detail_2="organization",  # Mismatch
                pos_detail_3="surname",
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "failed hierarchy check"):
            apply_training_mask([bad_surname_2])

        # Case 3: Country mismatch
        bad_country = Token(
            "Bad",
            features=TokenFeatures(
                pos="noun",
                pos_detail_1="proper-noun",
                pos_detail_2="person-name",  # Mismatch
                pos_detail_3="country",
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "failed hierarchy check"):
            apply_training_mask([bad_country])

    def test_numeral_masking(self):
        """Test that numerals are replaced with the number exemplar."""
        # "500円" -> "500" should be replaced with "1" (number exemplar)
        text = "500円"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        tokens = split_kotogram(kotogram)
        t0_features = extract_token_features(tokens[0])

        self.assertEqual(t0_features.surface, "1")  # number exemplar
        self.assertIn(t0_features.lemma, {"", "*"})
        self.assertEqual(t0_features.reading, "")
        self.assertEqual(t0_features.reading_gram, "<number>")
        self.assertEqual(t0_features.pos, "noun")
        self.assertEqual(t0_features.pos_detail_1, "numeral")

        # "円" is a common noun, not masked
        t1_features = extract_token_features(tokens[1])
        self.assertEqual(t1_features.surface, "円")


if __name__ == "__main__":
    unittest.main()
