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
        """Test that a given name is masked to '<given-name>'."""
        # 花子 (Hanako) is a given name
        text = "花子が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        # Check surface form in kotogram string (SHOULD BE PRESERVED)
        self.assertIn("ˢ花子", kotogram)
        self.assertNotIn("ˢ<given-name>", kotogram)
        # Check reading form (SHOULD BE CLEARED) -> No "ʳ<given-name>"
        self.assertNotIn("ʳ<given-name>", kotogram)
        # Check reading_gram form (SHOULD BE PRESENT via ᵍ)
        self.assertIn("ᵍ<given-name>", kotogram)

    def test_round_trip_masked(self):
        """Test round trip reconstruction of a masked sentence."""
        text = "花子が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        reconstructed = kotogram_to_japanese(kotogram)

        # Reconstructed text should use SURFACE, so it matches ORIGINAL text now
        self.assertEqual(reconstructed, "花子が走る")

    def test_pos_consistency(self):
        """Test that the masked token retains the grammatical role of the original."""
        original_text = "花子が走る"
        # 1. Parse original to find the target token
        original_kotogram = self.parser.japanese_to_kotogram(original_text)
        orig_tokens = split_kotogram(original_kotogram)

        # Identify the name token (first one)
        orig_features = extract_token_features(orig_tokens[0])
        self.assertEqual(orig_features.surface, "花子")
        self.assertEqual(orig_features.pos_detail_3, "given-name")

        # 2. Parse masked
        masked_kotogram = self.parser.japanese_to_kotogram(
            original_text, fmt=KotogramFormat.TRAINING_MASK
        )
        masked_tokens = split_kotogram(masked_kotogram)

        # Identify the masked token (first one)
        # Identify the masked token (first one)
        masked_features = extract_token_features(masked_tokens[0])
        # Surface should be preserved
        self.assertEqual(masked_features.surface, "花子")

        # 3. Assert POS tags are identical
        self.assertEqual(masked_features.pos, orig_features.pos)
        self.assertEqual(masked_features.pos_detail_1, orig_features.pos_detail_1)
        self.assertEqual(masked_features.pos_detail_2, orig_features.pos_detail_2)
        # Note: pos_detail_3 should also be "given-name" for placeholder
        self.assertEqual(masked_features.pos_detail_3, orig_features.pos_detail_3)

        # 4. Assert reading is CLEARED (empty)
        self.assertEqual(masked_features.reading, "")
        # 5. Assert reading_gram is MASKED to <given-name>
        self.assertEqual(masked_features.reading_gram, "<given-name>")

        # 5. Assert lemma is stripped (defaults to *)
        # Actually in kotogram parser: if feature.lemma == "*", it sets it to surface.
        # But here we want to assert the underlying token has lemma="*"
        # 6. Assert lemma is stripped (empty string, no marker)
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

    def test_surface_vocab_uses_masked_surface(self):
        """Surface vocab should collapse to masked token when reading_gram is masked."""
        text = "花子が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        tokens = split_kotogram(kotogram)
        features = extract_token_features(tokens[0])

        vocab_strings = get_vocab_strings(features)
        self.assertEqual(vocab_strings["surface"], "<given-name>")

        unmasked_kotogram = self.parser.japanese_to_kotogram(text)
        unmasked_tokens = split_kotogram(unmasked_kotogram)
        unmasked_features = extract_token_features(unmasked_tokens[0])
        unmasked_vocab_strings = get_vocab_strings(unmasked_features)
        self.assertEqual(unmasked_vocab_strings["surface"], "花子")

    def test_multiple_names(self):
        """Test masking multiple given names in one sentence."""
        text = "花子と次郎が遊ぶ"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        reconstructed = kotogram_to_japanese(kotogram)
        # Surface preserved
        self.assertEqual(reconstructed, "花子と次郎が遊ぶ")
        # Readings masked (no R marker, yes G marker)
        self.assertIn("ᵍ<given-name>", kotogram)

    def test_merge_prevention(self):
        """Regression test for surnames merging (e.g. 渡辺太郎)."""
        # "渡辺五郎" -> 2 tokens: 渡辺(Surname) + 五郎(Given)
        text = "こちらは渡辺五郎です。"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        # Verify surface replacement (SHOULD BE PRESERVED)
        self.assertIn("ˢ渡辺", kotogram)
        self.assertIn("ˢ五郎", kotogram)
        # Verify reading replacement (Cleared R, added G)
        self.assertIn("ᵍ<surname>", kotogram)
        self.assertIn("ᵍ<given-name>", kotogram)
        reconstructed = kotogram_to_japanese(kotogram)
        # Masking removes sentence-final punctuation for boundary noise reduction
        self.assertEqual(reconstructed, "こちらは渡辺五郎です")

    def test_pos_stability(self):
        """Regression test for adjacent particle POS stability ('ka')."""
        text = "「いらっしゃ～い」「よぉ」「なんだ、啓太か・・・」"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        # Should not raise RuntimeError
        # Should not raise RuntimeError
        self.assertIn("ˢ啓太", kotogram)
        self.assertIn("ᵍ<given-name>", kotogram)

    def test_generic_person_masking(self):
        """Test masking of generic person names (no given/surname detail)."""
        # "ジョン" (John) is parsed as noun:proper-noun:person-name (generic)
        text = "ジョン"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        tokens = split_kotogram(kotogram)
        t0 = extract_token_features(tokens[0])

        self.assertEqual(t0.surface, "ジョン")
        self.assertEqual(t0.reading, "")
        self.assertEqual(t0.reading_gram, "<person-name>")
        self.assertEqual(t0.pos_detail_1, "proper-noun")
        self.assertEqual(t0.pos_detail_2, "person-name")

    def test_generic_place_masking(self):
        """Test masking of generic place names (no country detail)."""
        # "東京" (Tokyo) is noun:proper-noun:place-name (generic/general)
        text = "東京"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        tokens = split_kotogram(kotogram)
        t0 = extract_token_features(tokens[0])

        self.assertEqual(t0.surface, "東京")
        self.assertEqual(t0.reading, "")
        self.assertEqual(t0.reading_gram, "<place-name>")
        self.assertEqual(t0.pos_detail_1, "proper-noun")
        self.assertEqual(t0.pos_detail_2, "place-name")

    def test_generic_proper_noun_masking(self):
        """Test masking of generic proper nouns (orgs, etc)."""
        # "トヨタ" (Toyota) is noun:proper-noun (generic, no detail2 in standard dict)
        # Actually my probe showed D2="" for "トヨタ"
        text = "トヨタ"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        tokens = split_kotogram(kotogram)
        t0 = extract_token_features(tokens[0])

        self.assertEqual(t0.surface, "トヨタ")
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
        """Test validation of numeral masking to <number> using real parsing."""
        # "500円" -> "500" should be masked to <number>
        text = "500円"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        tokens = split_kotogram(kotogram)
        # First token is 500
        t0_features = extract_token_features(tokens[0])

        self.assertEqual(t0_features.surface, "500")
        # Parser replaces '*' lemma with surface
        # Parser replaces '*' lemma with surface
        self.assertIn(t0_features.lemma, {"", "*"})
        self.assertEqual(t0_features.reading, "")
        self.assertEqual(t0_features.reading_gram, "<number>")
        self.assertEqual(t0_features.pos, "noun")
        self.assertEqual(t0_features.pos_detail_1, "numeral")

        # Second token "円" (counter/noun) should remain or be masked if it falls into specific rules?
        # It's a common noun / counter, usually not masked by existing rules unless it hits proper noun logic.
        # "円" is noun:common-noun:counter-possible.
        # So it stays as surface.
        t1_features = extract_token_features(tokens[1])
        self.assertEqual(t1_features.surface, "円")


if __name__ == "__main__":
    unittest.main()
