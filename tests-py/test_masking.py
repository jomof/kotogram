import unittest

from kotogram.japanese_parser import KotogramFormat
from kotogram.kotogram import (
    extract_token_features,
    kotogram_to_japanese,
    split_kotogram,
)
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser


class TestMasking(unittest.TestCase):
    def setUp(self):
        self.parser = SudachiJapaneseParser(dict_type="full")

    def test_basic_masking(self):
        """Test that a given name is masked to '<given-name>'."""
        # 花子 (Hanako) is a given name
        text = "花子が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        # Check surface form in kotogram string
        self.assertIn("ˢ<given-name>", kotogram)
        self.assertNotIn("ˢ花子", kotogram)

    def test_round_trip_masked(self):
        """Test round trip reconstruction of a masked sentence."""
        text = "花子が走る"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        reconstructed = kotogram_to_japanese(kotogram)

        self.assertEqual(reconstructed, "<given-name>が走る")

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
        masked_features = extract_token_features(masked_tokens[0])
        self.assertEqual(masked_features.surface, "<given-name>")

        # 3. Assert POS tags are identical
        self.assertEqual(masked_features.pos, orig_features.pos)
        self.assertEqual(masked_features.pos_detail_1, orig_features.pos_detail_1)
        self.assertEqual(masked_features.pos_detail_2, orig_features.pos_detail_2)
        # Note: pos_detail_3 should also be "given-name" for placeholder
        self.assertEqual(masked_features.pos_detail_3, orig_features.pos_detail_3)

        # 4. Assert reading is stripped (defaults to empty string in TokenFeatures if missing)
        self.assertEqual(masked_features.reading, "")

        # 5. Assert lemma is stripped (defaults to *)
        # Actually in kotogram parser: if feature.lemma == "*", it sets it to surface.
        # But here we want to assert the underlying token has lemma="*"
        # Let's check the kotogram string for "ᵈ*"
        self.assertIn("ᵈ*", masked_tokens[0])

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

    def test_multiple_names(self):
        """Test masking multiple given names in one sentence."""
        text = "花子と次郎が遊ぶ"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        reconstructed = kotogram_to_japanese(kotogram)
        self.assertEqual(reconstructed, "<given-name>と<given-name>が遊ぶ")

    def test_merge_prevention(self):
        """Regression test for surnames merging (e.g. 渡辺太郎)."""
        # "渡辺五郎" -> 2 tokens: 渡辺(Surname) + 五郎(Given)
        text = "こちらは渡辺五郎です。"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )

        # Verify surface replacement
        self.assertIn("ˢ<surname>", kotogram)
        self.assertIn("ˢ<given-name>", kotogram)
        reconstructed = kotogram_to_japanese(kotogram)
        self.assertEqual(reconstructed, "こちらは<surname><given-name>です。")

    def test_pos_stability(self):
        """Regression test for adjacent particle POS stability ('ka')."""
        text = "「いらっしゃ～い」「よぉ」「なんだ、啓太か・・・」"
        kotogram = self.parser.japanese_to_kotogram(
            text, fmt=KotogramFormat.TRAINING_MASK
        )
        # Should not raise RuntimeError
        self.assertIn("ˢ<given-name>", kotogram)

    def test_generic_person_masking(self):
        """Test masking of generic person names (no given/surname detail)."""
        from kotogram.kotogram import Token
        from kotogram.masking import apply_training_mask

        # Manually construct generic person token
        # noun:proper-noun:person-name:general (or empty)
        t = Token(
            "Somebody",
            features={
                "pos": "noun",
                "pos_detail_1": "proper-noun",
                "pos_detail_2": "person-name",
                "pos_detail_3": "general",
            },
        )
        tokens = [t]
        apply_training_mask(tokens)
        self.assertEqual(tokens[0].surface, "<person-name>")
        self.assertEqual(tokens[0].features["lemma"], "*")

    def test_generic_place_masking(self):
        """Test masking of generic place names (no country detail)."""
        from kotogram.kotogram import Token
        from kotogram.masking import apply_training_mask

        t = Token(
            "Somewhere",
            features={
                "pos": "noun",
                "pos_detail_1": "proper-noun",
                "pos_detail_2": "place-name",
            },
        )
        tokens = [t]
        apply_training_mask(tokens)
        self.assertEqual(tokens[0].surface, "<place-name>")

    def test_generic_proper_noun_masking(self):
        """Test masking of generic proper nouns (orgs, etc)."""
        from kotogram.kotogram import Token
        from kotogram.masking import apply_training_mask

        t = Token(
            "SomeCorp",
            features={
                "pos": "noun",
                "pos_detail_1": "proper-noun",
                "pos_detail_2": "organization",
            },
        )
        tokens = [t]
        apply_training_mask(tokens)
        self.assertEqual(tokens[0].surface, "<proper-noun>")

    def test_strict_hierarchy_assertions(self):
        """Test strict assertions verify hierarchy logic."""
        from kotogram.kotogram import Token
        from kotogram.masking import apply_training_mask

        # Case 1: Claims given-name but hierarchy logic fails
        # (e.g. wrong pos_detail_2)
        bad_given = Token(
            "Bad",
            features={
                "pos": "noun",
                "pos_detail_1": "proper-noun",
                "pos_detail_2": "place-name",  # Wrong category
                "pos_detail_3": "given-name",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "failed hierarchy check"):
            apply_training_mask([bad_given])

        # Retry Case 2: Noun:Proper-Noun but wrong sub-cat (e.g. place-name vs surname)
        bad_surname_2 = Token(
            "Bad",
            features={
                "pos": "noun",
                "pos_detail_1": "proper-noun",
                "pos_detail_2": "organization",  # Mismatch
                "pos_detail_3": "surname",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "failed hierarchy check"):
            apply_training_mask([bad_surname_2])

        # Case 3: Country mismatch
        bad_country = Token(
            "Bad",
            features={
                "pos": "noun",
                "pos_detail_1": "proper-noun",
                "pos_detail_2": "person-name",  # Mismatch
                "pos_detail_3": "country",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "failed hierarchy check"):
            apply_training_mask([bad_country])


if __name__ == "__main__":
    unittest.main()
