"""Hygiene tests for content canonicalization and exemplar consistency."""

import unittest

from kotogram.masking import (
    PRESERVED_READING_MASKS,
    SURFACE_EXEMPLARS,
    canonicalize_sentence,
)
from scripts.integrity import DataIntegrityException


class TestExemplarHygiene(unittest.TestCase):
    """Verify SURFACE_EXEMPLARS are consistent with masking categories."""

    def test_exemplars_cover_all_masks(self):
        """Every PRESERVED_READING_MASKS entry has a SURFACE_EXEMPLARS mapping."""
        for mask in PRESERVED_READING_MASKS:
            self.assertIn(
                mask,
                SURFACE_EXEMPLARS,
                f"Mask {mask!r} has no exemplar in SURFACE_EXEMPLARS",
            )

    def test_exemplar_surfaces_exist_in_chive(self):
        """Each exemplar surface string exists in the chiVe vocab."""
        from train.chive import download_chive, load_chive_vocab_set

        download_chive()
        vocab = load_chive_vocab_set()
        for mask, surface in SURFACE_EXEMPLARS.items():
            self.assertIn(
                surface,
                vocab,
                f"Exemplar {surface!r} (for {mask}) not in chiVe",
            )

    def test_canonicalize_idempotent(self):
        """canonicalize_sentence is idempotent on sample sentences."""
        samples = [
            "田中さんは東京に住んでいます",
            "1円の価値",
            "この本はとても面白い",
            "太郎と花子が遊んだ",
        ]
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

        parser = SudachiJapaneseParser(validate=False)
        for s in samples:
            c1 = canonicalize_sentence(s, _parser=parser)
            c2 = canonicalize_sentence(c1, _parser=parser)
            self.assertEqual(c1, c2, f"Not idempotent: {s!r} -> {c1!r} -> {c2!r}")

    def test_exemplar_sentences_are_canonical(self):
        """A sentence using only exemplar surfaces canonicalizes to itself."""
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

        parser = SudachiJapaneseParser(validate=False)
        exemplar_sentences = [
            "田中さんは東京に住んでいます",
            "1円で買える",
            "日本は東京が首都です",
        ]
        for s in exemplar_sentences:
            c = canonicalize_sentence(s, _parser=parser)
            self.assertEqual(s, c, f"Exemplar sentence changed: {s!r} -> {c!r}")

    def test_data_integrity_exception_importable(self):
        """DataIntegrityException is importable and is an Exception subclass."""
        self.assertTrue(issubclass(DataIntegrityException, Exception))
        exc = DataIntegrityException("[test] sample error")
        self.assertIn("[test]", str(exc))


class TestCanonicalDedupLogic(unittest.TestCase):
    """Verify canonicalization produces correct dedup behavior."""

    def setUp(self):
        from kotogram.sudachi_japanese_parser import SudachiJapaneseParser

        self.parser = SudachiJapaneseParser(validate=False)

    def test_number_variants_match(self):
        """Sentences differing only by numbers produce the same canonical form."""
        a = canonicalize_sentence("酒やめて500日", _parser=self.parser)
        b = canonicalize_sentence("酒やめて1246日", _parser=self.parser)
        self.assertEqual(a, b)

    def test_name_variants_match(self):
        """Sentences differing only by names produce the same canonical form."""
        a = canonicalize_sentence("佐藤さんは大阪に住んでいます", _parser=self.parser)
        b = canonicalize_sentence("田中さんは東京に住んでいます", _parser=self.parser)
        self.assertEqual(a, b)

    def test_no_mask_sentence_unchanged(self):
        """Sentences with no masked tokens are unchanged by canonicalization."""
        s = "この本はとても面白い"
        self.assertEqual(canonicalize_sentence(s, _parser=self.parser), s)


if __name__ == "__main__":
    unittest.main()
