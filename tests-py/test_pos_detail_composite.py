"""Tests for vocab string generation consistency.

This test ensures the vocabulary building in label.py produces the same
vocab strings as the tokenizer encoding. This was added to catch
regressions like the one where label.py was missing conjugated_form.
"""

from kotogram.kotogram import TokenFeatures
from kotogram.tokenizer import Tokenizer, get_vocab_strings


class TestGetVocabStringsConsistency:
    """Tests that get_vocab_strings produces correct vocab strings."""

    def test_pos_detail_1_includes_conjugated_form(self):
        """Verify pos_detail_1 composite includes conjugated_form when present."""
        features = TokenFeatures(
            surface="走れ",
            pos="verb",
            pos_detail_1="general",
            pos_detail_2="",
            pos_detail_3="",
            conjugated_type="godan-ra",
            conjugated_form="imperative",
            base_orth="走る",
            lemma="走る",
            reading="ハシレ",
            reading_gram="はしれ",
        )
        vocab_strings = get_vocab_strings(features)
        assert vocab_strings["pos_detail_1"] == "verb:general:imperative"

    def test_pos_detail_1_without_conjugated_form(self):
        """Verify pos_detail_1 composite works without conjugated_form."""
        features = TokenFeatures(
            surface="が",
            pos="particle",
            pos_detail_1="case-particle",
            pos_detail_2="",
            pos_detail_3="",
            conjugated_type="",
            conjugated_form="",
            base_orth="が",
            lemma="が",
            reading="ガ",
            reading_gram="が",
        )
        vocab_strings = get_vocab_strings(features)
        assert vocab_strings["pos_detail_1"] == "particle:case-particle"

    def test_pos_detail_2_composite(self):
        """Verify pos_detail_2 composite is correctly built."""
        features = TokenFeatures(
            surface="太郎",
            pos="noun",
            pos_detail_1="proper-noun",
            pos_detail_2="person-name",
            pos_detail_3="",
            conjugated_type="",
            conjugated_form="",
            base_orth="太郎",
            lemma="太郎",
            reading="タロウ",
            reading_gram="たろう",
        )
        vocab_strings = get_vocab_strings(features)
        assert vocab_strings["pos_detail_2"] == "noun:proper-noun:person-name"

    def test_raw_fields_passed_through(self):
        """Verify non-composite fields are passed through unchanged."""
        features = TokenFeatures(
            surface="走れ",
            pos="verb",
            pos_detail_1="general",
            pos_detail_2="",
            pos_detail_3="",
            conjugated_type="godan-ra",
            conjugated_form="imperative",
            base_orth="走る",
            lemma="走る",
            reading="ハシレ",
            reading_gram="はしれ",
        )
        vocab_strings = get_vocab_strings(features)
        assert vocab_strings["pos"] == "verb"
        assert vocab_strings["conjugated_type"] == "godan-ra"
        assert vocab_strings["reading_gram"] == "はしれ"

    def test_tokenizer_uses_get_vocab_strings(self):
        """Verify tokenizer encoding uses get_vocab_strings.

        This is the key regression test: if label.py and tokenizer.py
        produce different vocab strings, vocabulary lookup will fail.
        """
        tokenizer = Tokenizer()

        # Track what strings are passed to get_id
        strings_passed: dict = {}
        original_get_id = tokenizer.get_id

        def tracked_get_id(field: str, value: str) -> int:
            if value not in ("<CLS>", "<PAD>", "<UNK>", ""):
                strings_passed[field] = value
            return original_get_id(field, value)

        tokenizer.get_id = tracked_get_id

        features = TokenFeatures(
            surface="走れ",
            pos="verb",
            pos_detail_1="general",
            pos_detail_2="",
            pos_detail_3="",
            conjugated_type="godan-ra",
            conjugated_form="imperative",
            base_orth="走る",
            lemma="走る",
            reading="ハシレ",
            reading_gram="はしれ",
        )

        tokenizer.encode_features([features])

        # Tokenizer should use same strings as get_vocab_strings
        expected = get_vocab_strings(features)
        for field in ["pos", "pos_detail_1", "conjugated_type", "reading_gram"]:
            assert strings_passed.get(field) == expected[field], (
                f"Mismatch for {field}: tokenizer passed '{strings_passed.get(field)}' "
                f"but get_vocab_strings returns '{expected[field]}'"
            )

    def test_label_py_imports_get_vocab_strings(self):
        """Verify label.py imports get_vocab_strings from tokenizer."""
        from scripts.label import get_vocab_strings as label_fn

        # Verify it's the same function (not a copy)
        assert label_fn is get_vocab_strings, (
            "label.py should import get_vocab_strings from tokenizer, "
            "not define its own"
        )
