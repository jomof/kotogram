"""Unit tests for label.py Phase 2 encoding consistency."""

import pytest

from kotogram.kotogram import extract_token_features, split_kotogram
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.tokenizer import FEATURE_FIELDS, Tokenizer, get_vocab_strings


def _build_tokenizer_from_sentences(
    parser: SudachiJapaneseParser, sentences: list[str]
) -> Tokenizer:
    """Build a minimal Tokenizer with vocab from the given sentences."""
    tokenizer = Tokenizer()
    for sent in sentences:
        kotogram = parser.japanese_to_kotogram(sent)
        for token in split_kotogram(kotogram):
            feat = extract_token_features(token)
            vs = get_vocab_strings(feat)
            for field in FEATURE_FIELDS:
                vocab = tokenizer.field_vocabs[field]
                val = vs[field]
                if val and val not in vocab:
                    vocab[val] = len(vocab)
    return tokenizer


class TestPhase2EncodingConsistency:
    """Test that Phase 2 encoding uses the same vocab strings as Phase 1."""

    @pytest.fixture(scope="class")
    def parser(self) -> SudachiJapaneseParser:
        """Shared parser instance."""
        return SudachiJapaneseParser()

    @pytest.fixture(scope="class")
    def tokenizer(self, parser: SudachiJapaneseParser) -> Tokenizer:
        """Build a tokenizer from test sentences (no external files needed)."""
        return _build_tokenizer_from_sentences(
            parser, ["猫は可愛い。", "彼女は学校に行った。", "猫を見た"]
        )

    def test_compound_1_uses_composite_tokens(
        self, parser: SudachiJapaneseParser, tokenizer: Tokenizer
    ) -> None:
        """Verify compound_1 encoding uses composite tokens, not raw values.

        This test catches the bug where Phase 2 used getattr(token_feat, field)
        instead of get_vocab_strings(token_feat)[field], causing all compound_1
        values to become UNK.
        """
        kotogram = parser.japanese_to_kotogram("猫は可愛い。")
        tokens = split_kotogram(kotogram)

        unk_count = 0
        total_count = 0

        for token in tokens:
            token_feat = extract_token_features(token)
            vocab_strings = get_vocab_strings(token_feat)

            # Check that compound_1 composite token is in vocab
            composite_val = vocab_strings["compound_1"]
            if composite_val:  # Skip empty values
                fid = tokenizer.get_id("compound_1", composite_val)
                total_count += 1
                if fid == 1:  # UNK_ID
                    unk_count += 1

        # Most tokens should NOT be UNK
        assert total_count > 0, "No compound_1 tokens found"
        assert unk_count < total_count, (
            f"All {total_count} compound_1 tokens are UNK. "
            "This indicates Phase 2 encoding is not using composite tokens."
        )

    def test_vocab_strings_differ_from_raw_fields(
        self, parser: SudachiJapaneseParser
    ) -> None:
        """Verify get_vocab_strings produces different values than raw fields.

        For compound_1, the composite token includes POS prefix (e.g., 'particle:case-particle')
        whereas the raw field is just 'case-particle'.
        """
        kotogram = parser.japanese_to_kotogram("猫を見た")
        tokens = split_kotogram(kotogram)

        found_difference = False
        for token in tokens:
            token_feat = extract_token_features(token)
            vocab_strings = get_vocab_strings(token_feat)

            raw_val = token_feat.pos_detail_1
            composite_val = vocab_strings["compound_1"]

            if raw_val and composite_val and raw_val != composite_val:
                found_difference = True
                # Composite should contain the raw value
                assert raw_val in composite_val, (
                    f"Composite '{composite_val}' should contain raw '{raw_val}'"
                )

        assert found_difference, (
            "Expected to find tokens where composite differs from raw compound_1"
        )

    def test_encoding_roundtrip_consistency(
        self, parser: SudachiJapaneseParser, tokenizer: Tokenizer
    ) -> None:
        """Verify tokenizer.encode uses get_vocab_strings internally."""
        kotogram = parser.japanese_to_kotogram("彼女は学校に行った。")

        # Use the tokenizer's encode method (which should use get_vocab_strings)
        encoded = tokenizer.encode(kotogram)

        # Manually encode using get_vocab_strings (the correct way)
        tokens = split_kotogram(kotogram)
        manual_ids = []
        for token in tokens:
            token_feat = extract_token_features(token)
            vocab_strings = get_vocab_strings(token_feat)
            fid = tokenizer.get_id("compound_1", vocab_strings["compound_1"])
            manual_ids.append(fid)

        # Skip CLS token (first element) in encoded result
        encoded_ids = encoded["compound_1"][1:]  # Skip CLS

        assert list(encoded_ids) == manual_ids, (
            f"Tokenizer.encode IDs {list(encoded_ids)} don't match "
            f"manual get_vocab_strings IDs {manual_ids}"
        )
