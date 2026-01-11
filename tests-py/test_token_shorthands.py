"""Unit tests for kotogram token shorthand compression."""

import pytest

from kotogram.kotogram import TOKEN_SHORTHANDS, extract_token_features, split_kotogram
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser


class TestTokenShorthands:
    """Test that shorthand tokens are correctly generated and expanded."""

    @pytest.fixture(scope="class")
    def parser(self) -> SudachiJapaneseParser:
        """Shared parser instance for all tests."""
        return SudachiJapaneseParser()

    # Test sentences containing each shorthand token
    @pytest.mark.parametrize(
        "sentence,expected_shorthand",
        [
            ("うん。", "。"),  # Period
            ("猫は可愛い", "は"),  # Topic particle
            ("猫を見た", "を"),  # Object particle
            ("学校に行く", "に"),  # Dative/locative particle
            ("猫の名前", "の"),  # Genitive particle
            ("食べた", "た"),  # Past tense aux
            ("食べて", "て"),  # Conjunctive particle
            ("猫が好き", "が"),  # Subject particle
            ("笑、何？", "、"),  # Comma
            ("楢です。", "です"),  # Desu copula
            ("出発した", "し"),  # Suru continuative
            ("噛まれると", "と"),  # To quotative/case particle
            ("嫌だ！", "だ"),  # Da copula
            ("知ってるわ", "わ"),  # Sentence-final particle
        ],
    )
    def test_shorthand_in_output(
        self, parser: SudachiJapaneseParser, sentence: str, expected_shorthand: str
    ) -> None:
        """Verify that kotogram output contains shorthand (not full token)."""
        kotogram = parser.japanese_to_kotogram(sentence)

        # The shorthand should appear as a standalone character, not inside ⌈⌉
        assert expected_shorthand in kotogram, (
            f"Shorthand '{expected_shorthand}' not found in kotogram: {kotogram}"
        )

        # Split and verify the shorthand is returned as a token
        tokens = split_kotogram(kotogram)
        assert expected_shorthand in tokens, (
            f"Shorthand '{expected_shorthand}' not in tokens: {tokens}"
        )

    @pytest.mark.parametrize("shorthand,full_token", TOKEN_SHORTHANDS.items())
    def test_shorthand_expandable(self, shorthand: str, full_token: str) -> None:
        """Verify that each shorthand expands to correct features."""
        # Extract features from shorthand
        features_from_shorthand = extract_token_features(shorthand)

        # Extract features from full token
        features_from_full = extract_token_features(full_token)

        # They should be identical
        assert features_from_shorthand == features_from_full, (
            f"Features mismatch for '{shorthand}':\n"
            f"  From shorthand: {features_from_shorthand}\n"
            f"  From full:      {features_from_full}"
        )

    @pytest.mark.parametrize("shorthand,full_token", TOKEN_SHORTHANDS.items())
    def test_shorthand_surface_matches(self, shorthand: str, full_token: str) -> None:  # pylint: disable=unused-argument
        """Verify that shorthand surface equals the shorthand itself."""
        features = extract_token_features(shorthand)
        assert features.surface == shorthand, (
            f"Surface '{features.surface}' doesn't match shorthand '{shorthand}'"
        )

    def test_period_shorthand_structure(self, parser: SudachiJapaneseParser) -> None:
        """Test that period produces correct shorthand structure."""
        kotogram = parser.japanese_to_kotogram("終わり。")
        tokens = split_kotogram(kotogram)

        # Last token should be the period shorthand
        assert tokens[-1] == "。", f"Expected '。' as last token, got: {tokens[-1]}"

        # Features should be correctly extracted
        features = extract_token_features("。")
        assert features.surface == "。"
        assert features.pos == "aux-symbol"
        assert features.pos_detail_1 == "period"

    def test_particle_shorthand_structure(self, parser: SudachiJapaneseParser) -> None:
        """Test that particles produce correct shorthand structure."""
        kotogram = parser.japanese_to_kotogram("猫は犬を見た")
        tokens = split_kotogram(kotogram)

        # Check that shorthand particles are in tokens
        assert "は" in tokens, f"Expected 'は' in tokens: {tokens}"
        assert "を" in tokens, f"Expected 'を' in tokens: {tokens}"
        assert "た" in tokens, f"Expected 'た' in tokens: {tokens}"

        # Verify features
        ha_features = extract_token_features("は")
        assert ha_features.pos == "particle"
        assert ha_features.pos_detail_1 == "binding-particle"

        wo_features = extract_token_features("を")
        assert wo_features.pos == "particle"
        assert wo_features.pos_detail_1 == "case-particle"

    def test_all_shorthands_are_short(self) -> None:
        """Verify all shorthand keys are very short (1-2 characters)."""
        for shorthand in TOKEN_SHORTHANDS:
            assert len(shorthand) <= 2, (
                f"Shorthand '{shorthand}' is too long (max 2 chars)"
            )

    def test_shorthand_count(self) -> None:
        """Verify expected number of shorthands."""
        assert len(TOKEN_SHORTHANDS) == 23, (
            f"Expected 23 shorthands, got {len(TOKEN_SHORTHANDS)}"
        )
