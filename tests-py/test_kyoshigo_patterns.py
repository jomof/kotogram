"""Unit tests for KYOSHIGO register detection edge cases.

This test file verifies the refined KYOSHIGO detection rules that were fixed
to reduce false positives from overly broad patterns.
"""

import unittest

from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from scripts.rule_based_analysis import analyze_register

# Initialize parser once
_parser = SudachiJapaneseParser()


def analyze_sentence(sentence: str):
    """Analyze a raw Japanese sentence and return detected registers."""
    kotogram = _parser.japanese_to_kotogram(sentence)
    return analyze_register(kotogram)


def has_kyoshigo(sentence: str) -> bool:
    """Helper to check if a sentence is detected as KYOSHIGO."""
    registers = analyze_sentence(sentence)
    return any("kyoshigo" in str(r).lower() for r in registers)


class TestKyoshigoKaraNePattern(unittest.TestCase):
    """Test cases for 'からね' pattern refinement.

    The rule should only trigger for 'ですからね' (formal + kara + ne),
    NOT for 'だからね' (casual + kara + ne).
    """

    def test_casual_dakara_ne_not_kyoshigo(self):
        """Casual 'だからね' should NOT be detected as KYOSHIGO."""
        # This was the original false positive from the bug report
        sentence = "それはお腹ペコペコだからね！"
        assert not has_kyoshigo(sentence), (
            "Casual 'だからね' should not trigger KYOSHIGO"
        )

    def test_formal_desukara_ne_is_kyoshigo(self):
        """Formal 'ですからね' SHOULD be detected as KYOSHIGO."""
        sentence = "これは大切ですからね、覚えなさい。"
        assert has_kyoshigo(sentence), "Formal 'ですからね' should trigger KYOSHIGO"

    def test_another_casual_dakara_ne(self):
        """Another example of casual 'だからね' that shouldn't trigger."""
        sentence = "明日は休みだからね、ゆっくりしよう。"
        assert not has_kyoshigo(sentence), (
            "Casual 'だからね' in casual context should not trigger KYOSHIGO"
        )


class TestKyoshigoMachigaiPattern(unittest.TestCase):
    """Test cases for '間違い' pattern refinement.

    The rule should exclude:
    - '間違いなく' (undoubtedly) - adverbial usage
    - '間違った + noun' (wrong X) - casual attributive usage

    But should still detect actual correction contexts.
    """

    def test_machigainaku_not_kyoshigo(self):
        """'間違いなく' (undoubtedly) should NOT trigger KYOSHIGO."""
        sentence = "私たち５名の中で、間違いなく彼女が一番多くの言語を話せる。"
        assert not has_kyoshigo(sentence), (
            "'間違いなく' (undoubtedly) is not a correction context"
        )

    def test_machigatta_noun_not_kyoshigo(self):
        """'間違った + noun' (wrong X) should NOT trigger KYOSHIGO."""
        # Original false positive
        sentence = "彼女はまだ来ない。間違ったバスに乗ったのかもしれない。"
        assert not has_kyoshigo(sentence), (
            "'間違ったバス' (wrong bus) is casual attribution, not correction"
        )

    def test_machigatta_answer_not_kyoshigo(self):
        """Another '間違った + noun' example."""
        sentence = "間違った答えを選んでしまった。"
        assert not has_kyoshigo(sentence), (
            "'間違った答え' (wrong answer) is casual usage, not instructional"
        )

    def test_actual_correction_is_kyoshigo(self):
        """Actual correction context SHOULD trigger KYOSHIGO."""
        sentence = "君の作文には、間違いが２、３あります。"
        assert has_kyoshigo(sentence), (
            "Actual correction context should trigger KYOSHIGO"
        )

    def test_correction_warning_is_kyoshigo(self):
        """Correction warning SHOULD trigger KYOSHIGO."""
        sentence = "二度と同じ間違いをしないように気を付けます。"
        assert has_kyoshigo(sentence), (
            "Correction/warning context should trigger KYOSHIGO"
        )


class TestKyoshigoReliablePatterns(unittest.TestCase):
    """Test cases for reliable KYOSHIGO patterns that should still work."""

    def test_nasai_imperative(self):
        """'なさい' imperative should trigger KYOSHIGO."""
        sentence = "さあ、夕食を食べなさい。"
        assert has_kyoshigo(sentence), "'なさい' is a reliable KYOSHIGO marker"

    def test_shukudai_report_not_kyoshigo(self):
        """'宿題' (homework) in reportive context should NOT trigger KYOSHIGO."""
        sentence = "ベスは怠け者の彼氏に、歴史の宿題をやってくれと頼まれました。"
        assert not has_kyoshigo(sentence), (
            "'宿題' in reportive context is not instructional"
        )

    def test_shukudai_instruction_is_kyoshigo(self):
        """'宿題' as instruction should trigger KYOSHIGO."""
        sentence = "今日の宿題はこれです。"
        assert has_kyoshigo(sentence), "'宿題は' + formal ending is instructional"

    def test_sensei_teacher(self):
        """'先生' (teacher) should trigger KYOSHIGO."""
        sentence = "「明日テストをします」と先生は言った。"
        assert has_kyoshigo(sentence), "'先生' + 'テスト' indicates classroom context"

    def test_test_keyword(self):
        """'テスト' (test) should trigger KYOSHIGO."""
        sentence = "明日は数学のテストがあります。"
        assert has_kyoshigo(sentence), "'テスト' indicates academic/classroom context"
