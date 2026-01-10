"""Tests for parse_epoch_spec function in train.epoch_spec module."""

import unittest

from train.epoch_spec import parse_epoch_spec


class TestParseEpochSpec(unittest.TestCase):
    """Tests for the parse_epoch_spec helper function."""

    def test_none_value(self) -> None:
        """None input returns None."""
        result = parse_epoch_spec(None, current=5)
        self.assertIsNone(result)

    def test_absolute_value(self) -> None:
        """Absolute value '10' returns 10 regardless of current."""
        result = parse_epoch_spec("10", current=3)
        self.assertEqual(result, 10)

    def test_absolute_zero(self) -> None:
        """Absolute value '0' returns 0."""
        result = parse_epoch_spec("0", current=5)
        self.assertEqual(result, 0)

    def test_relative_additive(self) -> None:
        """Relative value '+5' with current=3 returns 8."""
        result = parse_epoch_spec("+5", current=3)
        self.assertEqual(result, 8)

    def test_relative_zero(self) -> None:
        """Relative value '+0' with current=5 returns 5 (no change)."""
        result = parse_epoch_spec("+0", current=5)
        self.assertEqual(result, 5)

    def test_relative_from_zero_current(self) -> None:
        """Relative value '+3' with current=0 returns 3."""
        result = parse_epoch_spec("+3", current=0)
        self.assertEqual(result, 3)

    def test_large_relative_increment(self) -> None:
        """Relative value '+100' with current=6 returns 106."""
        result = parse_epoch_spec("+100", current=6)
        self.assertEqual(result, 106)


if __name__ == "__main__":
    unittest.main()
