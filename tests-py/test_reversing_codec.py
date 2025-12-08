"""Tests for ReversingCodec."""

import unittest
from kotogram import ReversingCodec


class TestReversingCodec(unittest.TestCase):
    """Test cases for ReversingCodec."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.codec = ReversingCodec()

    def test_encode_simple_string(self) -> None:
        """Test encoding a simple string."""
        result = self.codec.encode("hello")
        self.assertEqual(result, "olleh")

    def test_decode_simple_string(self) -> None:
        """Test decoding a simple string."""
        result = self.codec.decode("olleh")
        self.assertEqual(result, "hello")

    def test_encode_decode_roundtrip(self) -> None:
        """Test that encode followed by decode returns original string."""
        original = "kotogram"
        encoded = self.codec.encode(original)
        decoded = self.codec.decode(encoded)
        self.assertEqual(decoded, original)

    def test_encode_empty_string(self) -> None:
        """Test encoding an empty string."""
        result = self.codec.encode("")
        self.assertEqual(result, "")

    def test_decode_empty_string(self) -> None:
        """Test decoding an empty string."""
        result = self.codec.decode("")
        self.assertEqual(result, "")

    def test_encode_palindrome(self) -> None:
        """Test encoding a palindrome."""
        palindrome = "racecar"
        result = self.codec.encode(palindrome)
        self.assertEqual(result, palindrome)

    def test_encode_with_spaces(self) -> None:
        """Test encoding a string with spaces."""
        result = self.codec.encode("hello world")
        self.assertEqual(result, "dlrow olleh")

    def test_encode_with_unicode(self) -> None:
        """Test encoding a string with unicode characters."""
        result = self.codec.encode("hello 世界")
        self.assertEqual(result, "界世 olleh")


if __name__ == "__main__":
    unittest.main()
