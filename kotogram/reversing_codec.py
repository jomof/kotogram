"""Implementation of a reversing codec."""

from .codec import Codec


class ReversingCodec(Codec):
    """A codec that reverses strings for both encoding and decoding."""

    def encode(self, text: str) -> str:
        """Encode a string by reversing it.

        Args:
            text: The string to encode

        Returns:
            The reversed string
        """
        return text[::-1]

    def decode(self, text: str) -> str:
        """Decode a string by reversing it.

        Args:
            text: The string to decode

        Returns:
            The reversed string
        """
        return text[::-1]
