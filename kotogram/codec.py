"""Abstract base class for codecs."""

from abc import ABC, abstractmethod


class Codec(ABC):
    """Base codec interface for encoding and decoding strings."""

    @abstractmethod
    def encode(self, text: str) -> str:
        """Encode a string.

        Args:
            text: The string to encode

        Returns:
            The encoded string
        """
        pass

    @abstractmethod
    def decode(self, text: str) -> str:
        """Decode a string.

        Args:
            text: The string to decode

        Returns:
            The decoded string
        """
        pass
