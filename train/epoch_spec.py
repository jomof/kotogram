"""Epoch specification parsing utilities."""

from typing import Optional


def parse_epoch_spec(spec: Optional[str], current: int) -> Optional[int]:
    """Parse an epoch specification.

    Args:
        spec: Epoch specification string. Can be:
            - None: returns None
            - Absolute: "10" returns 10
            - Relative: "+5" returns current + 5
        current: Current completed epoch count (used for relative specs)

    Returns:
        Target epoch count, or None if spec is None
    """
    if spec is None:
        return None
    if spec.startswith("+"):
        return current + int(spec[1:])
    return int(spec)
