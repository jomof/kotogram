"""Legacy wrappers for model-based analysis functions.

These utilities wrap the new consolidated grammar() function to provide
the old individual method interface if needed for scripts or tests.
"""

from typing import Optional
from kotogram import grammar, FormalityLevel

def formality(kotogram: str) -> FormalityLevel:
    """Legacy wrapper for formality analysis."""
    return grammar(kotogram).formality

def gender(kotogram: str) -> Optional[float]:
    """Legacy wrapper for gender analysis."""
    res = grammar(kotogram)
    if res.gender_is_pragmatic:
        return res.gender_score
    return None


