"""Legacy wrappers for model-based analysis functions.

These utilities wrap the new consolidated grammar() function to provide
the old individual method interface if needed for scripts or tests.
"""

from typing import Optional, Set, Tuple
from kotogram import grammar, FormalityLevel, RegisterLevel

def formality(kotogram: str) -> FormalityLevel:
    """Legacy wrapper for formality analysis."""
    return grammar(kotogram).formality

def gender(kotogram: str) -> Optional[float]:
    """Legacy wrapper for gender analysis."""
    res = grammar(kotogram)
    if res.gender_is_pragmatic:
        return res.gender_score
    return None

def register(kotogram: str) -> Set[RegisterLevel]:
    """Legacy wrapper for register analysis."""
    return grammar(kotogram).registers

def style(kotogram: str) -> Tuple[FormalityLevel, Optional[float], Set[RegisterLevel], bool]:
    """Legacy wrapper for consolidated style analysis."""
    res = grammar(kotogram)
    gender_val = res.gender_score if res.gender_is_pragmatic else None
    return (
        res.formality,
        gender_val,
        res.registers,
        res.is_grammatic
    )

def grammaticality(kotogram: str) -> bool:
    """Legacy wrapper for grammaticality analysis."""
    return grammar(kotogram).is_grammatic
