"""Validation utilities for runtime type checking."""

from typing import Any

def ensure_string(value: Any, name: str) -> None:
    """Ensure that the value is a string.
    
    Args:
        value: The value to check.
        name: The name of the parameter (for error message).
        
    Raises:
        TypeError: If value is not a string.
    """
    if not isinstance(value, str):
        raise TypeError(f"Parameter '{name}' must be a string, but got {type(value).__name__}")

def ensure_list_of_strings(value: Any, name: str) -> None:
    """Ensure that the value is a list of strings.
    
    Args:
        value: The value to check.
        name: The name of the parameter (for error message).
        
    Raises:
        TypeError: If value is not a list of strings.
    """
    if not isinstance(value, list):
        raise TypeError(f"Parameter '{name}' must be a list of strings, but got {type(value).__name__}")
        
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"Parameter '{name}' must be a list of strings, but item at index {i} is {type(item).__name__}")
