"""
Helper functions for Streamlit widget initialization.
"""
from typing import List, Any, Optional


def safe_option_index(options: List[str], value: Any, default: str) -> int:
    """
    Safely compute the index for a selectbox/radio given a value.
    
    Args:
        options: List of option strings
        value: The value to find (can be None, missing, or invalid)
        default: Default value to use if value is invalid
        
    Returns:
        Index into options list (always valid)
    """
    if value is None or value not in options:
        return options.index(default)
    return options.index(value)
