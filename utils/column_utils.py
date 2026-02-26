"""
Column name utilities for data processing.
"""
from typing import List, Any


def make_unique_columns(cols: List[Any]) -> List[str]:
    """Deduplicate column names by appending _1, _2, etc. for duplicates."""
    seen = {}
    result = []
    for c in cols:
        c_str = str(c)
        if c_str in seen:
            seen[c_str] += 1
            result.append(f"{c_str}_{seen[c_str]}")
        else:
            seen[c_str] = 0
            result.append(c_str)
    return result
