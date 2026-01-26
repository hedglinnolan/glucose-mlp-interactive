"""
Helper functions to reconcile target/features selections without resetting everything.
"""
import pandas as pd
from typing import List, Tuple, Optional, Union


def reconcile_target_features(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    selectable_cols: Union[List[str], Tuple[List[str], List[str]]],
) -> Tuple[Optional[str], List[str]]:
    """
    Reconcile target and feature selections, removing invalid choices.

    Rules:
    - Target cannot be in features
    - Features must exist in selectable pool (numeric + categorical) and df.columns
    - If target changes, remove it from features
    - Preserve valid features

    Args:
        df: DataFrame
        target: Current target selection (can be None or '')
        features: Current feature selections
        selectable_cols: Either a flat list of selectable column names, or
            (numeric_cols, categorical_cols) from get_selectable_columns.

    Returns:
        Tuple of (reconciled_target, reconciled_features)
    """
    if target == "":
        target = None

    if isinstance(selectable_cols, tuple) and len(selectable_cols) == 2:
        a, b = selectable_cols[0], selectable_cols[1]
        if isinstance(a, list) and isinstance(b, list):
            selectable = set(a) | set(b)
        else:
            selectable = set(selectable_cols)
    else:
        selectable = set(selectable_cols)

    valid_features = [f for f in features if f in selectable and f in df.columns]

    if target:
        valid_features = [f for f in valid_features if f != target]

    if target and target not in selectable:
        target = None

    return target, valid_features
