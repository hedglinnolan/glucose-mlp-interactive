"""
Helper functions to reconcile target/features selections without resetting everything.
"""
import pandas as pd
from typing import List, Tuple, Optional


def reconcile_target_features(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    numeric_cols: List[str]
) -> Tuple[Optional[str], List[str]]:
    """
    Reconcile target and feature selections, removing invalid choices.
    
    Rules:
    - Target cannot be in features
    - Features must exist in numeric_cols
    - If target changes, remove it from features
    - Preserve valid features
    
    Args:
        df: DataFrame
        target: Current target selection (can be None or '')
        features: Current feature selections
        numeric_cols: List of numeric column names
        
    Returns:
        Tuple of (reconciled_target, reconciled_features)
    """
    # Normalize target (empty string -> None)
    if target == '':
        target = None
    
    # Filter features to only numeric columns that exist
    valid_features = [f for f in features if f in numeric_cols and f in df.columns]
    
    # Remove target from features if it's selected
    if target:
        valid_features = [f for f in valid_features if f != target]
    
    # Ensure target is valid if set
    if target and target not in numeric_cols:
        target = None
    
    return target, valid_features
