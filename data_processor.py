"""
Data processing utilities for the interactive predictor.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional


def load_and_preview_csv(file_path: str, n_rows: int = 5) -> pd.DataFrame:
    """Load CSV and return preview."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric column names."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if df[col].notna().sum() > 0]


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42
) -> Tuple:
    """
    Prepare data for training.
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names)
    """
    # Check columns exist
    missing = set([target_col] + feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    # Convert to numeric, coercing errors to NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    
    # Drop rows with NaN in target
    mask = y.notna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    # Drop rows with all NaN features
    mask = X.notna().any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    # Fill remaining NaN with median
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=seed
    )
    
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1.0 - rel_val), random_state=seed
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train.values, y_val.values, y_test.values,
        scaler, feature_cols
    )


def validate_data_selection(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> Tuple[bool, str]:
    """Validate that data selection is valid."""
    if not target_col:
        return False, "Please select a target column"
    
    if not feature_cols:
        return False, "Please select at least one feature column"
    
    if target_col in feature_cols:
        return False, "Target column cannot be in feature columns"
    
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in data"
    
    missing = set(feature_cols) - set(df.columns)
    if missing:
        return False, f"Feature columns not found: {missing}"
    
    # Check for numeric data
    if target_col not in get_numeric_columns(df):
        return False, f"Target column '{target_col}' must be numeric"
    
    numeric_features = get_numeric_columns(df)
    non_numeric = set(feature_cols) - set(numeric_features)
    if non_numeric:
        return False, f"Non-numeric feature columns: {non_numeric}"
    
    return True, "OK"
