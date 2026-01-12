"""
Built-in toy datasets for educational purposes.
"""
import pandas as pd
import numpy as np
from typing import Dict


def generate_linear_with_outliers(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate linear regression dataset with outliers.
    Illustrates why Huber/robust loss is useful.
    
    Returns:
        DataFrame with features and target
    """
    np.random.seed(random_state)
    
    # Generate clean linear relationship
    X1 = np.random.randn(n_samples) * 2
    X2 = np.random.randn(n_samples) * 1.5
    noise = np.random.randn(n_samples) * 0.5
    
    # True relationship: y = 2*X1 + 1.5*X2 + noise
    y = 2 * X1 + 1.5 * X2 + noise
    
    # Add outliers (10% of data)
    n_outliers = int(n_samples * 0.1)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    y[outlier_indices] += np.random.randn(n_outliers) * 10  # Large outliers
    
    df = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'target': y
    })
    
    return df


def generate_nonlinear_regression(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate nonlinear regression dataset.
    Illustrates why RF/NN outperform linear models.
    
    Returns:
        DataFrame with features and target
    """
    np.random.seed(random_state)
    
    # Generate features
    X1 = np.random.uniform(-3, 3, n_samples)
    X2 = np.random.uniform(-2, 2, n_samples)
    X3 = np.random.randn(n_samples)
    
    # Nonlinear relationship: y = sin(X1) * X2^2 + X3 + noise
    y = np.sin(X1) * X2**2 + X3 + np.random.randn(n_samples) * 0.3
    
    df = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'feature_3': X3,
        'target': y
    })
    
    return df


def generate_imbalanced_classification(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate binary classification dataset with class imbalance.
    Illustrates importance of metrics beyond accuracy.
    
    Returns:
        DataFrame with features and target
    """
    np.random.seed(random_state)
    
    # Generate features
    X1 = np.random.randn(n_samples) * 2
    X2 = np.random.randn(n_samples) * 1.5
    X3 = np.random.randn(n_samples)
    
    # Create imbalanced classes (80% class 0, 20% class 1)
    # Class 1 has higher X1 and X2 values
    y = np.zeros(n_samples, dtype=int)
    class_1_mask = (X1 > 0.5) & (X2 > 0.3)
    y[class_1_mask] = 1
    
    # Add some noise
    noise_indices = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    y[noise_indices] = 1 - y[noise_indices]
    
    df = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'feature_3': X3,
        'target': y
    })
    
    return df


def get_builtin_datasets() -> Dict[str, callable]:
    """
    Get dictionary of built-in dataset generators.
    
    Returns:
        Dict mapping dataset name to generator function
    """
    return {
        'Linear Regression with Outliers': generate_linear_with_outliers,
        'Nonlinear Regression': generate_nonlinear_regression,
        'Imbalanced Classification': generate_imbalanced_classification
    }
