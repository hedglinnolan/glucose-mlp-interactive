"""
Helper functions for data splitting.
"""
import numpy as np
import pandas as pd
from typing import Union


def to_numpy_1d(x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
    """
    Convert input to 1D numpy array.
    
    Args:
        x: Can be numpy array, pandas Series, or DataFrame column
        
    Returns:
        1D numpy array
    """
    arr = np.asarray(x)
    return arr.reshape(-1)
