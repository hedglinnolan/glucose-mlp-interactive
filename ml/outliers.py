"""
Outlier detection utilities with configurable methods.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


DEFAULT_METHODS = ("iqr", "mad", "zscore", "percentile")


def detect_outliers(
    series: pd.Series,
    method: str = "iqr",
    params: Optional[Dict] = None
) -> Tuple[pd.Series, Dict[str, Optional[float]]]:
    """
    Detect outliers in a numeric series.

    Returns:
        mask: boolean Series aligned to input index
        info: dict with method and threshold details
    """
    params = params or {}
    method = method.lower()

    mask = pd.Series(False, index=series.index)
    clean = series.dropna()
    info: Dict[str, Optional[float]] = {"method": method, "lower": None, "upper": None, "threshold": None}

    if len(clean) < 3:
        return mask, info

    if method == "iqr":
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            return mask, info
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (clean < lower) | (clean > upper)
        info.update({"lower": float(lower), "upper": float(upper)})
        mask.loc[clean.index] = outliers
        return mask, info

    if method == "mad":
        median = clean.median()
        mad = np.median(np.abs(clean - median))
        threshold = float(params.get("threshold", 3.5))
        if mad == 0:
            return mask, info
        modified_z = 0.6745 * (clean - median) / mad
        outliers = np.abs(modified_z) > threshold
        approx_scale = 1.4826 * mad
        lower = median - threshold * approx_scale
        upper = median + threshold * approx_scale
        info.update({"lower": float(lower), "upper": float(upper), "threshold": threshold})
        mask.loc[clean.index] = outliers
        return mask, info

    if method == "zscore":
        mean = clean.mean()
        std = clean.std()
        threshold = float(params.get("threshold", 3.0))
        if std == 0:
            return mask, info
        zscores = (clean - mean) / std
        outliers = np.abs(zscores) > threshold
        lower = mean - threshold * std
        upper = mean + threshold * std
        info.update({"lower": float(lower), "upper": float(upper), "threshold": threshold})
        mask.loc[clean.index] = outliers
        return mask, info

    if method == "percentile":
        lower_q = float(params.get("lower_q", 0.01))
        upper_q = float(params.get("upper_q", 0.99))
        lower = clean.quantile(lower_q)
        upper = clean.quantile(upper_q)
        outliers = (clean < lower) | (clean > upper)
        info.update({"lower": float(lower), "upper": float(upper)})
        mask.loc[clean.index] = outliers
        return mask, info

    return mask, info


def outlier_rate(series: pd.Series, method: str = "iqr", params: Optional[Dict] = None) -> float:
    """Compute outlier rate for a series."""
    clean = series.dropna()
    if len(clean) == 0:
        return 0.0
    mask, _ = detect_outliers(series, method=method, params=params)
    return float(mask.sum() / len(clean))

