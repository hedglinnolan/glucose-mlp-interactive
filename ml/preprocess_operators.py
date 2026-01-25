"""
Custom preprocessing operators for unit harmonization and plausibility gating.
"""
from __future__ import annotations

from typing import Optional, Sequence, Dict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class UnitHarmonizer(BaseEstimator, TransformerMixin):
    """Convert numeric features to canonical units using per-feature factors.
    Store conversion_factors by reference (no copy) so sklearn clone works."""

    def __init__(self, conversion_factors: Sequence[float]):
        self.conversion_factors = conversion_factors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        factors = np.asarray(self.conversion_factors, dtype=float)
        return X_arr * factors


class PlausibilityGate(BaseEstimator, TransformerMixin):
    """Set values outside empirical plausibility bounds to NaN."""

    def __init__(self, lower_bounds: Sequence[Optional[float]], upper_bounds: Sequence[Optional[float]]):
        self.lower_bounds = np.array(
            [np.nan if v is None else float(v) for v in lower_bounds], dtype=float
        )
        self.upper_bounds = np.array(
            [np.nan if v is None else float(v) for v in upper_bounds], dtype=float
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float).copy()
        n_cols = min(X_arr.shape[1], len(self.lower_bounds), len(self.upper_bounds))
        for idx in range(n_cols):
            lower = self.lower_bounds[idx]
            upper = self.upper_bounds[idx]
            if not np.isnan(lower):
                mask_lo = X_arr[:, idx] < lower
                X_arr[mask_lo, idx] = np.nan
            if not np.isnan(upper):
                mask_hi = X_arr[:, idx] > upper
                X_arr[mask_hi, idx] = np.nan
        return X_arr


class OutlierCapping(BaseEstimator, TransformerMixin):
    """Cap outliers based on percentile or MAD bounds computed at fit time."""

    def __init__(self, method: str = "percentile", params: Optional[Dict] = None):
        self.method = method
        self.params = params or {}
        self.lower_bounds_: Optional[np.ndarray] = None
        self.upper_bounds_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        if self.method == "mad":
            threshold = float(self.params.get("threshold", 3.5))
            med = np.nanmedian(X_arr, axis=0)
            mad = np.nanmedian(np.abs(X_arr - med), axis=0)
            scale = 1.4826 * mad
            self.lower_bounds_ = med - threshold * scale
            self.upper_bounds_ = med + threshold * scale
        else:
            lower_q = float(self.params.get("lower_q", 0.01))
            upper_q = float(self.params.get("upper_q", 0.99))
            self.lower_bounds_ = np.nanpercentile(X_arr, lower_q * 100, axis=0)
            self.upper_bounds_ = np.nanpercentile(X_arr, upper_q * 100, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            return X_arr
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)
