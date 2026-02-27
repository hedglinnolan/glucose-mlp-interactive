"""Tests for calibration analysis."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.calibration import (
    calibration_classification, calibration_regression,
    CalibrationResult, decision_curve_analysis,
)


def test_classification_calibration():
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=500, p=[0.6, 0.4])
    y_proba = np.random.beta(2, 2, 500)
    y_proba[y_true == 1] += 0.2
    y_proba = np.clip(y_proba, 0, 1)

    result = calibration_classification(y_true, y_proba, model_name="test")
    assert isinstance(result, CalibrationResult)
    assert result.brier_score is not None
    assert 0 <= result.brier_score <= 1
    assert result.ece is not None
    assert 0 <= result.ece <= 1
    assert result.bin_edges is not None
    assert len(result.bin_edges) == 11  # 10 bins + 1


def test_regression_calibration():
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 200)
    y_pred = y_true * 0.9 + 5  # Slightly miscalibrated

    result = calibration_regression(y_true, y_pred, model_name="test")
    assert isinstance(result, CalibrationResult)
    assert result.calibration_slope is not None
    assert result.calibration_intercept is not None


def test_decision_curve():
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=200, p=[0.7, 0.3])
    y_proba = np.random.beta(2, 5, 200)
    y_proba[y_true == 1] += 0.3
    y_proba = np.clip(y_proba, 0, 1)

    fig = decision_curve_analysis(y_true, {"Model A": y_proba})
    assert fig is not None
