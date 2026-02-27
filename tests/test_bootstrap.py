"""Tests for bootstrap confidence intervals."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.bootstrap import (
    bootstrap_metric, bootstrap_all_regression_metrics,
    bootstrap_all_classification_metrics, format_metric_with_ci,
    BootstrapResult,
)


@pytest.fixture
def regression_data():
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 200)
    y_pred = y_true + np.random.normal(0, 10, 200)
    return y_true, y_pred


@pytest.fixture
def classification_data():
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=200, p=[0.6, 0.4])
    y_pred = y_true.copy()
    # Flip 20% of predictions
    flip = np.random.choice(200, size=40, replace=False)
    y_pred[flip] = 1 - y_pred[flip]
    y_proba = np.random.beta(2, 2, 200)
    y_proba[y_true == 1] += 0.3
    y_proba = np.clip(y_proba, 0, 1)
    return y_true, y_pred, y_proba


def test_bootstrap_metric(regression_data):
    y_true, y_pred = regression_data
    from sklearn.metrics import mean_squared_error
    rmse_fn = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))

    result = bootstrap_metric(y_true, y_pred, rmse_fn, n_resamples=100, metric_name="RMSE")
    assert isinstance(result, BootstrapResult)
    assert result.ci_lower <= result.estimate <= result.ci_upper
    assert result.metric_name == "RMSE"


def test_bootstrap_regression(regression_data):
    y_true, y_pred = regression_data
    results = bootstrap_all_regression_metrics(y_true, y_pred, n_resamples=100)
    assert "RMSE" in results
    assert "MAE" in results
    assert "R2" in results
    for name, res in results.items():
        assert res.ci_lower <= res.ci_upper


def test_bootstrap_classification(classification_data):
    y_true, y_pred, y_proba = classification_data
    results = bootstrap_all_classification_metrics(y_true, y_pred, y_proba, n_resamples=100)
    assert "Accuracy" in results
    assert "F1" in results


def test_format_metric_with_ci():
    result = BootstrapResult(
        estimate=0.85, ci_lower=0.80, ci_upper=0.90,
        ci_level=0.95, n_resamples=1000, metric_name="AUC"
    )
    formatted = format_metric_with_ci(result, decimal_places=2)
    assert "0.85" in formatted
    assert "0.80" in formatted
    assert "0.90" in formatted


def test_bootstrap_handles_errors():
    """Test that bootstrap handles metric computation errors gracefully."""
    def bad_metric(yt, yp):
        raise ValueError("test error")

    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    result = bootstrap_metric(y_true, y_pred, bad_metric, n_resamples=10)
    assert np.isnan(result.estimate)
