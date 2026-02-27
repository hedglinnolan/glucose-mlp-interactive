"""Tests for feature selection methods."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.feature_selection import (
    lasso_path_selection, rfe_cv_selection,
    univariate_screening, stability_selection,
    consensus_features, FeatureSelectionResult,
)


@pytest.fixture
def regression_data():
    """Create data where features 0-2 are informative, 3-4 are noise."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 5)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n) * 0.5
    features = ['important_1', 'important_2', 'important_3', 'noise_1', 'noise_2']
    return X, y, features


def test_lasso_selection(regression_data):
    X, y, features = regression_data
    result = lasso_path_selection(X, y, features, "regression", cv_folds=3)
    assert isinstance(result, FeatureSelectionResult)
    assert result.method == "LASSO"
    assert len(result.selected_features) > 0
    assert len(result.scores) == len(features)
    # Informative features should have higher scores
    assert result.scores['important_1'] > result.scores['noise_1']


def test_rfe_cv_selection(regression_data):
    X, y, features = regression_data
    result = rfe_cv_selection(X, y, features, "regression", cv_folds=3)
    assert isinstance(result, FeatureSelectionResult)
    assert result.method == "RFE-CV"
    assert len(result.selected_features) > 0


def test_univariate_screening(regression_data):
    X, y, features = regression_data
    result = univariate_screening(X, y, features, "regression", alpha=0.05)
    assert isinstance(result, FeatureSelectionResult)
    assert "Univariate" in result.method
    # Important features should be selected
    assert 'important_1' in result.selected_features


def test_stability_selection(regression_data):
    X, y, features = regression_data
    result = stability_selection(
        X, y, features, "regression",
        n_bootstrap=30, threshold=0.4,
    )
    assert isinstance(result, FeatureSelectionResult)
    assert result.method == "Stability Selection"


def test_consensus_features():
    results = [
        FeatureSelectionResult(
            method="A", selected_features=["a", "b", "c"],
            all_features=["a", "b", "c", "d"], scores={}, details={}, description="",
        ),
        FeatureSelectionResult(
            method="B", selected_features=["b", "c", "d"],
            all_features=["a", "b", "c", "d"], scores={}, details={}, description="",
        ),
    ]
    consensus = consensus_features(results, min_methods=2)
    assert "b" in consensus
    assert "c" in consensus
    assert "a" not in consensus  # Only selected by 1 method
