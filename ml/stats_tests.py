"""
Conventional statistical significance tests (distribution-informed).
Used for EDA association tests, missingness scan, and model comparison.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional, Union
from scipy import stats


def correlation_test(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pearson",
) -> Tuple[float, float, str]:
    """
    Correlation between two numeric vectors with p-value.

    Args:
        x: First numeric array
        y: Second numeric array
        method: 'pearson', 'spearman', or 'kendall'
    
    Returns:
        (r, p, test_name)
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    if len(x) < 3 or len(y) < 3:
        return (float("nan"), float("nan"), "correlation (insufficient n)")
    if method == "spearman":
        r, p = stats.spearmanr(x, y)
        return (float(r), float(p), "Spearman")
    elif method == "kendall":
        r, p = stats.kendalltau(x, y)
        return (float(r), float(p), "Kendall's Tau")
    else:
        r, p = stats.pearsonr(x, y)
        return (float(r), float(p), "Pearson")


def two_sample_location_test(
    x1: np.ndarray,
    x2: np.ndarray,
    parametric: bool,
) -> Tuple[float, float, str]:
    """
    Compare location of two independent samples.

    Returns:
        (statistic, p_value, test_name)
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]
    if len(x1) < 2 or len(x2) < 2:
        return (float("nan"), float("nan"), "two-sample (insufficient n)")
    if parametric:
        stat, p = stats.ttest_ind(x1, x2)
        return (float(stat), float(p), "t-test (ind.)")
    else:
        stat, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
        return (float(stat), float(p), "Mann–Whitney U")


def k_sample_location_test(
    groups: List[np.ndarray],
    parametric: bool,
) -> Tuple[float, float, str]:
    """
    Compare location across k independent groups.

    Returns:
        (statistic, p_value, test_name)
    """
    groups = [np.asarray(g)[~np.isnan(np.asarray(g))] for g in groups]
    groups = [g for g in groups if len(g) >= 1]
    if len(groups) < 2:
        return (float("nan"), float("nan"), "k-sample (insufficient groups)")
    if parametric:
        stat, p = stats.f_oneway(*groups)
        return (float(stat), float(p), "ANOVA")
    else:
        stat, p = stats.kruskal(*groups)
        return (float(stat), float(p), "Kruskal–Wallis")


def categorical_association_test(
    contingency: np.ndarray,
    use_fisher: bool = False,
) -> Tuple[float, float, str]:
    """
    Test association between two categorical variables (contingency table).

    use_fisher: use Fisher exact for 2x2 when appropriate.

    Returns:
        (statistic, p_value, test_name)
    """
    contingency = np.asarray(contingency)
    if contingency.size == 0:
        return (float("nan"), float("nan"), "chi2/Fisher")
    if use_fisher and contingency.shape == (2, 2):
        stat, p = stats.fisher_exact(contingency)
        return (float(stat), float(p), "Fisher exact")
    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        return (float(chi2), float(p), "chi-squared")
    except Exception:
        return (float("nan"), float("nan"), "chi-squared")


def normality_check(series: Union[np.ndarray, list]) -> Tuple[float, float, str]:
    """
    Shapiro–Wilk normality test. Sample capped at 5000 for performance.

    Returns:
        (statistic, p_value, test_name)
    """
    x = np.asarray(series)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return (float("nan"), float("nan"), "Shapiro–Wilk")
    n = min(5000, len(x))
    if n < len(x):
        rng = np.random.default_rng(42)
        x = rng.choice(x, size=n, replace=False)
    stat, p = stats.shapiro(x)
    return (float(stat), float(p), "Shapiro–Wilk")


def friedman_test(block_data: np.ndarray) -> Tuple[float, float, str]:
    """
    Friedman test for k related samples (e.g. CV metric across models).
    block_data: (n_blocks, k) e.g. (n_folds, n_models).

    Returns:
        (statistic, p_value, test_name)
    """
    block_data = np.asarray(block_data)
    if block_data.size == 0 or block_data.shape[1] < 2:
        return (float("nan"), float("nan"), "Friedman")
    try:
        stat, p = stats.friedmanchisquare(*block_data.T)
        return (float(stat), float(p), "Friedman")
    except Exception:
        return (float("nan"), float("nan"), "Friedman")


def paired_location_test(diff: np.ndarray, parametric: bool) -> Tuple[float, float, str]:
    """
    Test whether mean/median of paired differences is zero.

    Returns:
        (statistic, p_value, test_name)
    """
    d = np.asarray(diff)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (float("nan"), float("nan"), "paired (insufficient n)")
    if parametric:
        stat, p = stats.ttest_1samp(d, 0)
        return (float(stat), float(p), "paired t-test")
    else:
        stat, p = stats.wilcoxon(d, alternative="two-sided")
        return (float(stat), float(p), "Wilcoxon signed-rank")
