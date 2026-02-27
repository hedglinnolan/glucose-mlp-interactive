"""
Sensitivity analysis framework.

Re-runs the modeling pipeline under varying assumptions to assess robustness:
- Different imputation strategies
- Different random seeds
- Different outlier thresholds
- Different feature subsets
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class SensitivityResult:
    """Result of a single sensitivity analysis run."""
    variation_name: str
    variation_value: Any
    metrics: Dict[str, float]


@dataclass
class SensitivityAnalysis:
    """Complete sensitivity analysis results."""
    analysis_type: str  # e.g., "imputation", "seed", "outlier"
    baseline_metrics: Dict[str, float]
    variations: List[SensitivityResult]
    description: str

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{
            "Variation": "Baseline",
            **self.baseline_metrics,
        }]
        for v in self.variations:
            rows.append({
                "Variation": f"{v.variation_name}={v.variation_value}",
                **v.metrics,
            })
        return pd.DataFrame(rows)

    def is_robust(self, metric: str, tolerance: float = 0.1) -> bool:
        """Check if results are robust (all variations within tolerance of baseline)."""
        baseline_val = self.baseline_metrics.get(metric, 0)
        if baseline_val == 0:
            return True
        for v in self.variations:
            var_val = v.metrics.get(metric, 0)
            if abs(var_val - baseline_val) / abs(baseline_val) > tolerance:
                return False
        return True


def sensitivity_random_seeds(
    train_fn: Callable,
    eval_fn: Callable,
    seeds: List[int] = None,
    baseline_seed: int = 42,
) -> SensitivityAnalysis:
    """Test robustness to random seed choice.

    Args:
        train_fn: Function(seed) -> trained_model that trains a model with given seed
        eval_fn: Function(model) -> Dict[str, float] that evaluates the model
        seeds: List of seeds to try (default: [0, 1, 7, 13, 42, 99, 123, 456])
        baseline_seed: The seed used in the main analysis

    Returns:
        SensitivityAnalysis with results for each seed
    """
    if seeds is None:
        seeds = [0, 1, 7, 13, 99, 123, 456]

    # Baseline
    baseline_model = train_fn(baseline_seed)
    baseline_metrics = eval_fn(baseline_model)

    variations = []
    for seed in seeds:
        if seed == baseline_seed:
            continue
        try:
            model = train_fn(seed)
            metrics = eval_fn(model)
            variations.append(SensitivityResult(
                variation_name="seed",
                variation_value=seed,
                metrics=metrics,
            ))
        except Exception:
            pass

    return SensitivityAnalysis(
        analysis_type="random_seed",
        baseline_metrics=baseline_metrics,
        variations=variations,
        description=f"Results across {len(variations)} different random seeds "
                    f"(baseline seed={baseline_seed}).",
    )


def sensitivity_summary_table(analyses: List[SensitivityAnalysis], primary_metric: str) -> pd.DataFrame:
    """Create a summary table of all sensitivity analyses.

    Shows whether results are robust for each analysis type.
    """
    rows = []
    for analysis in analyses:
        baseline_val = analysis.baseline_metrics.get(primary_metric, float('nan'))
        var_vals = [v.metrics.get(primary_metric, float('nan')) for v in analysis.variations]

        if var_vals:
            min_val = min(var_vals)
            max_val = max(var_vals)
            mean_val = np.mean(var_vals)
            std_val = np.std(var_vals)
        else:
            min_val = max_val = mean_val = std_val = float('nan')

        rows.append({
            "Analysis": analysis.analysis_type,
            "Baseline": f"{baseline_val:.4f}",
            "Mean": f"{mean_val:.4f}",
            "Std": f"{std_val:.4f}",
            "Range": f"[{min_val:.4f}, {max_val:.4f}]",
            "N variations": len(analysis.variations),
            "Robust (10%)": "✅" if analysis.is_robust(primary_metric, 0.1) else "⚠️",
        })

    return pd.DataFrame(rows)
