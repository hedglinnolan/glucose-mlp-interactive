"""Tests for publication engine (methods generator, TRIPOD, flow diagram)."""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml.publication import (
    TRIPODTracker, TRIPOD_ITEMS,
    generate_methods_section,
    generate_flow_diagram_mermaid,
    subgroup_analysis,
)


def test_tripod_tracker():
    tracker = TRIPODTracker()
    done, total = tracker.get_progress()
    assert done == 0
    assert total == len(TRIPOD_ITEMS)

    tracker.mark_complete("outcome_defined", note="Glucose level", page_ref="Upload")
    done, total = tracker.get_progress()
    assert done == 1

    df = tracker.get_checklist_df()
    assert len(df) == len(TRIPOD_ITEMS)
    assert "✅" in df["Status"].values


def test_methods_section():
    text = generate_methods_section(
        data_config={},
        preprocessing_config={"numeric_scaling": "standard", "numeric_imputation": "median"},
        model_configs={"Ridge": {}, "Random Forest": {}},
        split_config={},
        n_total=1000,
        n_train=700,
        n_val=150,
        n_test=150,
        feature_names=["age", "bmi", "glucose"],
        target_name="outcome",
        task_type="regression",
        metrics_used=["RMSE", "MAE", "R2"],
    )
    assert "1,000 observations" in text
    assert "RMSE" in text
    assert "age, bmi, glucose" in text
    assert "RIDGE" in text
    assert "bootstrap" in text.lower()


def test_flow_diagram():
    mermaid = generate_flow_diagram_mermaid(
        n_total=1000,
        n_excluded=50,
        exclusion_reasons={"Missing outcome": 30, "Age < 18": 20},
        n_train=700,
        n_val=125,
        n_test=125,
    )
    assert "graph TD" in mermaid
    assert "1,000" in mermaid
    assert "700" in mermaid


def test_subgroup_analysis():
    np.random.seed(42)
    y_true = np.random.normal(100, 20, 100)
    y_pred = y_true + np.random.normal(0, 10, 100)
    subgroups = np.array(["Young"] * 50 + ["Old"] * 50)

    result = subgroup_analysis(y_true, y_pred, subgroups, n_bootstrap=50)
    assert isinstance(result, pd.DataFrame)
    assert "Overall" in result["Subgroup"].values
    assert "Young" in result["Subgroup"].values
    assert "Old" in result["Subgroup"].values
