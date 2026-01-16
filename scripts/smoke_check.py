#!/usr/bin/env python
"""
Smoke Check Script for Glucose MLP Interactive

Validates key functions and imports without running Streamlit.
Run with: python scripts/smoke_check.py
"""
import sys
import os
import traceback
from typing import List, Tuple
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Track test results
results: List[Tuple[str, bool, str]] = []


def test(name: str, requires: List[str] = None):
    """Decorator to wrap tests and track results."""
    def decorator(func):
        def wrapper():
            # Check for required packages
            if requires:
                for pkg in requires:
                    try:
                        __import__(pkg)
                    except ImportError:
                        results.append((name, None, f"SKIPPED (requires {pkg})"))
                        print(f"⏭️  {name}: SKIPPED (requires {pkg})")
                        return
            
            try:
                func()
                results.append((name, True, "PASS"))
                print(f"✅ {name}")
            except ImportError as e:
                # Skip tests that fail due to missing packages
                pkg_name = str(e).replace("No module named ", "").strip("'")
                if pkg_name in ['streamlit', 'torch', 'shap']:
                    results.append((name, None, f"SKIPPED (requires {pkg_name})"))
                    print(f"⏭️  {name}: SKIPPED (requires {pkg_name})")
                else:
                    results.append((name, False, str(e)))
                    print(f"❌ {name}: {e}")
                    if "--verbose" in sys.argv:
                        traceback.print_exc()
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"❌ {name}: {e}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
        return wrapper
    return decorator


# ============================================================
# Import Tests - Ensure modules can be loaded without errors
# ============================================================

@test("Import: utils.session_state")
def test_import_session_state():
    from utils.session_state import (
        init_session_state, get_data, set_data, DataConfig,
        TaskTypeDetection, CohortStructureDetection
    )


@test("Import: ml.model_registry")
def test_import_model_registry():
    from ml.model_registry import get_registry, ModelSpec, ModelCapabilities


@test("Import: ml.model_coach")
def test_import_model_coach():
    from ml.model_coach import (
        coach_recommendations, CoachRecommendation, GROUP_DISPLAY_NAMES,
        compute_model_recommendations, ModelRecommendation, RecommendationBucket,
        TrainingTimeTier, CoachOutput
    )


@test("Import: ml.dataset_profile")
def test_import_dataset_profile():
    from ml.dataset_profile import (
        compute_dataset_profile, DatasetProfile, FeatureProfile, TargetProfile,
        DataSufficiencyLevel, WarningLevel, DataWarning, get_profile_summary_text
    )


@test("Import: ml.eda_recommender")
def test_import_eda_recommender():
    from ml.eda_recommender import compute_dataset_signals, recommend_eda, DatasetSignals


@test("Import: ml.triage")
def test_import_triage():
    from ml.triage import detect_task_type, detect_cohort_structure


@test("Import: ml.eval")
def test_import_eval():
    from ml.eval import (
        calculate_regression_metrics, calculate_classification_metrics,
        perform_cross_validation, analyze_residuals
    )


@test("Import: models.nn_whuber")
def test_import_nn_wrapper():
    from models.nn_whuber import NNWeightedHuberWrapper, SimpleMLP


@test("Import: models.glm")
def test_import_glm():
    from models.glm import GLMWrapper


@test("Import: models.rf")
def test_import_rf():
    from models.rf import RFWrapper


@test("Import: data_processor")
def test_import_data_processor():
    from data_processor import load_and_preview_csv, get_numeric_columns


# ============================================================
# Functional Tests - Test key functionality
# ============================================================

@test("Registry: get_registry returns dict with models")
def test_registry_structure():
    from ml.model_registry import get_registry
    registry = get_registry()
    assert isinstance(registry, dict), "Registry should be a dict"
    assert len(registry) > 0, "Registry should not be empty"
    # Check required models exist
    required_models = ['nn', 'rf', 'glm', 'huber', 'ridge', 'logreg']
    for model in required_models:
        assert model in registry, f"Model '{model}' should be in registry"


@test("Registry: NN model has architecture params")
def test_nn_architecture_params():
    from ml.model_registry import get_registry
    registry = get_registry()
    nn_spec = registry.get('nn')
    assert nn_spec is not None, "NN spec should exist"
    schema = nn_spec.hyperparam_schema
    assert 'num_layers' in schema, "NN should have num_layers param"
    assert 'layer_width' in schema, "NN should have layer_width param"
    assert 'architecture_pattern' in schema, "NN should have architecture_pattern param"
    assert 'activation' in schema, "NN should have activation param"


@test("Coach: recommendations are merged by group")
def test_coach_merging():
    from ml.model_coach import coach_recommendations, _merge_recommendations_by_group, CoachRecommendation
    
    # Create test recommendations with same group
    recs = [
        CoachRecommendation(
            group='Linear',
            recommended_models=['glm'],
            why=['Reason 1'],
            when_not_to_use=['Caveat 1'],
            suggested_preprocessing=['Preprocess 1'],
            priority=1
        ),
        CoachRecommendation(
            group='Linear',
            recommended_models=['ridge'],
            why=['Reason 2'],
            when_not_to_use=['Caveat 2'],
            suggested_preprocessing=['Preprocess 2'],
            priority=2
        ),
    ]
    
    merged = _merge_recommendations_by_group(recs)
    assert len(merged) == 1, "Should merge into single Linear recommendation"
    assert 'glm' in merged[0].recommended_models, "Should include glm"
    assert 'ridge' in merged[0].recommended_models, "Should include ridge"
    assert merged[0].priority == 1, "Should use lowest priority"


@test("Coach: display_name property works")
def test_coach_display_name():
    from ml.model_coach import CoachRecommendation, GROUP_DISPLAY_NAMES
    
    rec = CoachRecommendation(
        group='Linear',
        recommended_models=['glm'],
        why=['Test'],
        when_not_to_use=[],
        suggested_preprocessing=[],
        priority=1
    )
    
    assert rec.display_name == 'Linear Models', f"Expected 'Linear Models', got '{rec.display_name}'"


@test("NN: SimpleMLP accepts activation parameter")
def test_nn_activation():
    import torch
    from models.nn_whuber import SimpleMLP
    
    # Test with different activations
    for activation in ['relu', 'tanh', 'leaky_relu', 'elu']:
        model = SimpleMLP(input_dim=10, hidden=[32, 32], dropout=0.1, output_dim=1, activation=activation)
        assert model.activation_name == activation, f"Activation should be {activation}"
        
        # Test forward pass
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 1), f"Output shape should be (5, 1)"


@test("NN: NNWeightedHuberWrapper accepts activation parameter")
def test_nn_wrapper_activation():
    from models.nn_whuber import NNWeightedHuberWrapper
    
    wrapper = NNWeightedHuberWrapper(
        hidden_layers=[32, 16],
        dropout=0.1,
        task_type='regression',
        activation='tanh'
    )
    
    assert wrapper.activation == 'tanh', "Activation should be tanh"
    assert wrapper.hidden_layers == [32, 16], "Hidden layers should be [32, 16]"


@test("Data: DataConfig can be created")
def test_data_config():
    from utils.session_state import DataConfig
    
    config = DataConfig(
        target_col='glucose',
        feature_cols=['feature1', 'feature2'],
        datetime_col=None,
        task_type='regression'
    )
    
    assert config.target_col == 'glucose'
    assert len(config.feature_cols) == 2


@test("Detection: TaskTypeDetection final property works")
def test_task_type_detection():
    from utils.session_state import TaskTypeDetection
    
    # Without override
    det = TaskTypeDetection(detected='regression', confidence='high', reasons=['Test'])
    assert det.final == 'regression', "Final should be detected value"
    
    # With override
    det.override_enabled = True
    det.override_value = 'classification'
    assert det.final == 'classification', "Final should be override value when enabled"


@test("DatasetProfile: compute_dataset_profile works")
def test_dataset_profile():
    import pandas as pd
    import numpy as np
    from ml.dataset_profile import compute_dataset_profile, DataSufficiencyLevel
    
    # Create simple test data
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randn(100)
    })
    
    profile = compute_dataset_profile(
        df, 
        target_col='target',
        feature_cols=['feature1', 'feature2', 'category'],
        task_type='regression'
    )
    
    assert profile.n_rows == 100, "Should have 100 rows"
    assert profile.n_features == 3, "Should have 3 features"
    assert profile.n_numeric == 2, "Should have 2 numeric features"
    assert profile.n_categorical == 1, "Should have 1 categorical feature"
    assert profile.target_profile is not None, "Should have target profile"
    assert profile.target_profile.task_type == 'regression', "Should be regression"
    assert profile.data_sufficiency in DataSufficiencyLevel, "Should have valid sufficiency level"


@test("DatasetProfile: classification target detection")
def test_profile_classification():
    import pandas as pd
    import numpy as np
    from ml.dataset_profile import compute_dataset_profile
    
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(200),
        'target': np.random.choice([0, 1], 200, p=[0.8, 0.2])  # Imbalanced
    })
    
    profile = compute_dataset_profile(
        df, 
        target_col='target',
        feature_cols=['feature1'],
        task_type='classification'
    )
    
    assert profile.target_profile.task_type == 'classification'
    assert profile.target_profile.n_classes == 2
    assert profile.target_profile.is_imbalanced, "Should detect imbalance"


@test("DatasetProfile: warnings generation")
def test_profile_warnings():
    import pandas as pd
    import numpy as np
    from ml.dataset_profile import compute_dataset_profile, WarningLevel
    
    # Create data with issues
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': [1.0] * 20 + [np.nan] * 5,  # Small sample with missing
        'target': list(range(25))
    })
    
    profile = compute_dataset_profile(
        df, 
        target_col='target',
        feature_cols=['feature1'],
        task_type='regression'
    )
    
    # Should have sample size warning
    assert len(profile.warnings) > 0, "Should have warnings for small sample"
    warning_categories = [w.category for w in profile.warnings]
    assert 'sample_size' in warning_categories, "Should warn about small sample"


@test("Coach: compute_model_recommendations works with profile")
def test_coach_with_profile():
    import pandas as pd
    import numpy as np
    from ml.dataset_profile import compute_dataset_profile
    from ml.model_coach import compute_model_recommendations, RecommendationBucket
    
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(500),
        'feature2': np.random.randn(500),
        'target': np.random.randn(500)
    })
    
    profile = compute_dataset_profile(
        df, 
        target_col='target',
        feature_cols=['feature1', 'feature2'],
        task_type='regression'
    )
    
    coach_output = compute_model_recommendations(profile)
    
    assert coach_output is not None, "Should return coach output"
    assert len(coach_output.recommended_models) > 0 or len(coach_output.worth_trying_models) > 0, \
        "Should have some recommendations"
    assert coach_output.dataset_summary, "Should have dataset summary"
    assert coach_output.preprocessing_recommendations is not None, "Should have preprocessing recs"
    assert coach_output.baseline_eda, "Should have baseline EDA recommendations"
    assert coach_output.advanced_eda_by_family, "Should have advanced EDA by family"


@test("Coach: ModelRecommendation has required fields")
def test_model_recommendation_fields():
    from ml.model_coach import ModelRecommendation, RecommendationBucket, TrainingTimeTier
    
    rec = ModelRecommendation(
        model_key='ridge',
        model_name='Ridge Regression',
        group='Linear',
        bucket=RecommendationBucket.RECOMMENDED,
        rationale='Good for this dataset',
        dataset_fit_summary='Good fit',
        strengths=['Interpretable'],
        weaknesses=[],
        risks=[],
        training_time=TrainingTimeTier.FAST,
        interpretability='high',
        requires_scaling=True,
        requires_encoding=True,
        handles_missing=False,
        plain_language_summary='Ridge is a regularized linear model.',
        when_to_use='When features are correlated',
        when_to_avoid='When relationships are nonlinear',
        priority=10
    )
    
    assert rec.model_key == 'ridge'
    assert rec.bucket == RecommendationBucket.RECOMMENDED
    assert rec.training_time == TrainingTimeTier.FAST


# ============================================================
# Main execution
# ============================================================

def run_all_tests():
    """Run all registered tests."""
    print("\n" + "=" * 60)
    print("Glucose MLP Interactive - Smoke Check")
    print("=" * 60 + "\n")
    
    # Run import tests
    print("Import Tests:")
    print("-" * 40)
    test_import_session_state()
    test_import_model_registry()
    test_import_model_coach()
    test_import_dataset_profile()
    test_import_eda_recommender()
    test_import_triage()
    test_import_eval()
    test_import_nn_wrapper()
    test_import_glm()
    test_import_rf()
    test_import_data_processor()
    
    print("\nFunctional Tests:")
    print("-" * 40)
    test_registry_structure()
    test_nn_architecture_params()
    test_coach_merging()
    test_coach_display_name()
    test_nn_activation()
    test_nn_wrapper_activation()
    test_data_config()
    test_task_type_detection()
    test_dataset_profile()
    test_profile_classification()
    test_profile_warnings()
    test_coach_with_profile()
    test_model_recommendation_fields()
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, success, _ in results if success is True)
    failed = sum(1 for _, success, _ in results if success is False)
    skipped = sum(1 for _, success, _ in results if success is None)
    total = len(results)
    
    if failed == 0:
        print(f"✅ All tests passed! ({passed} passed, {skipped} skipped)")
    else:
        print(f"❌ {failed}/{total} tests failed ({passed} passed, {skipped} skipped):")
        for name, success, msg in results:
            if success is False:
                print(f"   - {name}: {msg}")
    
    if skipped > 0:
        print(f"\nSkipped tests (missing optional dependencies):")
        for name, success, msg in results:
            if success is None:
                print(f"   - {name}: {msg}")
    
    print("=" * 60 + "\n")
    
    # Return success if no tests failed (skipped is OK)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
