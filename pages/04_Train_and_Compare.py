"""
Page 04: Train and Compare Models
Train models, evaluate, compare metrics, show diagnostics.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from utils.session_state import (
    init_session_state, get_data, get_preprocessing_pipeline,
    DataConfig, SplitConfig, ModelConfig, set_splits, add_trained_model,
    TaskTypeDetection, CohortStructureDetection
)
from utils.seed import set_global_seed, get_global_seed
from utils.storyline import render_progress_indicator, get_insights_by_category
from ml.splits import to_numpy_1d

logger = logging.getLogger(__name__)

# Lazy imports for heavy packages - only load when needed
def _get_plotly():
    """Lazy import plotly."""
    import plotly.graph_objects as go
    import plotly.express as px
    return go, px

def _get_sklearn_splits():
    """Lazy import sklearn model_selection."""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
    return train_test_split, GroupShuffleSplit, GroupKFold

def _get_model_wrappers():
    """Lazy import model wrappers - these load torch/sklearn models."""
    from models.nn_whuber import NNWeightedHuberWrapper
    from models.glm import GLMWrapper
    from models.huber_glm import HuberGLMWrapper
    from models.rf import RFWrapper
    from models.registry_wrappers import RegistryModelWrapper
    return NNWeightedHuberWrapper, GLMWrapper, HuberGLMWrapper, RFWrapper, RegistryModelWrapper

def _get_eval_functions():
    """Lazy import evaluation functions."""
    from ml.eval import (
        calculate_regression_metrics, calculate_classification_metrics,
        perform_cross_validation, analyze_residuals
    )
    return calculate_regression_metrics, calculate_classification_metrics, perform_cross_validation, analyze_residuals

def _get_visualization_functions():
    """Lazy import visualization functions with fallback."""
    try:
        from visualizations import plot_training_history, plot_predictions_vs_actual, plot_residuals
        return plot_training_history, plot_predictions_vs_actual, plot_residuals
    except ImportError:
        # Fallback if visualizations module not found
        import plotly.graph_objects as go
        def plot_training_history(history):
            fig = go.Figure()
            epochs = range(1, len(history['train_loss']) + 1)
            fig.add_trace(go.Scatter(x=list(epochs), y=history['train_loss'], name='Train Loss'))
            if 'val_loss' in history:
                fig.add_trace(go.Scatter(x=list(epochs), y=history['val_loss'], name='Val Loss'))
            return fig
        def plot_predictions_vs_actual(y_true, y_pred, title=""):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions'))
            return fig
        def plot_residuals(y_true, y_pred, title=""):
            residuals = y_true - y_pred
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
            return fig
        return plot_training_history, plot_predictions_vs_actual, plot_residuals

init_session_state()

# Set global seed
set_global_seed(st.session_state.get('random_seed', 42))

st.set_page_config(page_title="Train & Compare", page_icon="🏋️", layout="wide")
st.title("🏋️ Train & Compare Models")

# Progress indicator
render_progress_indicator("04_Train_and_Compare")

# Global random seed control
with st.sidebar:
    st.header("⚙️ Global Settings")
    random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=st.session_state.get('random_seed', 42),
        help="Controls randomness for reproducibility"
    )
    if random_seed != st.session_state.get('random_seed', 42):
        st.session_state.random_seed = random_seed
        set_global_seed(random_seed)
        st.info("Seed updated. Re-run splits and training to apply.")

# Check prerequisites
df = get_data()
if df is None:
    st.warning("⚠️ Please upload data first")
    st.stop()

data_config: DataConfig = st.session_state.get('data_config')
if data_config is None or not data_config.target_col:
    st.warning("⚠️ Please configure target and features")
    st.stop()

pipelines_by_model = st.session_state.get('preprocessing_pipelines_by_model', {})
pipeline = get_preprocessing_pipeline()
if pipeline is None and not pipelines_by_model:
    st.warning("⚠️ Please build preprocessing pipeline first")
    st.stop()

# Get final detection values
task_type_detection: TaskTypeDetection = st.session_state.get('task_type_detection', TaskTypeDetection())
cohort_structure_detection: CohortStructureDetection = st.session_state.get('cohort_structure_detection', CohortStructureDetection())

task_type_final = task_type_detection.final if task_type_detection.final else data_config.task_type
cohort_type_final = cohort_structure_detection.final if cohort_structure_detection.final else 'cross_sectional'
entity_id_final = cohort_structure_detection.entity_id_final

# Use final task type for downstream logic
if task_type_final:
    data_config.task_type = task_type_final

# Split configuration
st.header("📊 Data Splitting")

# Longitudinal data handling
use_group_split = False
if cohort_type_final == 'longitudinal' and entity_id_final:
    st.info(f"ℹ️ Longitudinal data detected. Entity ID: `{entity_id_final}`. Using group-based splitting to prevent data leakage.")
    use_group_split = True
    if entity_id_final not in df.columns:
        st.error(f"⚠️ Entity ID column '{entity_id_final}' not found in data. Please check Upload & Audit page.")
        st.stop()
elif cohort_type_final == 'longitudinal' and not entity_id_final:
    st.warning("⚠️ Longitudinal data detected but no entity ID column found. Consider selecting an entity ID in Upload & Audit page.")
    if data_config.datetime_col:
        st.info("ℹ️ Using time-based split as fallback for longitudinal data.")

# Time-series split option
use_time_split = False
if data_config.datetime_col:
    time_split_default = st.session_state.get('train_use_time_split')
    if time_split_default is None:
        time_split_default = (cohort_type_final == 'longitudinal' and not entity_id_final)
    use_time_split = st.checkbox(
        "Use Time-Based Split",
        value=time_split_default,
        disabled=use_group_split,
        key="train_use_time_split",
        help="Split data chronologically instead of randomly (recommended for time-series)"
    )
    if not use_time_split and not use_group_split:
        st.warning("⚠️ Datetime column detected but random split selected. Consider using time-based split for time-series data.")

col1, col2, col3 = st.columns(3)

# Read split sizes from session_state or use defaults
split_config_existing = st.session_state.get('split_config')
train_size_default = int((split_config_existing.train_size * 100) if split_config_existing and split_config_existing.train_size else 70)
val_size_default = int((split_config_existing.val_size * 100) if split_config_existing and split_config_existing.val_size else 15)
test_size_default = int((split_config_existing.test_size * 100) if split_config_existing and split_config_existing.test_size else 15)

with col1:
    train_size = st.slider("Train %", 50, 90, train_size_default, key="train_split_train_pct") / 100
with col2:
    val_size = st.slider("Val %", 5, 30, val_size_default, key="train_split_val_pct") / 100
with col3:
    test_size = st.slider("Test %", 5, 30, test_size_default, key="train_split_test_pct") / 100

if abs(train_size + val_size + test_size - 1.0) > 0.01:
    st.error("⚠️ Splits must sum to 100%")
    st.stop()

split_config = SplitConfig(
    train_size=train_size,
    val_size=val_size,
    test_size=test_size,
    random_state=st.session_state.get('random_seed', 42),
    stratify=(task_type_final == 'classification' and not use_time_split and not use_group_split),
    use_time_split=use_time_split,
    datetime_col=data_config.datetime_col if use_time_split else None
)
st.session_state.split_config = split_config

# Cross-validation option - read from session_state
use_cv_default = st.session_state.get('use_cv', False)
use_cv = st.checkbox("Enable Cross-Validation", value=use_cv_default, key="train_use_cv")
if use_cv:
    cv_folds_default = st.session_state.get('cv_folds', 5)
    cv_folds = st.slider("CV Folds", 3, 10, cv_folds_default, key="train_cv_folds")
    st.session_state.use_cv = True
    st.session_state.cv_folds = cv_folds
else:
    st.session_state.use_cv = False

# Prepare data splits
if st.button("🔄 Prepare Splits", type="primary"):
    try:
        # Lazy import sklearn splitting functions
        train_test_split, GroupShuffleSplit, GroupKFold = _get_sklearn_splits()
        
        X = df[data_config.feature_cols]
        y = df[data_config.target_col]
        indices = np.arange(len(df))
        
        # Split data (group-based, time-based, or random)
        if use_group_split and entity_id_final:
            # Group-based split for longitudinal data
            groups = to_numpy_1d(df[entity_id_final])
            y_arr = to_numpy_1d(y)
            
            gss = GroupShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=split_config.random_state)
            train_idx, temp_idx = next(gss.split(indices, y_arr, groups))
            
            # Split temp into val and test, maintaining groups
            groups_temp = groups[temp_idx]
            rel_val = val_size / (val_size + test_size)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=(1 - rel_val), random_state=split_config.random_state)
            val_idx, test_idx = next(gss2.split(indices[temp_idx], y_arr[temp_idx], groups_temp))
            
            X_train = X.iloc[train_idx]
            X_val = X.iloc[temp_idx[val_idx]]
            X_test = X.iloc[temp_idx[test_idx]]
            y_train = y_arr[train_idx]
            y_val = y_arr[temp_idx[val_idx]]
            y_test = y_arr[temp_idx[test_idx]]
            
            n_train_groups = len(np.unique(groups[train_idx]))
            n_val_groups = len(np.unique(groups[temp_idx[val_idx]]))
            n_test_groups = len(np.unique(groups[temp_idx[test_idx]]))
            st.info(f"👥 Group-based split: {n_train_groups} train groups, {n_val_groups} val groups, {n_test_groups} test groups")
        elif split_config.use_time_split and data_config.datetime_col:
            # Time-based split
            df_with_datetime = df.copy()
            df_with_datetime['_temp_index'] = df_with_datetime.index
            df_with_datetime = df_with_datetime.sort_values(data_config.datetime_col)
            
            # Calculate split indices
            n_total = len(df_with_datetime)
            n_train = int(n_total * train_size)
            n_val = int(n_total * val_size)
            
            train_indices = df_with_datetime.iloc[:n_train]['_temp_index'].values
            val_indices = df_with_datetime.iloc[n_train:n_train+n_val]['_temp_index'].values
            test_indices = df_with_datetime.iloc[n_train+n_val:]['_temp_index'].values
            
            X_train = X.iloc[train_indices]
            X_val = X.iloc[val_indices]
            X_test = X.iloc[test_indices]
            y_train = to_numpy_1d(y.iloc[train_indices])
            y_val = to_numpy_1d(y.iloc[val_indices])
            y_test = to_numpy_1d(y.iloc[test_indices])
            
            st.info(f"⏰ Time-based split: Train={df_with_datetime.iloc[0][data_config.datetime_col]} to {df_with_datetime.iloc[n_train-1][data_config.datetime_col]}")
        elif split_config.stratify and task_type_final == 'classification':
            idx_train, idx_temp, y_train, y_temp = train_test_split(
                indices, y, test_size=(val_size + test_size),
                random_state=split_config.random_state, stratify=y
            )
            rel_val = val_size / (val_size + test_size)
            idx_val, idx_test, y_val, y_test = train_test_split(
                idx_temp, y_temp, test_size=(1 - rel_val),
                random_state=split_config.random_state, stratify=y_temp
            )
            X_train = X.iloc[idx_train]
            X_val = X.iloc[idx_val]
            X_test = X.iloc[idx_test]
        else:
            idx_train, idx_temp, y_train, y_temp = train_test_split(
                indices, y, test_size=(val_size + test_size),
                random_state=split_config.random_state
            )
            rel_val = val_size / (val_size + test_size)
            idx_val, idx_test, y_val, y_test = train_test_split(
                idx_temp, y_temp, test_size=(1 - rel_val),
                random_state=split_config.random_state
            )
            X_train = X.iloc[idx_train]
            X_val = X.iloc[idx_val]
            X_test = X.iloc[idx_test]
        
        feature_names = list(data_config.feature_cols)
        set_splits(X_train, X_val, X_test, to_numpy_1d(y_train), to_numpy_1d(y_val), to_numpy_1d(y_test), feature_names)
        
        # Store indices for explainability (need raw data)
        if use_group_split and entity_id_final:
            st.session_state.train_indices = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
            st.session_state.test_indices = (temp_idx[test_idx].tolist() if hasattr(temp_idx[test_idx], 'tolist') else list(temp_idx[test_idx]))
        elif split_config.use_time_split and data_config.datetime_col:
            st.session_state.train_indices = train_indices.tolist() if hasattr(train_indices, 'tolist') else list(train_indices)
            st.session_state.test_indices = test_indices.tolist() if hasattr(test_indices, 'tolist') else list(test_indices)
        else:
            st.session_state.train_indices = idx_train.tolist() if hasattr(idx_train, 'tolist') else list(idx_train)
            st.session_state.test_indices = idx_test.tolist() if hasattr(idx_test, 'tolist') else list(idx_test)
        
        st.success(f"✅ Splits prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    except Exception as e:
        st.error(f"❌ Error preparing splits: {str(e)}")
        logger.exception(e)

# Check if splits are ready
splits = st.session_state.get('X_train')
if splits is None:
    st.stop()

X_train = st.session_state.X_train
X_val = st.session_state.X_val
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_val = st.session_state.y_val
y_test = st.session_state.y_test

# Model Selection Coach (top section) - cached for performance
@st.cache_data
def _compute_coach_recommendations(_df_hash, target_col, task_type, cohort_type, entity_id, eda_results_keys):
    """Cached coach recommendations computation."""
    from ml.model_coach import coach_recommendations
    from ml.eda_recommender import compute_dataset_signals
    
    df = get_data()  # Get actual dataframe
    signals = compute_dataset_signals(
        df,
        target_col,
        task_type,
        cohort_type,
        entity_id,
        outlier_method=st.session_state.get("eda_outlier_method", "iqr")
    )
    eda_results = st.session_state.get('eda_results')
    return coach_recommendations(signals, eda_results, get_insights_by_category())

# Key insights after pre-processing (EDA + preprocessing-specific)
insights = get_insights_by_category()
eda_only = [i for i in insights if i.get('category') != 'preprocessing']
prep_only = [i for i in insights if i.get('category') == 'preprocessing']
if eda_only or prep_only:
    with st.expander("💡 Key insights after pre-processing", expanded=True):
        if eda_only:
            st.markdown("**From EDA**")
            for insight in eda_only:
                st.markdown(f"• **{insight.get('category', 'General').title()}:** {insight['finding']}")
                st.caption(f"  → {insight['implication']}")
        if prep_only:
            st.markdown("**From preprocessing**")
            for insight in prep_only:
                st.markdown(f"• {insight['finding']}")
                st.caption(f"  → {insight['implication']}")

# Compute coach recommendations (using cached function)
# Create a hash for the dataframe to use as cache key
_df_hash = hash((len(df), tuple(df.columns)))
_eda_results_keys = tuple(sorted(st.session_state.get('eda_results', {}).keys()))
coach_recs = _compute_coach_recommendations(
    _df_hash, data_config.target_col, task_type_final, cohort_type_final, entity_id_final, _eda_results_keys
)

if coach_recs:
    with st.expander("🎓 Model Selection Coach", expanded=True):
        st.caption("Selections here sync with Preprocessing; pick models there first to build pipelines.")
        st.markdown("**Based on your data, try these first:**")
        for idx, rec in enumerate(coach_recs[:3]):
            display_name = rec.display_name if hasattr(rec, 'display_name') else f"{rec.group} Models"
            priority_label = "High" if rec.priority <= 2 else "Medium"
            st.markdown(f"**{display_name}** ({priority_label})")
            if hasattr(rec, 'readiness_checks') and rec.readiness_checks:
                st.caption("Prerequisites: " + "; ".join(rec.readiness_checks[:2]))
            with st.expander("Why?"):
                for reason in rec.why[:3]:
                    st.write(f"• {reason}")
                if rec.when_not_to_use:
                    st.caption("When not to use: " + "; ".join(rec.when_not_to_use[:2]))
                if rec.suggested_preprocessing:
                    st.caption("Preprocessing: " + "; ".join(rec.suggested_preprocessing[:2]))
            button_key = f"coach_select_{rec.group}_p{rec.priority}_i{idx}"
            if st.button(f"Select {display_name}", key=button_key):
                for model_key in rec.recommended_models:
                    st.session_state[f'train_model_{model_key}'] = True
                st.success(f"✅ Selected {len(rec.recommended_models)} {display_name}")
                st.rerun()

# Model selection and configuration
st.header("🤖 Model Configuration")
_prep_pipes = st.session_state.get("preprocessing_pipelines_by_model") or {}
_prep_models = [k for k in _prep_pipes.keys() if k != "default"]
if _prep_models:
    st.caption("Models with preprocessing pipelines are pre-selected. Adjust as needed.")

# Get registry and filter by task type (cached)
@st.cache_resource
def _get_registry_cached():
    """Cached registry access."""
    from ml.model_registry import get_registry
    return get_registry()

registry = _get_registry_cached()
available_models = {
    k: v for k, v in registry.items()
    if (task_type_final == 'regression' and v.capabilities.supports_regression) or
       (task_type_final == 'classification' and v.capabilities.supports_classification)
}

# Sync Train & Compare selections from Preprocessing (before any checkbox)
_prep_built = st.session_state.get("preprocess_built_model_keys", [])
for _k in _prep_built:
    if _k not in available_models:
        continue
    _key = f"train_model_{_k}"
    if _key not in st.session_state:
        st.session_state[_key] = True

# ============================================================================
# COACH-INTEGRATED MODEL BUCKETS
# ============================================================================
# Get comprehensive coach output if available
coach_output = st.session_state.get('coach_output')

# CSS for model buckets
st.markdown("""
<style>
.bucket-header {
    padding: 0.5rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-weight: 600;
}
.bucket-recommended { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
.bucket-worth-trying { background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }
.bucket-not-recommended { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
.model-rationale {
    font-size: 0.85rem;
    color: #666;
    padding: 0.25rem 0;
}
.training-time-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    margin-left: 0.5rem;
}
.time-fast { background: #d4edda; color: #155724; }
.time-medium { background: #fff3cd; color: #856404; }
.time-slow { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Initialize model_config and tracking variables (needed by both views and training)
model_config = st.session_state.get('model_config', ModelConfig())
models_to_train = []
selected_model_params = st.session_state.get('selected_model_params', {})

# Display models in buckets if we have coach output with detailed recommendations
if coach_output and hasattr(coach_output, 'recommended_models') and coach_output.recommended_models:
    st.caption("Selections sync with Preprocessing; pick models there first to build pipelines.")
    st.markdown("Models by fit: **Recommended** (best), **Worth Trying** (caveats), **Not Recommended** (limitations).")
    model_view = st.radio(
        "View models by:",
        ["Coach Recommendations", "Model Family"],
        horizontal=True,
        key="model_view_mode"
    )
    
    if model_view == "Coach Recommendations":
        # Tab-based view of buckets
        tab_rec, tab_try, tab_not = st.tabs([
            f"✅ Recommended ({len(coach_output.recommended_models)})",
            f"🔄 Worth Trying ({len(coach_output.worth_trying_models)})",
            f"⛔ Not Recommended ({len(coach_output.not_recommended_models)})"
        ])
        
        # Recommended Tab
        with tab_rec:
            st.markdown('<div class="bucket-header bucket-recommended">✅ Recommended Models</div>', 
                       unsafe_allow_html=True)
            st.caption("These models are well-suited to your dataset.")
            if st.button("Select all recommended", key="coach_select_all_recommended"):
                for rec in coach_output.recommended_models:
                    if rec.model_key in available_models:
                        st.session_state[f"train_model_{rec.model_key}"] = True
                st.success("Selected all recommended.")
                st.rerun()
            for rec in coach_output.recommended_models:
                if rec.model_key not in available_models:
                    continue
                    
                spec = available_models[rec.model_key]
                time_class = f"time-{rec.training_time.value}"
                time_label = rec.training_time.value.title()
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    checkbox_key = f"train_model_{rec.model_key}"
                    is_selected = st.checkbox(
                        f"**{rec.model_name}** ",
                        value=st.session_state.get(checkbox_key, False),
                        key=checkbox_key
                    )
                    st.markdown(f'<span class="model-rationale">{rec.dataset_fit_summary}</span>', 
                               unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<span class="training-time-badge {time_class}">{time_label}</span>',
                               unsafe_allow_html=True)
                
                if is_selected:
                    models_to_train.append(rec.model_key)
                    
                    # Show hyperparameters
                    if spec.hyperparam_schema:
                        with st.expander(f"⚙️ {rec.model_name} Settings"):
                            st.markdown(f"*{rec.rationale}*")
                            params = {}
                            for param_name, param_def in spec.hyperparam_schema.items():
                                param_key = f"{rec.model_key}_{param_name}"
                                if param_def['type'] == 'int':
                                    params[param_name] = st.number_input(
                                        param_def.get('help', param_name),
                                        min_value=param_def['min'],
                                        max_value=param_def['max'],
                                        value=param_def['default'],
                                        key=param_key
                                    )
                                elif param_def['type'] == 'float':
                                    format_str = "%.4f" if param_def.get('log', False) else "%.2f"
                                    params[param_name] = st.number_input(
                                        param_def.get('help', param_name),
                                        min_value=param_def['min'],
                                        max_value=param_def['max'],
                                        value=param_def['default'],
                                        format=format_str,
                                        key=param_key
                                    )
                                elif param_def['type'] == 'select':
                                    options = param_def['options']
                                    default_idx = options.index(param_def['default'])
                                    params[param_name] = st.selectbox(
                                        param_def.get('help', param_name),
                                        options=options,
                                        index=default_idx,
                                        key=param_key
                                    )
                            selected_model_params[rec.model_key] = params
                
                st.markdown("---")
        
        # Worth Trying Tab
        with tab_try:
            st.markdown('<div class="bucket-header bucket-worth-trying">🔄 Worth Trying</div>', 
                       unsafe_allow_html=True)
            st.caption("These models may work but have some caveats.")
            if st.button("Select all worth trying", key="coach_select_all_worth_trying"):
                for rec in coach_output.worth_trying_models:
                    if rec.model_key in available_models:
                        st.session_state[f"train_model_{rec.model_key}"] = True
                st.success("Selected all worth trying.")
                st.rerun()
            for rec in coach_output.worth_trying_models:
                if rec.model_key not in available_models:
                    continue
                    
                spec = available_models[rec.model_key]
                
                with st.expander(f"**{rec.model_name}** — {rec.dataset_fit_summary}"):
                    st.markdown(f"*{rec.rationale}*")
                    if rec.risks:
                        st.warning("**Risks:** " + "; ".join(rec.risks[:2]))
                    
                    checkbox_key = f"train_model_{rec.model_key}"
                    is_selected = st.checkbox(
                        f"Select {rec.model_name} for training",
                        value=st.session_state.get(checkbox_key, False),
                        key=checkbox_key
                    )
                    
                    if is_selected:
                        models_to_train.append(rec.model_key)
                        if spec.hyperparam_schema:
                            params = {}
                            for param_name, param_def in spec.hyperparam_schema.items():
                                param_key = f"{rec.model_key}_{param_name}"
                                if param_def['type'] == 'int':
                                    params[param_name] = st.number_input(
                                        param_def.get('help', param_name),
                                        min_value=param_def['min'],
                                        max_value=param_def['max'],
                                        value=param_def['default'],
                                        key=param_key
                                    )
                                elif param_def['type'] == 'float':
                                    params[param_name] = st.number_input(
                                        param_def.get('help', param_name),
                                        min_value=param_def['min'],
                                        max_value=param_def['max'],
                                        value=param_def['default'],
                                        format="%.4f",
                                        key=param_key
                                    )
                                elif param_def['type'] == 'select':
                                    options = param_def['options']
                                    default_idx = options.index(param_def['default'])
                                    params[param_name] = st.selectbox(
                                        param_def.get('help', param_name),
                                        options=options,
                                        index=default_idx,
                                        key=param_key
                                    )
                            selected_model_params[rec.model_key] = params
        
        # Not Recommended Tab
        with tab_not:
            st.markdown('<div class="bucket-header bucket-not-recommended">⛔ Not Recommended</div>', 
                       unsafe_allow_html=True)
            st.caption("These models are not well-suited for your current dataset. Use with caution.")
            
            for rec in coach_output.not_recommended_models:
                if rec.model_key not in available_models:
                    continue
                    
                spec = available_models[rec.model_key]
                
                with st.expander(f"**{rec.model_name}** — Why not recommended"):
                    st.error(f"**Reason:** {rec.rationale}")
                    if rec.risks:
                        for risk in rec.risks[:3]:
                            st.markdown(f"• ⚠️ {risk}")
                    
                    st.markdown(f"**When this model IS appropriate:** {rec.when_to_use}")
                    
                    checkbox_key = f"train_model_{rec.model_key}"
                    is_selected = st.checkbox(
                        f"Train anyway (not recommended)",
                        value=st.session_state.get(checkbox_key, False),
                        key=checkbox_key
                    )
                    
                    if is_selected:
                        models_to_train.append(rec.model_key)
                        st.warning("⚠️ You've selected a model that may not perform well on your data.")
    else:
        # Fall through to family-based view
        pass
else:
    model_view = "Model Family"  # Default to family view if no coach output

# Family-based model selection (original view or when coach not available)
if model_view == "Model Family" or not coach_output:
    # Group models by group
    model_groups = {}
    for key, spec in available_models.items():
        group = spec.group
        if group not in model_groups:
            model_groups[group] = []
        model_groups[group].append((key, spec))

    # Define advanced model groups
    advanced_groups = ['Margin', 'Probabilistic']  # SVM, Naive Bayes, LDA

    # Check if there are any advanced models available for the current task type
    advanced_model_count = sum(
        len(model_groups.get(group, [])) 
        for group in advanced_groups 
        if group in model_groups
    )

    # Advanced models toggle - only show if there are advanced models
    if advanced_model_count > 0:
        show_advanced = st.checkbox(
            f"Show Advanced Models ({advanced_model_count} available)", 
            value=st.session_state.get('show_advanced_models', False), 
            key="show_advanced_models",
            help="Advanced models include SVMs, Naive Bayes, and Linear Discriminant Analysis"
        )
    else:
        show_advanced = False
        st.caption("ℹ️ No advanced models available for this task type.")

    # Display models by group
    for group_name in sorted(model_groups.keys()):
        if group_name in advanced_groups and not show_advanced:
            continue
        
        st.subheader(f"{group_name} Models")
        group_models = model_groups[group_name]
        
        for model_key, spec in group_models:
            # Check if model is selected
            checkbox_key = f"train_model_{model_key}"
            is_selected = st.checkbox(
                spec.name,
                value=st.session_state.get(checkbox_key, False),
                key=checkbox_key,
                help=", ".join(spec.capabilities.notes) if spec.capabilities.notes else None
            )
            
            if is_selected:
                models_to_train.append(model_key)
                
                # Hyperparameter controls
                if spec.hyperparam_schema:
                    with st.expander(f"{spec.name} Hyperparameters"):
                        params = {}
                        for param_name, param_def in spec.hyperparam_schema.items():
                            param_key = f"{model_key}_{param_name}"
                            if param_def['type'] == 'int':
                                params[param_name] = st.number_input(
                                    param_def.get('help', param_name),
                                    min_value=param_def['min'],
                                    max_value=param_def['max'],
                                    value=param_def['default'],
                                    key=param_key
                                )
                            elif param_def['type'] == 'float':
                                format_str = "%.4f" if param_def.get('log', False) else "%.2f"
                                params[param_name] = st.number_input(
                                    param_def.get('help', param_name),
                                    min_value=param_def['min'],
                                    max_value=param_def['max'],
                                    value=param_def['default'],
                                    format=format_str,
                                    key=param_key
                                )
                            elif param_def['type'] == 'select':
                                options = param_def['options']
                                default_idx = options.index(param_def['default'])
                                params[param_name] = st.selectbox(
                                    param_def.get('help', param_name),
                                    options=options,
                                    index=default_idx,
                                    key=param_key
                                )
                            elif param_def['type'] == 'int_or_none':
                                # Special handling for max_depth=None
                                use_none = st.checkbox(f"{param_name} = None (unlimited)", value=param_def['default'] is None, key=f"{param_key}_none")
                                if use_none:
                                    params[param_name] = None
                                else:
                                    params[param_name] = st.number_input(
                                        param_def.get('help', param_name),
                                        min_value=param_def['min'],
                                        max_value=param_def['max'],
                                        value=param_def['min'] if param_def['default'] is None else param_def['default'],
                                        key=param_key
                                    )
                        
                        selected_model_params[model_key] = params

st.session_state.model_config = model_config

# Pre-training coach tips
coach_output = st.session_state.get('coach_output')
with st.expander("🎓 Pre-training Coach Tips", expanded=False):
    if coach_output and hasattr(coach_output, 'preprocessing_recommendations') and coach_output.preprocessing_recommendations:
        st.markdown("**Preprocessing checklist (from Coach):**")
        for prep in coach_output.preprocessing_recommendations[:5]:
            st.markdown(f"- **{prep.step_name}** ({prep.priority}): {prep.rationale}")
        st.caption("Configure these in the Preprocessing page before building the pipeline.")
    else:
        st.info("Run EDA and check the Model Selection Coach for preprocessing recommendations.")
    st.markdown("**Tip:** Ensure your preprocessing pipeline matches your selected models. Linear models and neural nets require scaling; tree models do not.")

# Training
if st.button("🚀 Train Models", type="primary", key="train_models_button") and models_to_train:
    # Lazy import model wrappers and evaluation functions only when training
    NNWeightedHuberWrapper, GLMWrapper, HuberGLMWrapper, RFWrapper, RegistryModelWrapper = _get_model_wrappers()
    calculate_regression_metrics, calculate_classification_metrics, perform_cross_validation, analyze_residuals = _get_eval_functions()
    from sklearn.pipeline import Pipeline as SklearnPipeline
    
    progress_container = st.container()
    
    for model_name in models_to_train:
        with progress_container:
            st.subheader(f"Training {model_name.upper()}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Get model spec from registry
                spec = registry.get(model_name)
                model_pipeline = get_preprocessing_pipeline(model_name) or pipeline
                if model_pipeline is None:
                    st.error("Preprocessing pipeline not found for this model.")
                    continue

                # Fit preprocessing on training data only
                model_pipeline.fit(X_train)
                X_train_model = model_pipeline.transform(X_train)
                X_val_model = model_pipeline.transform(X_val)
                X_test_model = model_pipeline.transform(X_test)
                if hasattr(X_train_model, 'toarray'):
                    X_train_model = X_train_model.toarray()
                    X_val_model = X_val_model.toarray()
                    X_test_model = X_test_model.toarray()
                
                # Handle existing wrappers (nn, rf, glm, huber) with special logic
                if model_name == 'nn':
                    params = selected_model_params.get(model_name, {})
                    
                    # Compute hidden_layers from architecture parameters
                    num_layers = params.get('num_layers', 2)
                    layer_width = params.get('layer_width', 32)
                    pattern = params.get('architecture_pattern', 'constant')
                    
                    if pattern == 'constant':
                        hidden_layers = [layer_width] * num_layers
                    elif pattern == 'pyramid':
                        # Increasing width: 32 -> 64 -> 128
                        hidden_layers = [layer_width * (2 ** i) for i in range(num_layers)]
                    elif pattern == 'funnel':
                        # Decreasing width: 128 -> 64 -> 32
                        max_width = layer_width * (2 ** (num_layers - 1))
                        hidden_layers = [max_width // (2 ** i) for i in range(num_layers)]
                    else:
                        hidden_layers = [layer_width] * num_layers
                    
                    status_text.text(f"Architecture: {hidden_layers} ({pattern})")
                    
                    model = NNWeightedHuberWrapper(
                        hidden_layers=hidden_layers,
                        dropout=params.get('dropout', model_config.nn_dropout),
                        task_type=task_type_final,
                        activation=params.get('activation', 'relu')
                    )
                    def progress_cb(epoch, train_loss, val_loss, val_metric):
                        epochs = params.get('epochs', model_config.nn_epochs)
                        progress = epoch / epochs
                        progress_bar.progress(progress)
                        if task_type_final == 'regression':
                            status_text.text(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val RMSE: {val_metric:.4f}")
                        else:
                            status_text.text(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Accuracy: {val_metric:.4f}")
                    
                    results = model.fit(
                        X_train_model, y_train, X_val_model, y_val,
                        epochs=params.get('epochs', model_config.nn_epochs),
                        batch_size=params.get('batch_size', model_config.nn_batch_size),
                        lr=params.get('lr', model_config.nn_lr),
                        weight_decay=params.get('weight_decay', model_config.nn_weight_decay),
                        patience=params.get('patience', model_config.nn_patience),
                        progress_callback=progress_cb,
                        random_seed=st.session_state.get('random_seed', 42)
                    )
                    
                    # Store architecture info in results for reporting
                    results['architecture'] = model.get_architecture_summary()
                
                elif model_name == 'rf':
                    params = selected_model_params.get(model_name, {})
                    model = RFWrapper(
                        n_estimators=params.get('n_estimators', model_config.rf_n_estimators),
                        max_depth=params.get('max_depth', model_config.rf_max_depth),
                        min_samples_leaf=params.get('min_samples_leaf', model_config.rf_min_samples_leaf),
                        task_type=task_type_final
                    )
                    results = model.fit(X_train_model, y_train, X_val_model, y_val)
                
                elif model_name == 'glm':
                    model = GLMWrapper(task_type=task_type_final)
                    results = model.fit(X_train_model, y_train, X_val_model, y_val)
                
                elif model_name == 'huber':
                    params = selected_model_params.get(model_name, {})
                    model = HuberGLMWrapper(
                        epsilon=params.get('epsilon', model_config.huber_epsilon),
                        alpha=params.get('alpha', model_config.huber_alpha)
                    )
                    results = model.fit(X_train_model, y_train, X_val_model, y_val)
                
                else:
                    # New registry models: create estimator and wrap
                    if spec is None:
                        st.error(f"Model spec not found for {model_name}")
                        continue
                    
                    params = selected_model_params.get(model_name, spec.default_params)
                    random_seed = st.session_state.get('random_seed', 42)
                    
                    # Create estimator from factory
                    estimator = spec.factory(task_type_final, random_seed)
                    
                    # Set hyperparameters
                    for param_name, param_value in params.items():
                        if hasattr(estimator, param_name):
                            setattr(estimator, param_name, param_value)
                    
                    # Wrap in generic wrapper
                    model = RegistryModelWrapper(estimator, spec.name)
                    
                    # Fit model
                    results = model.fit(X_train_model, y_train, X_val_model, y_val)
                
                # Evaluate on test set
                y_test_pred = model.predict(X_test_model)
                
                if task_type_final == 'regression':
                    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
                else:
                    y_test_proba = model.predict_proba(X_test_model) if model.supports_proba() else None
                    test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
                
                # Cross-validation if enabled (skip for NN - PyTorch models don't implement sklearn interface)
                cv_results = None
                if use_cv and model_name != 'nn':
                    try:
                        cv_results = perform_cross_validation(
                            model.get_model(), X_train_model, y_train,
                            cv_folds=cv_folds, task_type=data_config.task_type
                        )
                    except Exception as cv_error:
                        st.warning(f"⚠️ Cross-validation failed for {model_name}: {cv_error}. Skipping CV.")
                        logger.warning(f"CV failed for {model_name}: {cv_error}")
                elif use_cv and model_name == 'nn':
                    st.info(f"ℹ️ Cross-validation skipped for Neural Network (PyTorch models use their own validation loop during training)")
                
                # Store results
                model_results = {
                    'metrics': test_metrics,
                    'history': results.get('history', {}),
                    'y_test_pred': y_test_pred,
                    'y_test': y_test,
                    'cv_results': cv_results
                }
                
                add_trained_model(model_name, model, model_results)
                
                # Store fitted estimator for explainability
                # For explainability, we need a pipeline that can handle raw data
                # Store both the fitted model and the preprocessing pipeline
                if model_name == 'nn':
                    # NN needs special handling - store sklearn-compatible wrapper
                    fitted_estimator = model.get_sklearn_estimator()
                    if not (hasattr(fitted_estimator, 'is_fitted_') and fitted_estimator.is_fitted_):
                        fitted_estimator.fit(X_train_model[:1], y_train[:1])
                    st.session_state.fitted_estimators[model_name] = fitted_estimator
                else:
                    # For sklearn models, store the fitted model
                    sklearn_model = model.get_model()
                    st.session_state.fitted_estimators[model_name] = sklearn_model
                
                # Store preprocessing pipeline for all models (needed for explainability)
                st.session_state.fitted_preprocessing_pipelines[model_name] = model_pipeline
                from ml.pipeline import get_feature_names_after_transform
                st.session_state.feature_names_by_model[model_name] = get_feature_names_after_transform(
                    model_pipeline, data_config.feature_cols
                )
                
                progress_bar.progress(1.0)
                st.success(f"✅ {model_name.upper()} training complete!")
                
            except Exception as e:
                with st.expander(f"❌ Error training {model_name.upper()}", expanded=True):
                    st.error(f"Training failed: {str(e)}")
                    st.code(str(e), language='python')
                    logger.exception(e)

# Results comparison
if st.session_state.get('trained_models'):
    # Lazy import plotly and visualization functions for results display
    go, px = _get_plotly()
    plot_training_history, plot_predictions_vs_actual, plot_residuals = _get_visualization_functions()
    calculate_regression_metrics, calculate_classification_metrics, perform_cross_validation, analyze_residuals = _get_eval_functions()
    
    st.header("📊 Results Comparison")
    
    # How to read results explainer
    with st.expander("📚 How to Read These Results", expanded=False):
        if task_type_final == 'regression':
            st.markdown("""
            **Metrics:**
            - **RMSE (Root Mean Squared Error):** Average prediction error in target units. Lower is better.
            - **MAE (Mean Absolute Error):** Average absolute error. Less sensitive to outliers than RMSE.
            - **R² (R-squared):** Proportion of variance explained. 1.0 = perfect, 0 = no better than mean.
            - **MedianAE:** Median absolute error. Robust to outliers.
            
            **Cross-Validation vs Holdout:**
            - **Holdout:** Single train/test split. Fast but may be noisy.
            - **Cross-Validation:** Multiple splits. More stable estimate but slower.
            """)
        else:
            st.markdown("""
            **Metrics:**
            - **Accuracy:** Proportion of correct predictions. Can be misleading with class imbalance.
            - **F1 Score:** Harmonic mean of precision and recall. Better for imbalanced data.
            - **ROC-AUC:** Area under ROC curve. Measures separability of classes.
            - **PR-AUC:** Precision-Recall AUC. Better for imbalanced data than ROC-AUC.
            - **Log Loss:** Penalizes confident wrong predictions. Lower is better.
            
            **Calibration:**
            - Well-calibrated models: predicted probabilities match actual frequencies
            - Important for medical decision-making
            - Check calibration plots if available
            """)
    
    # Metrics table
    comparison_data = []
    for name, results in st.session_state.model_results.items():
        row = {'Model': name.upper()}
        row.update(results['metrics'])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if data_config.task_type == 'regression':
        comparison_df = comparison_df.sort_values('RMSE')
        st.dataframe(
            comparison_df.style.highlight_min(subset=['RMSE', 'MAE'], axis=0, color='lightgreen')
            .highlight_max(subset=['R2'], axis=0, color='lightgreen'),
            use_container_width=True
        )
    else:
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        st.dataframe(
            comparison_df.style.highlight_max(subset=['Accuracy', 'F1'], axis=0, color='lightgreen'),
            use_container_width=True
        )
    
    # CV results if available
    if use_cv:
        st.subheader("Cross-Validation Results")
        cv_data = []
        for name, results in st.session_state.model_results.items():
            if results.get('cv_results'):
                cv_data.append({
                    'Model': name.upper(),
                    'Mean Score': results['cv_results']['mean'],
                    'Std Score': results['cv_results']['std']
                })
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            st.dataframe(cv_df, use_container_width=True)
            
            # Boxplot of CV scores
            fig = go.Figure()
            for name, results in st.session_state.model_results.items():
                if results.get('cv_results'):
                    fig.add_trace(go.Box(
                        y=results['cv_results']['scores'],
                        name=name.upper()
                    ))
            fig.update_layout(title="CV Score Distribution", yaxis_title="Score")
            st.plotly_chart(fig, use_container_width=True)
    
    # Model diagnostics (one tab per model so pred-vs-actual etc. visible for all)
    st.header("🔍 Model Diagnostics")
    model_names = list(st.session_state.trained_models.keys())
    if not model_names:
        st.info("No models to show.")
    else:
        tabs = st.tabs([f"{n.upper()}" for n in model_names])
        _fn_by_model = st.session_state.get("feature_names_by_model", {})
        for tab, name in zip(tabs, model_names):
            with tab:
                model = st.session_state.trained_models[name]
                results = st.session_state.model_results[name]
                _feats = _fn_by_model.get(name) or (data_config.feature_cols if data_config else [])
                _n_test = len(results.get("y_test", []))
                _task = data_config.task_type if data_config else None

                fitted_prep = st.session_state.get("fitted_preprocessing_pipelines", {}).get(name)
                if fitted_prep is not None:
                    from ml.pipeline import get_pipeline_recipe
                    st.subheader("Preprocessing used")
                    st.caption(f"Pipeline for **{name.upper()}**")
                    st.code(get_pipeline_recipe(fitted_prep), language=None)
                    st.markdown("---")

                st.subheader("Test Set Metrics")
                metrics = results["metrics"]
                metric_cols = st.columns(len(metrics))
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    with metric_cols[i]:
                        st.metric(metric_name, f"{metric_value:.4f}")

                if name == "nn" and results.get("history", {}).get("train_loss"):
                    st.subheader("Learning Curves")
                    st.plotly_chart(plot_training_history(results["history"]), use_container_width=True, key=f"diag_lc_{name}")
                    from ml.plot_narrative import narrative_learning_curves
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                    nar = narrative_learning_curves(results["history"])
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    h = results["history"]
                    tl, vl = h.get("train_loss", []), h.get("val_loss", h.get("train_loss", []))
                    stats_summary = f"train_loss={tl[-1]:.4f}; val_loss={vl[-1]:.4f}" if tl else ""
                    ctx = build_llm_context("learning_curves", stats_summary, model_name=name, existing=nar or "", metrics=results.get("metrics"), feature_names=_feats, sample_size=_n_test, task_type=_task)
                    render_interpretation_with_llm_button(ctx, key=f"llm_lc_{name}", result_session_key=f"llm_result_lc_{name}")

                if data_config.task_type == "regression":
                    st.subheader("Predictions vs Actual")
                    st.plotly_chart(
                        plot_predictions_vs_actual(results["y_test"], results["y_test_pred"], title=f"{name.upper()} Predictions"),
                        use_container_width=True,
                        key=f"diag_pva_{name}",
                    )
                    st.caption("The dashed red line (y = x) represents perfect agreement. Points closer to it indicate better predictions.")
                    from ml.eval import analyze_pred_vs_actual
                    from ml.plot_narrative import narrative_pred_vs_actual
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                    pva_stats = analyze_pred_vs_actual(results["y_test"], results["y_test_pred"])
                    nar = narrative_pred_vs_actual(pva_stats, model_name=name)
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    stats_summary = f"corr={pva_stats.get('correlation', 0):.3f}; mean_err={pva_stats.get('mean_error', 0):.4f}"
                    ctx = build_llm_context("pred_vs_actual", stats_summary, model_name=name, existing=nar or "", metrics=results.get("metrics"), feature_names=_feats, sample_size=_n_test, task_type=_task)
                    render_interpretation_with_llm_button(ctx, key=f"llm_pva_{name}", result_session_key=f"llm_result_pva_{name}")

                    st.subheader("Residuals")
                    st.plotly_chart(
                        plot_residuals(results["y_test"], results["y_test_pred"], title=f"{name.upper()} Residuals"),
                        use_container_width=True,
                        key=f"diag_resid_{name}",
                    )
                    from ml.eval import analyze_residuals_extended
                    from ml.plot_narrative import narrative_residuals
                    resid_stats = analyze_residuals_extended(results["y_test"], results["y_test_pred"])
                    nar = narrative_residuals(resid_stats, model_name=name)
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    else:
                        res_basic = analyze_residuals(results["y_test"], results["y_test_pred"])
                        st.caption(f"Mean residual: {res_basic['mean_residual']:.4f} | Std: {res_basic['std_residual']:.4f}")
                    stats_summary = f"skew={resid_stats.get('skew', 0):.3f}; iqr={resid_stats.get('iqr', 0):.4f}; rvp={resid_stats.get('residual_vs_predicted_corr', 0):.3f}"
                    ctx = build_llm_context("residuals", stats_summary, model_name=name, existing=nar or "", metrics=results.get("metrics"), feature_names=_feats, sample_size=_n_test, task_type=_task)
                    render_interpretation_with_llm_button(ctx, key=f"llm_resid_{name}", result_session_key=f"llm_result_resid_{name}")
                else:
                    st.subheader("Classification Performance")
                    if model.supports_proba():
                        st.info("For classification, see Confusion Matrix and metrics above. ROC/PR curves can be added later.")
                    else:
                        st.info("This model does not support probability predictions. See Confusion Matrix above.")
                    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
                    from ml.eval import analyze_confusion_matrix
                    from ml.plot_narrative import narrative_confusion_matrix
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                    cm = sk_confusion_matrix(results["y_test"], results["y_test_pred"])
                    fig_cm = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues")
                    st.plotly_chart(fig_cm, use_container_width=True, key=f"diag_cm_{name}")
                    cm_stats = analyze_confusion_matrix(results["y_test"], results["y_test_pred"])
                    nar = narrative_confusion_matrix(cm_stats, model_name=name)
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    per = cm_stats.get("per_class", [])[:3]
                    stats_summary = "; ".join(f"{p.get('label','?')}: P={p.get('precision',0):.2f} R={p.get('recall',0):.2f}" for p in per) if per else ""
                    ctx = build_llm_context("confusion_matrix", stats_summary, model_name=name, existing=nar or "", metrics=results.get("metrics"), feature_names=_feats, sample_size=_n_test, task_type=_task)
                    render_interpretation_with_llm_button(ctx, key=f"llm_cm_{name}", result_session_key=f"llm_result_cm_{name}")
