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

pipeline = get_preprocessing_pipeline()
if pipeline is None:
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
        
        # Apply preprocessing (convert sparse to dense if needed)
        X_transformed = pipeline.transform(X)
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        # Split data (group-based, time-based, or random)
        if use_group_split and entity_id_final:
            # Group-based split for longitudinal data
            groups = to_numpy_1d(df[entity_id_final])
            y_arr = to_numpy_1d(y)
            
            gss = GroupShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=split_config.random_state)
            train_idx, temp_idx = next(gss.split(X_transformed, y_arr, groups))
            
            # Split temp into val and test, maintaining groups
            groups_temp = groups[temp_idx]
            rel_val = val_size / (val_size + test_size)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=(1 - rel_val), random_state=split_config.random_state)
            val_idx, test_idx = next(gss2.split(X_transformed[temp_idx], y_arr[temp_idx], groups_temp))
            
            X_train = X_transformed[train_idx]
            X_val = X_transformed[temp_idx[val_idx]]
            X_test = X_transformed[temp_idx[test_idx]]
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
            
            X_train = X_transformed[train_indices]
            X_val = X_transformed[val_indices]
            X_test = X_transformed[test_indices]
            y_train = to_numpy_1d(y.iloc[train_indices])
            y_val = to_numpy_1d(y.iloc[val_indices])
            y_test = to_numpy_1d(y.iloc[test_indices])
            
            st.info(f"⏰ Time-based split: Train={df_with_datetime.iloc[0][data_config.datetime_col]} to {df_with_datetime.iloc[n_train-1][data_config.datetime_col]}")
        elif split_config.stratify and task_type_final == 'classification':
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_transformed, y, test_size=(val_size + test_size),
                random_state=split_config.random_state, stratify=y
            )
            rel_val = val_size / (val_size + test_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - rel_val),
                random_state=split_config.random_state, stratify=y_temp
            )
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_transformed, y, test_size=(val_size + test_size),
                random_state=split_config.random_state
            )
            rel_val = val_size / (val_size + test_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=(1 - rel_val),
                random_state=split_config.random_state
            )
        
        # Get feature names after transformation (handle sparse matrices)
        from ml.pipeline import get_feature_names_after_transform
        feature_names = get_feature_names_after_transform(pipeline, data_config.feature_cols)
        
        set_splits(X_train, X_val, X_test, to_numpy_1d(y_train), to_numpy_1d(y_val), to_numpy_1d(y_test), list(feature_names))
        
        # Store indices for explainability (need raw data)
        if use_group_split and entity_id_final:
            st.session_state.train_indices = train_idx.tolist() if hasattr(train_idx, 'tolist') else list(train_idx)
            st.session_state.test_indices = list(temp_idx[test_idx]) if hasattr(test_idx, 'tolist') else temp_idx[test_idx]
        elif split_config.use_time_split and data_config.datetime_col:
            st.session_state.train_indices = train_indices.tolist() if hasattr(train_indices, 'tolist') else list(train_indices)
            st.session_state.test_indices = test_indices.tolist() if hasattr(test_indices, 'tolist') else list(test_indices)
        else:
            # For random splits, we don't have original indices easily, so use range
            # This is a limitation - explainability will need to reconstruct
            st.session_state.train_indices = None
            st.session_state.test_indices = None
        
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
    signals = compute_dataset_signals(df, target_col, task_type, cohort_type, entity_id)
    eda_results = st.session_state.get('eda_results')
    return coach_recommendations(signals, eda_results, get_insights_by_category())

# Show relevant insights
insights = get_insights_by_category()
if insights:
    with st.expander("💡 Key Insights from EDA", expanded=True):
        for insight in insights:
            st.markdown(f"**{insight.get('category', 'General').title()}:** {insight['finding']}")
            st.caption(f"→ {insight['implication']}")

# Compute coach recommendations (using cached function)
# Create a hash for the dataframe to use as cache key
_df_hash = hash((len(df), tuple(df.columns)))
_eda_results_keys = tuple(sorted(st.session_state.get('eda_results', {}).keys()))
coach_recs = _compute_coach_recommendations(
    _df_hash, data_config.target_col, task_type_final, cohort_type_final, entity_id_final, _eda_results_keys
)

if coach_recs:
    with st.expander("🎓 Model Selection Coach", expanded=True):
        st.markdown("**Based on your data, try these models first:**")
        for idx, rec in enumerate(coach_recs[:3]):  # Top 3
            with st.container():
                # Use display_name for consistent naming
                display_name = rec.display_name if hasattr(rec, 'display_name') else f"{rec.group} Models"
                priority_label = "High" if rec.priority <= 2 else "Medium"
                st.markdown(f"### {display_name} ({priority_label} Priority)")
                
                # Show readiness checks if any
                if hasattr(rec, 'readiness_checks') and rec.readiness_checks:
                    st.warning("⚠️ **Recommended prerequisites:**")
                    for check in rec.readiness_checks:
                        st.write(f"• {check}")
                
                st.markdown("**Why:**")
                for reason in rec.why[:5]:  # Limit to 5 reasons
                    st.write(f"• {reason}")
                if rec.when_not_to_use:
                    st.markdown("**When not to use:**")
                    for caveat in rec.when_not_to_use[:3]:  # Limit to 3
                        st.write(f"• {caveat}")
                if rec.suggested_preprocessing:
                    st.markdown("**Suggested preprocessing:**")
                    for prep in rec.suggested_preprocessing:
                        st.write(f"• {prep}")
                
                # Auto-select button with unique key (include priority and index)
                button_key = f"coach_select_{rec.group}_p{rec.priority}_i{idx}"
                if st.button(f"Select {display_name}", key=button_key):
                    # Store recommended models in session_state
                    for model_key in rec.recommended_models:
                        st.session_state[f'train_model_{model_key}'] = True
                    st.success(f"✅ Selected {len(rec.recommended_models)} {display_name}")
                    st.rerun()

# Model selection and configuration
st.header("🤖 Model Configuration")

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

# Group models by group
model_groups = {}
for key, spec in available_models.items():
    group = spec.group
    if group not in model_groups:
        model_groups[group] = []
    model_groups[group].append((key, spec))

# Advanced models toggle
show_advanced = st.checkbox("Show Advanced Models", value=st.session_state.get('show_advanced_models', False), key="show_advanced_models")
advanced_groups = ['Margin', 'Probabilistic']  # SVM, Naive Bayes, LDA

model_config = ModelConfig()
models_to_train = []
selected_model_params = {}

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
                
                # Handle existing wrappers (nn, rf, glm, huber) with special logic
                if model_name == 'nn':
                    params = selected_model_params.get(model_name, {})
                    model = NNWeightedHuberWrapper(
                        dropout=params.get('dropout', model_config.nn_dropout),
                        task_type=task_type_final
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
                        X_train, y_train, X_val, y_val,
                        epochs=params.get('epochs', model_config.nn_epochs),
                        batch_size=params.get('batch_size', model_config.nn_batch_size),
                        lr=params.get('lr', model_config.nn_lr),
                        weight_decay=params.get('weight_decay', model_config.nn_weight_decay),
                        patience=params.get('patience', model_config.nn_patience),
                        progress_callback=progress_cb,
                        random_seed=st.session_state.get('random_seed', 42)
                    )
                
                elif model_name == 'rf':
                    params = selected_model_params.get(model_name, {})
                    model = RFWrapper(
                        n_estimators=params.get('n_estimators', model_config.rf_n_estimators),
                        max_depth=params.get('max_depth', model_config.rf_max_depth),
                        min_samples_leaf=params.get('min_samples_leaf', model_config.rf_min_samples_leaf),
                        task_type=task_type_final
                    )
                    results = model.fit(X_train, y_train, X_val, y_val)
                
                elif model_name == 'glm':
                    model = GLMWrapper(task_type=task_type_final)
                    results = model.fit(X_train, y_train, X_val, y_val)
                
                elif model_name == 'huber':
                    params = selected_model_params.get(model_name, {})
                    model = HuberGLMWrapper(
                        epsilon=params.get('epsilon', model_config.huber_epsilon),
                        alpha=params.get('alpha', model_config.huber_alpha)
                    )
                    results = model.fit(X_train, y_train, X_val, y_val)
                
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
                    results = model.fit(X_train, y_train, X_val, y_val)
                
                # Evaluate on test set
                y_test_pred = model.predict(X_test)
                
                if task_type_final == 'regression':
                    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
                else:
                    y_test_proba = model.predict_proba(X_test) if model.supports_proba() else None
                    test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
                
                # Cross-validation if enabled (skip for NN - PyTorch models don't implement sklearn interface)
                cv_results = None
                if use_cv and model_name != 'nn':
                    try:
                        cv_results = perform_cross_validation(
                            model.get_model(), X_train, y_train,
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
                    if not hasattr(fitted_estimator, 'is_fitted_') or not fitted_estimator.is_fitted_:
                        fitted_estimator.fit(X_train[:1], y_train[:1])
                    st.session_state.fitted_estimators[model_name] = fitted_estimator
                else:
                    # For sklearn models, store the fitted model
                    sklearn_model = model.get_model()
                    st.session_state.fitted_estimators[model_name] = sklearn_model
                
                # Store preprocessing pipeline for all models (needed for explainability)
                st.session_state.fitted_preprocessing_pipelines[model_name] = pipeline
                
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
    
    # Model diagnostics
    st.header("🔍 Model Diagnostics")
    
    model_tabs = st.tabs([name.upper() for name in st.session_state.trained_models.keys()])
    
    for idx, (name, model) in enumerate(st.session_state.trained_models.items()):
        with model_tabs[idx]:
            results = st.session_state.model_results[name]
            
            # Metrics
            st.subheader("Test Set Metrics")
            metrics = results['metrics']
            metric_cols = st.columns(len(metrics))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                with metric_cols[i]:
                    st.metric(metric_name, f"{metric_value:.4f}")
            
            # Learning curves (for NN)
            if name == 'nn' and 'history' in results and results['history'].get('train_loss'):
                st.subheader("Learning Curves")
                fig_history = plot_training_history(results['history'])
                st.plotly_chart(fig_history, use_container_width=True)
            
            # Predictions vs Actual (regression only)
            if data_config.task_type == 'regression':
                st.subheader("Predictions vs Actual")
                fig_pred = plot_predictions_vs_actual(
                    results['y_test'],
                    results['y_test_pred'],
                    title=f"{name.upper()} Predictions"
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                st.caption("The dashed red line (y = x) represents perfect agreement between predictions and actual values. Points closer to this line indicate better predictions.")
            else:
                # Classification: show ROC/PR or note
                st.subheader("Classification Performance")
                if model.supports_proba():
                    st.info("ℹ️ For classification models, see the Confusion Matrix and metrics above. ROC/PR curves can be added in future updates.")
                else:
                    st.info("ℹ️ This model does not support probability predictions. See the Confusion Matrix above.")
            
            # Residuals (regression) or Confusion Matrix (classification)
            if data_config.task_type == 'regression':
                st.subheader("Residuals")
                fig_resid = plot_residuals(
                    results['y_test'],
                    results['y_test_pred'],
                    title=f"{name.upper()} Residuals"
                )
                st.plotly_chart(fig_resid, use_container_width=True)
                
                # Residual analysis
                residual_stats = analyze_residuals(results['y_test'], results['y_test_pred'])
                st.info(f"Mean residual: {residual_stats['mean_residual']:.4f} | "
                       f"Std residual: {residual_stats['std_residual']:.4f}")
            else:
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(results['y_test'], results['y_test_pred'])
                fig_cm = px.imshow(
                    cm, text_auto=True, aspect="auto",
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual"),
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
