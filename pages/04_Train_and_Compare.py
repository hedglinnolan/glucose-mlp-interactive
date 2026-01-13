"""
Page 04: Train and Compare Models
Train models, evaluate, compare metrics, show diagnostics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
import logging

from utils.session_state import (
    init_session_state, get_data, get_preprocessing_pipeline,
    DataConfig, SplitConfig, ModelConfig, set_splits, add_trained_model,
    TaskTypeDetection, CohortStructureDetection
)
from utils.seed import set_global_seed, get_global_seed
from models.nn_whuber import NNWeightedHuberWrapper
from models.glm import GLMWrapper
from models.huber_glm import HuberGLMWrapper
from models.rf import RFWrapper
from ml.eval import (
    calculate_regression_metrics, calculate_classification_metrics,
    perform_cross_validation, analyze_residuals
)
from ml.splits import to_numpy_1d
try:
    from visualizations import plot_training_history, plot_predictions_vs_actual, plot_residuals
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

logger = logging.getLogger(__name__)

init_session_state()

# Set global seed
set_global_seed(st.session_state.get('random_seed', 42))

st.set_page_config(page_title="Train & Compare", page_icon="🏋️", layout="wide")
st.title("🏋️ Train & Compare Models")

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

# Model selection and configuration
st.header("🤖 Model Configuration")

model_config = ModelConfig()
models_to_train = []

col1, col2 = st.columns(2)

with col1:
    train_nn = st.checkbox("Neural Network", value=st.session_state.get('train_model_nn', True), key="train_model_nn")
    if train_nn:
        if task_type_final == 'classification':
            st.info("ℹ️ Neural Network supports classification (BCE/CrossEntropy loss)")
        with st.expander("NN Hyperparameters"):
            model_config.nn_epochs = st.number_input("Epochs", 50, 500, 200, key="nn_epochs")
            model_config.nn_batch_size = st.number_input("Batch Size", 32, 512, 256, key="nn_batch")
            model_config.nn_lr = st.number_input("Learning Rate", 1e-5, 1e-2, 0.0015, format="%.4f", key="nn_lr")
            model_config.nn_weight_decay = st.number_input("Weight Decay", 0.0, 1e-2, 0.0002, format="%.4f", key="nn_wd")
            model_config.nn_patience = st.number_input("Early Stopping Patience", 5, 50, 30, key="nn_patience")
            model_config.nn_dropout = st.number_input("Dropout", 0.0, 0.5, 0.1, format="%.2f", key="nn_dropout")
        models_to_train.append('nn')

    train_rf = st.checkbox("Random Forest", value=st.session_state.get('train_model_rf', True), key="train_model_rf")
    if train_rf:
        with st.expander("RF Hyperparameters"):
            model_config.rf_n_estimators = st.number_input("N Estimators", 50, 1000, 500, key="rf_n_est")
            model_config.rf_max_depth = st.number_input("Max Depth", 1, 50, None, help="None = unlimited", key="rf_depth")
            model_config.rf_min_samples_leaf = st.number_input("Min Samples Leaf", 1, 20, 10, key="rf_leaf")
        models_to_train.append('rf')

with col2:
    train_glm = st.checkbox(
        "GLM (OLS)" if task_type_final == 'regression' else "GLM (Logistic)", 
        value=st.session_state.get('train_model_glm', True), 
        key="train_model_glm"
    )
    if train_glm:
        models_to_train.append('glm')

    train_huber = st.checkbox(
        "GLM (Huber)", 
        value=st.session_state.get('train_model_huber', task_type_final == 'regression'),
        key="train_model_huber"
    )
    if train_huber:
        if task_type_final == 'classification':
            st.warning("⚠️ Huber regression is for regression tasks only. Not suitable for classification.")
            train_huber = False
        else:
            with st.expander("Huber Hyperparameters"):
                model_config.huber_epsilon = st.number_input("Epsilon", 1.0, 2.0, 1.35, format="%.2f", key="huber_eps")
                model_config.huber_alpha = st.number_input("Alpha", 0.0, 1.0, 0.0, format="%.3f", key="huber_alpha")
            if train_huber:
                models_to_train.append('huber')

st.session_state.model_config = model_config

# Training
if st.button("🚀 Train Models", type="primary", key="train_models_button") and models_to_train:
    progress_container = st.container()
    
    for model_name in models_to_train:
        with progress_container:
            st.subheader(f"Training {model_name.upper()}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create model wrapper
                if model_name == 'nn':
                    model = NNWeightedHuberWrapper(
                        dropout=model_config.nn_dropout,
                        task_type=task_type_final
                    )
                    def progress_cb(epoch, train_loss, val_loss, val_metric):
                        progress = epoch / model_config.nn_epochs
                        progress_bar.progress(progress)
                        if task_type_final == 'regression':
                            status_text.text(f"Epoch {epoch}/{model_config.nn_epochs} | Loss: {train_loss:.4f} | Val RMSE: {val_metric:.4f}")
                        else:
                            status_text.text(f"Epoch {epoch}/{model_config.nn_epochs} | Loss: {train_loss:.4f} | Val Accuracy: {val_metric:.4f}")
                    
                    results = model.fit(
                        X_train, y_train, X_val, y_val,
                        epochs=model_config.nn_epochs,
                        batch_size=model_config.nn_batch_size,
                        lr=model_config.nn_lr,
                        weight_decay=model_config.nn_weight_decay,
                        patience=model_config.nn_patience,
                        progress_callback=progress_cb,
                        random_seed=st.session_state.get('random_seed', 42)
                    )
                
                elif model_name == 'rf':
                    model = RFWrapper(
                        n_estimators=model_config.rf_n_estimators,
                        max_depth=model_config.rf_max_depth,
                        min_samples_leaf=model_config.rf_min_samples_leaf,
                        task_type=task_type_final
                    )
                    results = model.fit(X_train, y_train, X_val, y_val)
                
                elif model_name == 'glm':
                    model = GLMWrapper(task_type=task_type_final)
                    results = model.fit(X_train, y_train, X_val, y_val)
                
                elif model_name == 'huber':
                    model = HuberGLMWrapper(
                        epsilon=model_config.huber_epsilon,
                        alpha=model_config.huber_alpha
                    )
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
                
                # Store fitted sklearn-compatible estimator for explainability
                # For NN, store the sklearn-compatible wrapper
                # For others, store the model itself (already sklearn-compatible)
                if model_name == 'nn':
                    fitted_estimator = model.get_sklearn_estimator()
                    # Ensure it's marked as fitted with correct attributes
                    # Use actual training data to set attributes properly
                    if not hasattr(fitted_estimator, 'is_fitted_') or not fitted_estimator.is_fitted_:
                        fitted_estimator.fit(X_train[:1], y_train[:1])
                    st.session_state.fitted_estimators[model_name] = fitted_estimator
                else:
                    # For sklearn models, store the model directly
                    # sklearn models are already fitted after model.fit() call above
                    # They don't have is_fitted_ but sklearn's check_is_fitted will work
                    sklearn_model = model.get_model()
                    st.session_state.fitted_estimators[model_name] = sklearn_model
                
                progress_bar.progress(1.0)
                st.success(f"✅ {model_name.upper()} training complete!")
                
            except Exception as e:
                with st.expander(f"❌ Error training {model_name.upper()}", expanded=True):
                    st.error(f"Training failed: {str(e)}")
                    st.code(str(e), language='python')
                    logger.exception(e)

# Results comparison
if st.session_state.get('trained_models'):
    st.header("📊 Results Comparison")
    
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
