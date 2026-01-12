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
from sklearn.model_selection import train_test_split
import logging

from utils.session_state import (
    init_session_state, get_data, get_preprocessing_pipeline,
    DataConfig, SplitConfig, ModelConfig, set_splits, add_trained_model
)
from models.nn_whuber import NNWeightedHuberWrapper
from models.glm import GLMWrapper
from models.huber_glm import HuberGLMWrapper
from models.rf import RFWrapper
from ml.eval import (
    calculate_regression_metrics, calculate_classification_metrics,
    perform_cross_validation, analyze_residuals
)
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

st.set_page_config(page_title="Train & Compare", page_icon="🏋️", layout="wide")
st.title("🏋️ Train & Compare Models")

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

# Split configuration
st.header("📊 Data Splitting")
col1, col2, col3 = st.columns(3)

with col1:
    train_size = st.slider("Train %", 50, 90, 70) / 100
with col2:
    val_size = st.slider("Val %", 5, 30, 15) / 100
with col3:
    test_size = st.slider("Test %", 5, 30, 15) / 100

if abs(train_size + val_size + test_size - 1.0) > 0.01:
    st.error("⚠️ Splits must sum to 100%")
    st.stop()

split_config = SplitConfig(
    train_size=train_size,
    val_size=val_size,
    test_size=test_size,
    stratify=(data_config.task_type == 'classification')
)
st.session_state.split_config = split_config

# Cross-validation option
use_cv = st.checkbox("Enable Cross-Validation", value=False)
if use_cv:
    cv_folds = st.slider("CV Folds", 3, 10, 5)
    st.session_state.use_cv = True
    st.session_state.cv_folds = cv_folds
else:
    st.session_state.use_cv = False

# Prepare data splits
if st.button("🔄 Prepare Splits", type="primary"):
    try:
        X = df[data_config.feature_cols]
        y = df[data_config.target_col]
        
        # Apply preprocessing
        X_transformed = pipeline.transform(X)
        
        # Split data
        if split_config.stratify and data_config.task_type == 'classification':
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
        
        # Get feature names after transformation
        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
        
        set_splits(X_train, X_val, X_test, y_train.values, y_val.values, y_test.values, list(feature_names))
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
    train_nn = st.checkbox("Neural Network", value=True)
    if train_nn:
        with st.expander("NN Hyperparameters"):
            model_config.nn_epochs = st.number_input("Epochs", 50, 500, 200)
            model_config.nn_batch_size = st.number_input("Batch Size", 32, 512, 256)
            model_config.nn_lr = st.number_input("Learning Rate", 1e-5, 1e-2, 0.0015, format="%.4f")
            model_config.nn_weight_decay = st.number_input("Weight Decay", 0.0, 1e-2, 0.0002, format="%.4f")
            model_config.nn_patience = st.number_input("Early Stopping Patience", 5, 50, 30)
            model_config.nn_dropout = st.number_input("Dropout", 0.0, 0.5, 0.1, format="%.2f")
        models_to_train.append('nn')

    train_rf = st.checkbox("Random Forest", value=True)
    if train_rf:
        with st.expander("RF Hyperparameters"):
            model_config.rf_n_estimators = st.number_input("N Estimators", 50, 1000, 500)
            model_config.rf_max_depth = st.number_input("Max Depth", 1, 50, None, help="None = unlimited")
            model_config.rf_min_samples_leaf = st.number_input("Min Samples Leaf", 1, 20, 10)
        models_to_train.append('rf')

with col2:
    train_glm = st.checkbox("GLM (OLS)", value=True)
    if train_glm:
        models_to_train.append('glm')

    train_huber = st.checkbox("GLM (Huber)", value=True)
    if train_huber:
        with st.expander("Huber Hyperparameters"):
            model_config.huber_epsilon = st.number_input("Epsilon", 1.0, 2.0, 1.35, format="%.2f")
            model_config.huber_alpha = st.number_input("Alpha", 0.0, 1.0, 0.0, format="%.3f")
        models_to_train.append('huber')

st.session_state.model_config = model_config

# Training
if st.button("🚀 Train Models", type="primary") and models_to_train:
    progress_container = st.container()
    
    for model_name in models_to_train:
        with progress_container:
            st.subheader(f"Training {model_name.upper()}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create model wrapper
                if model_name == 'nn':
                    model = NNWeightedHuberWrapper(dropout=model_config.nn_dropout)
                    def progress_cb(epoch, train_loss, val_loss, val_rmse):
                        progress = epoch / model_config.nn_epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch}/{model_config.nn_epochs} | Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")
                    
                    results = model.fit(
                        X_train, y_train, X_val, y_val,
                        epochs=model_config.nn_epochs,
                        batch_size=model_config.nn_batch_size,
                        lr=model_config.nn_lr,
                        weight_decay=model_config.nn_weight_decay,
                        patience=model_config.nn_patience,
                        progress_callback=progress_cb
                    )
                
                elif model_name == 'rf':
                    model = RFWrapper(
                        n_estimators=model_config.rf_n_estimators,
                        max_depth=model_config.rf_max_depth,
                        min_samples_leaf=model_config.rf_min_samples_leaf,
                        task_type=data_config.task_type
                    )
                    results = model.fit(X_train, y_train, X_val, y_val)
                
                elif model_name == 'glm':
                    model = GLMWrapper()
                    results = model.fit(X_train, y_train, X_val, y_val)
                
                elif model_name == 'huber':
                    model = HuberGLMWrapper(
                        epsilon=model_config.huber_epsilon,
                        alpha=model_config.huber_alpha
                    )
                    results = model.fit(X_train, y_train, X_val, y_val)
                
                # Evaluate on test set
                y_test_pred = model.predict(X_test)
                
                if data_config.task_type == 'regression':
                    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
                else:
                    y_test_proba = model.predict_proba(X_test) if model.supports_proba() else None
                    test_metrics = calculate_classification_metrics(y_test, y_test_pred, y_test_proba)
                
                # Cross-validation if enabled
                cv_results = None
                if use_cv:
                    cv_results = perform_cross_validation(
                        model.get_model(), X_train, y_train,
                        cv_folds=cv_folds, task_type=data_config.task_type
                    )
                
                # Store results
                model_results = {
                    'metrics': test_metrics,
                    'history': results.get('history', {}),
                    'y_test_pred': y_test_pred,
                    'y_test': y_test,
                    'cv_results': cv_results
                }
                
                add_trained_model(model_name, model, model_results)
                progress_bar.progress(1.0)
                st.success(f"✅ {model_name.upper()} training complete!")
                
            except Exception as e:
                st.error(f"❌ Error training {model_name}: {str(e)}")
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
            
            # Predictions vs Actual
            st.subheader("Predictions vs Actual")
            fig_pred = plot_predictions_vs_actual(
                results['y_test'],
                results['y_test_pred'],
                title=f"{name.upper()} Predictions"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
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
