"""
Page 05: Model Explainability
Permutation importance, partial dependence, optional SHAP.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
from sklearn.inspection import permutation_importance, partial_dependence
import logging

from utils.session_state import (
    init_session_state, get_preprocessing_pipeline, DataConfig, get_data
)
from utils.storyline import render_progress_indicator
from ml.estimator_utils import is_estimator_fitted
from ml.model_registry import get_registry
from sklearn.pipeline import Pipeline as SklearnPipeline

logger = logging.getLogger(__name__)

init_session_state()

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Model Explainability")

# Progress indicator
render_progress_indicator("05_Explainability")

# Check prerequisites
if not st.session_state.get('trained_models'):
    st.warning("⚠️ Please train models first in the Train & Compare page")
    st.info("**Next steps:** Go to Train & Compare page, prepare splits, and train at least one model.")
    st.stop()

data_config: DataConfig = st.session_state.get('data_config')
pipeline = get_preprocessing_pipeline()
X_test = st.session_state.get('X_test')
y_test = st.session_state.get('y_test')
feature_names = st.session_state.get('feature_names', [])

if X_test is None or y_test is None:
    st.warning("⚠️ Please prepare data splits first")
    st.info("**Next steps:** Go to Train & Compare page and click 'Prepare Splits'.")
    st.stop()

# Get registry for capability checks
registry = get_registry()

# Permutation Importance
st.header("🎯 Permutation Importance")
with st.expander("📚 What is Permutation Importance?", expanded=False):
    st.markdown("""
    **Definition:** Permutation importance measures how much model performance degrades when a feature's values are randomly shuffled.
    
    **How it works:**
    1. Calculate baseline model performance
    2. Shuffle one feature's values
    3. Recalculate performance
    4. Importance = baseline - shuffled performance
    
    **When it can mislead:**
    - Correlated features: shuffling one may not hurt if another is similar
    - Non-linear interactions: may underestimate importance of features that work together
    - Extrapolation: if shuffled values are outside training range, predictions may be unreliable
    """)
st.info("Available for all models with `predict` method.")

if st.button("Calculate Permutation Importance", key="explain_perm_importance_button"):
    perm_errors = []
    for name, model_wrapper in st.session_state.trained_models.items():
        try:
            # Get the fitted sklearn-compatible estimator from session_state
            if name not in st.session_state.get('fitted_estimators', {}):
                perm_errors.append(f"{name}: Fitted estimator not found in session_state. Please retrain the model.")
                continue
            
            # Use the stored fitted estimator (not creating a new instance)
            estimator = st.session_state.fitted_estimators[name]
            
            # Verify it's fitted (works for both sklearn models and custom wrappers)
            if not is_estimator_fitted(estimator):
                perm_errors.append(f"{name}: Estimator not marked as fitted")
                continue
            
            # Check if model supports permutation importance (all models with predict should)
            # Create full pipeline if preprocessing pipeline exists
            if name in st.session_state.get('fitted_preprocessing_pipelines', {}):
                prep_pipeline = st.session_state.fitted_preprocessing_pipelines[name]
                # Create full pipeline for explainability
                full_pipeline = SklearnPipeline([
                    ('preprocess', prep_pipeline),
                    ('model', estimator)
                ])
                # Get raw test data for explainability
                df_raw = st.session_state.get('raw_data')
                test_indices = st.session_state.get('test_indices')
                if df_raw is not None and data_config and test_indices is not None:
                    try:
                        X_test_raw = df_raw[data_config.feature_cols].iloc[test_indices]
                        y_test_for_perm = df_raw[data_config.target_col].iloc[test_indices].values
                    except:
                        # Fallback to preprocessed data
                        full_pipeline = estimator
                        X_test_raw = X_test
                        y_test_for_perm = y_test
                else:
                    # Fallback to preprocessed data
                    full_pipeline = estimator
                    X_test_raw = X_test
                    y_test_for_perm = y_test
            else:
                # No preprocessing pipeline, use estimator directly
                full_pipeline = estimator
                X_test_raw = X_test
                y_test_for_perm = y_test
            
            with st.spinner(f"Calculating permutation importance for {name.upper()} (this may take a while)..."):
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    full_pipeline, X_test_raw, y_test_for_perm,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
            
                # Store results
                st.session_state.permutation_importance[name] = {
                    'importances_mean': perm_importance.importances_mean,
                    'importances_std': perm_importance.importances_std,
                    'feature_names': feature_names[:len(perm_importance.importances_mean)]
                }
        except Exception as e:
            perm_errors.append(f"{name}: {str(e)}")
            logger.exception(f"Error calculating permutation importance for {name}: {e}")
    
    if perm_errors:
        with st.expander("⚠️ Permutation Importance Errors (click to view)", expanded=False):
            for err in perm_errors:
                st.text(err)
    
    if any(st.session_state.get('permutation_importance', {}).values()):
        st.success("✅ Permutation importance calculated!")
    else:
        st.warning("⚠️ Could not calculate permutation importance for any models. Check errors above.")

# Display permutation importance
if st.session_state.get('permutation_importance'):
    for name, perm_data in st.session_state.permutation_importance.items():
        st.subheader(f"{name.upper()} - Permutation Importance")
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': perm_data['feature_names'],
            'Importance': perm_data['importances_mean'],
            'Std': perm_data['importances_std']
        }).sort_values('Importance', ascending=False)
        
        # Show top features
        top_n = min(10, len(importance_df))
        st.dataframe(importance_df.head(top_n), use_container_width=True)
        
        # Plot
        fig = px.bar(
            importance_df.head(top_n),
            x='Importance',
            y='Feature',
            orientation='h',
            error_x='Std',
            title=f"{name.upper()} - Top {top_n} Features"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# Partial Dependence
st.header("📈 Partial Dependence Plots")
with st.expander("📚 What is Partial Dependence?", expanded=False):
    st.markdown("""
    **Definition:** Partial dependence shows how a feature affects predictions, averaged over all other features.
    
    **How it works:**
    1. Fix one feature at a specific value
    2. Average predictions over all other features
    3. Repeat for different values of the fixed feature
    4. Plot the average prediction vs feature value
    
    **When it can mislead:**
    - Extrapolation: if feature values are outside training range, predictions may be unreliable
    - Correlated features: assumes independence, may not reflect real-world interactions
    - Only shows average effect, not individual variation
    """)
st.info("Available for models with `predict` or `predict_proba` methods.")

if st.button("Calculate Partial Dependence"):
    # Get original feature names (pre-transform) for PD
    original_features = data_config.feature_cols if data_config else []
    
    pd_errors = []
    for name, model_wrapper in st.session_state.trained_models.items():
        try:
            # Explicitly disable PDP for NN (sklearn partial_dependence doesn't work reliably with custom NN wrappers)
            if name == 'nn':
                pd_errors.append(f"{name}: Partial dependence not supported for neural networks. Use SHAP or permutation importance instead.")
                continue
            
            # Check capability from registry
            spec = registry.get(name)
            if spec and not spec.capabilities.supports_partial_dependence:
                pd_errors.append(f"{name}: Partial dependence not supported for this model type")
                continue
            
            # Get the fitted sklearn-compatible estimator from session_state
            if name not in st.session_state.get('fitted_estimators', {}):
                pd_errors.append(f"{name}: Fitted estimator not found in session_state. Please retrain the model.")
                continue
            
            # Use the stored fitted estimator (not creating a new instance)
            estimator = st.session_state.fitted_estimators[name]
            
            # Verify it's fitted (works for both sklearn models and custom wrappers)
            if not is_estimator_fitted(estimator):
                pd_errors.append(f"{name}: Estimator not marked as fitted")
                continue
            
            # Create full pipeline if preprocessing exists
            if name in st.session_state.get('fitted_preprocessing_pipelines', {}):
                prep_pipeline = st.session_state.fitted_preprocessing_pipelines[name]
                full_pipeline = SklearnPipeline([
                    ('preprocess', prep_pipeline),
                    ('model', estimator)
                ])
                # Get raw test data for explainability
                df_raw = st.session_state.get('raw_data')
                test_indices = st.session_state.get('test_indices')
                if df_raw is not None and data_config and test_indices is not None:
                    try:
                        X_test_raw = df_raw[data_config.feature_cols].iloc[test_indices]
                    except:
                        full_pipeline = estimator
                        X_test_raw = X_test
                else:
                    full_pipeline = estimator
                    X_test_raw = X_test
            else:
                full_pipeline = estimator
                X_test_raw = X_test
            
            with st.spinner(f"Calculating partial dependence for {name.upper()}..."):
                # Get top features from permutation importance if available
                if name in st.session_state.get('permutation_importance', {}):
                    perm_data = st.session_state.permutation_importance[name]
                    top_indices = np.argsort(perm_data['importances_mean'])[-5:][::-1]
                    top_feature_names = [perm_data['feature_names'][i] for i in top_indices]
                else:
                    # Use first 5 original features
                    top_feature_names = original_features[:5] if original_features else feature_names[:5]
                    top_indices = list(range(min(5, len(top_feature_names))))
                
                # Calculate partial dependence for top numeric original features only
                pd_results = {}
                for feat_name in top_feature_names[:3]:  # Top 3
                    try:
                        # Find feature index in transformed space
                        if feat_name in feature_names:
                            feat_idx = feature_names.index(feat_name)
                        else:
                            # Try to find in original features
                            if feat_name in original_features:
                                # Map to transformed feature index
                                feat_idx = original_features.index(feat_name)
                                if feat_idx >= len(feature_names):
                                    pd_errors.append(f"{name}: {feat_name} - feature index out of range")
                                    continue
                            else:
                                pd_errors.append(f"{name}: {feat_name} - feature not found")
                                continue
                        
                        # Use a sample for faster computation (handle sparse)
                        X_sample_pd = X_test_raw[:min(500, len(X_test_raw))]
                        if hasattr(X_sample_pd, 'toarray'):
                            X_sample_pd = X_sample_pd.toarray()
                        
                        pd_result = partial_dependence(
                            full_pipeline, X_sample_pd, features=[feat_idx],
                            grid_resolution=20
                        )
                        pd_results[feat_name] = {
                            'values': pd_result['grid_values'][0] if isinstance(pd_result['grid_values'], list) else pd_result['grid_values'],
                            'average': pd_result['average'][0] if isinstance(pd_result['average'], list) else pd_result['average']
                        }
                    except Exception as e:
                        error_msg = f"{name}: {feat_name} - {str(e)}"
                        pd_errors.append(error_msg)
                        logger.warning(f"Error calculating PD for {feat_name}: {e}")
                
                st.session_state.partial_dependence[name] = pd_results
        except Exception as e:
            pd_errors.append(f"{name}: {str(e)}")
            logger.exception(f"Error in partial dependence calculation for {name}: {e}")
        
        if pd_errors:
            with st.expander("⚠️ Partial Dependence Errors (click to view)", expanded=False):
                for err in pd_errors:
                    st.text(err)
        
        if any(st.session_state.partial_dependence.values()):
            st.success("✅ Partial dependence calculated!")
        else:
            st.warning("⚠️ Could not calculate partial dependence for any features. Check errors above.")

# Display partial dependence
if st.session_state.get('partial_dependence'):
    for name, pd_data in st.session_state.partial_dependence.items():
        st.subheader(f"{name.upper()} - Partial Dependence")
        
        # Ensure valid column spec (must be positive integer)
        n_cols = max(1, min(3, len(pd_data)))
        cols = st.columns(n_cols)
        for idx, (feat_name, pd_values) in enumerate(pd_data.items()):
            col_idx = idx % n_cols
            with cols[col_idx]:
                try:
                    # Handle both array and list formats
                    values = pd_values['values']
                    average = pd_values['average']
                    
                    # Convert to numpy arrays if needed
                    if not isinstance(values, np.ndarray):
                        values = np.array(values)
                    if not isinstance(average, np.ndarray):
                        average = np.array(average)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=values.flatten() if values.ndim > 1 else values,
                        y=average.flatten() if average.ndim > 1 else average,
                        mode='lines',
                        name=feat_name
                    ))
                    fig.update_layout(
                        title=feat_name,
                        xaxis_title=feat_name,
                        yaxis_title="Partial Dependence"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error plotting PD for {feat_name}: {str(e)}")

# SHAP (Advanced)
st.header("🔬 SHAP Analysis (Advanced)")
with st.expander("📚 What is SHAP?", expanded=False):
    st.markdown("""
    **Definition:** SHAP (SHapley Additive exPlanations) provides feature-level explanations based on game theory.
    
    **How it works:**
    - Each feature gets a "contribution" to each prediction
    - Contributions sum to the difference between prediction and baseline
    - Based on Shapley values from cooperative game theory
    
    **Explainer types:**
    - **TreeExplainer:** Fast and exact for tree models (RF, ExtraTrees, HistGB)
    - **LinearExplainer:** Fast for linear models (Ridge, Lasso, Logistic)
    - **KernelExplainer:** Slow but works for any model (uses sampling)
    
    **When it can mislead:**
    - KernelExplainer uses sampling - may be slow or inaccurate with many features
    - Assumes feature independence (like permutation importance)
    - Values depend on background data distribution
    """)
st.info("Availability depends on model type: TreeExplainer for tree models, LinearExplainer for linear models, KernelExplainer for others (slower).")

use_shap = st.checkbox(
    "Enable SHAP (requires shap package)", 
    value=st.session_state.get('explain_shap_enable', False), 
    key="explain_shap_enable"
)

if use_shap:
    try:
        import shap
        import matplotlib.pyplot as plt
        
        # SHAP configuration
        with st.expander("⚙️ SHAP Configuration", expanded=False):
            background_size = st.slider(
                "Background Sample Size",
                min_value=50,
                max_value=200,
                value=100,
                step=10,
                help="Number of samples for background distribution (larger = more accurate but slower)"
            )
            eval_size = st.slider(
                "Evaluation Sample Size",
                min_value=100,
                max_value=500,
                value=200,
                step=50,
                help="Number of samples to compute SHAP values for (larger = more detailed but slower)"
            )
        
        # Model SHAP support summary
        st.markdown("**SHAP Support by Model:**")
        shap_support_info = []
        for name, model_wrapper in st.session_state.trained_models.items():
            spec = registry.get(name)
            if spec:
                support = spec.capabilities.supports_shap
                support_label = {
                    'tree': '🟢 Fast (TreeExplainer)',
                    'linear': '🟢 Fast (LinearExplainer)',
                    'kernel': '🟡 Slow (KernelExplainer)',
                    'none': '🔴 Not supported'
                }.get(support, '⚪ Unknown')
                shap_support_info.append(f"• **{name.upper()}**: {support_label}")
        st.markdown("\n".join(shap_support_info))
        
        # Run SHAP button
        run_shap = st.button("🚀 Run SHAP Analysis", type="primary", key="run_shap_button")
        
        if not run_shap:
            st.info("👆 Click the button above to compute SHAP values. This may take a while depending on your data and model types.")
            st.stop()
        
        for name, model_wrapper in st.session_state.trained_models.items():
            st.subheader(f"{name.upper()} - SHAP Values")
            
            # Check SHAP capability from registry
            spec = registry.get(name)
            if spec:
                shap_support = spec.capabilities.supports_shap
                if shap_support == 'none':
                    st.warning(f"⚠️ {name.upper()}: SHAP not supported for this model type.")
                    continue
                elif shap_support == 'kernel':
                    st.info(f"ℹ️ {name.upper()}: Using KernelExplainer (may be slow)")
            
            # Get the fitted sklearn-compatible estimator from session_state
            if name not in st.session_state.get('fitted_estimators', {}):
                st.warning(f"⚠️ {name.upper()} fitted estimator not found. Please retrain the model.")
                continue
            
            # Use the stored fitted estimator (not creating a new instance)
            estimator = st.session_state.fitted_estimators[name]
            
            # Verify it's fitted (works for both sklearn models and custom wrappers)
            if not is_estimator_fitted(estimator):
                st.warning(f"⚠️ {name.upper()} estimator not marked as fitted. Skipping SHAP.")
                continue
            
            # Create full pipeline if preprocessing exists
            if name in st.session_state.get('fitted_preprocessing_pipelines', {}):
                prep_pipeline = st.session_state.fitted_preprocessing_pipelines[name]
                full_pipeline = SklearnPipeline([
                    ('preprocess', prep_pipeline),
                    ('model', estimator)
                ])
                # Get raw test data for explainability
                df_raw = st.session_state.get('raw_data')
                test_indices = st.session_state.get('test_indices')
                if df_raw is not None and data_config and test_indices is not None:
                    try:
                        X_test_raw = df_raw[data_config.feature_cols].iloc[test_indices]
                    except:
                        full_pipeline = estimator
                        X_test_raw = X_test
                else:
                    full_pipeline = estimator
                    X_test_raw = X_test
            else:
                full_pipeline = estimator
                X_test_raw = X_test
            
            try:
                
                # Prepare samples (use raw data for full pipeline)
                X_background = X_test_raw[:min(background_size, len(X_test_raw))]
                X_eval = X_test_raw[:min(eval_size, len(X_test_raw))]
                
                # Handle sparse matrices
                if hasattr(X_background, 'toarray'):
                    X_background = X_background.toarray()
                if hasattr(X_eval, 'toarray'):
                    X_eval = X_eval.toarray()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create SHAP explainer
                status_text.text("Preparing SHAP explainer...")
                progress_bar.progress(0.2)
                
                # Determine explainer type based on capability
                if spec and spec.capabilities.supports_shap == 'tree':
                    # Tree-based model - use TreeExplainer
                    explainer = shap.TreeExplainer(full_pipeline.named_steps['model'] if isinstance(full_pipeline, SklearnPipeline) else full_pipeline)
                    status_text.text("Computing SHAP values (TreeExplainer)...")
                    progress_bar.progress(0.5)
                    shap_values = explainer.shap_values(X_eval)
                    progress_bar.progress(0.8)
                elif spec and spec.capabilities.supports_shap == 'linear':
                    # Linear model - use LinearExplainer
                    explainer = shap.LinearExplainer(
                        full_pipeline.named_steps['model'] if isinstance(full_pipeline, SklearnPipeline) else full_pipeline,
                        X_background
                    )
                    status_text.text("Computing SHAP values (LinearExplainer)...")
                    progress_bar.progress(0.5)
                    shap_values = explainer.shap_values(X_eval)
                    progress_bar.progress(0.8)
                else:
                    # Kernel or other explainer
                    task_type = data_config.task_type if data_config else 'regression'
                    if task_type == 'classification' and hasattr(full_pipeline, 'predict_proba'):
                        # Use predict_proba for classification
                        status_text.text("Preparing background data for KernelExplainer...")
                        progress_bar.progress(0.3)
                        explainer = shap.KernelExplainer(
                            full_pipeline.predict_proba,
                            X_background[:min(50, len(X_background))]
                        )
                        status_text.text("Computing SHAP values (this may take a while)...")
                        progress_bar.progress(0.5)
                        shap_values = explainer.shap_values(X_eval)
                        progress_bar.progress(0.8)
                    else:
                        # Regression or model without predict_proba
                        explainer = shap.KernelExplainer(
                            full_pipeline.predict,
                            X_background[:min(50, len(X_background))]
                        )
                        status_text.text("Computing SHAP values (this may take a while)...")
                        progress_bar.progress(0.5)
                        shap_values = explainer.shap_values(X_eval)
                        progress_bar.progress(0.8)
                
                # Handle SHAP values format
                if isinstance(shap_values, list):
                    # Multiclass: list of arrays
                    n_classes = len(shap_values)
                    if n_classes == 2:
                        # Binary: use positive class (index 1)
                        shap_values_to_plot = shap_values[1]
                        class_label = "Class 1 (Positive)"
                    else:
                        # Multiclass: let user select
                        selected_class = st.selectbox(
                            f"Select class to visualize ({name})",
                            options=list(range(n_classes)),
                            format_func=lambda x: f"Class {x}",
                            key=f"shap_class_{name}"
                        )
                        shap_values_to_plot = shap_values[selected_class]
                        class_label = f"Class {selected_class}"
                else:
                    # Single array (regression or binary)
                    shap_values_to_plot = shap_values
                    class_label = None
                
                # Prepare feature names
                plot_feature_names = feature_names[:X_eval.shape[1]] if len(feature_names) >= X_eval.shape[1] else [f"Feature {i}" for i in range(X_eval.shape[1])]
                
                # Dynamic figure sizing based on number of features
                n_features = X_eval.shape[1]
                if n_features <= 3:
                    fig_height = max(400, n_features * 150)
                    fig_width = 800
                else:
                    fig_height = max(400, min(800, n_features * 100))
                    fig_width = 1000
                
                status_text.text("Rendering SHAP summary plot...")
                progress_bar.progress(0.9)
                
                # Create summary plot
                fig, ax = plt.subplots(figsize=(fig_width/100, fig_height/100))
                shap.summary_plot(
                    shap_values_to_plot,
                    X_eval,
                    feature_names=plot_feature_names,
                    show=False,
                    plot_size=(fig_width/100, fig_height/100)
                )
                
                if class_label:
                    ax.set_title(f"{name.upper()} - SHAP Values ({class_label})", fontsize=12)
                
                st.pyplot(fig)
                plt.close(fig)
                
                progress_bar.progress(1.0)
                status_text.text("✅ SHAP analysis complete!")
                
                # Keep progress visible briefly, then clear
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error calculating SHAP for {name}: {str(e)}")
                with st.expander("Error details", expanded=False):
                    st.text(str(e))
                    logger.exception(e)
            
    except ImportError:
        st.warning("⚠️ SHAP not installed. Install with: `pip install shap`")
    except Exception as e:
        st.error(f"❌ Error setting up SHAP: {str(e)}")
        logger.exception(e)

# State Debug (Advanced)
with st.expander("🔧 Advanced / State Debug", expanded=False):
    st.markdown("**Current State:**")
    _df = get_data()  # Get data from session state
    st.write(f"• Data shape: {_df.shape if _df is not None else 'None'}")
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")
    st.write(f"• X_test shape: {X_test.shape if X_test is not None else 'None'}")
    task_det = st.session_state.get('task_type_detection')
    cohort_det = st.session_state.get('cohort_structure_detection')
    st.write(f"• Task type (final): {task_det.final if task_det else 'None'}")
    st.write(f"• Cohort type (final): {cohort_det.final if cohort_det else 'None'}")
    st.write(f"• Trained models: {len(st.session_state.get('trained_models', {}))}")
    st.write(f"• Permutation importance: {len(st.session_state.get('permutation_importance', {}))}")
    st.write(f"• Partial dependence: {len(st.session_state.get('partial_dependence', {}))}")
