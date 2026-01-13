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
    init_session_state, get_preprocessing_pipeline, DataConfig
)
from ml.estimator_utils import is_estimator_fitted

logger = logging.getLogger(__name__)

init_session_state()

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Model Explainability")

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

# Permutation Importance
st.header("🎯 Permutation Importance")

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
            
            with st.spinner(f"Calculating permutation importance for {name.upper()} (this may take a while)..."):
                # Calculate permutation importance
                perm_importance = permutation_importance(
                    estimator, X_test, y_test,
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

if st.button("Calculate Partial Dependence"):
    # Get original feature names (pre-transform) for PD
    original_features = data_config.feature_cols if data_config else []
    
    pd_errors = []
    for name, model_wrapper in st.session_state.trained_models.items():
        try:
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
                        X_sample_pd = X_test[:min(500, len(X_test))]
                        if hasattr(X_sample_pd, 'toarray'):
                            X_sample_pd = X_sample_pd.toarray()
                        
                        pd_result = partial_dependence(
                            estimator, X_sample_pd, features=[feat_idx],
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
        
        for name, model_wrapper in st.session_state.trained_models.items():
            st.subheader(f"{name.upper()} - SHAP Values")
            
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
            
            try:
                
                # Prepare samples
                X_background = X_test[:min(background_size, len(X_test))]
                X_eval = X_test[:min(eval_size, len(X_test))]
                
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
                
                # Determine explainer type
                if hasattr(estimator, 'tree_') or (hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, 'tree_')):
                    # Tree-based model
                    explainer = shap.TreeExplainer(estimator)
                    status_text.text("Computing SHAP values (TreeExplainer)...")
                    progress_bar.progress(0.5)
                    shap_values = explainer.shap_values(X_eval)
                    progress_bar.progress(0.8)
                else:
                    # Kernel or other explainer
                    if data_config.task_type == 'classification' and hasattr(estimator, 'predict_proba'):
                        # Use predict_proba for classification
                        status_text.text("Preparing background data for KernelExplainer...")
                        progress_bar.progress(0.3)
                        explainer = shap.KernelExplainer(
                            estimator.predict_proba,
                            X_background[:min(50, len(X_background))]
                        )
                        status_text.text("Computing SHAP values (this may take a while)...")
                        progress_bar.progress(0.5)
                        shap_values = explainer.shap_values(X_eval)
                        progress_bar.progress(0.8)
                    else:
                        # Regression or model without predict_proba
                        explainer = shap.KernelExplainer(
                            estimator.predict,
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
