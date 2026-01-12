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

logger = logging.getLogger(__name__)

init_session_state()

st.set_page_config(page_title="Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Model Explainability")

# Check prerequisites
if not st.session_state.get('trained_models'):
    st.warning("⚠️ Please train models first in the Train & Compare page")
    st.stop()

data_config: DataConfig = st.session_state.get('data_config')
pipeline = get_preprocessing_pipeline()
X_test = st.session_state.get('X_test')
y_test = st.session_state.get('y_test')
feature_names = st.session_state.get('feature_names', [])

if X_test is None or y_test is None:
    st.warning("⚠️ Please prepare data splits first")
    st.stop()

# Permutation Importance
st.header("🎯 Permutation Importance")

if st.button("Calculate Permutation Importance"):
    with st.spinner("Calculating permutation importance (this may take a while)..."):
        for name, model_wrapper in st.session_state.trained_models.items():
            model = model_wrapper.get_model()
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test,
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
    
    st.success("✅ Permutation importance calculated!")

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
    with st.spinner("Calculating partial dependence..."):
        for name, model_wrapper in st.session_state.trained_models.items():
            model = model_wrapper.get_model()
            
            # Get top features from permutation importance if available
            if name in st.session_state.get('permutation_importance', {}):
                perm_data = st.session_state.permutation_importance[name]
                top_indices = np.argsort(perm_data['importances_mean'])[-5:][::-1]
                top_features = [perm_data['feature_names'][i] for i in top_indices]
            else:
                # Use first 5 features
                top_features = feature_names[:5]
                top_indices = list(range(min(5, len(feature_names))))
            
            # Calculate partial dependence for top features
            pd_results = {}
            for feat_idx, feat_name in zip(top_indices[:3], top_features[:3]):  # Top 3
                try:
                    # Use a sample for faster computation
                    X_sample_pd = X_test[:min(500, len(X_test))]
                    pd_result = partial_dependence(
                        model, X_sample_pd, features=[feat_idx],
                        grid_resolution=20
                    )
                    pd_results[feat_name] = {
                        'values': pd_result['grid_values'][0] if isinstance(pd_result['grid_values'], list) else pd_result['grid_values'],
                        'average': pd_result['average'][0] if isinstance(pd_result['average'], list) else pd_result['average']
                    }
                except Exception as e:
                    logger.warning(f"Error calculating PD for {feat_name}: {e}")
                    st.warning(f"Could not calculate PD for {feat_name}: {str(e)}")
            
            st.session_state.partial_dependence[name] = pd_results
    
    st.success("✅ Partial dependence calculated!")

# Display partial dependence
if st.session_state.get('partial_dependence'):
    for name, pd_data in st.session_state.partial_dependence.items():
        st.subheader(f"{name.upper()} - Partial Dependence")
        
        cols = st.columns(min(3, len(pd_data)))
        for idx, (feat_name, pd_values) in enumerate(pd_data.items()):
            with cols[idx % len(cols)]:
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

use_shap = st.checkbox("Enable SHAP (requires shap package)", value=False)

if use_shap:
    try:
        import shap
        
        for name, model_wrapper in st.session_state.trained_models.items():
            st.subheader(f"{name.upper()} - SHAP Values")
            
            model = model_wrapper.get_model()
            
            # Create SHAP explainer
            # Use a sample for faster computation
            X_sample = X_test[:min(100, len(X_test))]
            
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict_proba, X_sample[:20])
            else:
                explainer = shap.TreeExplainer(model) if hasattr(model, 'tree_') else shap.KernelExplainer(model.predict, X_sample[:20])
            
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            try:
                import matplotlib.pyplot as plt
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names[:X_sample.shape[1]], show=False)
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.warning(f"Could not display SHAP summary plot: {e}")
            
    except ImportError:
        st.warning("⚠️ SHAP not installed. Install with: `pip install shap`")
    except Exception as e:
        st.error(f"❌ Error calculating SHAP: {str(e)}")
        logger.exception(e)
