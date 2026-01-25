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

# Interpretability coach guidance
trained = list(st.session_state.get('trained_models', {}).keys())
with st.expander("🎓 Interpretability Coach", expanded=False):
    st.markdown(
        "**Interpretability by model:** Linear models (Ridge, Lasso, LogReg, GLM) and Huber regression offer "
        "**high** interpretability (coefficients, direct feature–outcome links). Tree-based models and "
        "boosting offer **medium** (feature importance, partial dependence). Neural nets and kernel SVMs "
        "offer **low** interpretability (SHAP or permutation importance only)."
    )
    if trained:
        high_int = [m for m in trained if registry.get(m) and getattr(registry.get(m).capabilities, 'interpretability_tier', 'medium') == 'high']
        if high_int:
            st.markdown(f"**Your high-interpretability models:** {', '.join(high_int)}. Use coefficients or linear SHAP when available.")
    st.markdown(
        "**Permutation importance:** Works for all models; no extra dependencies. Use it as a baseline. "
        "**SHAP:** TreeExplainer is fast for trees; LinearExplainer for linear models; KernelExplainer is slow but model-agnostic."
    )

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
            
                # Store results (use per-model feature names if available; pad so lengths match)
                fn_by_model = st.session_state.get('feature_names_by_model', {})
                n = len(perm_importance.importances_mean)
                base = list(fn_by_model.get(name, feature_names) or [])
                fnames = (base + [f"feature_{i}" for i in range(len(base), n)])[:n]
                st.session_state.permutation_importance[name] = {
                    'importances_mean': perm_importance.importances_mean,
                    'importances_std': perm_importance.importances_std,
                    'feature_names': fnames
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
    from ml.plot_narrative import narrative_permutation_importance
    for name, perm_data in st.session_state.permutation_importance.items():
        st.subheader(f"{name.upper()} - Permutation Importance")
        _fn = perm_data.get('feature_names', [])
        _im = perm_data.get('importances_mean', [])
        _is = perm_data.get('importances_std', [])
        n = min(len(_fn), len(_im), len(_is))
        if n == 0:
            st.warning(f"⚠️ Skipping {name.upper()}: permutation importance data has empty arrays.")
            continue
        fn_slice = _fn[:n]
        im_slice = np.asarray(_im)[:n]
        is_slice = np.asarray(_is)[:n]
        importance_df = pd.DataFrame({
            'Feature': fn_slice,
            'Importance': im_slice,
            'Std': is_slice
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
        st.plotly_chart(fig, use_container_width=True, key=f"perm_importance_{name}")
        perm_aligned = {"feature_names": fn_slice, "importances_mean": im_slice, "importances_std": is_slice}
        nar = narrative_permutation_importance(perm_aligned, model_name=name)
        if nar:
            st.markdown(f"**Interpretation:** {nar}")
        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
        stats_summary = "; ".join(
            f"{f}={v:.3f}" for f, v in zip(fn_slice[:5], im_slice[:5])
        )
        ctx = build_llm_context(
            "permutation_importance", stats_summary, model_name=name, existing=nar or "",
            feature_names=fn_slice, sample_size=len(X_test) if X_test is not None else None,
            task_type=data_config.task_type if data_config else None,
        )
        render_interpretation_with_llm_button(
            ctx, key=f"llm_perm_{name}", result_session_key=f"llm_result_perm_{name}",
        )

# Cross-Model Robustness
st.header("🔄 Cross-Model Robustness")
with st.expander("📚 What is Cross-Model Robustness?", expanded=False):
    st.markdown(
        "Compare permutation importance **rankings** across models. When models agree on which features "
        "matter (high rank correlation, overlapping top features), explanations are more **robust**. "
        "Large disagreement may indicate instability or model-specific artifacts."
    )
perm_names = list(st.session_state.get('permutation_importance', {}).keys())
if len(perm_names) >= 2:
    if st.button("Run Robustness Check", key="explain_robustness_button"):
        from ml.eval import compare_importance_ranks
        perm = st.session_state.permutation_importance
        fn_by_model = st.session_state.get('feature_names_by_model', {})
        robustness = compare_importance_ranks(perm_names, perm, fn_by_model, top_k=5)
        st.session_state.explainability_robustness = robustness
        st.session_state.pop("llm_result_robustness", None)
        st.success("Robustness check complete.")
    if st.session_state.get('explainability_robustness'):
        rob = st.session_state.explainability_robustness
        rows = []
        for (ma, mb), v in rob.items():
            r = v.get('spearman')
            r_str = f"{r:.3f}" if r is not None else "N/A"
            rows.append({"Model A": ma, "Model B": mb, "Spearman ρ": r_str, "Top-5 overlap": v.get('top_k_overlap', 0)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            from ml.plot_narrative import narrative_robustness
            from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
            pairs = list(rob.keys())
            nar = narrative_robustness(rob, model_pairs=pairs)
            if nar:
                st.markdown(f"**Interpretation:** {nar}")
            stats_summary = "; ".join(
                f"{r['Model A']}-{r['Model B']}: ρ={r['Spearman ρ']}, overlap={r.get('Top-5 overlap', 'N/A')}"
                for r in rows[:5]
            )
            ctx = build_llm_context(
                "robustness", stats_summary, where="Cross-model robustness", existing=nar or "",
                task_type=data_config.task_type if data_config else None,
                feature_names=st.session_state.get("feature_names"),
            )
            render_interpretation_with_llm_button(
                ctx, key="llm_robustness_btn", result_session_key="llm_result_robustness",
            )
        else:
            st.info("No model pairs share the same feature set. Robustness comparison requires matching pipelines.")
else:
    st.info("Compute permutation importance for at least 2 models to run robustness checks.")

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
    original_features = data_config.feature_cols if data_config else []
    pd_errors = []
    nn_skipped = False
    if "partial_dependence" not in st.session_state or st.session_state.partial_dependence is None:
        st.session_state.partial_dependence = {}
    for name, model_wrapper in st.session_state.trained_models.items():
        try:
            if name == "nn":
                nn_skipped = True
                continue
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

    if nn_skipped:
        st.info("Partial dependence is not supported for neural networks. Use SHAP or permutation importance for model explainability.")
    if pd_errors:
        with st.expander("⚠️ Partial Dependence Errors (click to view)", expanded=False):
            for err in pd_errors:
                st.text(err)
    if any(st.session_state.get("partial_dependence", {}).values()):
        st.success("✅ Partial dependence calculated!")
    elif not nn_skipped and not pd_errors:
        st.warning("⚠️ Could not calculate partial dependence for any features.")

# Display partial dependence
if st.session_state.get('partial_dependence'):
    for name, pd_data in st.session_state.partial_dependence.items():
        st.subheader(f"{name.upper()} - Partial Dependence")
        
        n_cols = max(1, min(3, len(pd_data)))
        cols = st.columns(n_cols)
        for idx, (feat_name, pd_values) in enumerate(pd_data.items()):
            col_idx = idx % n_cols
            with cols[col_idx]:
                try:
                    values = pd_values['values']
                    average = pd_values['average']
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
                    st.plotly_chart(fig, use_container_width=True, key=f"pd_plot_{name}_{feat_name}")
                except Exception as e:
                    st.warning(f"Error plotting PD for {feat_name}: {str(e)}")
        
        from ml.plot_narrative import narrative_partial_dependence
        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
        nar = narrative_partial_dependence(pd_data, model_name=name)
        if nar:
            st.markdown(f"**Interpretation:** {nar}")
        feats = list(pd_data.keys())[:5]
        stats_summary = "; ".join(feats) if feats else ""
        ctx = build_llm_context(
            "partial_dependence", stats_summary, model_name=name, existing=nar or "",
            feature_names=feats, task_type=data_config.task_type if data_config else None,
            sample_size=len(X_test) if X_test is not None else None,
        )
        render_interpretation_with_llm_button(
            ctx, key=f"llm_pd_{name}", result_session_key=f"llm_result_pd_{name}",
        )

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

        class _ShapPredictWrapper:
            """Thin wrapper so SHAP can set feature_names_in_ without touching the pipeline."""
            feature_names_in_ = None

            def __init__(self, model):
                self._model = model

            def predict(self, X):
                return self._model.predict(X)

            def predict_proba(self, X):
                return self._model.predict_proba(X)

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
        if "shap_results" not in st.session_state:
            st.session_state.shap_results = {}

        if not run_shap and not st.session_state.get("shap_results"):
            st.info("👆 Click the button above to compute SHAP values. This may take a while depending on your data and model types.")
        elif run_shap:
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
            
                def _to_dense_numpy(arr):
                    if hasattr(arr, 'toarray'):
                        out = arr.toarray()
                    elif isinstance(arr, pd.DataFrame):
                        out = np.asarray(arr.values, dtype=float)
                    else:
                        out = np.asarray(arr, dtype=float)
                    return np.ascontiguousarray(out)

                use_tree_or_linear = (
                    spec
                    and spec.capabilities.supports_shap in ('tree', 'linear')
                    and isinstance(full_pipeline, SklearnPipeline)
                    and 'preprocess' in getattr(full_pipeline, 'named_steps', {})
                )
                if use_tree_or_linear:
                    prep = full_pipeline.named_steps['preprocess']
                    X_transformed = prep.transform(X_test_raw)
                    X_background = _to_dense_numpy(X_transformed[:min(background_size, len(X_transformed))])
                    X_eval = _to_dense_numpy(X_transformed[:min(eval_size, len(X_transformed))])
                else:
                    bg_n = min(background_size, len(X_test_raw))
                    ev_n = min(eval_size, len(X_test_raw))
                    if isinstance(X_test_raw, pd.DataFrame):
                        X_background = X_test_raw.iloc[:bg_n].copy()
                        X_eval = X_test_raw.iloc[:ev_n].copy()
                    else:
                        X_background = _to_dense_numpy(X_test_raw[:bg_n])
                        X_eval = _to_dense_numpy(X_test_raw[:ev_n])

                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Preparing SHAP explainer...")
                    progress_bar.progress(0.2)

                    if spec and spec.capabilities.supports_shap == 'tree':
                        model_step = full_pipeline.named_steps['model'] if isinstance(full_pipeline, SklearnPipeline) else full_pipeline
                        explainer = shap.TreeExplainer(model_step)
                        status_text.text("Computing SHAP values (TreeExplainer)...")
                        progress_bar.progress(0.5)
                        shap_values = explainer.shap_values(X_eval)
                        progress_bar.progress(0.8)
                    elif spec and spec.capabilities.supports_shap == 'linear':
                        model_step = full_pipeline.named_steps['model'] if isinstance(full_pipeline, SklearnPipeline) else full_pipeline
                        explainer = shap.LinearExplainer(model_step, X_background)
                        status_text.text("Computing SHAP values (LinearExplainer)...")
                        progress_bar.progress(0.5)
                        shap_values = explainer.shap_values(X_eval)
                        progress_bar.progress(0.8)
                    else:
                        task_type = data_config.task_type if data_config else 'regression'
                        _kernel_model = _ShapPredictWrapper(full_pipeline)
                        if task_type == 'classification' and hasattr(full_pipeline, 'predict_proba'):
                            status_text.text("Preparing background data for KernelExplainer...")
                            progress_bar.progress(0.3)
                            explainer = shap.KernelExplainer(
                                _kernel_model.predict_proba,
                                X_background[:min(50, len(X_background))]
                            )
                            status_text.text("Computing SHAP values (this may take a while)...")
                            progress_bar.progress(0.5)
                            shap_values = explainer.shap_values(X_eval)
                            progress_bar.progress(0.8)
                        else:
                            explainer = shap.KernelExplainer(
                                _kernel_model.predict,
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
                            shap_values_to_plot = shap_values[1]
                            class_label = "Class 1 (Positive)"
                        else:
                            selected_class = st.selectbox(
                                f"Select class to visualize ({name})",
                                options=list(range(n_classes)),
                                format_func=lambda x: f"Class {x}",
                                key=f"shap_class_{name}"
                            )
                            shap_values_to_plot = shap_values[selected_class]
                            class_label = f"Class {selected_class}"
                    else:
                        shap_values_to_plot = shap_values
                        class_label = None
                
                    fn_by_model = st.session_state.get('feature_names_by_model', {})
                    fn_for_model = fn_by_model.get(name, feature_names)
                    if use_tree_or_linear and len(fn_for_model) >= X_eval.shape[1]:
                        plot_feature_names = fn_for_model[:X_eval.shape[1]]
                    elif not use_tree_or_linear and isinstance(full_pipeline, SklearnPipeline) and 'preprocess' in getattr(full_pipeline, 'named_steps', {}) and data_config and data_config.feature_cols:
                        raw_names = list(data_config.feature_cols)
                        plot_feature_names = raw_names[:X_eval.shape[1]] if len(raw_names) >= X_eval.shape[1] else [f"Feature {i}" for i in range(X_eval.shape[1])]
                    else:
                        plot_feature_names = fn_for_model[:X_eval.shape[1]] if len(fn_for_model) >= X_eval.shape[1] else [f"Feature {i}" for i in range(X_eval.shape[1])]
                
                    n_features = X_eval.shape[1]
                    if n_features <= 3:
                        fig_height = max(400, n_features * 150)
                        fig_width = 800
                    else:
                        fig_height = max(400, min(800, n_features * 100))
                        fig_width = 1000
                
                    status_text.text("Rendering SHAP summary plot...")
                    progress_bar.progress(0.9)
                
                    n_cols = X_eval.shape[1]
                    fn = plot_feature_names[:n_cols] if len(plot_feature_names) >= n_cols else [f"Feature {i}" for i in range(n_cols)]
                    X_plot = pd.DataFrame(X_eval, columns=fn)
                    fig, ax = plt.subplots(figsize=(fig_width/100, fig_height/100))
                    shap.summary_plot(
                        shap_values_to_plot,
                        X_plot,
                        feature_names=fn,
                        show=False,
                        plot_size=(fig_width/100, fig_height/100)
                    )
                    if class_label:
                        ax.set_title(f"{name.upper()} - SHAP Values ({class_label})", fontsize=12)
                
                    st.pyplot(fig)
                    plt.close(fig)
                    from ml.plot_narrative import narrative_shap
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                    nar = narrative_shap(shap_values_to_plot, plot_feature_names, model_name=name)
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    mean_abs = np.abs(shap_values_to_plot).mean(axis=0)
                    top_idx = np.argsort(mean_abs)[::-1][:5]
                    stats_summary = "; ".join(
                        f"{plot_feature_names[i]}={mean_abs[i]:.3f}" for i in top_idx
                        if i < len(plot_feature_names)
                    )
                    st.session_state.shap_results[name] = {
                        "shap_values_to_plot": shap_values_to_plot,
                        "X_eval": X_eval,
                        "plot_feature_names": plot_feature_names,
                        "class_label": class_label,
                        "nar": nar,
                        "stats_summary": stats_summary,
                        "fig_width": fig_width,
                        "fig_height": fig_height,
                    }
                    ctx = build_llm_context(
                        "SHAP", stats_summary, model_name=name, existing=nar or "",
                        feature_names=plot_feature_names, sample_size=X_eval.shape[0],
                        task_type=data_config.task_type if data_config else None,
                    )
                    render_interpretation_with_llm_button(
                        ctx, key=f"llm_shap_{name}", result_session_key=f"llm_result_shap_{name}",
                    )
                    progress_bar.progress(1.0)
                    status_text.text("✅ SHAP analysis complete!")
                    progress_bar.empty()
                    status_text.empty()
                
                except Exception as e:
                    st.error(f"❌ Error calculating SHAP for {name}: {str(e)}")
                    with st.expander("Error details", expanded=False):
                        st.text(str(e))
                        logger.exception(e)
        elif st.session_state.get("shap_results"):
            for name, s in st.session_state.shap_results.items():
                st.subheader(f"{name.upper()} - SHAP Values")
                try:
                    sv = s["shap_values_to_plot"]
                    Xe = s["X_eval"]
                    fnames = s["plot_feature_names"]
                    cl = s.get("class_label")
                    nw = s.get("nar", "")
                    ss = s.get("stats_summary", "")
                    fw = s.get("fig_width", 1000)
                    fh = s.get("fig_height", 400)
                    n_cols = Xe.shape[1] if hasattr(Xe, "shape") and hasattr(Xe.shape, "__len__") and len(Xe.shape) > 1 else (len(Xe[0]) if Xe is not None and len(Xe) else 0)
                    fn = fnames[:n_cols] if fnames and len(fnames) >= n_cols else [f"Feature {i}" for i in range(n_cols)]
                    X_plot = pd.DataFrame(np.asarray(Xe), columns=fn)
                    fig2, ax2 = plt.subplots(figsize=(fw / 100, fh / 100))
                    shap.summary_plot(sv, X_plot, feature_names=fn, show=False, plot_size=(fw / 100, fh / 100))
                    if cl:
                        ax2.set_title(f"{name.upper()} - SHAP Values ({cl})", fontsize=12)
                    st.pyplot(fig2)
                    plt.close(fig2)
                    if nw:
                        st.markdown(f"**Interpretation:** {nw}")
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                    _n = Xe.shape[0] if hasattr(Xe, "shape") else (len(Xe) if Xe is not None else None)
                    ctx = build_llm_context("SHAP", ss, model_name=name, existing=nw or "", feature_names=fnames, sample_size=_n, task_type=data_config.task_type if data_config else None)
                    render_interpretation_with_llm_button(
                        ctx, key=f"llm_shap_{name}", result_session_key=f"llm_result_shap_{name}",
                    )
                except Exception as e:
                    st.warning(f"Could not redraw SHAP for {name}: {e}")
            
    except ImportError:
        st.warning("⚠️ SHAP not installed. Install with: `pip install shap`")
    except Exception as e:
        st.error(f"❌ Error setting up SHAP: {str(e)}")
        logger.exception(e)

# Bland–Altman (regression only)
st.header("📐 Bland–Altman Plot")
with st.expander("📚 What is a Bland–Altman Plot?", expanded=False):
    st.markdown(
        "Compares **agreement** between two measurement methods (e.g. two models' predictions). "
        "X-axis: mean of the two; Y-axis: difference. Lines show mean difference and limits of agreement (mean ± 1.96 SD). "
        "Useful to see systematic bias and spread of disagreement between models."
    )
mr = st.session_state.get('model_results', {})
task_det = st.session_state.get('task_type_detection')
task_final = (task_det.final if task_det and task_det.final else None) or (data_config.task_type if data_config else None)
if task_final == 'regression' and len(mr) >= 2:
    models_with_pred = [n for n in mr if 'y_test_pred' in mr[n] and mr[n]['y_test_pred'] is not None]
    if len(models_with_pred) >= 2:
        ma = st.selectbox("Model A", models_with_pred, key="bland_altman_a")
        mb = st.selectbox("Model B", [m for m in models_with_pred if m != ma], key="bland_altman_b")
        if ma and mb:
            from visualizations import plot_bland_altman
            from ml.eval import analyze_bland_altman
            from ml.plot_narrative import narrative_bland_altman
            pa = np.asarray(mr[ma]['y_test_pred']).ravel()
            pb = np.asarray(mr[mb]['y_test_pred']).ravel()
            if len(pa) == len(pb):
                fig_ba = plot_bland_altman(pa, pb, title=f"Bland–Altman: {ma.upper()} vs {mb.upper()}", label_a=ma, label_b=mb)
                st.plotly_chart(fig_ba, use_container_width=True, key=f"bland_altman_{ma}_{mb}")
                ba_stats = analyze_bland_altman(pa, pb)
                nar = narrative_bland_altman(ba_stats, label_a=ma, label_b=mb)
                if nar:
                    st.markdown(f"**Interpretation:** {nar}")
                from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                stats_summary = f"mean_diff={ba_stats.get('mean_diff', 0):.4f}; width_loa={ba_stats.get('width_loa', 0):.4f}; pct_outside={ba_stats.get('pct_outside_loa', 0):.1%}"
                ctx = build_llm_context(
                    "bland_altman", stats_summary, where=f"Bland-Altman ({ma} vs {mb})", existing=nar or "",
                    metrics=ba_stats, sample_size=len(pa), task_type=data_config.task_type if data_config else None,
                )
                render_interpretation_with_llm_button(
                    ctx, key="llm_bland_altman_btn", result_session_key="llm_result_bland_altman",
                )
            else:
                st.warning("Prediction lengths differ; cannot compare.")
    else:
        st.info("At least two models with test predictions are required.")
else:
    st.info("Bland–Altman is available for **regression** tasks with at least two trained models.")

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
