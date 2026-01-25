"""
Page 03: Preprocessing Builder
Build sklearn Pipeline with ColumnTransformer.
Integrates coach recommendations for intelligent preprocessing suggestions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import copy
from typing import List, Dict, Any, Optional

from utils.session_state import (
    init_session_state, get_data, DataConfig, set_preprocessing_pipeline, set_preprocessing_pipelines,
    TaskTypeDetection,
)
from utils.storyline import render_progress_indicator, get_insights_by_category, add_insight
from ml.pipeline import (
    build_preprocessing_pipeline,
    get_pipeline_recipe,
    get_feature_names_after_transform,
    build_unit_harmonization_config,
    build_plausibility_bounds,
)
from ml.model_registry import get_registry
from data_processor import get_numeric_columns
from utils.widget_helpers import safe_option_index

init_session_state()

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")
st.title("⚙️ Preprocessing Builder")

# Progress indicator
render_progress_indicator("03_Preprocess")

df = get_data()
if df is None:
    st.warning("⚠️ Please upload data in the Upload & Audit page first")
    st.stop()

data_config: Optional[DataConfig] = st.session_state.get('data_config')
if data_config is None or not data_config.target_col:
    st.warning("⚠️ Please select target and features in the Upload & Audit page first")
    st.stop()

# Identify feature types (safe access with defaults)
all_features = data_config.feature_cols if data_config else []
if not all_features:
    st.warning("⚠️ No features selected. Please select features in the Upload & Audit page first")
    st.stop()

numeric_cols = get_numeric_columns(df)
numeric_features = [f for f in all_features if f in numeric_cols]
categorical_features = [f for f in all_features if f not in numeric_cols]

st.info(f"**Numeric features:** {len(numeric_features)} | **Categorical features:** {len(categorical_features)}")

# Get profile, insights, EDA results for recommendations
profile = st.session_state.get('dataset_profile')
coach_output = st.session_state.get('coach_output')
insights = get_insights_by_category()
eda_results = st.session_state.get('eda_results', {})
relevant_insights = [i for i in insights if i.get('category') in ['feature_relationships', 'data_quality']]

# EDA-based recommendation cues (for display next to options)
_eda_outliers = bool(profile and profile.features_with_outliers)
_eda_missing = bool(profile and profile.n_features_with_missing > 0)
_eda_high_pn = bool(profile and getattr(profile, 'p_n_ratio', 0) > 0.3)
_eda_collinearity = any('collinearity' in str(k).lower() or 'multicollinearity' in str(k).lower() for k in (eda_results or {}))

# ============================================================================
# 1. MODEL SELECTION FIRST
# ============================================================================
st.markdown("---")
st.header("📋 Select models for preprocessing")
st.caption("Select models below; these choices drive pipeline options and are used on Train & Compare.")
task_type_det = st.session_state.get("task_type_detection") or TaskTypeDetection()
task_type_final = (getattr(task_type_det, "final", None) or (data_config.task_type if data_config else None) or "regression")
if data_config:
    data_config.task_type = task_type_final

registry_prep = get_registry()
available_prep = {
    k: v for k, v in registry_prep.items()
    if (task_type_final == "regression" and v.capabilities.supports_regression)
    or (task_type_final == "classification" and v.capabilities.supports_classification)
}
model_groups_prep: Dict[str, List[tuple]] = {}
for key, spec in available_prep.items():
    g = spec.group
    if g not in model_groups_prep:
        model_groups_prep[g] = []
    model_groups_prep[g].append((key, spec))

for group_name in sorted(model_groups_prep.keys()):
    st.subheader(f"{group_name} models")
    for model_key, spec in model_groups_prep[group_name]:
        ck = f"train_model_{model_key}"
        st.checkbox(
            spec.name,
            value=st.session_state.get(ck, False),
            key=ck,
            help=(", ".join(spec.capabilities.notes) if spec.capabilities.notes else None),
        )
selected_models = [k.replace("train_model_", "") for k, v in st.session_state.items() if k.startswith("train_model_") and v]
if selected_models:
    st.caption(f"**Selected:** {', '.join(selected_models)}. Each gets its own pipeline when you build.")
else:
    st.caption("Select at least one model to build model-specific pipelines; otherwise a single default pipeline is built.")

# ============================================================================
# 2. INTERPRETABILITY (GLOBAL) + CONFIGURE PIPELINE PER MODEL
# ============================================================================
st.markdown("---")
st.header("🔧 Configure pipeline per model")
st.caption("Click a model to expand and set preprocessing options. Then scroll down and click **Build Pipelines** once.")

preprocessing_config = st.session_state.get("preprocessing_config", {}) or {}
st.session_state.preprocessing_config = preprocessing_config

_imode_opts = ["high", "balanced", "performance"]
_imode_stored = st.session_state.get("interpretability_mode", "balanced")
_imode_idx = _imode_opts.index(_imode_stored) if _imode_stored in _imode_opts else 1
interpretability_mode = st.selectbox(
    "Interpretability preference (applies to all models)",
    _imode_opts,
    index=_imode_idx,
    key="interpretability_mode",
    help="High disables log transform, PCA, KMeans. Balanced/Performance allow them.",
)
with st.expander("📚 Interpretability vs Performance", expanded=False):
    st.markdown("**High:** Disables log, PCA, KMeans. **Performance:** Allows them. **Balanced:** Default.")

def _interpretability_guidance(
    profile: Optional[Any],
    insights: List[Dict],
    eda_results: Dict,
    selected: List[str],
    registry: Dict,
) -> List[str]:
    bullets = []
    pn = getattr(profile, "p_n_ratio", 0) if profile else 0
    has_collinearity = any("collinearity" in str(k).lower() or "multicollinearity" in str(k).lower() for k in (eda_results or {}))
    has_outliers = bool(profile and profile.features_with_outliers)
    linear = [m for m in selected if m in ["ridge", "lasso", "elasticnet", "glm", "logreg"]]
    trees = [m for m in selected if m in ["rf", "extratrees_reg", "extratrees_clf", "histgb_reg", "histgb_clf"]]
    nn_only = selected and all(m == "nn" for m in selected)
    if pn > 0.3 and linear:
        bullets.append(f"High feature-to-sample ratio ({pn:.2f}) and linear models → **performance** can help accuracy; **high** keeps pipelines simple for stakeholders.")
    if has_collinearity and linear:
        bullets.append("Collinearity detected and linear models → **balanced** or **performance**; consider PCA or regularization.")
    if has_outliers and (linear or selected and "nn" in selected):
        bullets.append("Outliers present → **performance** (e.g. robust scaling) or **balanced**; **high** avoids extra transforms.")
    if trees and not linear and not (selected and "nn" in selected):
        bullets.append("Mostly tree models → interpretability preference mainly affects optional preprocessing (log, PCA, KMeans); **balanced** is a reasonable default.")
    if nn_only:
        bullets.append("Neural network only → interpretability affects only preprocessing; use **performance** if you care more about accuracy than explainability.")
    if not bullets:
        bullets.append("**Balanced** is a reasonable default. Use **high** when you need simple, explainable pipelines; **performance** when accuracy matters most.")
    return bullets[:4]

_guidance = _interpretability_guidance(profile, insights, eda_results or {}, selected_models, registry_prep)
if _guidance:
    st.caption("**Interpretability guidance:**")
    for _g in _guidance:
        st.caption(f"• {_g}")

_config_keys = ["default"] if not selected_models else selected_models

def _cfg(mk: str, key: str, default: Any, from_global: bool = True) -> Any:
    k = f"preprocess_{mk}_{key}"
    v = st.session_state.get(k)
    if v is not None:
        return v
    if from_global and preprocessing_config:
        return preprocessing_config.get(key, default)
    return default

for _mk in _config_keys:
    with st.expander(f"Configure {_mk.upper()}", expanded=False):
        # Slicing / cutting
        st.subheader("✂️ Slicing / cutting")
        if _eda_outliers:
            st.caption("💡 EDA found outliers → consider outlier treatment or plausibility gating.")
        _c1, _c2 = st.columns(2)
        with _c1:
            if numeric_features:
                _opt = ["none", "percentile", "mad"]
                _v = _cfg(_mk, "numeric_outlier_treatment", "none")
                _idx = _opt.index(_v) if _v in _opt else 0
                _ot = st.selectbox("Outlier treatment", _opt, index=_idx, key=f"preprocess_{_mk}_numeric_outlier_treatment")
                if _ot == "percentile":
                    st.number_input("Lower percentile", 0.0, 0.1, 0.01, key=f"preprocess_{_mk}_outlier_lower_q")
                    st.number_input("Upper percentile", 0.9, 1.0, 0.99, key=f"preprocess_{_mk}_outlier_upper_q")
                elif _ot == "mad":
                    st.number_input("MAD threshold", 2.0, 6.0, 3.5, key=f"preprocess_{_mk}_outlier_mad_threshold")
            else:
                _ot = "none"
        with _c2:
            st.checkbox("Plausibility gating (NHANES)", value=bool(_cfg(_mk, "plausibility_gating", False)), key=f"preprocess_{_mk}_plausibility_gating")
            st.checkbox("Unit harmonization", value=bool(_cfg(_mk, "unit_harmonization", False)), key=f"preprocess_{_mk}_unit_harmonization")

        # Imputing
        st.subheader("🔄 Imputing")
        if _eda_missing:
            st.caption("💡 EDA found missing values → consider imputation and/or missing indicators.")
        _c3, _c4 = st.columns(2)
        with _c3:
            _nim = safe_option_index(["median", "mean", "constant"], _cfg(_mk, "numeric_imputation", "median"), "median")
            st.selectbox("Numeric imputation", ["median", "mean", "constant"], index=_nim, key=f"preprocess_{_mk}_numeric_imputation")
            st.checkbox("Add missing indicators", value=bool(_cfg(_mk, "numeric_missing_indicators", False)), key=f"preprocess_{_mk}_numeric_missing_indicators")
        with _c4:
            _cim = safe_option_index(["most_frequent", "constant"], _cfg(_mk, "categorical_imputation", "most_frequent"), "most_frequent")
            st.selectbox("Categorical imputation", ["most_frequent", "constant"], index=_cim, key=f"preprocess_{_mk}_categorical_imputation")

        # Transformations
        st.subheader("📐 Transformations")
        _scl = safe_option_index(["standard", "robust", "none"], _cfg(_mk, "numeric_scaling", "standard"), "standard")
        st.selectbox("Scaling", ["standard", "robust", "none"], index=_scl, key=f"preprocess_{_mk}_numeric_scaling")
        st.checkbox("Log transform (log(1+x))", value=bool(_cfg(_mk, "numeric_log_transform", False)), key=f"preprocess_{_mk}_numeric_log_transform")

        # Encoding
        st.subheader("📦 Encoding")
        st.caption("One-hot creates a binary column per category; high-cardinality variables increase feature count.")
        st.selectbox("Categorical encoding", ["onehot"], index=0, key=f"preprocess_{_mk}_categorical_encoding")

        # Feature augmentation
        st.subheader("🔬 Feature augmentation")
        if _eda_high_pn:
            st.caption("💡 High feature-to-sample ratio → PCA may help.")
        if _eda_collinearity:
            st.caption("💡 Collinearity detected → PCA or regularization can help.")
        _uk = bool(_cfg(_mk, "use_kmeans_features", False))
        st.checkbox("KMeans features (adds distances, optional one-hot labels)", value=_uk, key=f"preprocess_{_mk}_use_kmeans")
        st.caption("**What changes:** Adds columns: distances to each cluster centroid and, if enabled, one-hot cluster labels. Original features remain unless PCA is also used.")
        if _uk:
            st.number_input("Clusters", 2, 20, int(_cfg(_mk, "kmeans_n_clusters", 5)), key=f"preprocess_{_mk}_kmeans_n_clusters")
            st.checkbox("Add distances", value=bool(_cfg(_mk, "kmeans_add_distances", True)), key=f"preprocess_{_mk}_kmeans_distances")
            st.checkbox("Add one-hot labels", value=bool(_cfg(_mk, "kmeans_add_onehot", False)), key=f"preprocess_{_mk}_kmeans_onehot")
        _up = bool(_cfg(_mk, "use_pca", False))
        st.checkbox("PCA (output PC1, PC2, …)", value=_up, key=f"preprocess_{_mk}_use_pca")
        st.caption("**What changes:** Replaces the current numeric (and possibly KMeans) features with principal components (PC1, PC2, …). Original feature names are no longer in the output.")
        if _up:
            _maxc = max(1, min(50, len(numeric_features) + (len(categorical_features) * 5) if categorical_features else len(numeric_features)))
            _pn = _cfg(_mk, "pca_n_components", 10)
            _fix = isinstance(_pn, (int, type(1)))
            _pmode = st.radio("PCA mode", ["Fixed Components", "Variance Threshold"], index=0 if _fix else 1, key=f"preprocess_{_mk}_pca_mode")
            if _pmode == "Fixed Components":
                _defn = min(int(_pn), _maxc) if isinstance(_pn, (int, float)) else min(10, _maxc)
                st.number_input("Components", 1, _maxc, _defn, key=f"preprocess_{_mk}_pca_n_components")
            else:
                _pv = 0.95 if not isinstance(_pn, (int, float)) or _pn > 1 else float(_pn)
                st.slider("Variance", 0.5, 0.99, _pv, 0.05, key=f"preprocess_{_mk}_pca_n_components")
            st.checkbox("Whiten", value=bool(_cfg(_mk, "pca_whiten", False)), key=f"preprocess_{_mk}_pca_whiten")

st.markdown("---")
if st.button("🔨 Build Pipelines", type="primary", key="preprocess_build_button"):
    try:
        with st.spinner("Building pipelines..."):
            _sel = [k.replace("train_model_", "") for k, v in st.session_state.items() if k.startswith("train_model_") and v]
            registry = get_registry()
            model_keys = _sel if _sel else ["default"]

            def _get(mk: str, key: str, default: Any) -> Any:
                return st.session_state.get(f"preprocess_{mk}_{key}", default)

            any_unit = any(_get(mk, "unit_harmonization", False) for mk in model_keys)
            unit_overrides = st.session_state.get("unit_overrides", {})
            unit_config = build_unit_harmonization_config(df, numeric_features, unit_overrides) if any_unit else None
            any_plaus = any_unit and any(_get(mk, "plausibility_gating", False) for mk in model_keys)
            plausibility_bounds = build_plausibility_bounds(numeric_features, unit_config["conversion_factors"]) if (unit_config and any_plaus) else None

            def apply_interpretability_overrides(c: Dict[str, Any], imode: str) -> List[str]:
                notes = []
                if imode != "high":
                    return notes
                if c.get("numeric_log_transform"):
                    c["numeric_log_transform"] = False
                    notes.append("Disabled log transform to preserve interpretability.")
                if c.get("use_pca"):
                    c["use_pca"] = False
                    notes.append("Disabled PCA for interpretability.")
                if c.get("use_kmeans_features"):
                    c["use_kmeans_features"] = False
                    notes.append("Disabled KMeans features for interpretability.")
                return notes

            def apply_model_requirements(c: Dict[str, Any], caps: Any) -> List[str]:
                notes = []
                if caps and getattr(caps, "requires_scaled_numeric", False) and c.get("numeric_scaling") == "none":
                    c["numeric_scaling"] = "standard"
                    notes.append("Enabled standard scaling (model requires scaling).")
                return notes

            pipelines_by_model = {}
            configs_by_model = {}
            X_sample = df[all_features]
            imode = st.session_state.get("interpretability_mode", "balanced")

            for model_key in model_keys:
                ot = _get(model_key, "numeric_outlier_treatment", "none")
                params = {}
                if ot == "percentile":
                    params = {"lower_q": float(_get(model_key, "outlier_lower_q", 0.01)), "upper_q": float(_get(model_key, "outlier_upper_q", 0.99))}
                elif ot == "mad":
                    params = {"threshold": float(_get(model_key, "outlier_mad_threshold", 3.5))}

                use_unit = _get(model_key, "unit_harmonization", False)
                use_plaus = _get(model_key, "plausibility_gating", False)
                pca_mode = _get(model_key, "pca_mode", "Fixed Components")
                pn = _get(model_key, "pca_n_components", 10)
                pca_int = pca_mode == "Fixed Components" and (isinstance(pn, (int, float)) and (pn >= 1 and pn == int(pn)))

                model_config = {
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                    "numeric_imputation": _get(model_key, "numeric_imputation", "median"),
                    "numeric_scaling": _get(model_key, "numeric_scaling", "standard"),
                    "numeric_log_transform": bool(_get(model_key, "numeric_log_transform", False)),
                    "numeric_missing_indicators": bool(_get(model_key, "numeric_missing_indicators", False)),
                    "numeric_outlier_treatment": ot,
                    "numeric_outlier_params": params,
                    "categorical_imputation": _get(model_key, "categorical_imputation", "most_frequent"),
                    "categorical_encoding": _get(model_key, "categorical_encoding", "onehot"),
                    "use_kmeans_features": bool(_get(model_key, "use_kmeans", False)),
                    "kmeans_n_clusters": int(_get(model_key, "kmeans_n_clusters", 5)),
                    "kmeans_add_distances": bool(_get(model_key, "kmeans_distances", True)),
                    "kmeans_add_onehot": bool(_get(model_key, "kmeans_onehot", False)),
                    "use_pca": bool(_get(model_key, "use_pca", False)),
                    "pca_n_components": int(pn) if pca_int else (float(pn) if pca_mode == "Variance Threshold" and isinstance(pn, (int, float)) else (0.95 if _get(model_key, "use_pca", False) else None)),
                    "pca_whiten": bool(_get(model_key, "pca_whiten", False)),
                    "unit_harmonization": use_unit,
                    "plausibility_gating": use_plaus,
                    "interpretability_mode": imode,
                }
                if unit_config:
                    model_config["unit_harmonization_config"] = unit_config
                if plausibility_bounds:
                    model_config["plausibility_bounds"] = plausibility_bounds

                override_notes = []
                spec = registry.get(model_key)
                caps = spec.capabilities if spec else None
                override_notes.extend(apply_interpretability_overrides(model_config, imode))
                override_notes.extend(apply_model_requirements(model_config, caps))

                uf = unit_config["conversion_factors"] if unit_config and use_unit else None
                pb = plausibility_bounds if use_plaus and plausibility_bounds else None

                temp_pipeline = build_preprocessing_pipeline(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    numeric_imputation=model_config["numeric_imputation"],
                    numeric_scaling=model_config["numeric_scaling"],
                    numeric_log_transform=model_config["numeric_log_transform"],
                    numeric_missing_indicators=model_config["numeric_missing_indicators"],
                    numeric_outlier_treatment=model_config["numeric_outlier_treatment"],
                    numeric_outlier_params=model_config["numeric_outlier_params"],
                    unit_harmonization_factors=uf,
                    plausibility_bounds=pb,
                    categorical_imputation=model_config["categorical_imputation"],
                    categorical_encoding=model_config["categorical_encoding"],
                    use_kmeans_features=model_config["use_kmeans_features"],
                    kmeans_n_clusters=model_config["kmeans_n_clusters"],
                    kmeans_add_distances=model_config["kmeans_add_distances"],
                    kmeans_add_onehot=model_config["kmeans_add_onehot"],
                    use_pca=False,
                    random_state=st.session_state.get("random_seed", 42),
                )
                temp_pipeline.fit(X_sample)
                X_temp = temp_pipeline.transform(X_sample)
                if hasattr(X_temp, "toarray"):
                    X_temp = X_temp.toarray()
                actual_n = X_temp.shape[1]
                if model_config["use_pca"] and isinstance(model_config["pca_n_components"], int) and model_config["pca_n_components"] > actual_n:
                    model_config["pca_n_components"] = actual_n
                    override_notes.append(f"Adjusted PCA components to {actual_n} (available features).")

                pipeline = build_preprocessing_pipeline(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    numeric_imputation=model_config["numeric_imputation"],
                    numeric_scaling=model_config["numeric_scaling"],
                    numeric_log_transform=model_config["numeric_log_transform"],
                    numeric_missing_indicators=model_config["numeric_missing_indicators"],
                    numeric_outlier_treatment=model_config["numeric_outlier_treatment"],
                    numeric_outlier_params=model_config["numeric_outlier_params"],
                    unit_harmonization_factors=uf,
                    plausibility_bounds=pb,
                    categorical_imputation=model_config["categorical_imputation"],
                    categorical_encoding=model_config["categorical_encoding"],
                    use_kmeans_features=model_config["use_kmeans_features"],
                    kmeans_n_clusters=model_config["kmeans_n_clusters"],
                    kmeans_add_distances=model_config["kmeans_add_distances"],
                    kmeans_add_onehot=model_config["kmeans_add_onehot"],
                    use_pca=model_config["use_pca"],
                    pca_n_components=model_config["pca_n_components"],
                    pca_whiten=model_config["pca_whiten"],
                    random_state=st.session_state.get("random_seed", 42),
                )
                pipeline.fit(X_sample)
                X_transformed = pipeline.transform(X_sample)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()
                model_config["n_output_features"] = X_transformed.shape[1]
                model_config["overrides"] = override_notes
                pipelines_by_model[model_key] = pipeline
                configs_by_model[model_key] = model_config

            base_config = {"numeric_features": numeric_features, "categorical_features": categorical_features}
            set_preprocessing_pipelines(pipelines_by_model, configs_by_model, base_config)
            _built = [k for k in pipelines_by_model.keys() if k != "default"]
            st.session_state["preprocess_built_model_keys"] = _built

            # Model-aware preprocessing insights for Train & Compare and Report
            high_card = bool(profile and getattr(profile, "high_cardinality_features", None))
            model_check_bullets = []
            for mk, cfg in configs_by_model.items():
                spec = registry.get(mk)
                caps = spec.capabilities if spec else None
                scaling = cfg.get("numeric_scaling", "standard")
                ov = cfg.get("overrides", [])
                parts = [f"{mk.upper()}:"]
                if caps and getattr(caps, "requires_scaled_numeric", False):
                    if scaling == "none":
                        parts.append("model requires scaling but you used none — consider enabling scaling.")
                    else:
                        parts.append(f"scaling enabled ({scaling}); appropriate for this model.")
                else:
                    if scaling != "none":
                        parts.append(f"scaling {scaling} (optional for tree models).")
                    else:
                        parts.append("no scaling; fine for tree models.")
                if any("interpretability" in str(o).lower() for o in ov):
                    parts.append("Interpretability overrides applied (e.g. PCA/KMeans disabled).")
                if high_card and cfg.get("categorical_encoding") == "onehot":
                    parts.append("High cardinality in EDA; one-hot may inflate feature count — consider alternatives.")
                model_check_bullets.append(" ".join(parts))
            finding = " ".join(model_check_bullets[:5])
            if len(model_check_bullets) > 5:
                finding += " …"
            add_insight(
                "preprocessing_model_checks",
                finding,
                "Review that preprocessing matches each model; adjust and rebuild if needed.",
                category="preprocessing",
            )
            add_insight(
                "preprocessing_summary",
                f"Pipelines built for {len(pipelines_by_model)} model(s): {', '.join(m.upper() for m in pipelines_by_model.keys())}.",
                "Use Train & Compare to train models; preprocessing is applied per model.",
                category="preprocessing",
            )

        st.success("✅ Preprocessing pipelines built successfully! Expand each model below to view recipe and transformed data.")
        
    except Exception as e:
        st.error(f"❌ Error building pipeline: {str(e)}")
        st.exception(e)

# Per-model expanders: recipe, overrides, show table, CSV export
pipelines_by_model = st.session_state.get("preprocessing_pipelines_by_model") or {}
configs_by_model = st.session_state.get("preprocessing_config_by_model") or {}
if pipelines_by_model:
    st.markdown("---")
    st.header("📋 Pipelines by model")
    st.caption("Expand each model to view recipe and overrides. Use «Show transformed table» to preview values, then «Download as CSV» to export.")
    X_sample_preview = df[all_features]
    for model_key, pipeline in pipelines_by_model.items():
        _show = st.session_state.get(f"show_preview_{model_key}", False)
        with st.expander(f"Pipeline for {model_key.upper()}", expanded=(model_key == "default" or _show)):
            recipe = get_pipeline_recipe(pipeline)
            st.code(recipe, language=None)
            cfg = configs_by_model.get(model_key, {})
            overrides = cfg.get("overrides", [])
            if overrides:
                st.caption("Overrides applied:")
                for note in overrides:
                    st.write(f"• {note}")
            show_table = st.checkbox("Show transformed table", value=_show, key=f"show_preview_{model_key}")
            if show_table:
                _before = X_sample_preview.head(100)
                X_t = pipeline.transform(X_sample_preview)
                if hasattr(X_t, "toarray"):
                    X_t = X_t.toarray()
                col_names = get_feature_names_after_transform(pipeline, all_features)
                if len(col_names) != X_t.shape[1]:
                    col_names = [f"feature_{i}" for i in range(X_t.shape[1])]
                preview_df = pd.DataFrame(X_t, columns=col_names)
                _ba, _aa = st.columns(2)
                with _ba:
                    st.subheader("Before")
                    st.dataframe(_before, use_container_width=True)
                with _aa:
                    st.subheader("After")
                    st.dataframe(preview_df.head(100), use_container_width=True)
                csv_bytes = preview_df.to_csv(index=False).encode()
                st.download_button(
                    "Download as CSV",
                    data=csv_bytes,
                    file_name=f"transformed_{model_key}.csv",
                    mime="text/csv",
                    key=f"download_preview_{model_key}",
                )
    st.info("✅ Pipeline ready! Proceed to Train & Compare page.")

    if st.button("🔄 Rebuild Pipeline", key="preprocess_rebuild_button"):
        st.session_state.preprocessing_pipeline = None
        st.session_state.preprocessing_config = None
        st.session_state.preprocessing_pipelines_by_model = {}
        st.session_state.preprocessing_config_by_model = {}
        st.rerun()

# State Debug (Advanced)
with st.expander("🔧 Advanced / State Debug", expanded=False):
    st.markdown("**Current State:**")
    st.write(f"• Data shape: {df.shape if df is not None else 'None'}")
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")
    st.write(f"• Preprocessing pipeline: {'Built' if st.session_state.get('preprocessing_pipeline') else 'Not built'}")
    preprocessing_config = st.session_state.get('preprocessing_config')
    if preprocessing_config:
        st.write(f"• Numeric imputation: {preprocessing_config.get('numeric_imputation', 'N/A')}")
        st.write(f"• Numeric scaling: {preprocessing_config.get('numeric_scaling', 'N/A')}")
    if profile:
        st.write(f"• Dataset profile available: Yes")
        st.write(f"• Data sufficiency: {profile.data_sufficiency.value}")