"""
Page 03: Preprocessing Builder
Build sklearn Pipeline with ColumnTransformer.
Integrates coach recommendations for intelligent preprocessing suggestions.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from utils.session_state import init_session_state, get_data, DataConfig, set_preprocessing_pipeline
from utils.storyline import render_progress_indicator, get_insights_by_category
from ml.pipeline import build_preprocessing_pipeline, get_pipeline_recipe
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

# ============================================================================
# COACH-DRIVEN PREPROCESSING RECOMMENDATIONS
# ============================================================================
st.markdown("---")
st.header("🎓 Recommended Preprocessing")

# Get dataset profile and coach output from session state
profile = st.session_state.get('dataset_profile')
coach_output = st.session_state.get('coach_output')

# Custom CSS for recommendation cards
st.markdown("""
<style>
.prep-rec-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #fafafa;
}
.prep-priority-required {
    border-left: 4px solid #dc3545;
}
.prep-priority-recommended {
    border-left: 4px solid #ffc107;
}
.prep-priority-optional {
    border-left: 4px solid #28a745;
}
.prep-apply-btn {
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# If we have coach recommendations, show them
if coach_output and hasattr(coach_output, 'preprocessing_recommendations'):
    prep_recs = coach_output.preprocessing_recommendations
    
    if prep_recs:
        st.markdown("""
        Based on your **dataset profile** and **likely model families**, here are the preprocessing steps we recommend.
        Click "Apply" to automatically configure each step.
        """)
        
        # Group by priority
        required_recs = [r for r in prep_recs if r.priority == 'required']
        recommended_recs = [r for r in prep_recs if r.priority == 'recommended']
        optional_recs = [r for r in prep_recs if r.priority == 'optional']
        
        # Required preprocessing
        if required_recs:
            st.markdown("### 🔴 Required Preprocessing")
            st.caption("These steps are critical for your data and models to work properly.")
            
            for rec in required_recs:
                with st.expander(f"**{rec.step_name}**", expanded=True):
                    st.markdown(f"**Why:** {rec.rationale}")
                    st.info(rec.plain_language_explanation)
                    st.markdown(f"**How:** {rec.how_to_implement}")
                    if rec.affected_model_families:
                        st.caption(f"Affects: {', '.join(rec.affected_model_families)}")
                    
                    # Apply button with specific action
                    if rec.step_key == 'imputation':
                        if st.button("Apply: Use median imputation", key=f"apply_{rec.step_key}"):
                            st.session_state['preprocess_numeric_imputation'] = 'median'
                            st.success("✅ Applied! Numeric imputation set to median.")
                    elif rec.step_key == 'scaling':
                        if st.button("Apply: Use standard scaling", key=f"apply_{rec.step_key}"):
                            st.session_state['preprocess_numeric_scaling'] = 'standard'
                            st.success("✅ Applied! Scaling set to standard.")
                    elif rec.step_key == 'high_card_encoding':
                        st.info("💡 For high cardinality, consider target encoding or using tree-based models which handle it natively.")
                    elif rec.step_key == 'imbalance':
                        st.info("💡 Use `class_weight='balanced'` during training (configured in Train & Compare page).")
        
        # Recommended preprocessing
        if recommended_recs:
            st.markdown("### 🟡 Recommended Preprocessing")
            st.caption("These steps will likely improve your model performance.")
            
            for rec in recommended_recs:
                with st.expander(f"**{rec.step_name}**"):
                    st.markdown(f"**Why:** {rec.rationale}")
                    st.info(rec.plain_language_explanation)
                    st.markdown(f"**How:** {rec.how_to_implement}")
                    if rec.affected_model_families:
                        st.caption(f"Affects: {', '.join(rec.affected_model_families)}")
        
        # Optional preprocessing
        if optional_recs:
            st.markdown("### 🟢 Optional Preprocessing")
            st.caption("These steps may help in some cases.")
            
            for rec in optional_recs:
                with st.expander(f"**{rec.step_name}**"):
                    st.markdown(f"**Why:** {rec.rationale}")
                    st.info(rec.plain_language_explanation)
                    st.markdown(f"**How:** {rec.how_to_implement}")
    else:
        st.success("✅ No critical preprocessing steps identified based on your data profile.")
        
elif profile:
    # Fallback: generate recommendations from profile directly
    st.markdown("""
    Based on your **dataset characteristics**, here are preprocessing recommendations:
    """)
    
    recommendations_made = False
    
    # Check for missing values
    if profile.n_features_with_missing > 0:
        recommendations_made = True
        severity = "🔴 Required" if profile.n_features_high_missing > 0 else "🟡 Recommended"
        with st.expander(f"{severity}: Handle Missing Values", expanded=True):
            st.markdown(f"**{profile.n_features_with_missing}** features have missing values.")
            if profile.n_features_high_missing > 0:
                st.warning(f"**{profile.n_features_high_missing}** features have >10% missing - needs attention!")
            st.info("Use median imputation for numeric features. Tree models can handle missing values natively.")
            if st.button("Apply: Use median imputation", key="apply_imputation_fallback"):
                st.session_state['preprocess_numeric_imputation'] = 'median'
                st.success("✅ Applied!")
    
    # Check for scaling need
    if profile.n_numeric > 1:
        recommendations_made = True
        with st.expander("🟡 Recommended: Scale Numeric Features"):
            st.markdown(f"You have **{profile.n_numeric}** numeric features with different scales.")
            st.info("Standard scaling (mean=0, std=1) is recommended for linear models, neural networks, and distance-based models.")
            if st.button("Apply: Use standard scaling", key="apply_scaling_fallback"):
                st.session_state['preprocess_numeric_scaling'] = 'standard'
                st.success("✅ Applied!")
    
    # Check for high cardinality
    if profile.high_cardinality_features:
        recommendations_made = True
        with st.expander("🟡 Recommended: Address High-Cardinality Categoricals"):
            st.markdown(f"Features with many categories: **{', '.join(profile.high_cardinality_features[:3])}**")
            st.info("One-hot encoding will create many columns. Consider target encoding or using tree-based models.")
    
    # Check for outliers
    if profile.features_with_outliers:
        recommendations_made = True
        with st.expander("🟢 Optional: Address Outliers"):
            st.markdown(f"**{len(profile.features_with_outliers)}** features have outliers.")
            st.info("Options: Use robust scaling, Huber regression, or tree-based models which are outlier-robust.")
            if st.button("Apply: Use robust scaling", key="apply_robust_scaling"):
                st.session_state['preprocess_numeric_scaling'] = 'robust'
                st.success("✅ Applied!")
    
    if not recommendations_made:
        st.success("✅ Your data looks clean! Standard preprocessing should work well.")
else:
    st.info("💡 Run EDA first to get personalized preprocessing recommendations.")

# ============================================================================
# RECOMMENDED MODELS CONTEXT
# ============================================================================
# Show what models are selected/recommended to guide preprocessing
st.markdown("---")
with st.expander("📊 Preprocessing for Selected/Recommended Models"):
    # Check if any models are selected
    selected_models = [k.replace('train_model_', '') for k, v in st.session_state.items() 
                       if k.startswith('train_model_') and v]
    
    if selected_models:
        st.markdown(f"**Selected models:** {', '.join(selected_models)}")
        
        # Determine preprocessing needs
        needs_scaling = any(m in ['ridge', 'lasso', 'elasticnet', 'glm', 'nn', 'svc', 'svr', 'knn_reg', 'knn_clf', 'logreg'] 
                          for m in selected_models)
        handles_missing = any(m in ['rf', 'histgb_reg', 'histgb_clf', 'extratrees_reg', 'extratrees_clf'] 
                             for m in selected_models)
        
        if needs_scaling:
            st.info("🔧 **Scaling is important** for your selected linear models / neural networks.")
        if handles_missing:
            st.success("✅ Your selected tree models **handle missing values natively** - imputation is optional for them.")
    else:
        st.info("No models selected yet. Recommendations will be more specific after selecting models in Train & Compare.")
        
        # Show quick summary of preprocessing needs by model family
        st.markdown("""
        **Quick guide:**
        - **Linear Models** (Ridge, Lasso, GLM): Require scaling and encoding
        - **Tree Models** (Random Forest, Gradient Boosting): No scaling needed, handle missing values
        - **Neural Networks**: Require scaling, encoding, and benefit from more data
        """)

# ============================================================================
# WHY THIS MATTERS
# ============================================================================
with st.expander("📚 Why Preprocessing Matters", expanded=False):
    st.markdown("""
    **Scaling:**
    - Required for kNN, SVM, PCA, and Neural Networks
    - Standard scaling (mean=0, std=1) works well for most cases
    - Robust scaling (median/IQR) is better for outliers
    
    **Encoding:**
    - One-hot encoding creates binary columns for each category
    - High-cardinality categoricals can explode feature count
    - Consider alternatives (target encoding) if many categories
    
    **Missingness:**
    - Mean/median imputation works for numeric features
    - Most frequent works for categoricals
    - Tree models can handle missing values natively (no imputation needed)
    
    **Feature Engineering:**
    - PCA reduces dimensionality (useful for high p/n ratio)
    - KMeans features add cluster-based patterns
    """)

# Show relevant insights
insights = get_insights_by_category()
relevant_insights = [i for i in insights if i.get('category') in ['feature_relationships', 'data_quality']]
if relevant_insights:
    st.info("💡 **Relevant insights from EDA:**")
    for insight in relevant_insights:
        st.caption(f"• {insight['finding']} → {insight['implication']}")

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================
st.header("🔧 Preprocessing Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric Features")
    if numeric_features:
        # Read from session_state or use defaults - ensure config exists
        preprocessing_config = st.session_state.get('preprocessing_config', {})
        if not preprocessing_config:
            preprocessing_config = {}
            st.session_state.preprocessing_config = preprocessing_config
        
        # Safe index computation
        numeric_imputation_options = ['median', 'mean', 'constant']
        numeric_imputation_idx = safe_option_index(
            numeric_imputation_options,
            st.session_state.get('preprocess_numeric_imputation', preprocessing_config.get('numeric_imputation')),
            'median'
        )
        numeric_imputation = st.selectbox(
            "Imputation Strategy",
            numeric_imputation_options,
            index=numeric_imputation_idx,
            key="preprocess_numeric_imputation",
            help="How to handle missing numeric values"
        )
        
        numeric_scaling_options = ['standard', 'robust', 'none']
        numeric_scaling_idx = safe_option_index(
            numeric_scaling_options,
            st.session_state.get('preprocess_numeric_scaling', preprocessing_config.get('numeric_scaling')),
            'standard'
        )
        numeric_scaling = st.selectbox(
            "Scaling Strategy",
            numeric_scaling_options,
            index=numeric_scaling_idx,
            key="preprocess_numeric_scaling",
            help="Feature scaling method"
        )
        numeric_log_transform = st.checkbox(
            "Log Transform",
            value=st.session_state.get('preprocess_numeric_log_transform', False),
            key="preprocess_numeric_log_transform",
            help="Apply log(1+x) transformation"
        )
    else:
        st.info("No numeric features")
        numeric_imputation = 'median'
        numeric_scaling = 'standard'
        numeric_log_transform = False

with col2:
    st.subheader("Categorical Features")
    if categorical_features:
        categorical_imputation_options = ['most_frequent', 'constant']
        categorical_imputation_idx = safe_option_index(
            categorical_imputation_options,
            preprocessing_config.get('categorical_imputation'),
            'most_frequent'
        )
        categorical_imputation = st.selectbox(
            "Imputation Strategy",
            categorical_imputation_options,
            index=categorical_imputation_idx,
            key="preprocess_categorical_imputation",
            help="How to handle missing categorical values"
        )
        categorical_encoding = st.selectbox(
            "Encoding Strategy",
            ['onehot'],
            index=0,
            key="preprocess_categorical_encoding",
            help="Categorical encoding method"
        )
    else:
        st.info("No categorical features")
        categorical_imputation = 'most_frequent'
        categorical_encoding = 'onehot'

# ============================================================================
# OPTIONAL FEATURE ENGINEERING
# ============================================================================
st.header("🔬 Optional Feature Engineering")

preprocessing_config = st.session_state.get('preprocessing_config', {})
if not preprocessing_config:
    preprocessing_config = {}
    st.session_state.preprocessing_config = preprocessing_config

fe_col1, fe_col2 = st.columns(2)

with fe_col1:
    st.subheader("KMeans Features")
    use_kmeans = st.checkbox(
        "Enable KMeans Features",
        value=preprocessing_config.get('use_kmeans_features', False),
        key="preprocess_use_kmeans",
        help="Add cluster-based features (distances to centroids)"
    )
    if use_kmeans:
        kmeans_n_clusters = st.number_input(
            "Number of Clusters",
            min_value=2,
            max_value=20,
            value=preprocessing_config.get('kmeans_n_clusters', 5),
            key="preprocess_kmeans_n_clusters",
            help="Number of KMeans clusters"
        )
        kmeans_add_distances = st.checkbox(
            "Add Distance Features",
            value=preprocessing_config.get('kmeans_add_distances', True),
            key="preprocess_kmeans_distances",
            help="Add distances to each cluster centroid"
        )
        kmeans_add_onehot = st.checkbox(
            "Add One-Hot Cluster Labels",
            value=preprocessing_config.get('kmeans_add_onehot', False),
            key="preprocess_kmeans_onehot",
            help="Add one-hot encoded cluster assignments"
        )
    else:
        kmeans_n_clusters = 5
        kmeans_add_distances = True
        kmeans_add_onehot = False

with fe_col2:
    st.subheader("PCA (Principal Component Analysis)")
    
    # Check if PCA is recommended based on profile
    pca_recommended = False
    if profile and profile.p_n_ratio > 0.3:
        st.info(f"💡 PCA may help: you have a relatively high feature-to-sample ratio ({profile.p_n_ratio:.2f})")
        pca_recommended = True
    
    use_pca = st.checkbox(
        "Enable PCA",
        value=preprocessing_config.get('use_pca', False),
        key="preprocess_use_pca",
        help="Dimensionality reduction via PCA"
    )
    if use_pca:
        pca_mode = st.radio(
            "PCA Mode",
            options=['Fixed Components', 'Variance Threshold'],
            index=0 if preprocessing_config.get('pca_n_components') is None or isinstance(preprocessing_config.get('pca_n_components'), int) else 1,
            key="preprocess_pca_mode",
            help="Fixed number of components or variance threshold"
        )
        if pca_mode == 'Fixed Components':
            # Get max components based on available features after preprocessing
            max_components_estimate = len(numeric_features) + (len(categorical_features) * 5) if categorical_features else len(numeric_features)
            max_components = max(1, min(50, max_components_estimate))
            
            default_n_components = min(10, max_components)
            if isinstance(preprocessing_config.get('pca_n_components'), int):
                default_n_components = min(preprocessing_config.get('pca_n_components'), max_components)
            
            pca_n_components = st.number_input(
                "Number of Components",
                min_value=1,
                max_value=max_components,
                value=default_n_components,
                key="preprocess_pca_n_components",
                help=f"Number of principal components to keep (max: {max_components} based on estimated feature count)"
            )
        else:
            pca_variance = st.slider(
                "Variance Threshold",
                min_value=0.5,
                max_value=0.99,
                value=preprocessing_config.get('pca_n_components', 0.95) if isinstance(preprocessing_config.get('pca_n_components'), float) else 0.95,
                step=0.05,
                key="preprocess_pca_variance",
                help="Keep components that explain this fraction of variance"
            )
            pca_n_components = pca_variance
        
        pca_whiten = st.checkbox(
            "Whiten Components",
            value=preprocessing_config.get('pca_whiten', False),
            key="preprocess_pca_whiten",
            help="Whiten the components (unit variance)"
        )
    else:
        pca_n_components = None
        pca_whiten = False

# Mutual exclusion warning
if use_kmeans and use_pca:
    st.warning("⚠️ Both KMeans Features and PCA are enabled. KMeans will be applied first, then PCA on the result.")

# ============================================================================
# BUILD PIPELINE
# ============================================================================
if st.button("🔨 Build Preprocessing Pipeline", type="primary", key="preprocess_build_button"):
    try:
        with st.spinner("Building pipeline..."):
            # First, build pipeline without PCA to check feature count
            temp_pipeline = build_preprocessing_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                numeric_imputation=numeric_imputation,
                numeric_scaling=numeric_scaling,
                numeric_log_transform=numeric_log_transform,
                categorical_imputation=categorical_imputation,
                categorical_encoding=categorical_encoding,
                use_kmeans_features=use_kmeans,
                kmeans_n_clusters=kmeans_n_clusters if use_kmeans else 5,
                kmeans_add_distances=kmeans_add_distances if use_kmeans else True,
                kmeans_add_onehot=kmeans_add_onehot if use_kmeans else False,
                use_pca=False,  # Build without PCA first
                random_state=st.session_state.get('random_seed', 42)
            )
            
            # Fit temp pipeline to get actual feature count
            X_sample = df[all_features]
            temp_pipeline.fit(X_sample)
            X_temp_transformed = temp_pipeline.transform(X_sample)
            if hasattr(X_temp_transformed, 'toarray'):
                X_temp_transformed = X_temp_transformed.toarray()
            actual_feature_count = X_temp_transformed.shape[1]
            
            # Validate PCA n_components if enabled
            if use_pca and isinstance(pca_n_components, int):
                if pca_n_components > actual_feature_count:
                    st.warning(f"⚠️ PCA n_components ({pca_n_components}) exceeds available features ({actual_feature_count}). Adjusting to {actual_feature_count}.")
                    pca_n_components = actual_feature_count
            
            # Now build final pipeline with validated PCA
            pipeline = build_preprocessing_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                numeric_imputation=numeric_imputation,
                numeric_scaling=numeric_scaling,
                numeric_log_transform=numeric_log_transform,
                categorical_imputation=categorical_imputation,
                categorical_encoding=categorical_encoding,
                # Feature engineering steps
                use_kmeans_features=use_kmeans,
                kmeans_n_clusters=kmeans_n_clusters if use_kmeans else 5,
                kmeans_add_distances=kmeans_add_distances if use_kmeans else True,
                kmeans_add_onehot=kmeans_add_onehot if use_kmeans else False,
                use_pca=use_pca,
                pca_n_components=pca_n_components if use_pca else None,
                pca_whiten=pca_whiten if use_pca else False,
                random_state=st.session_state.get('random_seed', 42)
            )
            
            # Fit final pipeline
            pipeline.fit(X_sample)
            
            # Get transformed feature names
            X_transformed = pipeline.transform(X_sample)
            # Convert sparse to dense if needed
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            if hasattr(X_transformed, 'shape'):
                n_output_features = X_transformed.shape[1]
            else:
                n_output_features = len(X_transformed[0]) if X_transformed else 0
            
            # Store pipeline and config
            config = {
                'numeric_features': numeric_features,
                'categorical_features': categorical_features,
                'numeric_imputation': numeric_imputation,
                'numeric_scaling': numeric_scaling,
                'numeric_log_transform': numeric_log_transform,
                'categorical_imputation': categorical_imputation,
                'categorical_encoding': categorical_encoding,
                'n_output_features': n_output_features,
                # Feature engineering config
                'use_kmeans_features': use_kmeans,
                'kmeans_n_clusters': kmeans_n_clusters if use_kmeans else 5,
                'kmeans_add_distances': kmeans_add_distances if use_kmeans else True,
                'kmeans_add_onehot': kmeans_add_onehot if use_kmeans else False,
                'use_pca': use_pca,
                'pca_n_components': pca_n_components if use_pca else None,
                'pca_whiten': pca_whiten if use_pca else False
            }
            
            set_preprocessing_pipeline(pipeline, config)
        
        st.success(f"✅ Pipeline built successfully! Output features: {n_output_features}")
        
        # Show pipeline recipe
        st.header("📋 Pipeline Recipe")
        recipe = get_pipeline_recipe(pipeline)
        st.code(recipe, language=None)
        
        # Preview transformation
        st.header("👀 Transformation Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before")
            st.dataframe(X_sample.head(10), use_container_width=True)
        
        with col2:
            st.subheader("After")
            # Ensure dense array for DataFrame
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            X_transformed_df = pd.DataFrame(
                X_transformed[:10],
                columns=[f"feature_{i}" for i in range(n_output_features)]
            )
            st.dataframe(X_transformed_df, use_container_width=True)
        
        st.info("✅ Pipeline ready! Proceed to Train & Compare page.")
        
    except Exception as e:
        st.error(f"❌ Error building pipeline: {str(e)}")
        st.exception(e)

# Show existing pipeline if available
if st.session_state.get('preprocessing_pipeline') is not None:
    st.header("✅ Current Pipeline")
    existing_pipeline = st.session_state.preprocessing_pipeline
    recipe = get_pipeline_recipe(existing_pipeline)
    st.code(recipe, language=None)
    
    if st.button("🔄 Rebuild Pipeline", key="preprocess_rebuild_button"):
        st.session_state.preprocessing_pipeline = None
        st.session_state.preprocessing_config = None
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