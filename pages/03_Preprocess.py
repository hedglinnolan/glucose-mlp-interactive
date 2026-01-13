"""
Page 03: Preprocessing Builder
Build sklearn Pipeline with ColumnTransformer.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any

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

data_config: DataConfig = st.session_state.get('data_config')
if data_config is None or not data_config.target_col:
    st.warning("⚠️ Please select target and features in the Upload & Audit page first")
    st.stop()

# Identify feature types
all_features = data_config.feature_cols
numeric_cols = get_numeric_columns(df)
numeric_features = [f for f in all_features if f in numeric_cols]
categorical_features = [f for f in all_features if f not in numeric_cols]

st.info(f"**Numeric features:** {len(numeric_features)} | **Categorical features:** {len(categorical_features)}")

# Why this matters section
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

# Preprocessing configuration
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
            preprocessing_config.get('numeric_imputation'),
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
            preprocessing_config.get('numeric_scaling'),
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

# Optional Feature Engineering Steps
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
            # Estimate: numeric features (after encoding, could be more)
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

# Build pipeline
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
