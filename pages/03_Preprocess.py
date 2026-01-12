"""
Page 03: Preprocessing Builder
Build sklearn Pipeline with ColumnTransformer.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from utils.session_state import init_session_state, get_data, DataConfig, set_preprocessing_pipeline
from ml.pipeline import build_preprocessing_pipeline, get_pipeline_recipe
from data_processor import get_numeric_columns

init_session_state()

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")
st.title("⚙️ Preprocessing Builder")

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

# Preprocessing configuration
st.header("🔧 Preprocessing Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric Features")
    if numeric_features:
        numeric_imputation = st.selectbox(
            "Imputation Strategy",
            ['median', 'mean', 'constant'],
            help="How to handle missing numeric values"
        )
        numeric_scaling = st.selectbox(
            "Scaling Strategy",
            ['standard', 'robust', 'none'],
            help="Feature scaling method"
        )
        numeric_log_transform = st.checkbox(
            "Log Transform",
            value=False,
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
        categorical_imputation = st.selectbox(
            "Imputation Strategy",
            ['most_frequent', 'constant'],
            help="How to handle missing categorical values"
        )
        categorical_encoding = st.selectbox(
            "Encoding Strategy",
            ['onehot'],
            help="Categorical encoding method"
        )
    else:
        st.info("No categorical features")
        categorical_imputation = 'most_frequent'
        categorical_encoding = 'onehot'

# Build pipeline
if st.button("🔨 Build Preprocessing Pipeline", type="primary"):
    try:
        with st.spinner("Building pipeline..."):
            pipeline = build_preprocessing_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                numeric_imputation=numeric_imputation,
                numeric_scaling=numeric_scaling,
                numeric_log_transform=numeric_log_transform,
                categorical_imputation=categorical_imputation,
                categorical_encoding=categorical_encoding
            )
            
            # Fit on sample data to get feature names
            X_sample = df[all_features]
            pipeline.fit(X_sample)
            
            # Get transformed feature names
            X_transformed = pipeline.transform(X_sample)
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
                'n_output_features': n_output_features
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
    
    if st.button("🔄 Rebuild Pipeline"):
        st.session_state.preprocessing_pipeline = None
        st.session_state.preprocessing_config = None
        st.rerun()
