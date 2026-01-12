"""
Page 01: Upload and Data Audit
Validates dataset and provides audit summary.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from utils.session_state import init_session_state, set_data, get_data, DataConfig
from data_processor import load_and_preview_csv, get_numeric_columns

logger = logging.getLogger(__name__)

# Initialize session state
init_session_state()

st.set_page_config(
    page_title="Upload & Audit",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Upload & Data Audit")

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload your dataset as a CSV file"
)

if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("Loading data..."):
            df = load_and_preview_csv(uploaded_file)
            set_data(df)
        
        st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)
            st.info(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Data Audit
        st.header("🔍 Data Audit")
        
        audit_results = {}
        
        # Missingness
        st.subheader("Missing Values")
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            audit_results['missing'] = missing_df.to_dict('records')
        else:
            st.success("✅ No missing values found")
            audit_results['missing'] = []
        
        # Data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Type': df.dtypes.values.astype(str),
            'Non-null Count': df.count().values
        })
        st.dataframe(dtype_df, use_container_width=True)
        audit_results['dtypes'] = dtype_df.to_dict('records')
        
        # Duplicates
        st.subheader("Duplicate Rows")
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            st.warning(f"⚠️ Found {n_duplicates:,} duplicate rows ({n_duplicates/len(df)*100:.2f}%)")
            audit_results['duplicates'] = n_duplicates
        else:
            st.success("✅ No duplicate rows found")
            audit_results['duplicates'] = 0
        
        # Constant columns
        st.subheader("Constant Columns")
        constant_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            st.warning(f"⚠️ Found {len(constant_cols)} constant columns: {', '.join(constant_cols)}")
            audit_results['constant_cols'] = constant_cols
        else:
            st.success("✅ No constant columns found")
            audit_results['constant_cols'] = []
        
        # High cardinality categoricals
        st.subheader("High Cardinality Categoricals")
        high_card_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.5:  # More than 50% unique values
                high_card_cols.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    '% Unique': (df[col].nunique() / len(df)) * 100
                })
        
        if high_card_cols:
            st.warning(f"⚠️ Found {len(high_card_cols)} high-cardinality columns")
            st.dataframe(pd.DataFrame(high_card_cols), use_container_width=True)
            audit_results['high_cardinality'] = high_card_cols
        else:
            st.success("✅ No high-cardinality categorical columns")
            audit_results['high_cardinality'] = []
        
        # Target leakage candidates (correlation with potential targets)
        st.subheader("Target Leakage Analysis")
        numeric_cols = get_numeric_columns(df)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            # Find columns highly correlated with others (potential leakage)
            leakage_candidates = []
            for col in numeric_cols:
                max_corr = corr_matrix[col].drop(col).max()
                if max_corr > 0.95:  # Very high correlation
                    correlated_with = corr_matrix[col].drop(col).idxmax()
                    leakage_candidates.append({
                        'Column': col,
                        'Correlated With': correlated_with,
                        'Correlation': max_corr
                    })
            
            if leakage_candidates:
                st.warning(f"⚠️ Found {len(leakage_candidates)} potential leakage candidates")
                st.dataframe(pd.DataFrame(leakage_candidates), use_container_width=True)
                audit_results['leakage_candidates'] = leakage_candidates
            else:
                st.success("✅ No obvious target leakage candidates")
                audit_results['leakage_candidates'] = []
        else:
            audit_results['leakage_candidates'] = []
        
        # ID-like fields (high cardinality numeric columns)
        st.subheader("ID-like Fields")
        id_like_cols = []
        for col in numeric_cols:
            if df[col].nunique() == len(df) and df[col].dtype in [np.int64, np.int32]:
                id_like_cols.append(col)
        
        if id_like_cols:
            st.warning(f"⚠️ Found {len(id_like_cols)} ID-like columns: {', '.join(id_like_cols)}")
            audit_results['id_like_cols'] = id_like_cols
        else:
            st.success("✅ No obvious ID-like columns")
            audit_results['id_like_cols'] = []
        
        # Store audit results
        st.session_state.data_audit = audit_results
        
        # Recommendations
        st.header("💡 Recommendations")
        recommendations = []
        
        if audit_results['missing']:
            recommendations.append("Consider imputing missing values in preprocessing")
        if audit_results['duplicates'] > 0:
            recommendations.append("Consider removing duplicate rows")
        if audit_results['constant_cols']:
            recommendations.append(f"Remove constant columns: {', '.join(audit_results['constant_cols'])}")
        if audit_results['id_like_cols']:
            recommendations.append(f"Exclude ID-like columns from features: {', '.join(audit_results['id_like_cols'])}")
        if audit_results['leakage_candidates']:
            recommendations.append("Review highly correlated columns for potential target leakage")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.info(f"{i}. {rec}")
        else:
            st.success("✅ No major issues detected. Data looks good!")
        
        # Target and feature selection
        st.header("🎯 Select Target & Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target selection
            target_col = st.selectbox(
                "Target Variable",
                options=[''] + numeric_cols,
                help="Select the column you want to predict"
            )
        
        with col2:
            # Feature selection
            feature_options = [col for col in numeric_cols if col != target_col]
            selected_features = st.multiselect(
                "Feature Variables",
                options=feature_options,
                default=feature_options[:min(10, len(feature_options))],
                help="Select columns to use as predictors"
            )
        
        if target_col and selected_features:
            # Detect task type
            unique_targets = df[target_col].nunique()
            if unique_targets <= 20 and df[target_col].dtype in [np.int64, np.int32, 'category']:
                task_type = 'classification'
            else:
                task_type = 'regression'
            
            # Store configuration
            data_config = DataConfig(
                target_col=target_col,
                feature_cols=selected_features,
                task_type=task_type
            )
            st.session_state.data_config = data_config
            
            st.success(f"✅ Configuration saved: {task_type.title()} task with {len(selected_features)} features")
            st.info(f"**Next:** Go to EDA page to explore your data")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        logger.exception(e)
else:
    st.info("👈 Please upload a CSV file to get started")
