"""
Page 01: Upload and Data Audit
Validates dataset and provides audit summary.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from utils.session_state import (
    init_session_state, set_data, get_data, DataConfig,
    TaskTypeDetection, CohortStructureDetection
)
from utils.datasets import get_builtin_datasets
from utils.reconcile import reconcile_target_features
from data_processor import load_and_preview_csv, get_numeric_columns
from ml.triage import detect_task_type, detect_cohort_structure

logger = logging.getLogger(__name__)

# Initialize session state
init_session_state()

st.set_page_config(
    page_title="Upload & Audit",
    page_icon="📁",
    layout="wide"
)

st.title("📁 Upload & Data Audit")

# Built-in datasets
st.header("📚 Built-in Datasets (for Testing)")
dataset_options = ['None (Upload CSV)'] + list(get_builtin_datasets().keys())
selected_dataset = st.selectbox(
    "Or try a built-in dataset:",
    dataset_options,
    key="upload_dataset_select",
    help="Select a built-in educational dataset to explore the app"
)

df_builtin = None
if selected_dataset != 'None (Upload CSV)':
    # Check if data already exists
    existing_data = get_data()
    replace_existing = False
    if existing_data is not None:
        replace_existing = st.checkbox(
            "Replace existing dataset",
            value=False,
            key="replace_dataset_checkbox",
            help="Check to overwrite current dataset"
        )
    
    if st.button("Generate Dataset", key="generate_dataset_button"):
        if existing_data is not None and not replace_existing:
            st.warning("⚠️ Dataset already loaded. Check 'Replace existing dataset' to overwrite.")
        else:
            generator = get_builtin_datasets()[selected_dataset]
            df_builtin = generator(random_state=st.session_state.get('random_seed', 42))
            set_data(df_builtin)
            st.session_state.data_source = f"Built-in: {selected_dataset}"
            st.success(f"✅ Generated {len(df_builtin)} rows")
            st.rerun()

# File upload
st.header("📤 Upload CSV File")
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=['csv'],
    key="csv_uploader",
    help="Upload your dataset as a CSV file"
)

# Get current data from session state
df = get_data()

# Handle new uploads
if uploaded_file is not None:
    existing_data = get_data()
    replace_existing = False
    if existing_data is not None:
        replace_existing = st.checkbox(
            "Replace existing dataset",
            value=False,
            key="replace_upload_checkbox",
            help="Check to overwrite current dataset"
        )
    
    if existing_data is None or replace_existing:
        try:
            with st.spinner("Loading data..."):
                df = load_and_preview_csv(uploaded_file)
                set_data(df)
                st.session_state.data_source = "Uploaded CSV"
                st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            logger.exception(e)
            df = None

# Use existing data or builtin
if df is None and df_builtin is not None:
    df = df_builtin

if df is not None:
    try:
        
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
        
        # Run cohort structure detection (once when data is loaded)
        if 'cohort_structure_detection' not in st.session_state or st.session_state.cohort_structure_detection.detected is None:
            with st.spinner("Detecting cohort structure..."):
                cohort_result = detect_cohort_structure(df)
                cohort_detection = CohortStructureDetection(
                    detected=cohort_result['detected'],
                    confidence=cohort_result['confidence'],
                    reasons=cohort_result['reasons'],
                    entity_id_candidates=cohort_result['entity_id_candidates'],
                    entity_id_detected=cohort_result['entity_id_detected'],
                    time_column_candidates=cohort_result['time_column_candidates']
                )
                st.session_state.cohort_structure_detection = cohort_detection
        
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
        
        # Get existing config or initialize
        existing_config = st.session_state.get('data_config')
        existing_target = existing_config.target_col if existing_config else None
        existing_features = existing_config.feature_cols if existing_config else []
        
        # Reconcile with current data
        if existing_target:
            existing_target, existing_features = reconcile_target_features(
                df, existing_target, existing_features, numeric_cols
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target selection - use existing or default
            target_default_idx = 0
            if existing_target and existing_target in numeric_cols:
                target_default_idx = numeric_cols.index(existing_target) + 1
            
            target_col = st.selectbox(
                "Target Variable",
                options=[''] + numeric_cols,
                index=target_default_idx,
                key="target_selectbox",
                help="Select the column you want to predict"
            )
        
        with col2:
            # Feature selection - reconcile with target
            target_col_reconciled, _ = reconcile_target_features(
                df, target_col, existing_features, numeric_cols
            )
            feature_options = [col for col in numeric_cols if col != target_col_reconciled]
            
            # Default features: use existing if valid, otherwise first 10
            feature_defaults = []
            if existing_features:
                feature_defaults = [f for f in existing_features if f in feature_options]
            if not feature_defaults:
                feature_defaults = feature_options[:min(10, len(feature_options))]
            
            selected_features = st.multiselect(
                "Feature Variables",
                options=feature_options,
                default=feature_defaults,
                key="features_multiselect",
                help="Select columns to use as predictors"
            )
        
        # Datetime column selection (for time-series)
        datetime_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
        datetime_col = None
        if datetime_cols:
            existing_datetime = existing_config.datetime_col if existing_config else None
            datetime_default_idx = 0
            if existing_datetime and existing_datetime in datetime_cols:
                datetime_default_idx = datetime_cols.index(existing_datetime) + 1
            
            datetime_col = st.selectbox(
                "Datetime Column (Optional, for time-series splits)",
                options=['None'] + datetime_cols,
                index=datetime_default_idx,
                key="datetime_selectbox",
                help="Select if you want time-based splitting"
            )
            if datetime_col == 'None':
                datetime_col = None
        
        if target_col and selected_features:
            # Reconcile target and features
            target_col, selected_features = reconcile_target_features(
                df, target_col, selected_features, numeric_cols
            )
            
            # Show warning if features were adjusted
            if existing_features and set(selected_features) != set(existing_features):
                removed = set(existing_features) - set(selected_features)
                if removed:
                    st.info(f"ℹ️ Removed invalid features: {', '.join(removed)}")
            
            if not target_col or not selected_features:
                st.warning("⚠️ Please select both target and at least one feature")
            else:
                # Run task type detection when target is selected/changed
                task_detection = st.session_state.get('task_type_detection', TaskTypeDetection())
                should_redetect = (
                    task_detection.detected is None or
                    existing_config is None or
                    existing_config.target_col != target_col
                )
                
                if should_redetect:
                    with st.spinner("Detecting task type..."):
                        task_result = detect_task_type(df, target_col)
                        task_detection = TaskTypeDetection(
                            detected=task_result['detected'],
                            confidence=task_result['confidence'],
                            reasons=task_result['reasons']
                        )
                        st.session_state.task_type_detection = task_detection
                
                # Auto-detection panel
                st.subheader("🔍 Auto-Detection Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Task Type Detection**")
                    task_det = st.session_state.task_type_detection
                    if task_det.detected:
                        confidence_emoji = {"high": "🟢", "med": "🟡", "low": "🟠"}.get(task_det.confidence, "⚪")
                        st.write(f"{confidence_emoji} Detected: **{task_det.detected.title()}** ({task_det.confidence} confidence)")
                        with st.expander("Detection reasons"):
                            for reason in task_det.reasons:
                                st.write(f"• {reason}")
                    else:
                        st.write("⚪ Not detected yet")
                
                with col2:
                    st.markdown("**Cohort Structure Detection**")
                    cohort_det = st.session_state.cohort_structure_detection
                    if cohort_det.detected:
                        confidence_emoji = {"high": "🟢", "med": "🟡", "low": "🟠"}.get(cohort_det.confidence, "⚪")
                        st.write(f"{confidence_emoji} Detected: **{cohort_det.detected.replace('_', ' ').title()}** ({cohort_det.confidence} confidence)")
                        with st.expander("Detection reasons"):
                            for reason in cohort_det.reasons:
                                st.write(f"• {reason}")
                        if cohort_det.entity_id_detected:
                            st.write(f"Entity ID: `{cohort_det.entity_id_detected}`")
                    else:
                        st.write("⚪ Not detected yet")
                
                # Override controls
                st.subheader("⚙️ Manual Overrides")
                
                # Task type override
                task_override_enabled = st.checkbox(
                    "Override task type",
                    value=task_detection.override_enabled,
                    key="task_override_checkbox",
                    help="Manually set task type instead of using auto-detection"
                )
                
                task_override_value = None
                if task_override_enabled:
                    task_default_idx = 0
                    if task_detection.override_value == 'regression':
                        task_default_idx = 0
                    elif task_detection.override_value == 'classification':
                        task_default_idx = 1
                    elif task_detection.detected == 'regression':
                        task_default_idx = 0
                    else:
                        task_default_idx = 1
                    
                    task_override_value = st.radio(
                        "Task type",
                        ['regression', 'classification'],
                        index=task_default_idx,
                        key="task_override_radio",
                        horizontal=True
                    )
                
                # Update task detection with override
                task_detection.override_enabled = task_override_enabled
                task_detection.override_value = task_override_value
                st.session_state.task_type_detection = task_detection
                
                # Cohort type override
                cohort_override_enabled = st.checkbox(
                    "Override cohort structure",
                    value=cohort_det.override_enabled,
                    key="cohort_override_checkbox",
                    help="Manually set cohort structure instead of using auto-detection"
                )
                
                cohort_override_value = None
                if cohort_override_enabled:
                    cohort_default_idx = 0
                    if cohort_det.override_value == 'longitudinal':
                        cohort_default_idx = 1
                    elif cohort_det.detected == 'longitudinal':
                        cohort_default_idx = 1
                    else:
                        cohort_default_idx = 0
                    
                    cohort_override_value = st.radio(
                        "Cohort structure",
                        ['cross_sectional', 'longitudinal'],
                        index=cohort_default_idx,
                        key="cohort_override_radio",
                        horizontal=True,
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                
                # Entity ID override
                entity_override_enabled = st.checkbox(
                    "Override entity ID column",
                    value=cohort_det.entity_id_override_enabled,
                    key="entity_id_override_checkbox",
                    help="Manually select entity ID column for longitudinal data"
                )
                
                entity_override_value = None
                if entity_override_enabled:
                    entity_options = ['None'] + (cohort_det.entity_id_candidates or [])
                    entity_default_idx = 0
                    if cohort_det.entity_id_override_value:
                        if cohort_det.entity_id_override_value in entity_options:
                            entity_default_idx = entity_options.index(cohort_det.entity_id_override_value)
                    elif cohort_det.entity_id_detected and cohort_det.entity_id_detected in entity_options:
                        entity_default_idx = entity_options.index(cohort_det.entity_id_detected)
                    
                    entity_override_value = st.selectbox(
                        "Entity ID column",
                        options=entity_options,
                        index=entity_default_idx,
                        key="entity_id_override_selectbox"
                    )
                    if entity_override_value == 'None':
                        entity_override_value = None
                
                # Update cohort detection with overrides
                cohort_det.override_enabled = cohort_override_enabled
                cohort_det.override_value = cohort_override_value
                cohort_det.entity_id_override_enabled = entity_override_enabled
                cohort_det.entity_id_override_value = entity_override_value
                st.session_state.cohort_structure_detection = cohort_det
                
                # Get final values
                task_type_final = task_detection.final
                cohort_type_final = cohort_det.final
                entity_id_final = cohort_det.entity_id_final
                
                # Warn if task type changed
                if existing_config and existing_config.task_type and existing_config.task_type != task_type_final:
                    if task_type_final == 'classification' and existing_config.task_type == 'regression':
                        st.info("ℹ️ Switched to classification. Ensure target has discrete values.")
            
            # Store configuration (use final values)
            data_config = DataConfig(
                target_col=target_col,
                feature_cols=selected_features,
                datetime_col=datetime_col,
                task_type=task_type_final if target_col and selected_features else None
            )
            st.session_state.data_config = data_config
            
            # Time-series warning (will check in Train page)
            if datetime_col:
                st.info("ℹ️ Datetime column selected. You can enable time-based splitting in the Train & Compare page.")
            
            st.success(f"✅ Configuration saved: {task_type.title()} task with {len(selected_features)} features")
            st.info(f"**Next:** Go to EDA page to explore your data")
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        logger.exception(e)
else:
    st.info("👈 Please upload a CSV file to get started")
