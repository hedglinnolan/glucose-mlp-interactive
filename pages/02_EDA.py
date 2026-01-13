"""
Page 02: Exploratory Data Analysis
Shows summary stats, distributions, correlations, and target analysis.
Includes EDA recommendation cards.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any

from utils.session_state import (
    init_session_state, get_data, DataConfig,
    TaskTypeDetection, CohortStructureDetection
)
from utils.storyline import render_progress_indicator, add_insight, get_insights_by_category
from data_processor import get_numeric_columns
from ml.eda_recommender import compute_dataset_signals, recommend_eda, DatasetSignals
from ml import eda_actions
from ml.model_coach import coach_recommendations

init_session_state()

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

# Progress indicator
render_progress_indicator("02_EDA")

df = get_data()
if df is None:
    st.warning("⚠️ Please upload data in the Upload & Audit page first")
    st.stop()

data_config: Optional[DataConfig] = st.session_state.get('data_config')
if data_config is None or not data_config.target_col:
    st.warning("⚠️ Please select target and features in the Upload & Audit page first")
    st.stop()

target_col = data_config.target_col
feature_cols = data_config.feature_cols

# Get final detection values
task_type_detection: TaskTypeDetection = st.session_state.get('task_type_detection', TaskTypeDetection())
cohort_structure_detection: CohortStructureDetection = st.session_state.get('cohort_structure_detection', CohortStructureDetection())

task_type_final = task_type_detection.final if task_type_detection.final else data_config.task_type
cohort_type_final = cohort_structure_detection.final if cohort_structure_detection.final else 'cross_sectional'
entity_id_final = cohort_structure_detection.entity_id_final

st.info(f"**Target:** {target_col} | **Features:** {len(feature_cols)} | **Task:** {task_type_final} | **Cohort:** {cohort_type_final}")

# Key insights panel
insights = get_insights_by_category()
if insights:
    with st.expander("💡 Key Insights So Far", expanded=True):
        st.markdown("**Insights collected from EDA analyses:**")
        for insight in insights:
            st.markdown(f"**{insight.get('category', 'General').title()}:** {insight['finding']}")
            st.caption(f"→ Implication: {insight['implication']}")
else:
    st.info("💡 Run EDA analyses below to collect insights that will guide model selection and preprocessing.")

# Model Selection Coach (at top, before EDA recommendations)
st.header("🎓 Model Selection Coach")
st.markdown("**Based on your dataset characteristics, here are recommended model families to try:**")

# Compute signals for coach
signals = compute_dataset_signals(
    df, target_col, task_type_final, cohort_type_final, entity_id_final
)
coach_recs = coach_recommendations(
    signals, 
    st.session_state.get('eda_results'),
    get_insights_by_category()
)

if coach_recs:
    for idx, rec in enumerate(coach_recs[:3]):  # Top 3 recommendations
        with st.container():
            # Use priority label for clarity
            priority_label = "High" if rec.priority <= 2 else "Medium"
            st.markdown(f"### {rec.group} Models ({priority_label} Priority)")
            
            # Show readiness checks if any
            if hasattr(rec, 'readiness_checks') and rec.readiness_checks:
                st.warning("⚠️ **Recommended prerequisites:**")
                for check in rec.readiness_checks:
                    st.write(f"• {check}")
            
            st.markdown("**Why:**")
            for reason in rec.why:
                st.write(f"• {reason}")
            if rec.when_not_to_use:
                st.markdown("**When not to use:**")
                for caveat in rec.when_not_to_use:
                    st.write(f"• {caveat}")
            if rec.suggested_preprocessing:
                st.markdown("**Suggested preprocessing:**")
                for prep in rec.suggested_preprocessing:
                    st.write(f"• {prep}")
            
            # Show recommended models
            st.markdown(f"**Recommended models:** {', '.join(rec.recommended_models)}")
            st.markdown("---")
else:
    st.info("Complete EDA analyses to get more specific model recommendations.")

# Initialize EDA results storage
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = {}

# Compute signals (cached by Streamlit automatically based on inputs)
@st.cache_data
def compute_signals_cached(_df: pd.DataFrame, target: str, task_type: Optional[str], 
                           cohort_type: Optional[str], entity_id: Optional[str]):
    """Cached signal computation."""
    return compute_dataset_signals(
        _df, target, task_type, cohort_type, entity_id
    )

# Compute signals
signals = compute_signals_cached(
    df, target_col, task_type_final, cohort_type_final, entity_id_final
)

# Generate recommendations
recommendations = recommend_eda(signals)

# Display recommendations section
st.header("🔍 Recommended Next Analyses")

# Show top 5 by default, expander for all
top_n = 5
show_all = st.checkbox("Show all recommendations", value=False, key="show_all_recommendations")

recs_to_show = recommendations if show_all else recommendations[:top_n]

for rec in recs_to_show:
    with st.container():
        # Card-like container
        st.markdown("---")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {rec.title}")
            cost_badge = {"low": "🟢 Low Cost", "medium": "🟡 Medium Cost", "high": "🔴 High Cost"}.get(rec.cost, "⚪ Unknown")
            st.markdown(f"**Priority:** {rec.priority} | **Cost:** {cost_badge}")
        
        with col2:
            # Run button
            run_key = f"run_{rec.id}"
            if st.button("▶️ Run", key=run_key, type="primary"):
                # Execute action
                try:
                    action_func = getattr(eda_actions, rec.run_action, None)
                    if action_func:
                        with st.spinner(f"Running {rec.title}..."):
                            result = action_func(df, target_col, feature_cols, signals, st.session_state)
                            # Store results
                            st.session_state.eda_results[rec.id] = result
                            st.rerun()
                    else:
                        st.error(f"Action '{rec.run_action}' not found")
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
                    st.session_state.eda_results[rec.id] = {
                        'findings': [],
                        'warnings': [f"Error: {str(e)}"],
                        'figures': []
                    }
        
        # Why recommended
        with st.expander("Why recommended"):
            for reason in rec.why:
                st.write(f"• {reason}")
        
        # What you'll learn
        with st.expander("What you'll learn"):
            for item in rec.what_you_learn:
                st.write(f"• {item}")
        
        # Model implications
        with st.expander("Model implications"):
            for item in rec.model_implications:
                st.write(f"• {item}")
        
        # Educational explanation
        if rec.description:
            with st.expander("📚 Explain this analysis"):
                st.markdown(rec.description)
        
        # Display results if available
        if rec.id in st.session_state.eda_results:
            result = st.session_state.eda_results[rec.id]
            
            st.markdown("**Results:**")
            
            # Findings
            if result.get('findings'):
                st.success("**Findings:**")
                for finding in result['findings']:
                    st.write(f"✓ {finding}")
            
            # Warnings
            if result.get('warnings'):
                for warning in result['warnings']:
                    st.warning(warning)
            
            # Figures
            if result.get('figures'):
                for fig_type, fig_data in result['figures']:
                    if fig_type == 'plotly':
                        st.plotly_chart(fig_data, use_container_width=True)
                    elif fig_type == 'table':
                        st.dataframe(fig_data, use_container_width=True)

st.markdown("---")

# Manual mode (separate section)
st.header("🔧 Manual Mode - Run Any Analysis")

action_names = [
    'plausibility_check',
    'missingness_scan',
    'cohort_split_guidance',
    'target_profile',
    'dose_response_trends',
    'collinearity_map',
    'quick_probe_baselines'
]

selected_action = st.selectbox("Select analysis to run", action_names, key="eda_manual_action_select")

if st.button("Run Selected Analysis", key="eda_manual_run_button"):
    try:
        action_func = getattr(eda_actions, selected_action, None)
        if action_func:
            with st.spinner(f"Running {selected_action}..."):
                result = action_func(df, target_col, feature_cols, signals, st.session_state)
                st.session_state.eda_results[f"manual_{selected_action}"] = result
                st.rerun()
        else:
            st.error(f"Action '{selected_action}' not found")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Show manual results
manual_key = f"manual_{selected_action}"
if manual_key in st.session_state.eda_results:
    result = st.session_state.eda_results[manual_key]
    st.markdown("**Results:**")
    
    if result.get('findings'):
        for finding in result['findings']:
            st.write(f"✓ {finding}")
    
    if result.get('warnings'):
        for warning in result['warnings']:
            st.warning(warning)
    
    if result.get('figures'):
        for fig_type, fig_data in result['figures']:
            if fig_type == 'plotly':
                st.plotly_chart(fig_data, use_container_width=True)
            elif fig_type == 'table':
                st.dataframe(fig_data, use_container_width=True)

# Explain recommendations
with st.expander("📊 Explain Recommendations - Dataset Signals"):
    st.markdown("**Dataset Summary:**")
    st.write(f"• Rows: {signals.n_rows:,}")
    st.write(f"• Columns: {signals.n_cols}")
    st.write(f"• Numeric columns: {len(signals.numeric_cols)}")
    st.write(f"• Categorical columns: {len(signals.categorical_cols)}")
    st.write(f"• High missing columns (>5%): {len(signals.high_missing_cols)}")
    st.write(f"• Duplicate row rate: {signals.duplicate_row_rate:.1%}")
    
    if signals.target_stats:
        st.markdown("**Target Statistics:**")
        for key, value in signals.target_stats.items():
            if isinstance(value, (int, float)):
                st.write(f"• {key}: {value:.3f}")
            else:
                st.write(f"• {key}: {value}")
    
    if signals.collinearity_summary:
        st.markdown("**Collinearity:**")
        st.write(f"• Max correlation: {signals.collinearity_summary.get('max_corr', 0):.3f}")
    
    if signals.unit_sanity_flags:
        st.markdown("**Unit Sanity Flags:**")
        for flag in signals.unit_sanity_flags:
            st.write(f"• {flag}")

st.markdown("---")
st.header("📈 Standard EDA Views")

# Summary statistics
st.header("📈 Summary Statistics")
st.dataframe(df[feature_cols + [target_col]].describe(), use_container_width=True)

# Distribution plots
st.header("📊 Distributions")

# Target distribution
st.subheader(f"Target Distribution: {target_col}")
col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    fig_box = px.box(df, y=target_col, title=f"Box Plot of {target_col}")
    st.plotly_chart(fig_box, use_container_width=True)

# Classification: class balance
if data_config.task_type == 'classification':
    st.subheader("Class Balance")
    class_counts = df[target_col].value_counts().sort_index()
    fig_bar = px.bar(x=class_counts.index.astype(str), y=class_counts.values,
                     title="Class Distribution", labels={'x': 'Class', 'y': 'Count'})
    st.plotly_chart(fig_bar, use_container_width=True)
    st.info(f"Classes: {len(class_counts)} | Imbalance ratio: {class_counts.max()/class_counts.min():.2f}")

# Feature distributions (top 6)
st.subheader("Feature Distributions")
n_features_show = min(6, len(feature_cols))
cols_per_row = 3

for i in range(0, n_features_show, cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if i + j < n_features_show:
            feat = feature_cols[i + j]
            with col:
                fig = px.histogram(df, x=feat, nbins=20, title=feat)
                st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
st.header("🔗 Correlation Analysis")
numeric_cols = get_numeric_columns(df)
corr_cols = [c for c in feature_cols + [target_col] if c in numeric_cols]

if len(corr_cols) > 1:
    corr_matrix = df[corr_cols].corr()
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Target vs feature correlations
    st.subheader("Target-Feature Correlations")
    target_corrs = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    fig_bar = px.bar(
        x=target_corrs.index,
        y=target_corrs.values,
        title=f"Correlation with {target_col}",
        labels={'x': 'Feature', 'y': 'Correlation'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Target vs feature plots
st.header("🎯 Target vs Features")

# Regression: scatter plots
if data_config.task_type == 'regression':
    n_plots = min(6, len(feature_cols))
    cols_per_row = 3
    
    for i in range(0, n_plots, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < n_plots:
                feat = feature_cols[i + j]
                with col:
                    fig = px.scatter(df, x=feat, y=target_col, title=f"{target_col} vs {feat}")
                    st.plotly_chart(fig, use_container_width=True)

# Classification: box plots
else:
    n_plots = min(6, len(feature_cols))
    cols_per_row = 3
    
    for i in range(0, n_plots, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < n_plots:
                feat = feature_cols[i + j]
                with col:
                    fig = px.box(df, x=target_col, y=feat, title=f"{feat} by {target_col}")
                    st.plotly_chart(fig, use_container_width=True)

st.success("✅ EDA complete. Proceed to Preprocessing page.")
