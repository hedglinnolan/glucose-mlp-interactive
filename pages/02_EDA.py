"""
Page 02: Exploratory Data Analysis
Shows summary stats, distributions, correlations, and target analysis.
Includes intelligent Model Selection Coach with data-aware recommendations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any, List

from utils.session_state import (
    init_session_state, get_data, DataConfig,
    TaskTypeDetection, CohortStructureDetection
)
from utils.storyline import render_progress_indicator, add_insight, get_insights_by_category
from data_processor import get_numeric_columns
from ml.eda_recommender import compute_dataset_signals, recommend_eda, DatasetSignals
from ml import eda_actions

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

# ============================================================================
# DATASET PROFILE - Compute comprehensive profile for intelligent coaching
# ============================================================================
@st.cache_data
def compute_profile_cached(_df: pd.DataFrame, target: str, features: List[str], task_type: str):
    """Compute dataset profile with caching."""
    from ml.dataset_profile import compute_dataset_profile
    return compute_dataset_profile(_df, target, features, task_type)

# Compute the dataset profile
profile = compute_profile_cached(df, target_col, feature_cols, task_type_final)
st.session_state['dataset_profile'] = profile  # Store for other pages

# ============================================================================
# DATASET OVERVIEW PANEL
# ============================================================================
st.markdown("---")

# Quick stats in a clean row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Samples", f"{profile.n_rows:,}")
with col2:
    st.metric("Features", f"{profile.n_features}")
with col3:
    st.metric("Numeric", f"{profile.n_numeric}")
with col4:
    st.metric("Categorical", f"{profile.n_categorical}")
with col5:
    sufficiency_emoji = {
        "abundant": "🟢",
        "adequate": "🟡", 
        "limited": "🟠",
        "scarce": "🔴",
        "critical": "⛔"
    }.get(profile.data_sufficiency.value, "⚪")
    st.metric("Data Sufficiency", f"{sufficiency_emoji} {profile.data_sufficiency.value.title()}")

# Data sufficiency narrative
with st.expander("📊 Data Sufficiency Analysis", expanded=True):
    st.markdown(f"**{profile.sufficiency_narrative}**")
    
    # Show feature-to-sample ratio context
    st.markdown(f"""
    **What this means for your models:**
    - **Feature-to-sample ratio:** {profile.p_n_ratio:.3f} (1 feature per {1/profile.p_n_ratio:.0f} samples)
    - **Numeric features:** {profile.n_numeric} ({profile.n_numeric/profile.n_features*100:.0f}% of total)
    - **Categorical features:** {profile.n_categorical} ({profile.n_categorical/profile.n_features*100:.0f}% of total)
    """)
    
    if profile.target_profile and profile.target_profile.task_type == 'classification':
        if profile.events_per_variable:
            st.markdown(f"- **Events per variable:** {profile.events_per_variable:.1f} "
                       f"(minority class has {profile.target_profile.minority_class_size:,} samples)")

# Warnings panel
if profile.warnings:
    st.markdown("### ⚠️ Data Warnings")
    
    # Group warnings by severity
    critical = [w for w in profile.warnings if w.level.value == 'critical']
    warnings = [w for w in profile.warnings if w.level.value == 'warning']
    cautions = [w for w in profile.warnings if w.level.value == 'caution']
    
    if critical:
        for w in critical:
            st.error(f"**{w.short_message}:** {w.detailed_message}")
            if w.suggested_actions:
                st.markdown("**Suggested actions:**")
                for action in w.suggested_actions:
                    st.markdown(f"  • {action}")
    
    if warnings:
        for w in warnings:
            st.warning(f"**{w.short_message}:** {w.detailed_message}")
            with st.expander("Suggested actions"):
                for action in w.suggested_actions:
                    st.markdown(f"• {action}")
    
    if cautions:
        with st.expander(f"ℹ️ {len(cautions)} additional caution(s)"):
            for w in cautions:
                st.info(f"**{w.short_message}:** {w.detailed_message}")

# ============================================================================
# MODEL SELECTION COACH - Intelligent, Educational Assistant
# ============================================================================
st.markdown("---")
st.header("🎓 Model Selection Coach")

st.markdown("""
The Model Selection Coach analyzes your dataset and recommends models based on:
- **Sample size and feature count** — Some models need more data than others
- **Feature types** — Numeric vs categorical, high cardinality
- **Data quality** — Missing values, outliers, imbalance
- **Task complexity** — Linear vs nonlinear relationships
""")

# Compute comprehensive coach recommendations
@st.cache_data
def compute_coach_cached(_profile_dict: dict):
    """Compute coach recommendations with caching."""
    from ml.model_coach import compute_model_recommendations
    from ml.dataset_profile import DatasetProfile
    # Reconstruct profile from dict for caching compatibility
    return compute_model_recommendations(_profile_dict['_obj'])

# Create a cacheable representation
profile_cache_key = {
    'n_rows': profile.n_rows,
    'n_features': profile.n_features,
    'task_type': profile.target_profile.task_type if profile.target_profile else None,
    '_obj': profile  # Pass actual object
}

try:
    from ml.model_coach import compute_model_recommendations, RecommendationBucket, TrainingTimeTier
    coach_output = compute_model_recommendations(profile)
    st.session_state['coach_output'] = coach_output
except Exception as e:
    st.error(f"Error computing recommendations: {e}")
    coach_output = None

# Custom CSS for coach UI
st.markdown("""
<style>
.coach-section {
    border: 2px solid #e8e8e8;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    background: linear-gradient(180deg, #fafafa 0%, #ffffff 100%);
}
.coach-bucket-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.bucket-recommended { border-left: 4px solid #28a745; }
.bucket-worth-trying { border-left: 4px solid #ffc107; }
.bucket-not-recommended { border-left: 4px solid #dc3545; }
.model-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.model-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.time-badge {
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 600;
}
.time-fast { background: #d4edda; color: #155724; }
.time-medium { background: #fff3cd; color: #856404; }
.time-slow { background: #f8d7da; color: #721c24; }
.plain-explanation {
    background: #f8f9fa;
    border-left: 3px solid #007bff;
    padding: 1rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

if coach_output:
    # Tabs for different views
    tab_summary, tab_recommended, tab_worth_trying, tab_not_recommended, tab_preprocessing, tab_advanced_eda = st.tabs([
        "📋 Summary",
        f"✅ Recommended ({len(coach_output.recommended_models)})",
        f"🔄 Worth Trying ({len(coach_output.worth_trying_models)})",
        f"⛔ Not Recommended ({len(coach_output.not_recommended_models)})",
        "🔧 Preprocessing",
        "🔬 Advanced EDA"
    ])
    
    # Summary Tab
    with tab_summary:
        st.markdown(f"### {coach_output.dataset_summary}")
        
        # Plain language narrative
        st.markdown(f'<div class="plain-explanation">{coach_output.data_sufficiency_narrative}</div>', 
                   unsafe_allow_html=True)
        
        # Quick recommendation summary
        st.markdown("### Quick Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**✅ Recommended Models**")
            for rec in coach_output.recommended_models[:5]:
                time_class = f"time-{rec.training_time.value}"
                st.markdown(f"- **{rec.model_name}** <span class='time-badge {time_class}'>{rec.training_time.value}</span>", 
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown("**🔄 Worth Trying**")
            for rec in coach_output.worth_trying_models[:5]:
                st.markdown(f"- {rec.model_name}")
        
        with col3:
            st.markdown("**⛔ Not Recommended**")
            for rec in coach_output.not_recommended_models[:3]:
                st.markdown(f"- {rec.model_name}")
        
        # Key warnings
        if coach_output.warnings_summary:
            st.markdown("### ⚠️ Key Warnings")
            for warning in coach_output.warnings_summary[:3]:
                st.warning(warning)
    
    # Recommended Models Tab
    with tab_recommended:
        st.markdown("### ✅ Recommended Models")
        st.markdown("These models are well-suited to your dataset based on sample size, feature types, and data quality.")
        
        if not coach_output.recommended_models:
            st.info("No models strongly recommended. Check the 'Worth Trying' tab.")
        
        for rec in coach_output.recommended_models:
            with st.container():
                st.markdown(f'<div class="model-card bucket-recommended">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"### {rec.model_name}")
                    st.caption(f"Group: {rec.display_name} | Key: `{rec.model_key}`")
                with col2:
                    time_emoji = {"fast": "⚡", "medium": "⏱️", "slow": "🐢"}.get(rec.training_time.value, "")
                    st.metric("Training Time", f"{time_emoji} {rec.training_time.value.title()}")
                with col3:
                    interp_emoji = {"high": "📖", "medium": "📊", "low": "🔮"}.get(rec.interpretability, "")
                    st.metric("Interpretability", f"{interp_emoji} {rec.interpretability.title()}")
                
                # Plain language summary
                st.markdown(f'<div class="plain-explanation">{rec.plain_language_summary}</div>', 
                           unsafe_allow_html=True)
                
                # Dataset fit
                st.markdown(f"**Dataset Fit:** {rec.dataset_fit_summary}")
                st.markdown(f"**Rationale:** {rec.rationale}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if rec.strengths:
                        st.markdown("**✅ Strengths:**")
                        for s in rec.strengths:
                            st.markdown(f"• {s}")
                with col2:
                    if rec.weaknesses or rec.risks:
                        st.markdown("**⚠️ Considerations:**")
                        for w in rec.weaknesses:
                            st.markdown(f"• {w}")
                        for r in rec.risks:
                            st.markdown(f"• ⚠️ {r}")
                
                # Expandable details
                with st.expander("When to use / When to avoid"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**When to use:** {rec.when_to_use}")
                    with col2:
                        st.markdown(f"**When to avoid:** {rec.when_to_avoid}")
                
                with st.expander("Prerequisites"):
                    st.markdown(f"• **Requires scaling:** {'Yes' if rec.requires_scaling else 'No'}")
                    st.markdown(f"• **Requires categorical encoding:** {'Yes' if rec.requires_encoding else 'No'}")
                    st.markdown(f"• **Handles missing values:** {'Yes' if rec.handles_missing else 'No (needs imputation)'}")
                
                # Select button
                if st.button(f"Select {rec.model_key} for training", key=f"select_{rec.model_key}"):
                    st.session_state[f'train_model_{rec.model_key}'] = True
                    st.success(f"✅ {rec.model_name} selected for training!")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")  # spacing
    
    # Worth Trying Tab
    with tab_worth_trying:
        st.markdown("### 🔄 Worth Trying")
        st.markdown("These models may work for your dataset but have some caveats. Consider them if recommended models underperform.")
        
        if not coach_output.worth_trying_models:
            st.info("No models in this category.")
        
        for rec in coach_output.worth_trying_models:
            with st.expander(f"**{rec.model_name}** ({rec.display_name})"):
                st.markdown(f'<div class="plain-explanation">{rec.plain_language_summary}</div>', 
                           unsafe_allow_html=True)
                
                st.markdown(f"**Rationale:** {rec.rationale}")
                
                if rec.risks:
                    st.warning("**Risks:** " + "; ".join(rec.risks))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**When to use:** {rec.when_to_use}")
                with col2:
                    st.markdown(f"**When to avoid:** {rec.when_to_avoid}")
                
                if st.button(f"Select {rec.model_key}", key=f"select_wt_{rec.model_key}"):
                    st.session_state[f'train_model_{rec.model_key}'] = True
                    st.success(f"✅ {rec.model_name} selected!")
    
    # Not Recommended Tab
    with tab_not_recommended:
        st.markdown("### ⛔ Not Recommended")
        st.markdown("These models are **not well-suited** for your current dataset. Use with caution.")
        
        if not coach_output.not_recommended_models:
            st.success("All models are at least worth trying for your dataset!")
        
        for rec in coach_output.not_recommended_models:
            with st.expander(f"**{rec.model_name}** — Why not recommended"):
                st.markdown(f'<div class="plain-explanation">{rec.plain_language_summary}</div>', 
                           unsafe_allow_html=True)
                
                st.error(f"**Why not recommended:** {rec.rationale}")
                
                if rec.risks:
                    st.markdown("**Key risks:**")
                    for r in rec.risks:
                        st.markdown(f"• ⚠️ {r}")
                
                st.markdown(f"**When this model IS appropriate:** {rec.when_to_use}")
    
    # Preprocessing Tab
    with tab_preprocessing:
        st.markdown("### 🔧 Recommended Preprocessing")
        st.markdown("Based on your dataset characteristics and the recommended models, here are preprocessing steps to consider.")
        
        if not coach_output.preprocessing_recommendations:
            st.success("No critical preprocessing steps identified.")
        
        for prep in coach_output.preprocessing_recommendations:
            priority_color = {"required": "🔴", "recommended": "🟡", "optional": "🟢"}.get(prep.priority, "⚪")
            
            with st.expander(f"{priority_color} **{prep.step_name}** ({prep.priority.upper()})"):
                st.markdown(f"**Why:** {prep.rationale}")
                
                st.markdown(f'<div class="plain-explanation">{prep.plain_language_explanation}</div>', 
                           unsafe_allow_html=True)
                
                st.markdown(f"**How to implement:** {prep.how_to_implement}")
                
                if prep.affected_model_families:
                    st.caption(f"Affects: {', '.join(prep.affected_model_families)}")
    
    # Advanced EDA Tab
    with tab_advanced_eda:
        st.markdown("### 🔬 Advanced EDA by Model Family")
        st.markdown("These analyses can help you better understand whether specific model families will work for your data.")
        
        st.markdown("#### 📊 Baseline EDA (Always Recommended)")
        for eda_item in coach_output.baseline_eda:
            st.markdown(f"• {eda_item}")
        
        st.markdown("---")
        st.markdown("#### 🔍 Model-Family Specific EDA")
        
        for family, eda_items in coach_output.advanced_eda_by_family.items():
            with st.expander(f"**{family}** — Advanced Analyses"):
                st.markdown(f"*These analyses are particularly relevant if you plan to use {family}:*")
                for item in eda_items:
                    st.markdown(f"• {item}")

# ============================================================================
# LEGACY COACH (for backward compatibility with signals)
# ============================================================================
# Compute signals for legacy features
@st.cache_data
def compute_signals_cached(_df: pd.DataFrame, target: str, task_type: Optional[str], 
                           cohort_type: Optional[str], entity_id: Optional[str]):
    """Cached signal computation."""
    return compute_dataset_signals(
        _df, target, task_type, cohort_type, entity_id
    )

signals = compute_signals_cached(
    df, target_col, task_type_final, cohort_type_final, entity_id_final
)

# ============================================================================
# KEY INSIGHTS PANEL
# ============================================================================
st.markdown("---")
insights = get_insights_by_category()
if insights:
    with st.expander("💡 Key Insights So Far", expanded=False):
        st.markdown("**Insights collected from EDA analyses:**")
        for insight in insights:
            st.markdown(f"**{insight.get('category', 'General').title()}:** {insight['finding']}")
            st.caption(f"→ Implication: {insight['implication']}")
else:
    st.info("💡 Run EDA analyses below to collect insights that will guide model selection and preprocessing.")

# ============================================================================
# EDA RECOMMENDATIONS ENGINE
# ============================================================================
# Initialize EDA results storage
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = {}

# Generate recommendations
recommendations = recommend_eda(signals)

# Display recommendations section
st.header("🔍 Recommended EDA Analyses")

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

# ============================================================================
# MANUAL MODE
# ============================================================================
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

# ============================================================================
# DATASET SIGNALS EXPLAINER
# ============================================================================
with st.expander("📊 Dataset Signals Detail"):
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

# ============================================================================
# STANDARD EDA VIEWS
# ============================================================================
st.header("📈 Standard EDA Views")

# Summary statistics
st.subheader("📈 Summary Statistics")
st.dataframe(df[feature_cols + [target_col]].describe(), use_container_width=True)

# Distribution plots
st.subheader("📊 Distributions")

# Target distribution
st.markdown(f"**Target Distribution: {target_col}**")
col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    fig_box = px.box(df, y=target_col, title=f"Box Plot of {target_col}")
    st.plotly_chart(fig_box, use_container_width=True)

# Classification: class balance
if task_type_final == 'classification':
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
if task_type_final == 'regression':
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
