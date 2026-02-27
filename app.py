"""
Tabular ML Lab — Publication-grade machine learning for tabular research data.

A guided, interactive platform for researchers working with tabular data
who need defensible methodology and publication-ready outputs.
"""
import streamlit as st

from utils.session_state import get_data, init_session_state
from utils.llm_ui import render_llm_settings_sidebar

# Initialize session state
init_session_state()

# Page config
st.set_page_config(
    page_title="Tabular ML Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar: LLM settings
render_llm_settings_sidebar()

# Sidebar: Learning Checklist
with st.sidebar:
    st.header("Workflow Progress")

    data_uploaded = get_data() is not None
    data_configured = st.session_state.get('data_config') is not None and st.session_state.get('data_config').target_col is not None
    audit_complete = st.session_state.get('data_audit') is not None
    pipeline_built = st.session_state.get('preprocessing_pipeline') is not None
    models_trained = len(st.session_state.get('trained_models', {})) > 0
    explainability_run = st.session_state.get('permutation_importance') is not None
    report_generated = st.session_state.get('report_data') is not None

    checklist_items = [
        ("Upload & Configure Data", data_uploaded),
        ("Review Data Quality", audit_complete),
        ("Explore (EDA)", data_configured),
        ("Select Features", False),  # new feature selection page
        ("Build Preprocessing", pipeline_built),
        ("Train & Compare Models", models_trained),
        ("Explain & Validate", explainability_run),
        ("Export Report", report_generated),
    ]

    for item, completed in checklist_items:
        status = "✅" if completed else "⬜"
        st.markdown(f"{status} {item}")

    st.divider()
    completed_count = sum(1 for _, completed in checklist_items if completed)
    progress_pct = completed_count / len(checklist_items)
    st.progress(progress_pct)
    st.caption(f"{completed_count}/{len(checklist_items)} steps complete")

# ============================================================================
# Main page
# ============================================================================

st.title("🔬 Tabular ML Lab")
st.markdown("**Publication-grade machine learning for tabular research data**")

st.markdown("---")

# Getting started - clean and not overwhelming
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Welcome

    Tabular ML Lab guides you through the complete modeling workflow — from raw data
    to publication-ready results. Built for researchers who need:

    - **Defensible methodology** — proper validation, bootstrap confidence intervals, baseline comparisons
    - **Publication-ready outputs** — Table 1, methods sections, TRIPOD checklists, journal-quality figures
    - **Intelligent guidance** — the app coaches you through decisions and flags reviewer concerns

    ### How It Works

    Follow the pages in order using the sidebar. Each step builds on the previous one:
    """)

    # Visual workflow
    steps = [
        ("1️⃣", "**Upload & Audit**", "Load your data, merge files, configure target & features"),
        ("2️⃣", "**Explore (EDA)**", "Distributions, correlations, Table 1, data quality assessment"),
        ("3️⃣", "**Feature Selection**", "LASSO, RFE-CV, stability selection with FDR correction"),
        ("4️⃣", "**Preprocess**", "Imputation, scaling, encoding — per-model pipelines"),
        ("5️⃣", "**Train & Compare**", "Multiple models with bootstrap CIs & baseline comparison"),
        ("6️⃣", "**Explain & Validate**", "SHAP, calibration, external validation, subgroup analysis"),
        ("7️⃣", "**Export Report**", "Methods section, TRIPOD checklist, figures & tables"),
    ]

    for emoji, title, desc in steps:
        st.markdown(f"{emoji} {title} — {desc}")

with col2:
    st.markdown("### Quick Start")
    st.info("""
    **First time?**

    1. Click **Upload & Audit** in the sidebar
    2. Upload a CSV or Excel file
    3. Select your target variable
    4. Follow the guided workflow

    The app will coach you through each step.
    """)

    st.markdown("### Task Modes")
    st.markdown("""
    - **Prediction** — Build and validate ML models
    - **Hypothesis Testing** — Run statistical tests (t-tests, ANOVA, etc.)
    """)

st.markdown("---")

# What's new / capabilities
with st.expander("📋 Capabilities", expanded=False):
    st.markdown("""
    **Data Handling:**
    - CSV, Excel, Parquet, TSV upload
    - Multi-file projects with intelligent merge builder
    - Missing data characterization (MCAR/MAR analysis)
    - Automatic data type detection

    **Models:**
    - Linear (Ridge, Lasso, ElasticNet)
    - Trees (Random Forest, ExtraTrees, HistGradientBoosting)
    - Distance (KNN) · Margin (SVM) · Probabilistic (Naive Bayes, LDA)
    - Neural Networks (PyTorch, configurable architecture)
    - Automatic baseline comparison (null model + simple regression)

    **Evaluation:**
    - Bootstrap 95% CIs on all metrics
    - Calibration analysis (reliability diagrams, Brier score, ECE)
    - Decision curve analysis (clinical utility)
    - Subgroup analysis with forest plots
    - Cross-validation with paired statistical comparisons

    **Publication Tools:**
    - Table 1 generator (stratified, with p-values and SMD)
    - Auto-generated methods section
    - TRIPOD checklist tracker
    - CONSORT-style flow diagrams
    - Journal-quality figure export (Nature/JAMA themes)
    - LaTeX/Word table export

    **Feature Selection:**
    - LASSO path visualization
    - Recursive Feature Elimination with CV
    - Univariate screening with FDR correction
    - Stability selection
    - Consensus features across methods

    **Explainability:**
    - SHAP values (linear, tree, kernel)
    - Permutation importance
    - Partial dependence plots
    - AI-powered interpretation (Ollama/OpenAI/Anthropic)
    """)

# Sidebar extras
st.sidebar.title("🔬 Tabular ML Lab")

with st.sidebar.expander("📖 Guided Tour", expanded=st.session_state.get('show_guided_tour', False)):
    st.markdown("""
    **Follow these steps in order:**

    1. **Upload & Audit** — Create project, upload files, select target & features
    2. **EDA** — Explore data, generate Table 1, assess quality
    3. **Feature Selection** — Identify the most important predictors
    4. **Preprocess** — Build preprocessing pipelines
    5. **Train & Compare** — Train models, compare with baselines
    6. **Explainability** — Understand model behavior, validate
    7. **Report Export** — Generate publication-ready outputs

    *Or use **Hypothesis Testing** for statistical tests (no ML).*
    """)
    if st.button("I've read this", key="tour_done"):
        st.session_state.has_completed_tour = True
        st.session_state.show_guided_tour = False
        st.rerun()

# Debug (collapsed)
if st.sidebar.checkbox("Show Session State", value=False):
    st.sidebar.json({
        'has_data': get_data() is not None,
        'task_mode': st.session_state.get('task_mode'),
        'n_datasets': len(st.session_state.get('datasets_registry', {})),
        'configured': st.session_state.get('data_config') is not None,
        'pipeline': st.session_state.get('preprocessing_pipeline') is not None,
        'splits': st.session_state.get('X_train') is not None,
        'n_models': len(st.session_state.get('trained_models', {})),
    })
