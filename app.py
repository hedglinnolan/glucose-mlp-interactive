"""
Main entry point for Modeling Lab multi-page Streamlit app.
Streamlit automatically detects pages in the pages/ directory.
"""
import streamlit as st

from utils.session_state import get_data, init_session_state

# Initialize session state (for tour and checklist)
init_session_state()

# Page config
st.set_page_config(
    page_title="Home",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Learning Checklist Sidebar (plain text; storyline progress indicator on pages keeps emojis)
with st.sidebar:
    st.header("Learning Checklist")
    
    # Check completion status
    data_uploaded = get_data() is not None
    data_configured = st.session_state.get('data_config') is not None and st.session_state.get('data_config').target_col is not None
    audit_complete = st.session_state.get('data_audit') is not None
    pipeline_built = st.session_state.get('preprocessing_pipeline') is not None
    models_trained = len(st.session_state.get('trained_models', {})) > 0
    explainability_run = st.session_state.get('permutation_importance') is not None
    report_generated = st.session_state.get('report_data') is not None
    
    checklist_items = [
        ("Upload Data", data_uploaded),
        ("Review Audit", audit_complete),
        ("Explore EDA", data_configured),
        ("Build Pipeline", pipeline_built),
        ("Train Models", models_trained),
        ("Run Explainability", explainability_run),
        ("Export Report", report_generated)
    ]
    
    for item, completed in checklist_items:
        status = "[x]" if completed else "[ ]"
        st.markdown(f"{status} {item}")
    
    st.divider()
    
    # Progress indicator
    completed_count = sum(1 for _, completed in checklist_items if completed)
    progress_pct = (completed_count / len(checklist_items)) * 100
    st.progress(progress_pct / 100)
    st.caption(f"Progress: {completed_count}/{len(checklist_items)} steps complete")

# Guided Tour (first-time users)
if not st.session_state.get('has_completed_tour', False):
    with st.container():
        st.info("**New here?** Take the Guided Tour to learn the workflow. Complete steps 1-7 in order for the best experience.")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Start Guided Tour", key="start_tour"):
                st.session_state.show_guided_tour = True
                st.rerun()
            if st.button("Dismiss", key="dismiss_tour"):
                st.session_state.has_completed_tour = True
                st.rerun()
        st.divider()

# Main page (home)
st.title("Home")
st.markdown("""
Welcome to the **Modeling Lab** - An educational platform for machine learning experimentation.

### Workflow

Navigate through the pages using the sidebar:

1. **Upload & Audit** - Create projects, upload related files, merge datasets, and configure your analysis
2. **EDA** - Explore your data with visualizations and statistics
3. **Preprocess** - Build preprocessing pipelines for your features (Prediction mode only)
4. **Train & Compare** - Train multiple models and compare performance (Prediction mode only)
5. **Explainability** - Understand model predictions with feature importance (Prediction mode only)
6. **Report Export** - Generate comprehensive modeling reports (Prediction mode only)
7. **Hypothesis Testing** - Run statistical tests to test hypotheses (Hypothesis Testing mode only)

### Features

- **Project-Based Organization**: Group related datasets into projects for organized analysis
- **Intelligent Data Merging**: Upload multiple related files and merge them with an interactive merge builder
- **Automatic Join Key Detection**: The system suggests potential join keys based on column analysis
- **Task Modes**: Choose between Prediction (ML models) or Hypothesis Testing (statistical tests)
- **Data Reshaping**: Transpose data during upload if features are in rows instead of columns
- **Multiple Formats**: Support for CSV, Excel, Parquet, and TSV files
- **Consistent Preprocessing**: All models use the same preprocessing pipeline
- **Honest Evaluation**: Proper train/val/test splits with optional cross-validation
- **Multiple Models**: Neural Networks, Random Forest, GLM (OLS & Huber)
- **Interpretability**: Permutation importance, partial dependence plots, optional SHAP
- **Statistical Testing**: Correlation, t-tests, ANOVA, chi-square, normality tests, and more

### Getting Started

**Step 1: Set Up Your Project**
1. Go to **Upload & Audit** page
2. Create a new project to organize your related datasets
3. Upload all data files that belong together (e.g., patient demographics + lab results)

**Step 2: Merge Your Data (if needed)**
4. If you have multiple files, use the Merge Builder to combine them
5. The system will detect common columns and suggest join keys
6. Define your merge steps and create a working table

**Step 3: Configure Your Analysis**
7. Select your task mode (Prediction or Hypothesis Testing)
8. For Prediction: Select target and feature variables
9. For Hypothesis Testing: Navigate to the Hypothesis Testing page

**Step 4: Complete Your Analysis**
- **Prediction**: EDA → Preprocess → Train & Compare → Explainability → Report Export
- **Hypothesis Testing**: Navigate to Hypothesis Testing page to run statistical tests

---

**Note:** This app uses Streamlit's multi-page feature. Pages are automatically detected from the `pages/` directory.
""")

# Sidebar navigation info
st.sidebar.title("Modeling Lab")

# Guided Tour expander (always available)
with st.sidebar.expander("Guided Tour", expanded=st.session_state.get('show_guided_tour', False)):
    st.markdown("""
    **Complete these steps in order:**
    
    1. **Upload & Audit** - Create project, upload files, merge if needed, select target & features
    2. **EDA** - Explore distributions, correlations, target vs features
    3. **Preprocess** - Build pipeline (imputation, scaling, encoding)
    4. **Train & Compare** - Prepare splits, train models, compare metrics
    5. **Explainability** - Permutation importance, partial dependence
    6. **Report Export** - Download comprehensive report
    
    **Hypothesis Testing** - Alternative path for statistical tests (no ML).
    """)
    if st.button("I've completed the tour", key="tour_done"):
        st.session_state.has_completed_tour = True
        st.session_state.show_guided_tour = False
        st.rerun()

st.sidebar.markdown("""
### Navigation

Use the sidebar to navigate between pages, or use the page selector at the top.

### Quick Links

- **Upload & Audit**: Create projects, upload files, merge datasets, and configure analysis
- **EDA**: Explore your dataset visually
- **Preprocess**: Build preprocessing pipelines (Prediction only)
- **Train & Compare**: Train and evaluate models (Prediction only)
- **Explainability**: Understand model behavior (Prediction only)
- **Report Export**: Generate comprehensive reports (Prediction only)
- **Hypothesis Testing**: Run statistical tests (Hypothesis Testing only)
""")

# Show session state info (for debugging, can be removed in production)
if st.sidebar.checkbox("Show Session State Info", value=False):
    st.sidebar.json({
        'has_working_table': st.session_state.get('working_table') is not None,
        'task_mode': st.session_state.get('task_mode'),
        'n_datasets_loaded': len(st.session_state.get('datasets_registry', {})),
        'has_config': st.session_state.get('data_config') is not None and st.session_state.get('data_config').target_col is not None,
        'has_pipeline': st.session_state.get('preprocessing_pipeline') is not None,
        'has_splits': st.session_state.get('X_train') is not None,
        'n_trained_models': len(st.session_state.get('trained_models', {}))
    })
