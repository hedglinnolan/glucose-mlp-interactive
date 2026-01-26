"""
Main entry point for Modeling Lab multi-page Streamlit app.
Streamlit automatically detects pages in the pages/ directory.
"""
import streamlit as st

# Page config
st.set_page_config(
    page_title="Modeling Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Learning Checklist Sidebar (plain text; storyline progress indicator on pages keeps emojis)
with st.sidebar:
    st.header("Learning Checklist")
    
    # Check completion status
    data_uploaded = st.session_state.get('df') is not None
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

# Main page (home)
st.title("Modeling Lab")
st.markdown("""
Welcome to the **Modeling Lab** - An educational platform for machine learning experimentation.

### Workflow

Navigate through the pages using the sidebar:

1. **Upload & Audit** - Upload your dataset and perform data quality checks
2. **EDA** - Explore your data with visualizations and statistics
3. **Preprocess** - Build preprocessing pipelines for your features
4. **Train & Compare** - Train multiple models and compare performance
5. **Explainability** - Understand model predictions with feature importance
6. **Report Export** - Generate comprehensive modeling reports

### Features

- **Consistent Preprocessing**: All models use the same preprocessing pipeline
- **Honest Evaluation**: Proper train/val/test splits with optional cross-validation
- **Multiple Models**: Neural Networks, Random Forest, GLM (OLS & Huber)
- **Interpretability**: Permutation importance, partial dependence plots, optional SHAP
- **Production Ready**: Clean code structure with proper abstractions

### Getting Started

1. Start by uploading your CSV file in the **Upload & Audit** page
2. Select your target variable and features
3. Explore your data in the **EDA** page
4. Build a preprocessing pipeline in the **Preprocess** page
5. Train and compare models in the **Train & Compare** page
6. Understand your models in the **Explainability** page
7. Export your results in the **Report Export** page

---

**Note:** This app uses Streamlit's multi-page feature. Pages are automatically detected from the `pages/` directory.
""")

# Sidebar navigation info
st.sidebar.title("Modeling Lab")
st.sidebar.markdown("""
### Navigation

Use the sidebar to navigate between pages, or use the page selector at the top.

### Quick Links

- **Upload & Audit**: Start here to upload your data
- **EDA**: Explore your dataset
- **Preprocess**: Build preprocessing pipelines
- **Train & Compare**: Train and evaluate models
- **Explainability**: Understand model behavior
- **Report Export**: Generate reports
""")

# Show session state info (for debugging, can be removed in production)
if st.sidebar.checkbox("Show Session State Info", value=False):
    st.sidebar.json({
        'has_data': st.session_state.get('raw_data') is not None,
        'has_config': st.session_state.get('data_config') is not None,
        'has_pipeline': st.session_state.get('preprocessing_pipeline') is not None,
        'has_splits': st.session_state.get('X_train') is not None,
        'n_trained_models': len(st.session_state.get('trained_models', {}))
    })
