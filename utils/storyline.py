"""
Storyline and progress tracking for the educational modeling lab.
"""
from typing import List, Dict, Optional
import streamlit as st


class StorylinePhase:
    """Represents a phase in the modeling lab storyline."""
    def __init__(self, id: str, name: str, description: str, page: str):
        self.id = id
        self.name = name
        self.description = description
        self.page = page


PHASES = [
    StorylinePhase("data_loaded", "Data Loaded", "Dataset uploaded and basic info confirmed", "01_Upload_and_Audit"),
    StorylinePhase("target_confirmed", "Target & Task Confirmed", "Target variable and task type (regression/classification) set", "01_Upload_and_Audit"),
    StorylinePhase("cohort_confirmed", "Cohort Structure Confirmed", "Cross-sectional vs longitudinal structure identified", "01_Upload_and_Audit"),
    StorylinePhase("eda_insights", "EDA Insights Gathered", "Key patterns and relationships explored", "02_EDA"),
    StorylinePhase("preprocessing", "Preprocessing Configured", "Data transformation pipeline built", "03_Preprocess"),
    StorylinePhase("models_trained", "Models Trained & Compared", "Models trained and performance evaluated", "04_Train_and_Compare"),
    StorylinePhase("explainability", "Explainability Completed", "Model interpretations and feature importance analyzed", "05_Explainability"),
    StorylinePhase("report_exported", "Report Exported", "Comprehensive report generated and downloaded", "06_Report_Export"),
]


def get_completed_phases() -> List[str]:
    """Get list of completed phase IDs from session state."""
    completed = []
    
    # Check each phase
    if st.session_state.get('raw_data') is not None:
        completed.append("data_loaded")
    
    data_config = st.session_state.get('data_config')
    if data_config and data_config.target_col:
        completed.append("target_confirmed")
    
    cohort_detection = st.session_state.get('cohort_structure_detection')
    if cohort_detection and cohort_detection.final:
        completed.append("cohort_confirmed")
    
    if st.session_state.get('eda_insights'):
        completed.append("eda_insights")
    
    if st.session_state.get('preprocessing_pipeline'):
        completed.append("preprocessing")
    
    if st.session_state.get('trained_models'):
        completed.append("models_trained")
    
    if st.session_state.get('permutation_importance') or st.session_state.get('partial_dependence'):
        completed.append("explainability")
    
    if st.session_state.get('report_data'):
        completed.append("report_exported")
    
    return completed


def render_progress_indicator(current_page: str):
    """Render the progress indicator showing where user is in the lab."""
    completed = get_completed_phases()
    
    st.sidebar.header("📍 Modeling Lab Progress")
    
    for phase in PHASES:
        is_completed = phase.id in completed
        is_current = phase.page == current_page
        
        if is_completed:
            icon = "✅"
            status = "Completed"
        elif is_current:
            icon = "🔄"
            status = "Current"
        else:
            icon = "⏳"
            status = "Pending"
        
        st.sidebar.markdown(f"{icon} **{phase.name}**")
        if is_current:
            st.sidebar.caption(phase.description)
    
    # Progress percentage
    progress_pct = len(completed) / len(PHASES) * 100
    st.sidebar.progress(progress_pct / 100)
    st.sidebar.caption(f"{len(completed)}/{len(PHASES)} phases complete ({progress_pct:.0f}%)")


def add_insight(insight_id: str, finding: str, implication: str, category: str = "general"):
    """Add an EDA insight to session state."""
    if 'eda_insights' not in st.session_state:
        st.session_state.eda_insights = []
    
    insight = {
        'id': insight_id,
        'finding': finding,
        'implication': implication,
        'category': category
    }
    
    # Avoid duplicates
    existing_ids = [i['id'] for i in st.session_state.eda_insights]
    if insight_id not in existing_ids:
        st.session_state.eda_insights.append(insight)


def get_insights_by_category(category: Optional[str] = None) -> List[Dict]:
    """Get EDA insights, optionally filtered by category."""
    insights = st.session_state.get('eda_insights', [])
    if category:
        return [i for i in insights if i.get('category') == category]
    return insights
