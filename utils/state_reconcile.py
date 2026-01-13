"""
State reconciliation functions to maintain consistency when data changes.
"""
import pandas as pd
from typing import Optional, List
from utils.session_state import DataConfig, TaskTypeDetection, CohortStructureDetection


def reconcile_state_with_df(df: pd.DataFrame, session_state) -> None:
    """
    Reconcile session state when DataFrame changes.
    
    Rules:
    - Drop invalid columns from feature list
    - Clear target if missing
    - Clear entity_id if missing
    - Preserve valid selections
    - Do NOT reset task/cohort overrides unless data is incompatible
    
    Args:
        df: New DataFrame
        session_state: Streamlit session state
    """
    if df is None or len(df) == 0:
        return
    
    # Get current config
    data_config: Optional[DataConfig] = session_state.get('data_config')
    if data_config is None:
        return
    
    # Reconcile target
    if data_config.target_col and data_config.target_col not in df.columns:
        # Target column missing - clear it
        data_config.target_col = None
        # Also clear task type detection if target was the trigger
        task_detection = session_state.get('task_type_detection')
        if task_detection:
            task_detection.detected = None
    
    # Reconcile features - keep only valid ones
    if data_config.feature_cols:
        valid_features = [
            f for f in data_config.feature_cols 
            if f in df.columns
        ]
        data_config.feature_cols = valid_features
    
    # Reconcile datetime column
    if data_config.datetime_col and data_config.datetime_col not in df.columns:
        data_config.datetime_col = None
    
    # Reconcile entity ID
    cohort_detection = session_state.get('cohort_structure_detection')
    if cohort_detection:
        if cohort_detection.entity_id_final and cohort_detection.entity_id_final not in df.columns:
            cohort_detection.entity_id_detected = None
            cohort_detection.entity_id_final = None
            cohort_detection.entity_id_override_value = None
        
        # Remove invalid entity ID candidates
        if cohort_detection.entity_id_candidates:
            cohort_detection.entity_id_candidates = [
                c for c in cohort_detection.entity_id_candidates 
                if c in df.columns
            ]
    
    # Update session state
    session_state.data_config = data_config
