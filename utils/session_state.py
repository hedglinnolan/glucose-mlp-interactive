"""
Session state management for multi-page Streamlit app.
Defines schema and initialization functions.
"""
import streamlit as st
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


@dataclass
class DataConfig:
    """Configuration for dataset and target/feature selection."""
    target_col: Optional[str] = None
    feature_cols: List[str] = field(default_factory=list)
    datetime_col: Optional[str] = None  # For time-series splits
    task_type: Optional[str] = None  # 'regression' or 'classification'


@dataclass
class SplitConfig:
    """Configuration for train/val/test splits."""
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    stratify: bool = False  # For classification
    use_time_split: bool = False  # Use datetime_col for splitting


@dataclass
class ModelConfig:
    """Configuration for model hyperparameters."""
    # Neural Network
    nn_epochs: int = 200
    nn_batch_size: int = 256
    nn_lr: float = 0.0015
    nn_weight_decay: float = 0.0002
    nn_patience: int = 30
    nn_dropout: float = 0.1
    
    # Random Forest
    rf_n_estimators: int = 500
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 10
    
    # GLM/Huber
    huber_epsilon: float = 1.35
    huber_alpha: float = 0.0


def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        # Data
        'raw_data': None,
        'data_config': DataConfig(),
        'data_audit': None,
        
        # Preprocessing
        'preprocessing_pipeline': None,
        'preprocessing_config': None,
        
        # Splits
        'split_config': SplitConfig(),
        'X_train': None,
        'X_val': None,
        'X_test': None,
        'y_train': None,
        'y_val': None,
        'y_test': None,
        'feature_names': None,
        
        # Models
        'model_config': ModelConfig(),
        'trained_models': {},  # Dict[str, Any] - model name -> model object
        'model_results': {},  # Dict[str, Dict] - model name -> metrics/history
        
        # Evaluation
        'cv_results': None,  # For k-fold CV
        'use_cv': False,
        'cv_folds': 5,
        
        # Explainability
        'permutation_importance': {},
        'partial_dependence': {},
        
        # Report
        'report_data': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_data() -> Optional[pd.DataFrame]:
    """Get raw data from session state."""
    return st.session_state.get('raw_data')


def set_data(df: pd.DataFrame):
    """Set raw data in session state."""
    st.session_state.raw_data = df


def get_preprocessing_pipeline() -> Optional[Pipeline]:
    """Get preprocessing pipeline from session state."""
    return st.session_state.get('preprocessing_pipeline')


def set_preprocessing_pipeline(pipeline: Pipeline, config: Dict[str, Any]):
    """Set preprocessing pipeline and config."""
    st.session_state.preprocessing_pipeline = pipeline
    st.session_state.preprocessing_config = config


def get_splits() -> Optional[tuple]:
    """Get train/val/test splits from session state."""
    if st.session_state.get('X_train') is None:
        return None
    return (
        st.session_state.X_train,
        st.session_state.X_val,
        st.session_state.X_test,
        st.session_state.y_train,
        st.session_state.y_val,
        st.session_state.y_test,
    )


def set_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_names: List[str]):
    """Set train/val/test splits in session state."""
    st.session_state.X_train = X_train
    st.session_state.X_val = X_val
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_val = y_val
    st.session_state.y_test = y_test
    st.session_state.feature_names = feature_names


def add_trained_model(name: str, model: Any, results: Dict[str, Any]):
    """Add a trained model and its results to session state."""
    st.session_state.trained_models[name] = model
    st.session_state.model_results[name] = results
