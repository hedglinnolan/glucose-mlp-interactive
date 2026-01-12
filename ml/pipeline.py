"""
Preprocessing pipeline builder.
Creates sklearn Pipeline with ColumnTransformer for mixed data types.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    numeric_imputation: str = 'median',  # 'mean', 'median', 'constant'
    numeric_scaling: str = 'standard',  # 'standard', 'robust', 'none'
    numeric_log_transform: bool = False,
    categorical_imputation: str = 'most_frequent',  # 'most_frequent', 'constant'
    categorical_encoding: str = 'onehot',  # 'onehot', 'target' (if enabled)
    handle_unknown: str = 'ignore'  # For one-hot encoding
) -> Pipeline:
    """
    Build preprocessing pipeline using ColumnTransformer.
    
    Args:
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names
        numeric_imputation: Strategy for imputing numeric missing values
        numeric_scaling: Scaling strategy for numeric features
        numeric_log_transform: Whether to apply log transform to numeric features
        categorical_imputation: Strategy for imputing categorical missing values
        categorical_encoding: Encoding strategy for categorical features
        handle_unknown: How to handle unknown categories in one-hot encoding
        
    Returns:
        sklearn Pipeline with ColumnTransformer
    """
    transformers = []
    
    # Numeric preprocessing
    if numeric_features:
        numeric_steps = []
        
        # Imputation
        if numeric_imputation == 'mean':
            numeric_steps.append(('imputer', SimpleImputer(strategy='mean')))
        elif numeric_imputation == 'median':
            numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
        elif numeric_imputation == 'constant':
            numeric_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
        
        # Log transform (optional)
        if numeric_log_transform:
            def log_transform(X):
                return np.log1p(np.maximum(X, 0))  # log1p handles zeros
            numeric_steps.append(('log', FunctionTransformer(log_transform)))
        
        # Scaling
        if numeric_scaling == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif numeric_scaling == 'robust':
            numeric_steps.append(('scaler', RobustScaler()))
        # 'none' means no scaling
        
        numeric_pipeline = Pipeline(numeric_steps)
        transformers.append(('numeric', numeric_pipeline, numeric_features))
    
    # Categorical preprocessing
    if categorical_features:
        categorical_steps = []
        
        # Imputation
        if categorical_imputation == 'most_frequent':
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        elif categorical_imputation == 'constant':
            categorical_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
        
        # Encoding
        if categorical_encoding == 'onehot':
            categorical_steps.append(('encoder', OneHotEncoder(
                sparse_output=False,
                handle_unknown=handle_unknown,
                drop='if_binary'  # Drop one column for binary features
            )))
        # Note: Target encoding would require target variable, handled separately if needed
        
        categorical_pipeline = Pipeline(categorical_steps)
        transformers.append(('categorical', categorical_pipeline, categorical_features))
    
    # Create ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any columns not specified
        verbose_feature_names_out=False
    )
    
    # Wrap in Pipeline (allows for future steps)
    pipeline = Pipeline([('preprocessor', preprocessor)])
    
    return pipeline


def get_pipeline_recipe(pipeline: Pipeline) -> str:
    """
    Get human-readable description of pipeline steps.
    
    Args:
        pipeline: sklearn Pipeline
        
    Returns:
        String description of pipeline
    """
    steps = []
    
    if hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
        for name, transformer, columns in pipeline.named_steps['preprocessor'].transformers_:
            if name == 'numeric':
                # Get numeric pipeline steps
                numeric_pipe = transformer
                step_desc = f"Numeric features ({len(columns)}): "
                step_parts = []
                for step_name, step_transformer in numeric_pipe.steps:
                    if step_name == 'imputer':
                        strategy = step_transformer.strategy
                        step_parts.append(f"Impute ({strategy})")
                    elif step_name == 'log':
                        step_parts.append("Log transform")
                    elif step_name == 'scaler':
                        if isinstance(step_transformer, StandardScaler):
                            step_parts.append("Standard scaling")
                        elif isinstance(step_transformer, RobustScaler):
                            step_parts.append("Robust scaling")
                step_desc += " → ".join(step_parts) if step_parts else "No transformation"
                steps.append(step_desc)
            
            elif name == 'categorical':
                # Get categorical pipeline steps
                categorical_pipe = transformer
                step_desc = f"Categorical features ({len(columns)}): "
                step_parts = []
                for step_name, step_transformer in categorical_pipe.steps:
                    if step_name == 'imputer':
                        strategy = step_transformer.strategy
                        step_parts.append(f"Impute ({strategy})")
                    elif step_name == 'encoder':
                        if isinstance(step_transformer, OneHotEncoder):
                            step_parts.append("One-hot encoding")
                step_desc += " → ".join(step_parts) if step_parts else "No transformation"
                steps.append(step_desc)
    
    return "\n".join(steps) if steps else "No preprocessing steps"
