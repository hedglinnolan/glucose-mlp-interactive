"""
Preprocessing pipeline builder.
Creates sklearn Pipeline with ColumnTransformer for mixed data types.
"""
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from ml.feature_steps import create_pca_step, KMeansFeatures


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    numeric_imputation: str = 'median',  # 'mean', 'median', 'constant'
    numeric_scaling: str = 'standard',  # 'standard', 'robust', 'none'
    numeric_log_transform: bool = False,
    categorical_imputation: str = 'most_frequent',  # 'most_frequent', 'constant'
    categorical_encoding: str = 'onehot',  # 'onehot', 'target' (if enabled)
    handle_unknown: str = 'ignore',  # For one-hot encoding
    # Optional feature engineering steps
    use_kmeans_features: bool = False,
    kmeans_n_clusters: int = 5,
    kmeans_add_distances: bool = True,
    kmeans_add_onehot: bool = False,
    use_pca: bool = False,
    pca_n_components: Optional[Union[int, float]] = None,
    pca_whiten: bool = False,
    random_state: int = 42
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
            # Use sparse_output=True for memory efficiency, convert only when needed
            categorical_steps.append(('encoder', OneHotEncoder(
                sparse_output=True,  # Sparse for memory efficiency
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
    
    # Build pipeline steps
    steps = [('preprocessor', preprocessor)]
    
    # Optional: KMeansFeatures (must come before PCA if both enabled)
    if use_kmeans_features:
        kmeans_transformer = KMeansFeatures(
            n_clusters=kmeans_n_clusters,
            add_distances=kmeans_add_distances,
            add_onehot_label=kmeans_add_onehot,
            random_state=random_state
        )
        steps.append(('kmeans_features', kmeans_transformer))
    
    # Optional: PCA (dimensionality reduction)
    # Note: PCA n_components validation will happen at fit time
    # We can't know exact feature count until after ColumnTransformer + KMeans
    # So we'll validate in the create_pca_step or handle gracefully
    if use_pca:
        pca_transformer = create_pca_step(
            enabled=True,
            n_components=pca_n_components,
            whiten=pca_whiten,
            random_state=random_state
        )
        if pca_transformer:
            steps.append(('pca', pca_transformer))
    
    return Pipeline(steps)


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
                            step_parts.append("One-hot encoding (sparse)")
                step_desc += " → ".join(step_parts) if step_parts else "No transformation"
                steps.append(step_desc)
    
    # Add optional feature engineering steps
    if 'kmeans_features' in pipeline.named_steps:
        kmeans = pipeline.named_steps['kmeans_features']
        kmeans_desc = f"KMeans Features: {kmeans.n_clusters} clusters"
        if kmeans.add_distances:
            kmeans_desc += ", distances"
        if kmeans.add_onehot_label:
            kmeans_desc += ", one-hot labels"
        steps.append(kmeans_desc)
    
    if 'pca' in pipeline.named_steps:
        pca = pipeline.named_steps['pca']
        n_comp = pca.n_components_
        if isinstance(n_comp, (int, np.integer)):
            steps.append(f"PCA: {n_comp} components")
        else:
            steps.append(f"PCA: {n_comp} components (variance threshold)")
        if pca.whiten:
            steps[-1] += ", whitened"
    
    return "\n".join(steps) if steps else "No preprocessing steps"


def get_feature_names_after_transform(pipeline: Pipeline, original_feature_names: List[str]) -> List[str]:
    """
    Get feature names after pipeline transformation.
    Handles sparse matrices and OneHotEncoder properly.
    
    Args:
        pipeline: Fitted sklearn Pipeline
        original_feature_names: Original feature names before transformation
        
    Returns:
        List of feature names after transformation
    """
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        if hasattr(preprocessor, 'get_feature_names_out'):
            return list(preprocessor.get_feature_names_out())
    except Exception:
        pass
    
    # Fallback: construct names manually
    feature_names = []
    
    if hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
        for name, transformer, columns in pipeline.named_steps['preprocessor'].transformers_:
            if name == 'numeric':
                # Numeric features keep their names (unless scaled, but we keep original)
                feature_names.extend(columns)
            elif name == 'categorical':
                # Categorical: one-hot encoding creates multiple columns
                categorical_pipe = transformer
                for step_name, step_transformer in categorical_pipe.steps:
                    if step_name == 'encoder' and isinstance(step_transformer, OneHotEncoder):
                        # Get categories for each original column
                        if hasattr(step_transformer, 'categories_'):
                            for col_idx, col_name in enumerate(columns):
                                categories = step_transformer.categories_[col_idx]
                                for cat in categories:
                                    feature_names.append(f"{col_name}_{cat}")
                        else:
                            # Fallback: use column names
                            feature_names.extend([f"{col}_encoded" for col in columns])
    
    return feature_names if feature_names else [f"feature_{i}" for i in range(pipeline.transform([[0]*len(original_feature_names)]).shape[1])]
