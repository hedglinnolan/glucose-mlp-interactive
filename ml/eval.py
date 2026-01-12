"""
Evaluation utilities: metrics, cross-validation, residual analysis.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, log_loss,
    average_precision_score, confusion_matrix
)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Returns:
        Dictionary with MAE, RMSE, R2, MedianAE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    median_ae = np.median(np.abs(y_true - y_pred))
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MedianAE': float(median_ae)
    }


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with Accuracy, F1, ROC-AUC (if probas), LogLoss, PR-AUC
    """
    metrics = {}
    
    metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['F1'] = float(f1_score(y_true, y_pred, average='weighted'))
    
    if y_proba is not None:
        try:
            # ROC-AUC (binary or multiclass)
            if len(np.unique(y_true)) == 2:
                metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics['ROC-AUC'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
            
            # Log Loss
            metrics['LogLoss'] = float(log_loss(y_true, y_proba))
            
            # PR-AUC
            if len(np.unique(y_true)) == 2:
                metrics['PR-AUC'] = float(average_precision_score(y_true, y_proba[:, 1]))
        except Exception as e:
            # If metrics fail, skip them
            pass
    
    return metrics


def perform_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    task_type: str = 'regression',
    scoring: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model: Model with fit/predict interface
        X: Features
        y: Targets
        cv_folds: Number of folds
        task_type: 'regression' or 'classification'
        scoring: Scoring metric (if None, uses default for task type)
        
    Returns:
        Dictionary with metric arrays across folds
    """
    if scoring is None:
        scoring = 'neg_mean_squared_error' if task_type == 'regression' else 'accuracy'
    
    # Choose CV strategy
    if task_type == 'classification':
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform CV
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Convert to positive if using negative MSE
    if 'neg_' in scoring:
        scores = -scores
    
    return {
        'scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'folds': cv_folds
    }


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Analyze residuals for regression models.
    
    Returns:
        Dictionary with residual statistics and arrays
    """
    residuals = y_true - y_pred
    
    return {
        'residuals': residuals,
        'mean_residual': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'min_residual': float(np.min(residuals)),
        'max_residual': float(np.max(residuals)),
        'median_residual': float(np.median(residuals))
    }
