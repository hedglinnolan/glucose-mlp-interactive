"""GLM (OLS Linear Regression / Logistic Regression) wrapper."""
import numpy as np
from typing import Dict, Optional, Any
from sklearn.linear_model import LinearRegression, LogisticRegression

from models.base import BaseModelWrapper


class GLMWrapper(BaseModelWrapper):
    """Wrapper for OLS Linear Regression (regression) or Logistic Regression (classification)."""
    
    def __init__(self, task_type: str = 'regression'):
        """
        Initialize GLM wrapper.
        
        Args:
            task_type: 'regression' or 'classification'
        """
        super().__init__("GLM (OLS)" if task_type == 'regression' else "GLM (Logistic)")
        self.task_type = task_type
        
        if task_type == 'regression':
            self.model = LinearRegression()
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> Dict[str, Any]:
        """Train the model."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate validation metrics if available
        val_metric = None
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            if self.task_type == 'regression':
                val_metric = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
            else:
                val_metric = np.mean(y_val_pred == y_val)  # Accuracy
        
        return {
            'history': {'val_rmse': [val_metric] if val_metric is not None else []},
            'best_val_rmse': val_metric
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities (for classification)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.task_type == 'classification' and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def supports_proba(self) -> bool:
        """Check if model supports probability predictions."""
        return self.task_type == 'classification'