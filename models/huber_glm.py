"""Huber GLM wrapper."""
import numpy as np
from typing import Dict, Optional, Any
from sklearn.linear_model import HuberRegressor

from models.base import BaseModelWrapper


class HuberGLMWrapper(BaseModelWrapper):
    """Wrapper for Huber Regression."""
    
    def __init__(self, epsilon: float = 1.35, alpha: float = 0.0):
        """
        Initialize Huber GLM wrapper.
        
        Args:
            epsilon: Epsilon parameter for Huber loss
            alpha: Regularization strength
        """
        super().__init__("GLM (Huber)")
        self.epsilon = epsilon
        self.alpha = alpha
        self.model = HuberRegressor(epsilon=epsilon, alpha=alpha)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> Dict[str, Any]:
        """Train the model."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate validation metrics if available
        val_rmse = None
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
        
        return {
            'history': {'val_rmse': [val_rmse] if val_rmse is not None else []},
            'best_val_rmse': val_rmse
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
