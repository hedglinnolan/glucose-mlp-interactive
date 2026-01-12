"""
Base model wrapper interface.
All model wrappers should inherit from BaseModelWrapper.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BaseModelWrapper(ABC):
    """Base class for all model wrappers."""
    
    def __init__(self, name: str):
        """
        Initialize model wrapper.
        
        Args:
            name: Model name/identifier
        """
        self.name = name
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities (for classification).
        Returns None if not supported.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities or None
        """
        return None
    
    def get_model(self) -> Any:
        """Get the underlying model object."""
        return self.model
    
    def supports_proba(self) -> bool:
        """Check if model supports probability predictions."""
        return False
