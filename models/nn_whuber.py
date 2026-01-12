"""
Neural Network wrapper using weighted Huber loss.
Wraps the existing NN training implementation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from models.base import BaseModelWrapper

logger = logging.getLogger(__name__)

# Copy SimpleMLP and weighted_huber_loss from existing models.py
# This wraps the existing implementation cleanly
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simplified MLP for regression (from existing models.py)."""
    
    def __init__(self, input_dim: int, hidden: list = [32, 32], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, 1))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


def weighted_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, 
                       t0: float = 180.0, s: float = 20.0, alpha: float = 2.5) -> torch.Tensor:
    """Weighted Huber loss focusing on high values (from existing models.py)."""
    errors = y_true - y_pred
    abs_errors = torch.abs(errors)
    
    # Huber loss component
    delta = 1.0
    huber_loss = torch.where(
        abs_errors <= delta,
        0.5 * errors ** 2,
        delta * abs_errors - 0.5 * delta ** 2
    )
    
    # Weight based on target value
    w = 1.0 + alpha * torch.exp(-((y_true - t0) / s) ** 2)
    w = torch.clamp(w, min=0.1, max=10.0)
    
    return (w * huber_loss).mean()


class NNWeightedHuberWrapper(BaseModelWrapper):
    """Wrapper for Neural Network with weighted Huber loss."""
    
    def __init__(self, hidden_layers: List[int] = None, dropout: float = 0.1):
        """
        Initialize NN wrapper.
        
        Args:
            hidden_layers: List of hidden layer sizes (default: [32, 32])
            dropout: Dropout rate
        """
        super().__init__("Neural Network (Weighted Huber)")
        self.hidden_layers = hidden_layers or [32, 32]
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 200,
            batch_size: int = 256,
            lr: float = 0.0015,
            weight_decay: float = 0.0002,
            patience: int = 30,
            progress_callback: Optional[callable] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            patience: Early stopping patience
            progress_callback: Optional callback function(epoch, train_loss, val_loss, val_rmse)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dictionary with training history
        """
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        else:
            X_val_t = None
            y_val_t = None
        
        # Create model
        self.model = SimpleMLP(
            input_dim=X_train.shape[1],
            hidden=self.hidden_layers,
            dropout=self.dropout
        )
        self.model = self.model.to(self.device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        if X_val_t is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
            )
        
        # Training loop
        best_val_rmse = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': []
        }
        
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_t, y_train_t),
            batch_size=batch_size,
            shuffle=True
        )
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = weighted_huber_loss(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
            
            # Validation
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_t)
                    val_loss = weighted_huber_loss(y_val_pred, y_val_t)
                    val_rmse = torch.sqrt(torch.mean((y_val_pred - y_val_t) ** 2)).item()
                
                history['val_loss'].append(val_loss.item())
                history['val_rmse'].append(val_rmse)
                
                # Progress callback
                if progress_callback:
                    progress_callback(epoch + 1, train_loss, val_loss.item(), val_rmse)
                
                # Learning rate scheduling
                scheduler.step(val_rmse)
                
                # Early stopping
                if val_rmse < best_val_rmse - 0.0001:
                    best_val_rmse = val_rmse
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.model.load_state_dict(best_model_state)
                        break
            else:
                # No validation set
                if progress_callback:
                    progress_callback(epoch + 1, train_loss, train_loss, 0.0)
        
        # Load best model if validation was used
        if X_val_t is not None and 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        self.history = history
        
        return {
            'history': history,
            'best_val_rmse': best_val_rmse if X_val_t is not None else None
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_pred = self.model(X_t).cpu().numpy().flatten()
        return y_pred


