"""
Model implementations for the interactive predictor.
Includes Neural Network, Random Forest, and GLM models.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler


class SimpleMLP(nn.Module):
    """Simplified MLP for regression."""
    
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
    """Weighted Huber loss focusing on high values."""
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


class ModelTrainer:
    """Unified interface for training different model types."""
    
    def __init__(self, model_type: str = "neural_network"):
        """
        Initialize model trainer.
        
        Args:
            model_type: One of "neural_network", "random_forest", "glm_ols", "glm_huber"
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.history = None
        
    def train(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        epochs: int = 200,
        batch_size: int = 256,
        lr: float = 0.0015,
        weight_decay: float = 0.0002,
        patience: int = 30,
        progress_callback: Optional[callable] = None,
        **kwargs
    ) -> Dict:
        """
        Train the model.
        
        Returns:
            Dictionary with training history and model info
        """
        self.feature_names = feature_names
        
        if self.model_type == "neural_network":
            return self._train_neural_network(
                X_train, X_val, y_train, y_val,
                epochs, batch_size, lr, weight_decay, patience, progress_callback
            )
        elif self.model_type == "random_forest":
            return self._train_random_forest(
                X_train, X_val, y_train, y_val, progress_callback, **kwargs
            )
        elif self.model_type == "glm_ols":
            return self._train_glm_ols(
                X_train, X_val, y_train, y_val, progress_callback
            )
        elif self.model_type == "glm_huber":
            return self._train_glm_huber(
                X_train, X_val, y_train, y_val, progress_callback
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _train_neural_network(
        self, X_train, X_val, y_train, y_val,
        epochs, batch_size, lr, weight_decay, patience, progress_callback
    ):
        """Train neural network model."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_train_t = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
        y_val_t = torch.FloatTensor(y_val.reshape(-1, 1)).to(device)
        
        # Create model
        model = SimpleMLP(input_dim=X_train.shape[1], hidden=[32, 32], dropout=0.1)
        model = model.to(device)
        
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = weighted_huber_loss(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_t)
                val_loss = weighted_huber_loss(y_val_pred, y_val_t)
                val_rmse = torch.sqrt(torch.mean((y_val_pred - y_val_t) ** 2)).item()
            
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)
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
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    model.load_state_dict(best_model_state)
                    break
        
        # Load best model
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        self.model = model
        self.history = history
        
        return {
            'history': history,
            'model_type': 'neural_network',
            'best_val_rmse': best_val_rmse
        }
    
    def _train_random_forest(
        self, X_train, X_val, y_train, y_val, progress_callback, **kwargs
    ):
        """Train Random Forest model."""
        n_estimators = kwargs.get('n_estimators', 500)
        min_samples_leaf = kwargs.get('min_samples_leaf', 10)
        
        if progress_callback:
            progress_callback(1, 0, 0, 0)  # Initialize
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
        
        if progress_callback:
            progress_callback(1, 0, 0, val_rmse)
        
        self.model = model
        self.history = {
            'train_loss': [0],
            'val_loss': [0],
            'val_rmse': [val_rmse]
        }
        
        return {
            'history': self.history,
            'model_type': 'random_forest',
            'best_val_rmse': val_rmse
        }
    
    def _train_glm_ols(
        self, X_train, X_val, y_train, y_val, progress_callback
    ):
        """Train OLS Linear Regression model."""
        if progress_callback:
            progress_callback(1, 0, 0, 0)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
        
        if progress_callback:
            progress_callback(1, 0, 0, val_rmse)
        
        self.model = model
        self.history = {
            'train_loss': [0],
            'val_loss': [0],
            'val_rmse': [val_rmse]
        }
        
        return {
            'history': self.history,
            'model_type': 'glm_ols',
            'best_val_rmse': val_rmse
        }
    
    def _train_glm_huber(
        self, X_train, X_val, y_train, y_val, progress_callback
    ):
        """Train Huber Regression model."""
        if progress_callback:
            progress_callback(1, 0, 0, 0)
        
        model = HuberRegressor(epsilon=1.35, alpha=0.0)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
        
        if progress_callback:
            progress_callback(1, 0, 0, val_rmse)
        
        self.model = model
        self.history = {
            'train_loss': [0],
            'val_loss': [0],
            'val_rmse': [val_rmse]
        }
        
        return {
            'history': self.history,
            'model_type': 'glm_huber',
            'best_val_rmse': val_rmse
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if self.model_type == "neural_network":
            device = next(self.model.parameters()).device
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(device)
                y_pred = self.model(X_t).cpu().numpy().flatten()
            return y_pred
        else:
            return self.model.predict(X)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - (ssr / sst) if sst > 0 else 0.0
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2)
    }
