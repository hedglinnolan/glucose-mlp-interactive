"""
Visualization utilities for the interactive predictor.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List


def plot_training_history(history: Dict[str, List[float]]) -> go.Figure:
    """Create interactive training history plot."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=history['train_loss'],
        mode='lines',
        name='Train Loss',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=history['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=history['val_rmse'],
        mode='lines',
        name='Validation RMSE',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss / RMSE',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                               title: str = "Predictions vs Actual") -> go.Figure:
    """Create scatter plot of predictions vs actual values."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.6,
            color='blue'
        ),
        name='Predictions'
    ))
    
    # y = x reference line (perfect agreement)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='y = x reference (perfect agreement)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Actual',
        yaxis_title='Predicted',
        height=400
    )
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                   title: str = "Residuals") -> go.Figure:
    """Create residual plot."""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.6,
            color='blue'
        ),
        name='Residuals'
    ))
    
    # Zero line
    fig.add_trace(go.Scatter(
        x=[y_pred.min(), y_pred.max()],
        y=[0, 0],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Zero'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='Residual (Actual - Predicted)',
        height=400
    )
    
    return fig


def plot_bland_altman(
    a: np.ndarray,
    b: np.ndarray,
    title: str = "Bland–Altman Plot",
    label_a: str = "Method A",
    label_b: str = "Method B"
) -> go.Figure:
    """
    Bland–Altman plot: agreement between two measurement methods.
    x = (a + b) / 2 (mean), y = a - b (difference).
    Shows mean difference and limits of agreement (mean ± 1.96 * std).
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mean_ab = (a + b) / 2
    diff = a - b
    mean_diff = float(np.nanmean(diff))
    std_diff = float(np.nanstd(diff))
    loa_low = mean_diff - 1.96 * std_diff
    loa_high = mean_diff + 1.96 * std_diff

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_ab,
        y=diff,
        mode='markers',
        marker=dict(size=5, opacity=0.6, color='blue'),
        name='Difference'
    ))
    fig.add_hline(y=mean_diff, line_dash='solid', line_color='green', annotation_text=f'Mean diff: {mean_diff:.3f}')
    fig.add_hline(y=loa_low, line_dash='dash', line_color='gray', annotation_text=f'LoA −1.96 SD: {loa_low:.3f}')
    fig.add_hline(y=loa_high, line_dash='dash', line_color='gray', annotation_text=f'LoA +1.96 SD: {loa_high:.3f}')
    fig.update_layout(
        title=title,
        xaxis_title=f'Mean of ({label_a} and {label_b})',
        yaxis_title=f'Difference ({label_a} − {label_b})',
        height=400
    )
    return fig


def create_metrics_display(metrics: Dict[str, float]) -> str:
    """Create formatted metrics display."""
    return f"""
    **Test Set Performance:**
    - **RMSE**: {metrics['RMSE']:.4f}
    - **MAE**: {metrics['MAE']:.4f}
    - **R²**: {metrics['R2']:.4f}
    """
