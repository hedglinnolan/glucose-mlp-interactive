"""
Lightweight conceptual MLP visualization for training display.
Shows simplified network with actual feature names and target label.
"""
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional
import torch
import torch.nn as nn


def create_conceptual_training_viz(
    feature_names: List[str],
    target_name: str,
    sample_input: np.ndarray,
    model: nn.Module,
    epoch: int,
    phase: str,  # "forward" or "backward"
    target_value: float,
    prediction: float,
    loss_value: float,
    loss_type: str = "Weighted Huber"
) -> go.Figure:
    """
    Create a lightweight conceptual visualization of the neural network.
    Shows simplified architecture with actual feature names.
    
    Args:
        feature_names: List of input feature names
        target_name: Name of target variable
        sample_input: Sample input values
        model: Current model
        epoch: Current epoch number
        phase: "forward" or "backward"
        target_value: Actual target value
        prediction: Model prediction
        loss_value: Current loss value
        loss_type: Type of loss function used
        
    Returns:
        Plotly figure
    """
    # Simplify: show max 5 features, 3 hidden nodes
    max_features = min(5, len(feature_names))
    num_hidden = 3
    
    # Select top features by absolute value
    feature_indices = np.argsort(np.abs(sample_input))[-max_features:][::-1]
    selected_features = [feature_names[i] for i in feature_indices]
    selected_values = sample_input[feature_indices]
    
    # Normalize values for display
    max_val = max(np.abs(selected_values).max(), abs(prediction), abs(target_value), 1.0)
    selected_values_norm = selected_values / max_val
    
    fig = go.Figure()
    
    # Layer positions
    input_x = 1
    hidden_x = 4
    output_x = 7
    
    # Input layer nodes
    input_y_positions = np.linspace(6, 2, max_features)
    input_nodes = []
    for i, (name, val) in enumerate(zip(selected_features, selected_values)):
        y_pos = input_y_positions[i]
        input_nodes.append((input_x, y_pos, name, val))
        
        # Draw input node
        color_intensity = min(abs(val) / max_val, 1.0)
        node_color = f'rgba(0, 150, 255, {0.4 + color_intensity * 0.6})'
        
        fig.add_trace(go.Scatter(
            x=[input_x],
            y=[y_pos],
            mode='markers+text',
            marker=dict(size=40, color=node_color, line=dict(width=2, color='white')),
            text=[f'{val:.2f}'],
            textposition='middle center',
            textfont=dict(size=10, color='white', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Feature name label
        fig.add_annotation(
            x=input_x,
            y=y_pos + 0.4,
            text=name[:15] + ('...' if len(name) > 15 else ''),
            showarrow=False,
            font=dict(size=9, color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    
    # Hidden layer nodes
    hidden_y_positions = np.linspace(5.5, 2.5, num_hidden)
    hidden_nodes = []
    
    # Simulate hidden layer activations (simplified)
    if phase == "forward":
        # Forward: show activations flowing
        hidden_activations = np.random.rand(num_hidden) * 0.5 + 0.3
    else:
        # Backward: show gradients
        hidden_activations = np.random.rand(num_hidden) * 0.3 - 0.15
    
    for i, act in enumerate(hidden_activations):
        y_pos = hidden_y_positions[i]
        hidden_nodes.append((hidden_x, y_pos, act))
        
        if phase == "forward":
            node_color = f'rgba(100, 200, 100, {0.4 + abs(act) * 0.6})'
        else:
            # Backward: red for negative, green for positive gradients
            if act < 0:
                node_color = f'rgba(255, 100, 100, {0.4 + abs(act) * 2})'
            else:
                node_color = f'rgba(100, 255, 100, {0.4 + abs(act) * 2})'
        
        fig.add_trace(go.Scatter(
            x=[hidden_x],
            y=[y_pos],
            mode='markers+text',
            marker=dict(size=35, color=node_color, line=dict(width=2, color='white')),
            text=[f'{act:.2f}'],
            textposition='middle center',
            textfont=dict(size=9, color='white', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Output node
    output_y = 4
    error = prediction - target_value
    
    if phase == "forward":
        output_color = f'rgba(255, 200, 0, 0.8)'
        output_text = f'{prediction:.2f}'
    else:
        # Backward: show error
        error_intensity = min(abs(error) / max(abs(error), 1.0), 1.0)
        if error < 0:
            output_color = f'rgba(255, 0, 0, {0.5 + error_intensity * 0.5})'
        else:
            output_color = f'rgba(0, 255, 0, {0.5 + error_intensity * 0.5})'
        output_text = f'{error:.2f}'
    
    fig.add_trace(go.Scatter(
        x=[output_x],
        y=[output_y],
        mode='markers+text',
        marker=dict(size=50, color=output_color, line=dict(width=3, color='white')),
        text=[output_text],
        textposition='middle center',
        textfont=dict(size=12, color='white', family='Arial Black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Output label
    fig.add_annotation(
        x=output_x,
        y=output_y + 0.5,
        text=f'{target_name[:20]}',
        showarrow=False,
        font=dict(size=10, color='black', family='Arial Black'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=2
    )
    
    # Target value annotation
    fig.add_annotation(
        x=output_x + 1.5,
        y=output_y,
        text=f'Target: {target_value:.2f}',
        showarrow=True,
        arrowhead=2,
        arrowcolor='red',
        font=dict(size=9, color='red'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='red',
        borderwidth=1
    )
    
    # Draw connections (forward pass)
    if phase == "forward":
        edge_color = 'rgba(0, 150, 255, 0.3)'
        for input_node in input_nodes:
            for hidden_node in hidden_nodes:
                fig.add_trace(go.Scatter(
                    x=[input_node[0], hidden_node[0]],
                    y=[input_node[1], hidden_node[1]],
                    mode='lines',
                    line=dict(color=edge_color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        for hidden_node in hidden_nodes:
            fig.add_trace(go.Scatter(
                x=[hidden_node[0], output_x],
                y=[hidden_node[1], output_y],
                mode='lines',
                line=dict(color=edge_color, width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        # Backward pass: show error flow
        edge_color = 'rgba(255, 100, 0, 0.4)'
        # Error flows from output back
        fig.add_trace(go.Scatter(
            x=[output_x, hidden_x],
            y=[output_y, hidden_y_positions[1]],
            mode='lines',
            line=dict(color=edge_color, width=3, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        for i, hidden_node in enumerate(hidden_nodes):
            fig.add_trace(go.Scatter(
                x=[hidden_x, input_x],
                y=[hidden_node[1], input_y_positions[min(i, len(input_nodes)-1)]],
                mode='lines',
                line=dict(color=edge_color, width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add loss function info with explanation
    phase_text = "🔵 Forward Pass" if phase == "forward" else "🔴 Backpropagation"
    title = f"🧠 Neural Network Training - Epoch {epoch} | {phase_text}"
    
    # Loss info box with explanation
    if phase == "backward":
        loss_text = f"<b>Loss ({loss_type}): {loss_value:.4f}</b><br>Error: {error:.4f}"
        if loss_type == "Weighted Huber":
            loss_text += "<br><i>Huber: robust to outliers | Weighted: focuses on high values</i>"
    else:
        loss_text = f"<b>Loss ({loss_type}): {loss_value:.4f}</b>"
        if loss_type == "Weighted Huber":
            loss_text += "<br><i>Combines Huber (outlier-resistant) with weighting</i>"
    
    fig.add_annotation(
        x=4,
        y=7.5,
        text=loss_text,
        showarrow=False,
        font=dict(size=10, color='black', family='Arial'),
        bgcolor='rgba(255, 255, 200, 0.95)',
        bordercolor='black',
        borderwidth=2,
        align='left'
    )
    
    # Layer labels
    fig.add_annotation(
        x=input_x,
        y=1,
        text="Input<br>Features",
        showarrow=False,
        font=dict(size=11, color='black', family='Arial Black'),
        bgcolor='rgba(200, 230, 255, 0.9)',
        bordercolor='blue',
        borderwidth=2
    )
    
    fig.add_annotation(
        x=hidden_x,
        y=1,
        text="Hidden<br>Layer",
        showarrow=False,
        font=dict(size=11, color='black', family='Arial Black'),
        bgcolor='rgba(200, 255, 200, 0.9)',
        bordercolor='green',
        borderwidth=2
    )
    
    fig.add_annotation(
        x=output_x,
        y=1,
        text="Output<br>Prediction",
        showarrow=False,
        font=dict(size=11, color='black', family='Arial Black'),
        bgcolor='rgba(255, 230, 200, 0.9)',
        bordercolor='orange',
        borderwidth=2
    )
    
    # Add Weighted Huber explanation box
    if loss_type == "Weighted Huber":
        explanation_y = 0.7
        huber_explanation = (
            "<b>Weighted Huber Loss Explained:</b><br>"
            "• <b>Huber Loss:</b> Like MSE for small errors, like MAE for large errors<br>"
            "• <b>Weighted:</b> Gives more importance to high target values<br>"
            "• <b>Why:</b> More robust to outliers than MSE, focuses on important predictions"
        )
        fig.add_annotation(
            x=4.5,
            y=explanation_y,
            text=huber_explanation,
            showarrow=False,
            font=dict(size=9, color='black', family='Arial'),
            bgcolor='rgba(230, 240, 255, 0.95)',
            bordercolor='blue',
            borderwidth=2,
            align='left'
        )
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 8]),
        plot_bgcolor='white',
        height=500,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig
