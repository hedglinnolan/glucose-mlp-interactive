"""
MLP Architecture Visualization Module
Visualizes neural network architecture with forward and backpropagation animations.
"""
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
import streamlit as st


def extract_architecture(model: nn.Module, feature_names: List[str]) -> Dict:
    """
    Extract architecture information from a PyTorch model.
    
    Args:
        model: PyTorch neural network model
        feature_names: List of input feature names
        
    Returns:
        Dictionary with architecture details:
        {
            'layer_sizes': [input_dim, hidden1, hidden2, ..., output_dim],
            'layer_types': ['input', 'linear', 'linear', ..., 'output'],
            'activation_types': ['', 'relu', 'relu', ..., ''],
            'feature_names': [...],
            'num_layers': int
        }
    """
    layer_sizes = []
    layer_types = []
    activation_types = []
    
    # Extract linear layers in order
    linear_layers = []
    current_activation = ''
    
    # Handle Sequential models
    if isinstance(model, nn.Sequential):
        modules = list(model.children())
    elif hasattr(model, 'layers') and isinstance(model.layers, nn.Sequential):
        modules = list(model.layers.children())
    else:
        # Fallback: iterate through all modules
        modules = []
        for module in model.modules():
            if not isinstance(module, nn.Sequential) and module != model:
                modules.append(module)
    
    # Process modules in order
    for module in modules:
        if isinstance(module, nn.Linear):
            linear_layers.append({
                'in_features': module.in_features,
                'out_features': module.out_features,
                'activation': current_activation
            })
            current_activation = ''
        elif isinstance(module, nn.ReLU):
            if linear_layers:
                linear_layers[-1]['activation'] = 'relu'
            current_activation = 'relu'
        elif isinstance(module, nn.Tanh):
            if linear_layers:
                linear_layers[-1]['activation'] = 'tanh'
            current_activation = 'tanh'
        elif isinstance(module, nn.Sigmoid):
            if linear_layers:
                linear_layers[-1]['activation'] = 'sigmoid'
            current_activation = 'sigmoid'
    
    # If no linear layers found, try direct inspection
    if not linear_layers:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if not layer_sizes:
                    # First layer - input size
                    layer_sizes.append(module.in_features)
                    layer_types.append('input')
                    activation_types.append('')
                layer_sizes.append(module.out_features)
                layer_types.append('linear')
                activation_types.append('')
    
    # Build from linear_layers if we found them
    if linear_layers:
        # Input layer
        layer_sizes.append(linear_layers[0]['in_features'])
        layer_types.append('input')
        activation_types.append('')
        
        # Hidden and output layers
        for i, layer in enumerate(linear_layers):
            layer_sizes.append(layer['out_features'])
            if i == len(linear_layers) - 1:
                layer_types.append('output')
            else:
                layer_types.append('linear')
            activation_types.append(layer.get('activation', ''))
    
    # Fallback: if still no layers, try to get from first Linear layer
    if not layer_sizes:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_sizes.append(module.in_features)
                layer_types.append('input')
                activation_types.append('')
                break
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_sizes.append(module.out_features)
                if len(layer_sizes) == 2:  # First output layer
                    layer_types.append('linear')
                else:
                    layer_types.append('output')
                activation_types.append('')
    
    # Ensure we have feature names
    if feature_names and len(feature_names) > 0:
        feature_list = feature_names
    else:
        input_size = layer_sizes[0] if layer_sizes else 1
        feature_list = [f'Feature {i+1}' for i in range(input_size)]
    
    # Mark output layer
    if layer_types and len(layer_types) > 1:
        layer_types[-1] = 'output'
    
    return {
        'layer_sizes': layer_sizes,
        'layer_types': layer_types,
        'activation_types': activation_types,
        'feature_names': feature_list,
        'num_layers': len(layer_sizes)
    }


def calculate_node_positions(architecture: Dict, width: float = 10.0, 
                            height: float = 8.0) -> Tuple[List[Tuple[float, float]], Dict]:
    """
    Calculate positions for nodes in the network visualization.
    
    Args:
        architecture: Architecture dictionary from extract_architecture
        width: Total width of visualization
        height: Total height of visualization
        
    Returns:
        Tuple of (node_positions, layer_info)
        node_positions: List of (x, y) positions for each node
        layer_info: Dict mapping layer index to node indices
    """
    layer_sizes = architecture['layer_sizes']
    num_layers = len(layer_sizes)
    
    # Calculate x positions (layers are evenly spaced)
    x_positions = np.linspace(0, width, num_layers)
    
    node_positions = []
    layer_info = {}
    node_idx = 0
    
    for layer_idx, layer_size in enumerate(layer_sizes):
        # Calculate y positions for nodes in this layer
        # Center them vertically
        y_spacing = height / max(layer_size, 1)
        y_start = (height - (layer_size - 1) * y_spacing) / 2
        
        layer_node_indices = []
        for node_in_layer in range(layer_size):
            y_pos = y_start + node_in_layer * y_spacing
            node_positions.append((x_positions[layer_idx], y_pos))
            layer_node_indices.append(node_idx)
            node_idx += 1
        
        layer_info[layer_idx] = layer_node_indices
    
    return node_positions, layer_info


def create_network_graph(architecture: Dict, node_positions: List[Tuple[float, float]], 
                         layer_info: Dict, highlight_nodes: Optional[List[int]] = None,
                         highlight_edges: Optional[List[Tuple[int, int]]] = None,
                         node_values: Optional[List[float]] = None,
                         edge_weights: Optional[Dict[Tuple[int, int], float]] = None) -> go.Figure:
    """
    Create a Plotly figure showing the neural network architecture.
    
    Args:
        architecture: Architecture dictionary
        node_positions: List of (x, y) positions for nodes
        layer_info: Dict mapping layer index to node indices
        highlight_nodes: List of node indices to highlight
        highlight_edges: List of (from_node, to_node) tuples to highlight
        node_values: Optional values to display on nodes (for activations)
        edge_weights: Optional dict mapping (from, to) -> weight value
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    layer_sizes = architecture['layer_sizes']
    num_layers = len(layer_sizes)
    
    # Draw edges (connections between layers)
    for layer_idx in range(num_layers - 1):
        from_nodes = layer_info[layer_idx]
        to_nodes = layer_info[layer_idx + 1]
        
        for from_node in from_nodes:
            for to_node in to_nodes:
                from_pos = node_positions[from_node]
                to_pos = node_positions[to_node]
                
                # Determine edge color and width
                is_highlighted = highlight_edges and (from_node, to_node) in highlight_edges
                
                if edge_weights and (from_node, to_node) in edge_weights:
                    weight = edge_weights[(from_node, to_node)]
                    # Color by weight: red for negative, blue for positive
                    edge_color = f'rgba(255, 0, 0, {min(abs(weight) * 0.5, 0.8)})' if weight < 0 else f'rgba(0, 0, 255, {min(abs(weight) * 0.5, 0.8)})'
                    edge_width = max(0.5, abs(weight) * 2)
                else:
                    edge_color = 'rgba(200, 200, 200, 0.3)' if not is_highlighted else 'rgba(255, 165, 0, 0.8)'
                    edge_width = 0.5 if not is_highlighted else 2.0
                
                fig.add_trace(go.Scatter(
                    x=[from_pos[0], to_pos[0]],
                    y=[from_pos[1], to_pos[1]],
                    mode='lines',
                    line=dict(color=edge_color, width=edge_width),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    node_texts = []
    
    for node_idx, (x, y) in enumerate(node_positions):
        # Determine if node should be highlighted
        is_highlighted = highlight_nodes and node_idx in highlight_nodes
        
        # Get node value if available
        if node_values and node_idx < len(node_values):
            value = node_values[node_idx]
            node_texts.append(f'{value:.2f}')
            # Color by activation value
            intensity = min(abs(value) * 0.1, 1.0)
            if value > 0:
                node_color = f'rgba(0, 100, 255, {intensity})'
            else:
                node_color = f'rgba(255, 100, 0, {intensity})'
        else:
            node_texts.append('')
            node_color = 'rgba(100, 100, 100, 0.6)' if not is_highlighted else 'rgba(255, 165, 0, 0.9)'
        
        node_colors.append(node_color)
        node_sizes.append(15 if not is_highlighted else 25)
    
    # Add node scatter plot
    fig.add_trace(go.Scatter(
        x=[pos[0] for pos in node_positions],
        y=[pos[1] for pos in node_positions],
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=node_texts,
        textposition='middle center',
        textfont=dict(size=8, color='white'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add layer labels
    for layer_idx in range(num_layers):
        x_pos = node_positions[layer_info[layer_idx][0]][0]
        layer_type = architecture['layer_types'][layer_idx]
        layer_size = layer_sizes[layer_idx]
        
        if layer_idx == 0:
            label = f'Input\n({layer_size})'
        elif layer_idx == num_layers - 1:
            label = f'Output\n({layer_size})'
        else:
            label = f'Hidden {layer_idx}\n({layer_size})'
        
        fig.add_annotation(
            x=x_pos,
            y=-0.5,
            text=label,
            showarrow=False,
            font=dict(size=12, color='black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    
    # Update layout
    fig.update_layout(
        title='Neural Network Architecture',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 11]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 9]),
        plot_bgcolor='white',
        height=600,
        margin=dict(l=20, r=20, t=50, b=50)
    )
    
    return fig


def simulate_forward_pass(architecture: Dict, sample_input: np.ndarray, 
                         model: Optional[nn.Module] = None) -> List[np.ndarray]:
    """
    Simulate forward pass through the network.
    
    Args:
        architecture: Architecture dictionary
        sample_input: Single input sample (1D array)
        model: Optional model to get actual activations
        
    Returns:
        List of activations for each layer
    """
    activations = [sample_input.copy()]
    
    if model is not None:
        # Use actual model to get real activations
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(sample_input).unsqueeze(0)
            
            # Hook to capture activations
            layer_activations = []
            
            def hook_fn(module, input, output):
                layer_activations.append(output.squeeze(0).cpu().numpy())
            
            hooks = []
            for module in model.modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.ReLU):
                    hooks.append(module.register_forward_hook(hook_fn))
            
            _ = model(x)
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            
            # Combine activations appropriately
            # This is simplified - in practice, you'd need to handle ReLU separately
            activations = [sample_input]
            idx = 0
            for i, size in enumerate(architecture['layer_sizes'][1:], 1):
                if idx < len(layer_activations):
                    # Take first size elements
                    act = layer_activations[idx][:size] if len(layer_activations[idx]) >= size else layer_activations[idx]
                    activations.append(act)
                    idx += 1
                else:
                    # Fallback to random values for visualization
                    activations.append(np.random.randn(size) * 0.1)
    else:
        # Simulate forward pass with random activations
        current = sample_input
        for layer_size in architecture['layer_sizes'][1:]:
            # Simple simulation: random activations
            current = np.random.randn(layer_size) * 0.5 + np.mean(current)
            # Apply ReLU-like behavior
            current = np.maximum(current, 0)
            activations.append(current)
    
    return activations


def simulate_backward_pass(architecture: Dict, activations: List[np.ndarray],
                          target: float, prediction: float) -> List[np.ndarray]:
    """
    Simulate backward pass (gradient flow) through the network.
    
    Args:
        architecture: Architecture dictionary
        activations: Activations from forward pass
        target: Target value
        prediction: Predicted value
        
    Returns:
        List of gradients for each layer (backward order)
    """
    # Start with output gradient (error)
    error = prediction - target
    gradients = [np.array([error])]
    
    # Simulate gradient flow backward
    # In reality, gradients would be computed via chain rule
    # Here we simulate for visualization purposes
    for i in range(len(activations) - 2, -1, -1):
        # Gradient flows backward, typically getting smaller
        prev_grad = gradients[0]
        # Simulate gradient propagation
        grad = np.random.randn(len(activations[i])) * np.mean(np.abs(prev_grad)) * 0.3
        # Apply some structure based on activations
        grad = grad * (activations[i] > 0).astype(float)  # ReLU gradient
        gradients.insert(0, grad)
    
    return gradients


def create_animated_forward_pass(architecture: Dict, node_positions: List[Tuple[float, float]],
                                 layer_info: Dict, sample_input: np.ndarray,
                                 model: Optional[nn.Module] = None, 
                                 num_frames: int = 30) -> go.Figure:
    """
    Create a single Plotly figure with animation frames for forward pass.
    Uses Plotly's built-in animation support.
    """
    activations = simulate_forward_pass(architecture, sample_input, model)
    layer_sizes = architecture['layer_sizes']
    num_layers = len(layer_sizes)
    
    # Prepare all frame data
    all_frames_data = []
    for frame_idx in range(num_frames):
        progress = (frame_idx + 1) / num_frames
        num_active_layers = int(np.ceil(progress * len(activations)))
        
        node_values = []
        highlight_nodes = []
        highlight_edges = []
        
        node_idx = 0
        for layer_idx, layer_size in enumerate(layer_sizes):
            if layer_idx < num_active_layers:
                layer_activation = activations[layer_idx]
                for i in range(layer_size):
                    if i < len(layer_activation):
                        node_values.append(layer_activation[i])
                    else:
                        node_values.append(0.0)
                    highlight_nodes.append(node_idx)
                    
                    if layer_idx > 0:
                        prev_layer_nodes = layer_info[layer_idx - 1]
                        for prev_node in prev_layer_nodes:
                            highlight_edges.append((prev_node, node_idx))
                    
                    node_idx += 1
            else:
                for _ in range(layer_size):
                    node_values.append(0.0)
                    node_idx += 1
        
        all_frames_data.append({
            'node_values': node_values,
            'highlight_nodes': highlight_nodes,
            'highlight_edges': highlight_edges
        })
    
    # Create base figure (first frame)
    base_data = all_frames_data[0]
    fig = create_network_graph(
        architecture, node_positions, layer_info,
        highlight_nodes=base_data['highlight_nodes'],
        highlight_edges=base_data['highlight_edges'],
        node_values=base_data['node_values']
    )
    
    # Create frames
    frames = []
    for frame_idx, frame_data in enumerate(all_frames_data):
        # Create traces for this frame
        frame_traces = []
        
        # Edges
        for layer_idx in range(num_layers - 1):
            from_nodes = layer_info[layer_idx]
            to_nodes = layer_info[layer_idx + 1]
            
            for from_node in from_nodes:
                for to_node in to_nodes:
                    from_pos = node_positions[from_node]
                    to_pos = node_positions[to_node]
                    
                    is_highlighted = (from_node, to_node) in frame_data['highlight_edges']
                    edge_color = 'rgba(255, 165, 0, 0.8)' if is_highlighted else 'rgba(200, 200, 200, 0.3)'
                    edge_width = 2.0 if is_highlighted else 0.5
                    
                    frame_traces.append(go.Scatter(
                        x=[from_pos[0], to_pos[0]],
                        y=[from_pos[1], to_pos[1]],
                        mode='lines',
                        line=dict(color=edge_color, width=edge_width),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Nodes
        node_colors = []
        node_sizes = []
        node_texts = []
        
        for node_idx, (x, y) in enumerate(node_positions):
            is_highlighted = node_idx in frame_data['highlight_nodes']
            
            if node_idx < len(frame_data['node_values']):
                value = frame_data['node_values'][node_idx]
                node_texts.append(f'{value:.2f}')
                intensity = min(abs(value) * 0.1, 1.0)
                if value > 0:
                    node_color = f'rgba(0, 100, 255, {intensity})'
                else:
                    node_color = f'rgba(255, 100, 0, {intensity})'
            else:
                node_texts.append('')
                node_color = 'rgba(100, 100, 100, 0.6)' if not is_highlighted else 'rgba(255, 165, 0, 0.9)'
            
            node_colors.append(node_color)
            node_sizes.append(25 if is_highlighted else 15)
        
        frame_traces.append(go.Scatter(
            x=[pos[0] for pos in node_positions],
            y=[pos[1] for pos in node_positions],
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_texts,
            textposition='middle center',
            textfont=dict(size=8, color='white'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        frames.append(go.Frame(data=frame_traces, name=str(frame_idx)))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0,
            'xanchor': 'left',
            'yanchor': 'bottom'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Frame:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 50, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 50, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 50}
                    }],
                    'label': f'{i+1}',
                    'method': 'animate'
                }
                for i, f in enumerate(frames)
            ]
        }]
    )
    
    return fig


def animate_forward_pass(architecture: Dict, node_positions: List[Tuple[float, float]],
                        layer_info: Dict, sample_input: np.ndarray,
                        model: Optional[nn.Module] = None, 
                        num_frames: int = 20) -> List[go.Figure]:
    """
    Create animation frames for forward pass.
    
    Args:
        architecture: Architecture dictionary
        node_positions: Node positions
        layer_info: Layer information
        sample_input: Input sample
        model: Optional model for real activations
        num_frames: Number of animation frames
        
    Returns:
        List of Plotly figures (one per frame)
    """
    activations = simulate_forward_pass(architecture, sample_input, model)
    frames = []
    
    # Create frames showing progressive activation
    for frame_idx in range(num_frames):
        progress = (frame_idx + 1) / num_frames
        
        # Determine which layers are active at this frame
        num_active_layers = int(np.ceil(progress * len(activations)))
        
        # Build node values and highlights
        node_values = []
        highlight_nodes = []
        highlight_edges = []
        
        node_idx = 0
        for layer_idx, layer_size in enumerate(architecture['layer_sizes']):
            if layer_idx < num_active_layers:
                # Layer is active
                layer_activation = activations[layer_idx]
                for i in range(layer_size):
                    if i < len(layer_activation):
                        node_values.append(layer_activation[i])
                    else:
                        node_values.append(0.0)
                    highlight_nodes.append(node_idx)
                    
                    # Highlight edges from previous layer
                    if layer_idx > 0:
                        prev_layer_nodes = layer_info[layer_idx - 1]
                        for prev_node in prev_layer_nodes:
                            highlight_edges.append((prev_node, node_idx))
                    
                    node_idx += 1
            else:
                # Layer not yet active
                for _ in range(layer_size):
                    node_values.append(0.0)
                    node_idx += 1
        
        # Create figure for this frame
        fig = create_network_graph(
            architecture, node_positions, layer_info,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            node_values=node_values
        )
        frames.append(fig)
    
    return frames


def create_animated_backward_pass(architecture: Dict, node_positions: List[Tuple[float, float]],
                                  layer_info: Dict, activations: List[np.ndarray],
                                  target: float, prediction: float,
                                  num_frames: int = 30) -> go.Figure:
    """
    Create a single Plotly figure with animation frames for backward pass.
    Uses Plotly's built-in animation support.
    """
    gradients = simulate_backward_pass(architecture, activations, target, prediction)
    layer_sizes = architecture['layer_sizes']
    num_layers = len(layer_sizes)
    
    # Prepare all frame data
    all_frames_data = []
    for frame_idx in range(num_frames):
        progress = (frame_idx + 1) / num_frames
        num_gradient_layers = int(np.ceil(progress * len(gradients)))
        start_layer = len(layer_sizes) - num_gradient_layers
        
        node_values = []
        highlight_nodes = []
        highlight_edges = []
        
        node_idx = 0
        for layer_idx, layer_size in enumerate(layer_sizes):
            if layer_idx >= start_layer:
                grad_idx = layer_idx - start_layer
                if grad_idx < len(gradients):
                    layer_grad = gradients[grad_idx]
                    for i in range(layer_size):
                        if i < len(layer_grad):
                            node_values.append(layer_grad[i])
                        else:
                            node_values.append(0.0)
                        highlight_nodes.append(node_idx)
                        
                        if layer_idx < len(layer_sizes) - 1:
                            next_layer_nodes = layer_info[layer_idx + 1]
                            for next_node in next_layer_nodes:
                                highlight_edges.append((node_idx, next_node))
                        
                        node_idx += 1
                else:
                    for _ in range(layer_size):
                        node_values.append(0.0)
                        node_idx += 1
            else:
                for _ in range(layer_size):
                    node_values.append(0.0)
                    node_idx += 1
        
        all_frames_data.append({
            'node_values': node_values,
            'highlight_nodes': highlight_nodes,
            'highlight_edges': highlight_edges
        })
    
    # Create base figure (first frame)
    base_data = all_frames_data[0]
    fig = create_network_graph(
        architecture, node_positions, layer_info,
        highlight_nodes=base_data['highlight_nodes'],
        highlight_edges=base_data['highlight_edges'],
        node_values=base_data['node_values']
    )
    
    # Create frames
    frames = []
    for frame_idx, frame_data in enumerate(all_frames_data):
        # Create traces for this frame
        frame_traces = []
        
        # Edges
        for layer_idx in range(num_layers - 1):
            from_nodes = layer_info[layer_idx]
            to_nodes = layer_info[layer_idx + 1]
            
            for from_node in from_nodes:
                for to_node in to_nodes:
                    from_pos = node_positions[from_node]
                    to_pos = node_positions[to_node]
                    
                    is_highlighted = (from_node, to_node) in frame_data['highlight_edges']
                    edge_color = 'rgba(255, 0, 0, 0.8)' if is_highlighted else 'rgba(200, 200, 200, 0.3)'
                    edge_width = 2.0 if is_highlighted else 0.5
                    
                    frame_traces.append(go.Scatter(
                        x=[from_pos[0], to_pos[0]],
                        y=[from_pos[1], to_pos[1]],
                        mode='lines',
                        line=dict(color=edge_color, width=edge_width),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Nodes
        node_colors = []
        node_sizes = []
        node_texts = []
        
        for node_idx, (x, y) in enumerate(node_positions):
            is_highlighted = node_idx in frame_data['highlight_nodes']
            
            if node_idx < len(frame_data['node_values']):
                value = frame_data['node_values'][node_idx]
                node_texts.append(f'{value:.3f}')
                intensity = min(abs(value) * 2.0, 1.0)
                if value < 0:
                    node_color = f'rgba(255, 0, 0, {intensity})'  # Red for negative gradients
                else:
                    node_color = f'rgba(0, 255, 0, {intensity})'  # Green for positive gradients
            else:
                node_texts.append('')
                node_color = 'rgba(100, 100, 100, 0.6)' if not is_highlighted else 'rgba(255, 0, 0, 0.9)'
            
            node_colors.append(node_color)
            node_sizes.append(25 if is_highlighted else 15)
        
        frame_traces.append(go.Scatter(
            x=[pos[0] for pos in node_positions],
            y=[pos[1] for pos in node_positions],
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=node_texts,
            textposition='middle center',
            textfont=dict(size=8, color='white'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        frames.append(go.Frame(data=frame_traces, name=str(frame_idx)))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 100, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 50}
                    }]
                },
                {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 0,
            'xanchor': 'left',
            'yanchor': 'bottom'
        }],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Frame:',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 50, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [[f.name], {
                        'frame': {'duration': 50, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 50}
                    }],
                    'label': f'{i+1}',
                    'method': 'animate'
                }
                for i, f in enumerate(frames)
            ]
        }]
    )
    
    return fig


def animate_backward_pass(architecture: Dict, node_positions: List[Tuple[float, float]],
                         layer_info: Dict, activations: List[np.ndarray],
                         target: float, prediction: float,
                         num_frames: int = 20) -> List[go.Figure]:
    """
    Create animation frames for backward pass (backpropagation).
    
    Args:
        architecture: Architecture dictionary
        node_positions: Node positions
        layer_info: Layer information
        activations: Activations from forward pass
        target: Target value
        prediction: Predicted value
        num_frames: Number of animation frames
        
    Returns:
        List of Plotly figures (one per frame)
    """
    gradients = simulate_backward_pass(architecture, activations, target, prediction)
    frames = []
    
    # Create frames showing progressive gradient flow (backward)
    for frame_idx in range(num_frames):
        progress = (frame_idx + 1) / num_frames
        
        # Determine which layers have gradients at this frame
        # Gradients flow backward, so we count from output
        num_gradient_layers = int(np.ceil(progress * len(gradients)))
        start_layer = len(architecture['layer_sizes']) - num_gradient_layers
        
        # Build node values and highlights
        node_values = []
        highlight_nodes = []
        highlight_edges = []
        
        node_idx = 0
        for layer_idx, layer_size in enumerate(architecture['layer_sizes']):
            if layer_idx >= start_layer:
                # Layer has gradient
                grad_idx = layer_idx - start_layer
                if grad_idx < len(gradients):
                    layer_grad = gradients[grad_idx]
                    for i in range(layer_size):
                        if i < len(layer_grad):
                            node_values.append(layer_grad[i])
                        else:
                            node_values.append(0.0)
                        highlight_nodes.append(node_idx)
                        
                        # Highlight edges to next layer (gradients flow backward)
                        if layer_idx < len(architecture['layer_sizes']) - 1:
                            next_layer_nodes = layer_info[layer_idx + 1]
                            for next_node in next_layer_nodes:
                                highlight_edges.append((node_idx, next_node))
                        
                        node_idx += 1
                else:
                    for _ in range(layer_size):
                        node_values.append(0.0)
                        node_idx += 1
            else:
                # Layer doesn't have gradient yet
                for _ in range(layer_size):
                    node_values.append(0.0)
                    node_idx += 1
        
        # Create figure for this frame
        fig = create_network_graph(
            architecture, node_positions, layer_info,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            node_values=node_values
        )
        frames.append(fig)
    
    return frames
