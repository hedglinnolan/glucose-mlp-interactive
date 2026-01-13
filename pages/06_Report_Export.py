"""
Page 06: Report Export
Generate and download comprehensive modeling report.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import io
import zipfile
import plotly.graph_objects as go
import plotly.express as px

from utils.session_state import (
    init_session_state, get_data, get_preprocessing_pipeline,
    DataConfig, SplitConfig, ModelConfig
)

init_session_state()

st.set_page_config(page_title="Report Export", page_icon="📄", layout="wide")
st.title("📄 Report Export")

# Check prerequisites
df = get_data()
if df is None:
    st.warning("⚠️ Please complete the modeling workflow first")
    st.stop()

data_config: DataConfig = st.session_state.get('data_config')
split_config: SplitConfig = st.session_state.get('split_config')
model_config: ModelConfig = st.session_state.get('model_config')
pipeline = get_preprocessing_pipeline()
trained_models = st.session_state.get('trained_models', {})
model_results = st.session_state.get('model_results', {})
data_audit = st.session_state.get('data_audit')

if not trained_models:
    st.warning("⚠️ Please train models first")
    st.stop()

# Generate report
def generate_report() -> str:
    """Generate markdown report."""
    report_lines = []
    
    # Header
    report_lines.append("# Modeling Lab Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset Summary
    report_lines.append("## Dataset Summary")
    report_lines.append(f"- **Rows:** {len(df):,}")
    report_lines.append(f"- **Columns:** {len(df.columns)}")
    report_lines.append(f"- **Target:** {data_config.target_col}")
    report_lines.append(f"- **Features:** {len(data_config.feature_cols)}")
    report_lines.append(f"- **Task Type:** {data_config.task_type}")
    report_lines.append("")
    
    # Data Audit Summary
    if data_audit:
        report_lines.append("## Data Audit Summary")
        if data_audit.get('missing'):
            report_lines.append(f"- **Missing Values:** {len(data_audit['missing'])} columns with missing data")
        if data_audit.get('duplicates', 0) > 0:
            report_lines.append(f"- **Duplicates:** {data_audit['duplicates']} duplicate rows")
        if data_audit.get('constant_cols'):
            report_lines.append(f"- **Constant Columns:** {len(data_audit['constant_cols'])}")
        report_lines.append("")
    
    # Split Strategy
    report_lines.append("## Split Strategy")
    report_lines.append(f"- **Train:** {split_config.train_size*100:.1f}%")
    report_lines.append(f"- **Validation:** {split_config.val_size*100:.1f}%")
    report_lines.append(f"- **Test:** {split_config.test_size*100:.1f}%")
    report_lines.append(f"- **Random State:** {split_config.random_state}")
    report_lines.append(f"- **Global Random Seed:** {st.session_state.get('random_seed', 42)}")
    if split_config.use_time_split:
        report_lines.append(f"- **Split Type:** Time-based (using {split_config.datetime_col})")
    else:
        report_lines.append(f"- **Split Type:** Random")
    if split_config.stratify:
        report_lines.append(f"- **Stratification:** Enabled (for classification)")
    report_lines.append("")
    
    # Preprocessing Recipe
    if pipeline:
        report_lines.append("## Preprocessing Pipeline")
        from ml.pipeline import get_pipeline_recipe
        recipe = get_pipeline_recipe(pipeline)
        report_lines.append("```")
        report_lines.append(recipe)
        report_lines.append("```")
        report_lines.append("")
    
    # Model Hyperparameters
    report_lines.append("## Model Hyperparameters")
    
    if 'nn' in trained_models:
        report_lines.append("### Neural Network")
        report_lines.append(f"- Epochs: {model_config.nn_epochs}")
        report_lines.append(f"- Batch Size: {model_config.nn_batch_size}")
        report_lines.append(f"- Learning Rate: {model_config.nn_lr}")
        report_lines.append(f"- Weight Decay: {model_config.nn_weight_decay}")
        report_lines.append(f"- Early Stopping Patience: {model_config.nn_patience}")
        report_lines.append(f"- Dropout: {model_config.nn_dropout}")
        report_lines.append("")
    
    if 'rf' in trained_models:
        report_lines.append("### Random Forest")
        report_lines.append(f"- N Estimators: {model_config.rf_n_estimators}")
        report_lines.append(f"- Max Depth: {model_config.rf_max_depth or 'None'}")
        report_lines.append(f"- Min Samples Leaf: {model_config.rf_min_samples_leaf}")
        report_lines.append("")
    
    if 'huber' in trained_models:
        report_lines.append("### GLM (Huber)")
        report_lines.append(f"- Epsilon: {model_config.huber_epsilon}")
        report_lines.append(f"- Alpha: {model_config.huber_alpha}")
        report_lines.append("")
    
    # Model Performance
    report_lines.append("## Model Performance")
    report_lines.append("")
    
    # Metrics table
    comparison_data = []
    for name, results in model_results.items():
        row = {'Model': name.upper()}
        row.update(results['metrics'])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    # Generate markdown table (with fallback if tabulate missing)
    try:
        report_lines.append(comparison_df.to_markdown(index=False))
    except ImportError:
        # Fallback: create markdown table manually
        headers = '| ' + ' | '.join(comparison_df.columns) + ' |'
        separators = '| ' + ' | '.join(['---'] * len(comparison_df.columns)) + ' |'
        report_lines.append(headers)
        report_lines.append(separators)
        for _, row in comparison_df.iterrows():
            values = '| ' + ' | '.join([str(v) for v in row.values]) + ' |'
            report_lines.append(values)
    report_lines.append("")
    
    # Best Model
    if data_config.task_type == 'regression':
        best_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
        best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'RMSE']
    else:
        best_model = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        best_acc = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Accuracy']
    
    report_lines.append(f"**Best Model:** {best_model}")
    report_lines.append("")
    
    # Cross-Validation Results (if available)
    if st.session_state.get('use_cv'):
        report_lines.append("## Cross-Validation Results")
        for name, results in model_results.items():
            if results.get('cv_results'):
                cv = results['cv_results']
                report_lines.append(f"### {name.upper()}")
                report_lines.append(f"- Mean Score: {cv['mean']:.4f}")
                report_lines.append(f"- Std Score: {cv['std']:.4f}")
                report_lines.append("")
    
    # Feature Importance (if available)
    if st.session_state.get('permutation_importance'):
        report_lines.append("## Feature Importance (Permutation)")
        for name, perm_data in st.session_state.permutation_importance.items():
            report_lines.append(f"### {name.upper()}")
            importance_df = pd.DataFrame({
                'Feature': perm_data['feature_names'],
                'Importance': perm_data['importances_mean']
            }).sort_values('Importance', ascending=False)
            # Generate markdown table (with fallback if tabulate missing)
            try:
                report_lines.append(importance_df.head(10).to_markdown(index=False))
            except ImportError:
                # Fallback: create markdown table manually
                headers = '| ' + ' | '.join(importance_df.head(10).columns) + ' |'
                separators = '| ' + ' | '.join(['---'] * len(importance_df.head(10).columns)) + ' |'
                report_lines.append(headers)
                report_lines.append(separators)
                for _, row in importance_df.head(10).iterrows():
                    values = '| ' + ' | '.join([str(v) for v in row.values]) + ' |'
                    report_lines.append(values)
            report_lines.append("")
    
    # Notes
    report_lines.append("## Notes")
    report_lines.append("- This report was generated automatically by the Modeling Lab")
    report_lines.append("- All models were evaluated on the same test set")
    report_lines.append("- Preprocessing was applied consistently across all models")
    
    return "\n".join(report_lines)

# Generate and display report
report_text = generate_report()

st.header("📋 Generated Report")
st.markdown(report_text)

# Download buttons
# Helper function to save plotly figures as images
def save_plotly_fig(fig, filename: str) -> Optional[bytes]:
    """Save plotly figure as PNG bytes."""
    try:
        return fig.to_image(format="png", width=1200, height=800)
    except Exception:
        # Fallback: try with kaleido
        try:
            return fig.to_image(format="png", width=1200, height=800, engine="kaleido")
        except Exception:
            return None

st.header("💾 Download")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📄 Download Report (Markdown)",
        data=report_text,
        file_name=f"modeling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

with col2:
    # Create zip with report and data
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("report.md", report_text)
        
        # Add metrics CSV
        comparison_data = []
        for name, results in model_results.items():
            row = {'Model': name.upper()}
            row.update(results['metrics'])
            comparison_data.append(row)
        comparison_df = pd.DataFrame(comparison_data)
        zip_file.writestr("metrics.csv", comparison_df.to_csv(index=False))
        
        # Add predictions CSV
        for name, results in model_results.items():
            pred_df = pd.DataFrame({
                'Actual': results['y_test'],
                'Predicted': results['y_test_pred']
            })
            zip_file.writestr(f"predictions_{name}.csv", pred_df.to_csv(index=False))
            
            # Save prediction plot
            if data_config.task_type == 'regression':
                fig = px.scatter(
                    x=results['y_test'], y=results['y_test_pred'],
                    labels={'x': 'Actual', 'y': 'Predicted'},
                    title=f"{name.upper()} - Predictions vs Actual"
                )
                fig.add_trace(go.Scatter(
                    x=[min(results['y_test']), max(results['y_test'])],
                    y=[min(results['y_test']), max(results['y_test'])],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
                plot_bytes = save_plotly_fig(fig, f"plot_{name}_predictions.png")
                if plot_bytes:
                    zip_file.writestr(f"plot_{name}_predictions.png", plot_bytes)
        
        # Add learning curves if available
        for name, model_wrapper in trained_models.items():
            if name == 'nn' and hasattr(model_wrapper, 'get_training_history'):
                try:
                    history = model_wrapper.get_training_history()
                    if history and 'train_loss' in history:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=history['train_loss'], name='Train Loss', mode='lines'))
                        if 'val_loss' in history:
                            fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines'))
                        fig.update_layout(title=f"{name.upper()} - Learning Curves", xaxis_title="Epoch", yaxis_title="Loss")
                        plot_bytes = save_plotly_fig(fig, f"plot_{name}_learning_curves.png")
                        if plot_bytes:
                            zip_file.writestr(f"plot_{name}_learning_curves.png", plot_bytes)
                except Exception:
                    pass
        
        # Add feature importance if available
        perm_importance = st.session_state.get('permutation_importance', {})
        for name, perm_data in perm_importance.items():
            try:
                importance_df = pd.DataFrame({
                    'Feature': perm_data['feature_names'],
                    'Importance': perm_data['importances_mean']
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"{name.upper()} - Feature Importance"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                plot_bytes = save_plotly_fig(fig, f"plot_{name}_importance.png")
                if plot_bytes:
                    zip_file.writestr(f"plot_{name}_importance.png", plot_bytes)
            except Exception:
                pass
    
    st.download_button(
        label="📦 Download Complete Package (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"modeling_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )

st.success("✅ Report generated successfully!")

# State Debug (Advanced)
with st.expander("🔧 Advanced / State Debug", expanded=False):
    st.markdown("**Current State:**")
    st.write(f"• Data shape: {df.shape if df is not None else 'None'}")
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")
    st.write(f"• Trained models: {len(trained_models)}")
    st.write(f"• Report data: {'Available' if st.session_state.get('report_data') else 'Not generated'}")
