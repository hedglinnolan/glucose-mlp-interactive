"""
Page 06: Report Export
Generate and download comprehensive modeling report with trained artifacts.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
import io
import zipfile
import json
import plotly.graph_objects as go
import plotly.express as px
import logging

from utils.session_state import (
    init_session_state, get_data, get_preprocessing_pipeline,
    DataConfig, SplitConfig, ModelConfig
)
from utils.storyline import render_progress_indicator, get_insights_by_category
from ml.model_registry import get_registry

logger = logging.getLogger(__name__)

init_session_state()

st.set_page_config(page_title="Report Export", page_icon="📄", layout="wide")
st.title("📄 Report Export")

# Progress indicator
render_progress_indicator("06_Report_Export")

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
profile = st.session_state.get('dataset_profile')
coach_output = st.session_state.get('coach_output')

if not trained_models:
    st.warning("⚠️ Please train models first")
    st.stop()

# Custom CSS for better report aesthetics
st.markdown("""
<style>
.report-section {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    background: #fafafa;
}
.report-section h3 {
    margin-top: 0;
    color: #333;
    border-bottom: 2px solid #1e88e5;
    padding-bottom: 0.5rem;
}
.metric-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e88e5;
}
.metric-label {
    font-size: 0.85rem;
    color: #666;
}
.coef-table {
    font-size: 0.9rem;
}
.model-detail-section {
    background: #f8f9fa;
    border-left: 4px solid #1e88e5;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


def get_git_info() -> Dict[str, str]:
    """Get git commit hash and branch if available."""
    try:
        import subprocess
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()[:8]
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        return {'commit': commit_hash, 'branch': branch}
    except:
        return {'commit': 'unknown', 'branch': 'unknown'}


def generate_metadata() -> Dict[str, Any]:
    """Generate comprehensive metadata for export."""
    git_info = get_git_info()
    
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'app_version': '1.0.0',  # You could read this from a version file
        'git_commit': git_info['commit'],
        'git_branch': git_info['branch'],
        'random_seed': st.session_state.get('random_seed', 42),
        'dataset': {
            'n_rows': len(df),
            'n_features': len(data_config.feature_cols),
            'target': data_config.target_col,
            'task_type': data_config.task_type,
            'features': data_config.feature_cols
        },
        'splits': {
            'train_size': split_config.train_size,
            'val_size': split_config.val_size,
            'test_size': split_config.test_size,
            'stratify': split_config.stratify,
            'use_time_split': split_config.use_time_split
        },
        'preprocessing': st.session_state.get('preprocessing_config', {}),
        'models_trained': list(trained_models.keys())
    }
    
    # Add dataset profile summary if available
    if profile:
        metadata['dataset_profile'] = {
            'data_sufficiency': profile.data_sufficiency.value,
            'n_numeric': profile.n_numeric,
            'n_categorical': profile.n_categorical,
            'p_n_ratio': profile.p_n_ratio,
            'total_missing_rate': profile.total_missing_rate,
            'n_features_with_outliers': len(profile.features_with_outliers),
            'warnings': [w.short_message for w in profile.warnings]
        }
    
    return metadata


def extract_model_coefficients(model, model_key: str, feature_names: list) -> Optional[pd.DataFrame]:
    """Extract coefficients from linear models."""
    try:
        # Try to get coefficients from the model
        coef = None
        intercept = None
        
        if hasattr(model, 'model') and hasattr(model.model, 'coef_'):
            coef = model.model.coef_
            intercept = model.model.intercept_ if hasattr(model.model, 'intercept_') else None
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            intercept = model.intercept_ if hasattr(model, 'intercept_') else None
        
        if coef is not None:
            # Handle multi-class case
            if len(coef.shape) > 1:
                coef = coef[0]  # Take first class for binary classification
            
            coef_df = pd.DataFrame({
                'Feature': feature_names[:len(coef)],
                'Coefficient': coef
            }).sort_values('Coefficient', key=abs, ascending=False)
            
            if intercept is not None:
                intercept_row = pd.DataFrame({
                    'Feature': ['(Intercept)'],
                    'Coefficient': [intercept if np.isscalar(intercept) else intercept[0]]
                })
                coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
            
            return coef_df
    except Exception as e:
        logger.debug(f"Could not extract coefficients for {model_key}: {e}")
    
    return None


def generate_report() -> str:
    """Generate markdown report with improved structure and aesthetics."""
    report_lines = []
    
    git_info = get_git_info()
    
    # Header with metadata
    report_lines.append("# 📊 Modeling Lab Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Git Commit:** {git_info['commit']} ({git_info['branch']})")
    report_lines.append(f"**Random Seed:** {st.session_state.get('random_seed', 42)}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## 🎯 Executive Summary")
    report_lines.append("")
    
    # Best model summary
    if data_config.task_type == 'regression':
        best_model = min(model_results.items(), key=lambda x: x[1]['metrics'].get('RMSE', float('inf')))
        report_lines.append(f"**Best Model:** {best_model[0].upper()}")
        report_lines.append(f"**Test RMSE:** {best_model[1]['metrics']['RMSE']:.4f}")
        report_lines.append(f"**Test R²:** {best_model[1]['metrics']['R2']:.4f}")
    else:
        best_model = max(model_results.items(), key=lambda x: x[1]['metrics'].get('F1', x[1]['metrics'].get('Accuracy', 0)))
        report_lines.append(f"**Best Model:** {best_model[0].upper()}")
        report_lines.append(f"**Test Accuracy:** {best_model[1]['metrics'].get('Accuracy', 'N/A'):.4f}")
        if 'F1' in best_model[1]['metrics']:
            report_lines.append(f"**Test F1:** {best_model[1]['metrics']['F1']:.4f}")
    
    report_lines.append("")
    
    # Key findings
    if profile and profile.warnings:
        report_lines.append("**Key Data Warnings:**")
        for w in profile.warnings[:3]:
            report_lines.append(f"- ⚠️ {w.short_message}")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Dataset Summary
    report_lines.append("## 📁 Dataset Summary")
    report_lines.append("")
    report_lines.append("| Property | Value |")
    report_lines.append("|----------|-------|")
    report_lines.append(f"| Rows | {len(df):,} |")
    report_lines.append(f"| Features | {len(data_config.feature_cols)} |")
    report_lines.append(f"| Target | `{data_config.target_col}` |")
    report_lines.append(f"| Task Type | {data_config.task_type.title()} |")
    
    if profile:
        report_lines.append(f"| Numeric Features | {profile.n_numeric} |")
        report_lines.append(f"| Categorical Features | {profile.n_categorical} |")
        report_lines.append(f"| Data Sufficiency | {profile.data_sufficiency.value.title()} |")
        report_lines.append(f"| Feature/Sample Ratio | {profile.p_n_ratio:.4f} |")
    
    report_lines.append("")
    
    # Data Sufficiency Narrative
    if profile and profile.sufficiency_narrative:
        report_lines.append("### Data Sufficiency Analysis")
        report_lines.append("")
        report_lines.append(f"> {profile.sufficiency_narrative}")
        report_lines.append("")
    
    # Task and cohort detection
    task_det = st.session_state.get('task_type_detection')
    cohort_det = st.session_state.get('cohort_structure_detection')
    if task_det or cohort_det:
        report_lines.append("### Automatic Detection")
        report_lines.append("")
        if task_det and task_det.detected:
            report_lines.append(f"- **Task Type:** {task_det.detected} ({task_det.confidence} confidence)")
        if cohort_det and cohort_det.detected:
            report_lines.append(f"- **Cohort Structure:** {cohort_det.detected} ({cohort_det.confidence} confidence)")
            if cohort_det.entity_id_final:
                report_lines.append(f"- **Entity ID Column:** `{cohort_det.entity_id_final}`")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Key EDA Insights
    insights = get_insights_by_category()
    if insights:
        report_lines.append("## 💡 Key EDA Insights")
        report_lines.append("")
        for insight in insights:
            report_lines.append(f"### {insight.get('category', 'General').title()}")
            report_lines.append(f"**Finding:** {insight['finding']}")
            report_lines.append("")
            report_lines.append(f"**Implication:** {insight['implication']}")
            report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Split Strategy
    report_lines.append("## 🔀 Data Split Strategy")
    report_lines.append("")
    report_lines.append("| Split | Percentage | Samples |")
    report_lines.append("|-------|------------|---------|")
    train_n = len(st.session_state.get('X_train', []))
    val_n = len(st.session_state.get('X_val', []))
    test_n = len(st.session_state.get('X_test', []))
    report_lines.append(f"| Train | {split_config.train_size*100:.1f}% | {train_n:,} |")
    report_lines.append(f"| Validation | {split_config.val_size*100:.1f}% | {val_n:,} |")
    report_lines.append(f"| Test | {split_config.test_size*100:.1f}% | {test_n:,} |")
    report_lines.append("")
    
    split_type = "Time-based" if split_config.use_time_split else ("Stratified" if split_config.stratify else "Random")
    report_lines.append(f"**Split Type:** {split_type}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Preprocessing Pipeline
    if pipeline:
        report_lines.append("## ⚙️ Preprocessing Pipeline")
        report_lines.append("")
        from ml.pipeline import get_pipeline_recipe
        recipe = get_pipeline_recipe(pipeline)
        report_lines.append("```")
        report_lines.append(recipe)
        report_lines.append("```")
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Model Performance Comparison
    report_lines.append("## 📈 Model Performance")
    report_lines.append("")
    
    # Metrics table
    comparison_data = []
    for name, results in model_results.items():
        row = {'Model': name.upper()}
        row.update(results['metrics'])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format the table nicely
    report_lines.append("### Performance Metrics (Test Set)")
    report_lines.append("")
    try:
        report_lines.append(comparison_df.to_markdown(index=False, floatfmt='.4f'))
    except:
        headers = '| ' + ' | '.join(comparison_df.columns) + ' |'
        separators = '| ' + ' | '.join(['---'] * len(comparison_df.columns)) + ' |'
        report_lines.append(headers)
        report_lines.append(separators)
        for _, row in comparison_df.iterrows():
            values = '| ' + ' | '.join([f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values]) + ' |'
            report_lines.append(values)
    report_lines.append("")
    
    # Cross-Validation Results
    cv_results_exist = any(r.get('cv_results') for r in model_results.values())
    if cv_results_exist:
        report_lines.append("### Cross-Validation Results")
        report_lines.append("")
        report_lines.append("| Model | Mean Score | Std Dev |")
        report_lines.append("|-------|------------|---------|")
        for name, results in model_results.items():
            if results.get('cv_results'):
                cv = results['cv_results']
                report_lines.append(f"| {name.upper()} | {cv['mean']:.4f} | ±{cv['std']:.4f} |")
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Model-Specific Details
    report_lines.append("## 🔬 Model-Specific Details")
    report_lines.append("")
    
    registry = get_registry()
    selected_model_params = st.session_state.get('selected_model_params', {})
    feature_names = st.session_state.get('feature_names', [])
    
    for model_key, model_wrapper in trained_models.items():
        spec = registry.get(model_key)
        model_name = spec.name if spec else model_key.upper()
        results = model_results.get(model_key, {})
        
        report_lines.append(f"### {model_name}")
        report_lines.append("")
        
        # Hyperparameters
        params = selected_model_params.get(model_key, spec.default_params if spec else {})
        if params:
            report_lines.append("**Hyperparameters:**")
            report_lines.append("")
            for param_name, param_value in params.items():
                report_lines.append(f"- `{param_name}`: {param_value}")
            report_lines.append("")
        
        # Linear model coefficients
        if model_key in ['ridge', 'lasso', 'elasticnet', 'glm', 'huber', 'logreg']:
            coef_df = extract_model_coefficients(model_wrapper, model_key, feature_names)
            if coef_df is not None:
                report_lines.append("**Model Coefficients (Top 10 by magnitude):**")
                report_lines.append("")
                try:
                    report_lines.append(coef_df.head(10).to_markdown(index=False, floatfmt='.4f'))
                except:
                    for _, row in coef_df.head(10).iterrows():
                        report_lines.append(f"- {row['Feature']}: {row['Coefficient']:.4f}")
                report_lines.append("")
                
                # Interpretation note
                report_lines.append("> **Interpretation:** A positive coefficient means the feature increases the target value")
                report_lines.append("> (or log-odds for classification). Coefficients are on the scale of standardized features.")
                report_lines.append("")
        
        # Neural network architecture
        if model_key == 'nn' and hasattr(model_wrapper, 'get_architecture_summary'):
            try:
                arch_summary = model_wrapper.get_architecture_summary()
                if arch_summary:
                    report_lines.append(f"**Architecture:** {arch_summary}")
                    report_lines.append("")
            except:
                pass
        
        # Training history for NN
        if model_key == 'nn' and hasattr(model_wrapper, 'get_training_history'):
            try:
                history = model_wrapper.get_training_history()
                if history and 'train_loss' in history:
                    final_train_loss = history['train_loss'][-1]
                    final_val_loss = history['val_loss'][-1] if 'val_loss' in history else 'N/A'
                    report_lines.append(f"**Training Summary:** {len(history['train_loss'])} epochs")
                    report_lines.append(f"- Final train loss: {final_train_loss:.4f}")
                    if final_val_loss != 'N/A':
                        report_lines.append(f"- Final validation loss: {final_val_loss:.4f}")
                    report_lines.append("")
            except:
                pass
        
        # Classification-specific: confusion matrix summary
        if data_config.task_type == 'classification' and 'y_test' in results and 'y_test_pred' in results:
            try:
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(results['y_test'], results['y_test_pred'])
                report_lines.append("**Confusion Matrix:**")
                report_lines.append("")
                report_lines.append("```")
                report_lines.append(str(cm))
                report_lines.append("```")
                report_lines.append("")
            except:
                pass
        
        report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    
    # Feature Importance
    perm_importance = st.session_state.get('permutation_importance', {})
    if perm_importance:
        report_lines.append("## 📊 Feature Importance (Permutation)")
        report_lines.append("")
        for name, perm_data in perm_importance.items():
            report_lines.append(f"### {name.upper()}")
            importance_df = pd.DataFrame({
                'Feature': perm_data['feature_names'],
                'Importance': perm_data['importances_mean']
            }).sort_values('Importance', ascending=False)
            try:
                report_lines.append(importance_df.head(10).to_markdown(index=False, floatfmt='.4f'))
            except:
                for _, row in importance_df.head(10).iterrows():
                    report_lines.append(f"- {row['Feature']}: {row['Importance']:.4f}")
            report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Recommendations
    if coach_output:
        report_lines.append("## 💡 Model Selection Coach Insights")
        report_lines.append("")
        report_lines.append(f"> {coach_output.data_sufficiency_narrative}")
        report_lines.append("")
        
        if coach_output.warnings_summary:
            report_lines.append("**Warnings:**")
            for warning in coach_output.warnings_summary[:3]:
                report_lines.append(f"- ⚠️ {warning}")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # Notes and Reproducibility
    report_lines.append("## 📝 Notes")
    report_lines.append("")
    report_lines.append("- This report was generated automatically by the Modeling Lab")
    report_lines.append("- All models were evaluated on the same held-out test set")
    report_lines.append("- Preprocessing was applied consistently across all models")
    report_lines.append(f"- Random seed: {st.session_state.get('random_seed', 42)} (for reproducibility)")
    report_lines.append("")
    
    preprocessing_config = st.session_state.get('preprocessing_config', {})
    if preprocessing_config.get('use_pca') or preprocessing_config.get('use_kmeans_features'):
        report_lines.append("**Feature Engineering:**")
        if preprocessing_config.get('use_pca'):
            report_lines.append(f"- PCA enabled with {preprocessing_config.get('pca_n_components')} components")
        if preprocessing_config.get('use_kmeans_features'):
            report_lines.append(f"- KMeans clustering with {preprocessing_config.get('kmeans_n_clusters')} clusters")
    
    return "\n".join(report_lines)


# ============================================================================
# REPORT PREVIEW
# ============================================================================
st.header("📋 Report Preview")

# Generate report
report_text = generate_report()

# Display in a nice container
with st.container():
    st.markdown(report_text)

# ============================================================================
# EXPORT OPTIONS
# ============================================================================
st.header("💾 Export Options")

# Export configuration
with st.expander("⚙️ Export Configuration"):
    export_models = st.checkbox("Include trained model artifacts (joblib/pickle)", value=True)
    export_predictions = st.checkbox("Include predictions CSV", value=True)
    export_plots = st.checkbox("Include plots (requires kaleido)", value=False)
    include_raw_data = st.checkbox("Include raw data sample (first 100 rows)", value=False)


# Helper function to save plotly figures as images
def save_plotly_fig(fig, filename: str) -> Optional[bytes]:
    """Save plotly figure as PNG bytes."""
    try:
        return fig.to_image(format="png", width=1200, height=800)
    except Exception:
        try:
            return fig.to_image(format="png", width=1200, height=800, engine="kaleido")
        except Exception:
            return None


def export_model_artifact(model_wrapper, model_key: str) -> Optional[bytes]:
    """Export trained model as bytes."""
    try:
        import joblib
        buffer = io.BytesIO()
        
        # For neural networks, export the sklearn-compatible wrapper
        if model_key == 'nn' and hasattr(model_wrapper, '_sklearn_estimator'):
            # Export the sklearn estimator (lighter weight)
            joblib.dump(model_wrapper._sklearn_estimator, buffer)
        elif hasattr(model_wrapper, 'model'):
            # Export the underlying model
            joblib.dump(model_wrapper.model, buffer)
        else:
            # Export the wrapper itself
            joblib.dump(model_wrapper, buffer)
        
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Could not export model {model_key}: {e}")
        return None


# Download buttons
col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="📄 Download Report (Markdown)",
        data=report_text,
        file_name=f"modeling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        type="primary"
    )

with col2:
    # Quick metrics CSV
    comparison_data = []
    for name, results in model_results.items():
        row = {'Model': name.upper()}
        row.update(results['metrics'])
        comparison_data.append(row)
    comparison_df = pd.DataFrame(comparison_data)
    
    st.download_button(
        label="📊 Download Metrics (CSV)",
        data=comparison_df.to_csv(index=False),
        file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col3:
    # Create comprehensive zip package
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Report
        zip_file.writestr("report.md", report_text)
        
        # Metadata JSON
        metadata = generate_metadata()
        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
        
        # Metrics CSV
        zip_file.writestr("metrics.csv", comparison_df.to_csv(index=False))
        
        # Predictions CSV
        if export_predictions:
            for name, results in model_results.items():
                pred_df = pd.DataFrame({
                    'Actual': results['y_test'],
                    'Predicted': results['y_test_pred']
                })
                zip_file.writestr(f"predictions/{name}_predictions.csv", pred_df.to_csv(index=False))
        
        # Model artifacts
        if export_models:
            for model_key, model_wrapper in trained_models.items():
                model_bytes = export_model_artifact(model_wrapper, model_key)
                if model_bytes:
                    zip_file.writestr(f"models/{model_key}_model.joblib", model_bytes)
                    
                    # For NN, also export model info
                    if model_key == 'nn':
                        model_info = {
                            'type': 'neural_network',
                            'params': selected_model_params.get(model_key, {}),
                            'note': 'Use joblib.load() to load the sklearn-compatible estimator'
                        }
                        zip_file.writestr(f"models/{model_key}_info.json", json.dumps(model_info, indent=2))
        
        # Preprocessing pipeline
        if pipeline:
            try:
                import joblib
                pipeline_buffer = io.BytesIO()
                joblib.dump(pipeline, pipeline_buffer)
                zip_file.writestr("preprocessing_pipeline.joblib", pipeline_buffer.getvalue())
            except Exception as e:
                logger.warning(f"Could not export pipeline: {e}")
        
        # Plots
        if export_plots:
            for name, results in model_results.items():
                if data_config.task_type == 'regression':
                    fig = px.scatter(
                        x=results['y_test'], y=results['y_test_pred'],
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title=f"{name.upper()} - Predictions vs Actual"
                    )
                    fig.add_trace(go.Scatter(
                        x=[min(results['y_test']), max(results['y_test'])],
                        y=[min(results['y_test']), max(results['y_test'])],
                        mode='lines', name='Perfect', line=dict(dash='dash', color='red')
                    ))
                    plot_bytes = save_plotly_fig(fig, f"plot_{name}.png")
                    if plot_bytes:
                        zip_file.writestr(f"plots/{name}_predictions.png", plot_bytes)
        
        # Raw data sample
        if include_raw_data:
            zip_file.writestr("data_sample.csv", df.head(100).to_csv(index=False))
        
        # Feature names
        feature_names = st.session_state.get('feature_names', [])
        if feature_names:
            zip_file.writestr("feature_names.txt", "\n".join(feature_names))
    
    st.download_button(
        label="📦 Download Complete Package (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"modeling_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )

st.success("✅ Report generated successfully!")

# ============================================================================
# STATE DEBUG
# ============================================================================
with st.expander("🔧 Advanced / State Debug", expanded=False):
    st.markdown("**Current State:**")
    st.write(f"• Data shape: {df.shape if df is not None else 'None'}")
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")
    st.write(f"• Trained models: {len(trained_models)}")
    st.write(f"• Dataset profile: {'Available' if profile else 'Not computed'}")
    st.write(f"• Coach output: {'Available' if coach_output else 'Not computed'}")
    st.write(f"• Git info: {get_git_info()}")
