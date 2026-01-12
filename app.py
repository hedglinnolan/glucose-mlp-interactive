"""
Interactive Streamlit app for regression model training.
Supports Neural Network, Random Forest, GLM OLS, and GLM Huber models.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from data_processor import (
        load_and_preview_csv,
        get_numeric_columns,
        prepare_data,
        validate_data_selection
    )
    from models import ModelTrainer, calculate_metrics
    from visualizations import (
        plot_training_history,
        plot_predictions_vs_actual,
        plot_residuals
    )
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.error("Please ensure all dependencies are installed: `pip install -r requirements.txt`")
    st.stop()

# Page config
st.set_page_config(
    page_title="Regression Model Trainer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'model': None,
        'scaler': None,
        'feature_names': None,
        'history': None,
        'test_metrics': None,
        'test_predictions': None,
        'test_actual': None,
        'model_type': None,
        'all_results': {},  # Store results for all models
        'animation_state': {},  # Store animation state
        'X_test': None,
        'y_test': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Title and description
st.title("📊 Interactive Regression Model Trainer")
st.markdown("""
Upload your CSV, select features and target, and train multiple regression models.
Compare Neural Networks, Random Forest, and Linear Models side-by-side.
""")

# Sidebar for file upload
st.sidebar.header("📁 Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file with your data. Must have numeric columns."
)

# Main content area
if uploaded_file is not None:
    try:
        # Load data
        with st.spinner("Loading data..."):
            df = load_and_preview_csv(uploaded_file)
        
        st.sidebar.success(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Show data info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
        
        # Get numeric columns
        numeric_cols = get_numeric_columns(df)
        
        if not numeric_cols:
            st.error("❌ **No numeric columns found in the dataset!**")
            st.info("Please ensure your CSV contains numeric data columns.")
            st.stop()
        
        # Target selection
        st.sidebar.header("🎯 Step 2: Select Target")
        target_col = st.sidebar.selectbox(
            "Target variable to predict",
            options=numeric_cols,
            help="Choose the column you want to predict",
            key="target_select"
        )
        
        # Feature selection
        st.sidebar.header("🔧 Step 3: Select Features")
        st.sidebar.markdown("Select columns to use as predictors:")
        
        # Exclude target from features
        feature_options = [col for col in numeric_cols if col != target_col]
        
        if not feature_options:
            st.error("❌ **No features available** (all columns are numeric but target is selected)")
            st.stop()
        
        selected_features = st.sidebar.multiselect(
            "Feature columns",
            options=feature_options,
            default=feature_options[:min(10, len(feature_options))],
            help="Choose which columns to use as predictors",
            key="feature_select"
        )
        
        # Model selection
        st.sidebar.header("🤖 Step 4: Select Models")
        st.sidebar.markdown("Choose which models to train:")
        
        train_nn = st.sidebar.checkbox("Neural Network", value=True, help="2-layer MLP (32-32)")
        train_rf = st.sidebar.checkbox("Random Forest", value=True, help="500 trees")
        train_glm_ols = st.sidebar.checkbox("GLM OLS", value=True, help="Linear Regression")
        train_glm_huber = st.sidebar.checkbox("GLM Huber", value=True, help="Robust Linear Regression")
        
        models_to_train = []
        if train_nn:
            models_to_train.append("neural_network")
        if train_rf:
            models_to_train.append("random_forest")
        if train_glm_ols:
            models_to_train.append("glm_ols")
        if train_glm_huber:
            models_to_train.append("glm_huber")
        
        if not models_to_train:
            st.sidebar.warning("⚠️ Please select at least one model to train")
        
        # Validate selection
        is_valid, error_msg = validate_data_selection(df, target_col, selected_features)
        
        if not is_valid:
            st.sidebar.error(f"❌ {error_msg}")
        else:
            st.sidebar.success("✅ Selection valid")
            
            # Training configuration
            st.header("⚙️ Training Configuration")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                epochs = st.number_input("Epochs (NN only)", min_value=50, max_value=500, value=200, step=50)
            with col2:
                batch_size = st.number_input("Batch Size (NN only)", min_value=32, max_value=512, value=256, step=32)
            with col3:
                lr = st.number_input("Learning Rate (NN only)", min_value=1e-5, max_value=1e-2, value=0.0015, step=1e-4, format="%.4f")
            with col4:
                rf_trees = st.number_input("RF Trees", min_value=50, max_value=1000, value=500, step=50)
            
            train_button = st.button("🏋️ Train Models", type="primary", use_container_width=True)
            
            if train_button and models_to_train:
                # Prepare data once
                with st.spinner("Preparing data..."):
                    try:
                        (X_train, X_val, X_test,
                         y_train, y_val, y_test,
                         scaler, feature_names) = prepare_data(
                            df, target_col, selected_features
                        )
                        
                        st.session_state.scaler = scaler
                        st.session_state.feature_names = feature_names
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        
                        st.success(f"✅ Data prepared: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
                    except Exception as e:
                        st.error(f"❌ Error preparing data: {str(e)}")
                        logger.exception(e)
                        st.stop()
                
                # Train each model
                results_container = st.container()
                
                for model_type in models_to_train:
                    with results_container:
                        st.subheader(f"Training {model_type.replace('_', ' ').title()}")
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        metrics_placeholder = st.empty()
                        
                        # Progress callback
                        def update_progress(epoch, train_loss, val_loss, val_rmse, **kwargs):
                            if model_type == "neural_network":
                                progress = epoch / epochs
                                progress_bar.progress(progress)
                                status_text.text(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")
                            else:
                                progress_bar.progress(1.0)
                                status_text.text(f"Training {model_type.replace('_', ' ').title()}...")
                            metrics_placeholder.metric("Validation RMSE", f"{val_rmse:.4f}")
                        
                        # Train model
                        try:
                            trainer = ModelTrainer(model_type=model_type)
                            
                            train_kwargs = {
                                'epochs': epochs if model_type == "neural_network" else 1,
                                'batch_size': batch_size if model_type == "neural_network" else 256,
                                'lr': lr if model_type == "neural_network" else 0.0015,
                                'progress_callback': update_progress,
                                'n_estimators': rf_trees if model_type == "random_forest" else 500
                            }
                            
                            train_result = trainer.train(
                                X_train, X_val, y_train, y_val,
                                feature_names, **train_kwargs
                            )
                            
                            # Test predictions
                            test_pred = trainer.predict(X_test)
                            test_metrics = calculate_metrics(y_test, test_pred)
                            
                            # Store results
                            st.session_state.all_results[model_type] = {
                                'trainer': trainer,
                                'metrics': test_metrics,
                                'predictions': test_pred,
                                'actual': y_test,
                                'history': train_result['history']
                            }
                            
                            st.success(f"✅ {model_type.replace('_', ' ').title()} training complete!")
                            progress_bar.progress(1.0)
                            
                        except Exception as e:
                            st.error(f"❌ Error training {model_type}: {str(e)}")
                            logger.exception(e)
                            continue
                
                st.balloons()
            
            # Results section
            if st.session_state.all_results:
                st.header("📈 Results Comparison")
                
                # Comparison table
                comparison_data = []
                for model_type, results in st.session_state.all_results.items():
                    comparison_data.append({
                        'Model': model_type.replace('_', ' ').title(),
                        'RMSE': results['metrics']['RMSE'],
                        'MAE': results['metrics']['MAE'],
                        'R²': results['metrics']['R2']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df = comparison_df.sort_values('RMSE')
                
                st.subheader("Performance Comparison")
                st.dataframe(
                    comparison_df.style.highlight_min(subset=['RMSE', 'MAE'], axis=0, color='lightgreen')
                    .highlight_max(subset=['R²'], axis=0, color='lightgreen'),
                    use_container_width=True
                )
                
                # Best model
                best_model = comparison_df.iloc[0]['Model']
                st.info(f"🏆 **Best Model:** {best_model} (RMSE: {comparison_df.iloc[0]['RMSE']:.4f})")
                
                # Individual model results
                st.subheader("Detailed Results")
                
                tabs = st.tabs([model.replace('_', ' ').title() for model in st.session_state.all_results.keys()])
                
                for idx, (model_type, results) in enumerate(st.session_state.all_results.items()):
                    with tabs[idx]:
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RMSE", f"{results['metrics']['RMSE']:.4f}")
                        with col2:
                            st.metric("MAE", f"{results['metrics']['MAE']:.4f}")
                        with col3:
                            st.metric("R²", f"{results['metrics']['R2']:.4f}")
                        
                        # Visualizations
                        if model_type == "neural_network" and len(results['history']['train_loss']) > 1:
                            st.subheader("Training History")
                            fig_history = plot_training_history(results['history'])
                            st.plotly_chart(fig_history, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Predictions vs Actual")
                            fig_pred = plot_predictions_vs_actual(
                                results['actual'],
                                results['predictions'],
                                title=f"{model_type.replace('_', ' ').title()} Predictions"
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                        
                        with col2:
                            st.subheader("Residuals")
                            fig_resid = plot_residuals(
                                results['actual'],
                                results['predictions'],
                                title=f"{model_type.replace('_', ' ').title()} Residuals"
                            )
                            st.plotly_chart(fig_resid, use_container_width=True)
                        
                        # Download predictions
                        st.subheader("📥 Download Results")
                        results_df = pd.DataFrame({
                            'Actual': results['actual'],
                            'Predicted': results['predictions'],
                            'Residual': results['actual'] - results['predictions']
                        })
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {model_type.replace('_', ' ').title()} Predictions (CSV)",
                            data=csv,
                            file_name=f"{model_type}_predictions.csv",
                            mime="text/csv",
                            key=f"download_{model_type}"
                        )
    
    except Exception as e:
        st.error(f"❌ **Error:** {str(e)}")
        logger.exception(e)
        with st.expander("Error Details"):
            st.exception(e)

else:
    # Welcome message
    st.info("👈 **Please upload a CSV file using the sidebar to get started.**")
    
    st.markdown("""
    ### How to Use:
    
    1. **📁 Upload CSV**: Use the sidebar to upload your dataset
    2. **🎯 Select Target**: Choose which column you want to predict
    3. **🔧 Select Features**: Check the boxes for columns to use as predictors
    4. **🤖 Choose Models**: Select which models to train (NN, RF, GLM OLS, GLM Huber)
    5. **🏋️ Train Models**: Click "Train Models" and watch the progress
    6. **📈 View Results**: Compare models side-by-side, see metrics and visualizations
    
    ### Supported Models:
    
    - **Neural Network**: 2-layer MLP (32-32) with Weighted Huber loss
    - **Random Forest**: 500 trees, robust to non-linear relationships
    - **GLM OLS**: Ordinary Least Squares linear regression
    - **GLM Huber**: Robust linear regression (outlier-resistant)
    
    ### Model Details:
    
    All models use the same train/validation/test split (70/15/15) and feature standardization
    for fair comparison. The Neural Network uses early stopping and learning rate scheduling.
    
    ### Requirements:
    
    - CSV file with numeric columns
    - At least 100 rows recommended
    - Target column must be numeric
    - Feature columns must be numeric
    """)
    
    # Example data format
    with st.expander("📝 Example Data Format"):
        example_data = {
            'age': [25, 30, 35, 40, 45],
            'bmi': [22.5, 24.1, 26.3, 23.8, 27.2],
            'glucose': [95, 102, 110, 98, 115],
            'protein': [50, 55, 60, 52, 58],
            'carb': [200, 220, 240, 210, 250]
        }
        st.dataframe(pd.DataFrame(example_data))
        st.caption("Example: Predict 'glucose' using 'age', 'bmi', 'protein', 'carb'")
