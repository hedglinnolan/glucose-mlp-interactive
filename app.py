"""
Interactive Streamlit app for glucose prediction model training.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time

try:
    from data_processor import (
        load_and_preview_csv,
        get_numeric_columns,
        prepare_data,
        validate_data_selection
    )
    from model_trainer import train_model, predict, calculate_metrics
    from visualizations import (
        plot_training_history,
        plot_predictions_vs_actual,
        plot_residuals,
        create_metrics_display
    )
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all dependencies are installed.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Glucose MLP Predictor",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'test_metrics' not in st.session_state:
    st.session_state.test_metrics = None
if 'test_predictions' not in st.session_state:
    st.session_state.test_predictions = None
if 'test_actual' not in st.session_state:
    st.session_state.test_actual = None

# Title
st.title("📊 Interactive Glucose Prediction Model Trainer")
st.markdown("Upload your CSV, select features and target, and train a neural network model.")

# Sidebar for file upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file with your data"
)

# Main content area
if uploaded_file is not None:
    try:
        # Load data
        df = load_and_preview_csv(uploaded_file)
        
        st.sidebar.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Get numeric columns
        numeric_cols = get_numeric_columns(df)
        
        if not numeric_cols:
            st.error("❌ No numeric columns found in the dataset!")
            st.stop()
        
        # Target selection
        st.sidebar.header("🎯 Target Selection")
        target_col = st.sidebar.selectbox(
            "Select target variable to predict",
            options=numeric_cols,
            help="Choose the column you want to predict"
        )
        
        # Feature selection
        st.sidebar.header("🔧 Feature Selection")
        st.sidebar.markdown("Select columns to use as predictors:")
        
        # Exclude target from features
        feature_options = [col for col in numeric_cols if col != target_col]
        
        if not feature_options:
            st.error("❌ No features available (all columns are numeric but target is selected)")
            st.stop()
        
        selected_features = st.sidebar.multiselect(
            "Select features",
            options=feature_options,
            default=feature_options[:min(10, len(feature_options))],
            help="Choose which columns to use as predictors"
        )
        
        # Validate selection
        is_valid, error_msg = validate_data_selection(df, target_col, selected_features)
        
        if not is_valid:
            st.sidebar.error(f"❌ {error_msg}")
        else:
            st.sidebar.success("✅ Selection valid")
            
            # Training section
            st.header("🚀 Model Training")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.number_input("Epochs", min_value=50, max_value=500, value=200, step=50)
            with col2:
                batch_size = st.number_input("Batch Size", min_value=32, max_value=512, value=256, step=32)
            with col3:
                lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=0.0015, step=1e-4, format="%.4f")
            
            train_button = st.button("🏋️ Train Model", type="primary", use_container_width=True)
            
            if train_button:
                # Prepare data
                with st.spinner("Preparing data..."):
                    try:
                        (X_train, X_val, X_test,
                         y_train, y_val, y_test,
                         scaler, feature_names) = prepare_data(
                            df, target_col, selected_features
                        )
                        
                        st.session_state.scaler = scaler
                        st.session_state.feature_names = feature_names
                        
                        st.success(f"✅ Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                    except Exception as e:
                        st.error(f"❌ Error preparing data: {str(e)}")
                        st.stop()
                
                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                metrics_placeholder = st.empty()
                
                # Progress callback
                def update_progress(epoch, train_loss, val_loss, val_rmse):
                    progress = epoch / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f}")
                    metrics_placeholder.metric("Validation RMSE", f"{val_rmse:.4f}")
                
                # Train model
                with st.spinner("Training model..."):
                    try:
                        model, history = train_model(
                            X_train, X_val, y_train, y_val,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=lr,
                            progress_callback=update_progress
                        )
                        
                        st.session_state.model = model
                        st.session_state.history = history
                        
                        # Test predictions
                        test_pred = predict(model, X_test)
                        test_metrics = calculate_metrics(y_test, test_pred)
                        
                        st.session_state.test_metrics = test_metrics
                        st.session_state.test_predictions = test_pred
                        st.session_state.test_actual = y_test
                        
                        st.success("✅ Training complete!")
                        progress_bar.progress(1.0)
                        
                    except Exception as e:
                        st.error(f"❌ Training error: {str(e)}")
                        st.stop()
            
            # Results section
            if st.session_state.model is not None:
                st.header("📈 Results")
                
                # Metrics
                if st.session_state.test_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{st.session_state.test_metrics['RMSE']:.4f}")
                    with col2:
                        st.metric("MAE", f"{st.session_state.test_metrics['MAE']:.4f}")
                    with col3:
                        st.metric("R²", f"{st.session_state.test_metrics['R2']:.4f}")
                
                # Visualizations
                if st.session_state.history:
                    st.subheader("Training History")
                    fig_history = plot_training_history(st.session_state.history)
                    st.plotly_chart(fig_history, use_container_width=True)
                
                if st.session_state.test_predictions is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Predictions vs Actual")
                        fig_pred = plot_predictions_vs_actual(
                            st.session_state.test_actual,
                            st.session_state.test_predictions
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    with col2:
                        st.subheader("Residuals")
                        fig_resid = plot_residuals(
                            st.session_state.test_actual,
                            st.session_state.test_predictions
                        )
                        st.plotly_chart(fig_resid, use_container_width=True)
                
                # Download predictions
                if st.session_state.test_predictions is not None:
                    st.subheader("📥 Download Results")
                    results_df = pd.DataFrame({
                        'Actual': st.session_state.test_actual,
                        'Predicted': st.session_state.test_predictions,
                        'Residual': st.session_state.test_actual - st.session_state.test_predictions
                    })
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

else:
    # Welcome message
    st.info("👈 Please upload a CSV file using the sidebar to get started.")
    
    st.markdown("""
    ### How to use:
    1. **Upload CSV**: Use the sidebar to upload your dataset
    2. **Select Target**: Choose which column you want to predict
    3. **Select Features**: Check the boxes for columns to use as predictors
    4. **Train Model**: Click "Train Model" and watch the progress
    5. **View Results**: See metrics, visualizations, and download predictions
    
    ### Model Details:
    - **Architecture**: 2-layer MLP (32 → 32 → 1)
    - **Loss Function**: Weighted Huber (optimized for regression)
    - **Features**: Automatically standardized
    - **Training**: Adam optimizer with learning rate scheduling and early stopping
    """)
