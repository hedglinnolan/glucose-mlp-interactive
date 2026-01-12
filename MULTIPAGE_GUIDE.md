# Multi-Page Modeling Lab Guide

## Overview

The app has been refactored into a comprehensive 6-page modeling lab with proper separation of concerns, consistent preprocessing, and production-quality code structure.

## Architecture

### Directory Structure

```
glucose-mlp-interactive/
├── app.py                 # Main entry point (home page)
├── pages/                 # Streamlit multi-page app pages
│   ├── 01_Upload_and_Audit.py
│   ├── 02_EDA.py
│   ├── 03_Preprocess.py
│   ├── 04_Train_and_Compare.py
│   ├── 05_Explainability.py
│   └── 06_Report_Export.py
├── models/                # Model wrappers
│   ├── base.py           # BaseModelWrapper interface
│   ├── nn_whuber.py      # Neural Network wrapper (wraps existing training)
│   ├── glm.py            # OLS Linear Regression wrapper
│   ├── huber_glm.py      # Huber Regression wrapper
│   └── rf.py             # Random Forest wrapper
├── ml/                    # ML utilities
│   ├── pipeline.py       # Preprocessing pipeline builder
│   └── eval.py           # Evaluation metrics and CV
├── utils/                 # Utilities
│   └── session_state.py  # Session state management
└── [existing files]      # data_processor.py, visualizations.py, etc.
```

## Page Flow

1. **Upload & Audit** → Upload CSV, validate data, select target/features
2. **EDA** → Explore data with visualizations and statistics
3. **Preprocess** → Build preprocessing pipeline
4. **Train & Compare** → Train models, evaluate, compare performance
5. **Explainability** → Understand models with feature importance
6. **Report Export** → Generate and download comprehensive reports

## Key Features

### Data Audit (Page 01)
- Missing value analysis
- Data type validation
- Duplicate detection
- Constant column detection
- High-cardinality categorical detection
- Target leakage candidate detection
- ID-like field detection

### EDA (Page 02)
- Summary statistics
- Distribution plots (target and features)
- Correlation heatmap
- Target vs feature plots
- Class balance analysis (for classification)

### Preprocessing (Page 03)
- Numeric: imputation (mean/median/constant), scaling (standard/robust/none), optional log transform
- Categorical: imputation (most_frequent/constant), encoding (one-hot)
- Pipeline recipe display
- Transformation preview

### Train & Compare (Page 04)
- Configurable train/val/test splits
- Optional k-fold cross-validation
- Model hyperparameter controls
- Learning curves (for NN)
- Comprehensive metrics (regression: MAE, RMSE, R2, MedianAE; classification: Accuracy, F1, ROC-AUC, LogLoss, PR-AUC)
- Residual analysis / Confusion matrices
- CV score distributions

### Explainability (Page 05)
- Permutation importance for all models
- Partial dependence plots (top 3-5 features)
- Optional SHAP analysis (gated behind checkbox, handles missing package gracefully)

### Report Export (Page 06)
- Comprehensive markdown report
- Includes: dataset summary, audit, splits, preprocessing recipe, hyperparameters, metrics, feature importance
- Download as markdown or ZIP (with CSV exports)

## Model Wrappers

All models implement `BaseModelWrapper` interface:
- `fit(X_train, y_train, X_val=None, y_val=None, **kwargs)` → returns history dict
- `predict(X)` → returns predictions
- `predict_proba(X)` → returns probabilities (optional, for classification)
- `get_model()` → returns underlying model object

### Neural Network (nn_whuber.py)
- Wraps existing weighted Huber loss training
- Preserves original training loop exactly
- Supports early stopping, learning rate scheduling
- Returns training history with train/val loss and RMSE

### Other Models
- Random Forest: supports regression and classification
- GLM (OLS): standard linear regression
- GLM (Huber): robust regression

## Session State Schema

Data flows between pages via `st.session_state`:

- `raw_data`: Original DataFrame
- `data_config`: DataConfig (target, features, task_type)
- `data_audit`: Audit results dictionary
- `preprocessing_pipeline`: sklearn Pipeline
- `preprocessing_config`: Pipeline configuration dict
- `split_config`: SplitConfig (train/val/test percentages)
- `X_train/X_val/X_test`: Preprocessed feature arrays
- `y_train/y_val/y_test`: Target arrays
- `feature_names`: Transformed feature names
- `model_config`: ModelConfig (hyperparameters)
- `trained_models`: Dict[str, BaseModelWrapper]
- `model_results`: Dict[str, Dict] (metrics, history, predictions)
- `permutation_importance`: Dict[str, Dict]
- `partial_dependence`: Dict[str, Dict]

## Usage

1. Run the app: `.\run.ps1` (Windows) or `./run.sh` (macOS/Linux)
2. Navigate through pages using sidebar or page selector
3. Follow the workflow: Upload → Audit → EDA → Preprocess → Train → Explain → Export

## Notes

- All models use the same preprocessing pipeline for fair comparison
- NN training preserves original weighted Huber implementation
- SHAP is optional and gracefully handles missing package
- Cross-validation is optional and shows metric distributions
- Report generation includes all key information for reproducibility

## Dependencies

- Existing dependencies (streamlit, torch, pandas, numpy, scikit-learn, plotly)
- Added: shap>=0.42.0 (optional, for advanced explainability)
