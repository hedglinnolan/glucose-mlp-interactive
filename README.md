# 🧪 Modeling Lab

A comprehensive, educational machine learning modeling platform built with Streamlit. Upload your data, explore it, build preprocessing pipelines, train multiple models, understand their behavior, and export detailed reports—all with consistent preprocessing and honest evaluation.

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Happy Path Walkthrough](#happy-path-walkthrough)
- [Features](#features)
- [Architecture](#architecture)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for first-time dependency installation

### Setup Steps

**Windows:**
```powershell
# Run setup script (creates venv and installs dependencies)
.\setup.ps1

# Run the app
.\run.ps1
```

**macOS/Linux:**
```bash
# Make scripts executable
chmod +x setup.sh run.sh

# Run setup script
./setup.sh

# Run the app
./run.sh
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## ⚡ Quick Start

1. **Upload Data**: Use the sidebar or Upload page to upload a CSV file
2. **Select Target & Features**: Choose what to predict and which columns to use
3. **Explore**: Review data audit and EDA visualizations
4. **Preprocess**: Build a preprocessing pipeline
5. **Train**: Train multiple models and compare performance
6. **Explain**: Understand model behavior with feature importance
7. **Export**: Download comprehensive reports

## 🎯 Happy Path Walkthrough

### Step 1: Upload & Audit (Page 01)

1. Navigate to **📁 Upload & Audit** page
2. Click "Upload CSV file" and select your dataset
3. Review the data audit:
   - Missing values summary
   - Data types
   - Duplicate rows
   - Constant columns
   - High-cardinality categoricals
   - Target leakage candidates
4. Select your **Target Variable** (what you want to predict)
5. Select your **Feature Variables** (predictors)
6. Review recommendations and address any issues

**Expected Output**: Dataset loaded, target/features selected, task type detected

### Step 2: Exploratory Data Analysis (Page 02)

1. Navigate to **📊 EDA** page
2. Review summary statistics
3. Examine target distribution (histogram and box plot)
4. For classification: check class balance
5. Review correlation heatmap
6. Explore target vs feature relationships (scatter/box plots)

**Expected Output**: Understanding of data distributions and relationships

### Step 3: Preprocessing (Page 03)

1. Navigate to **⚙️ Preprocess** page
2. Configure numeric preprocessing:
   - Imputation strategy (mean/median/constant)
   - Scaling (standard/robust/none)
   - Optional log transform
3. Configure categorical preprocessing:
   - Imputation strategy (most_frequent/constant)
   - Encoding (one-hot)
4. Click **"Build Preprocessing Pipeline"**
5. Review the pipeline recipe
6. Preview transformation (before/after)

**Expected Output**: Preprocessing pipeline built and stored

### Step 4: Train & Compare (Page 04)

1. Navigate to **🏋️ Train & Compare** page
2. Configure data splits:
   - Set train/val/test percentages (must sum to 100%)
   - Optionally enable cross-validation
3. Click **"Prepare Splits"**
4. Select models to train:
   - Neural Network (configure epochs, LR, batch size, etc.)
   - Random Forest (configure trees, depth, etc.)
   - GLM (OLS)
   - GLM (Huber)
5. Click **"Train Models"**
6. Review results:
   - Metrics comparison table
   - CV results (if enabled)
   - Learning curves (for NN)
   - Predictions vs Actual plots
   - Residual plots / Confusion matrices

**Expected Output**: Trained models with performance metrics

### Step 5: Explainability (Page 05)

1. Navigate to **🔍 Explainability** page
2. Click **"Calculate Permutation Importance"**
3. Review feature importance rankings and plots
4. Click **"Calculate Partial Dependence"**
5. Review partial dependence plots for top features
6. Optionally enable SHAP analysis (requires shap package)

**Expected Output**: Understanding of which features matter most

### Step 6: Report Export (Page 06)

1. Navigate to **📄 Report Export** page
2. Review the generated markdown report
3. Click **"Download Report (Markdown)"** or **"Download Complete Package (ZIP)"**
4. ZIP includes:
   - Report markdown
   - Metrics CSV
   - Predictions CSV for each model

**Expected Output**: Comprehensive report downloaded

## ✨ Features

### Multi-Page Structure
- **6 dedicated pages** for focused workflows
- Clean navigation via sidebar
- Session state maintains data flow between pages

### Data Audit
- Missing value analysis
- Duplicate detection
- Constant column detection
- High-cardinality categorical detection
- Target leakage candidate detection
- ID-like field detection

### Exploratory Data Analysis
- Summary statistics
- Distribution visualizations
- Correlation heatmaps
- Target vs feature analysis
- Class balance analysis (classification)

### Preprocessing Pipeline
- Numeric: imputation, scaling, optional log transform
- Categorical: imputation, one-hot encoding
- sklearn Pipeline with ColumnTransformer
- Human-readable pipeline recipe
- Transformation preview

### Model Training & Evaluation
- **Neural Network**: Weighted Huber loss, early stopping, learning rate scheduling
- **Random Forest**: Configurable trees and depth
- **GLM (OLS)**: Standard linear regression
- **GLM (Huber)**: Robust regression
- Configurable train/val/test splits
- Optional k-fold cross-validation
- Comprehensive metrics (regression & classification)
- Learning curves
- Residual analysis / Confusion matrices

### Explainability
- Permutation importance
- Partial dependence plots
- Optional SHAP analysis (gracefully handles missing package)

### Report Export
- Comprehensive markdown report
- Includes: dataset summary, audit, splits, preprocessing, hyperparameters, metrics, feature importance
- Download as markdown or ZIP (with CSV exports)

## 🏗️ Architecture

```
glucose-mlp-interactive/
├── app.py                 # Main entry point (home page)
├── pages/                 # Streamlit multi-page app
│   ├── 01_Upload_and_Audit.py
│   ├── 02_EDA.py
│   ├── 03_Preprocess.py
│   ├── 04_Train_and_Compare.py
│   ├── 05_Explainability.py
│   └── 06_Report_Export.py
├── models/                # Model wrappers
│   ├── base.py           # BaseModelWrapper interface
│   ├── nn_whuber.py      # Neural Network (wraps existing training)
│   ├── glm.py            # OLS Linear Regression
│   ├── huber_glm.py      # Huber Regression
│   └── rf.py             # Random Forest
├── ml/                    # ML utilities
│   ├── pipeline.py       # Preprocessing pipeline builder
│   └── eval.py           # Evaluation metrics and CV
├── utils/                 # Utilities
│   └── session_state.py  # Session state management
└── [existing files]      # data_processor.py, visualizations.py, etc.
```

### Key Design Principles

- **Consistent Preprocessing**: All models use the same preprocessing pipeline
- **Honest Evaluation**: Proper train/val/test splits with optional CV
- **Modular Architecture**: Clean separation of concerns
- **Production Quality**: Type hints, docstrings, error handling
- **Educational Focus**: Clear explanations and visualizations

## ⚠️ Known Limitations & Assumptions

### Data Assumptions
- **CSV format**: Only CSV files are supported
- **Numeric target**: Target variable must be numeric (for regression) or integer/categorical (for classification)
- **Memory**: Large datasets (>100K rows) may be slow; consider sampling
- **Missing values**: Handled via preprocessing, but extensive missingness may affect model quality

### Model Limitations
- **Neural Network**: Currently supports regression only (weighted Huber loss optimized for regression)
- **Huber GLM**: Regression only (not suitable for classification)
- **Classification**: Logistic regression used for GLM in classification mode
- **Feature count**: Very high-dimensional data (>1000 features) may be slow

### Preprocessing Limitations
- **Categorical encoding**: Only one-hot encoding supported (no target encoding in base version)
- **Sparse matrices**: OneHotEncoder outputs are converted to dense arrays (may be memory-intensive for high-cardinality categoricals)
- **Feature names**: Feature names preserved through preprocessing, but very long names may be truncated in visualizations

### Evaluation Assumptions
- **Stratification**: Automatic for classification tasks
- **Time-series**: Basic time-based split support (requires datetime column selection)
- **Cross-validation**: Uses random splits (not time-aware for time-series)

### Technical Limitations
- **SHAP**: Optional dependency; app works without it
- **Plotly**: Some complex plots may be slow with very large datasets
- **Session state**: Data persists only during session (refresh clears state)

### Browser Compatibility
- Tested on Chrome, Firefox, Edge
- Some features may not work on older browsers

## 🐛 Troubleshooting

### "streamlit: command not found"
- **Solution**: Make sure virtual environment is activated
- Use `.\run.ps1` (Windows) or `./run.sh` (macOS/Linux) which handles this automatically

### Import errors
- **Solution**: Run `pip install -r requirements.txt` in activated venv

### Memory errors
- **Solution**: Reduce dataset size, use fewer features, or reduce model complexity

### Model training fails
- **Check**: Ensure preprocessing pipeline is built first
- **Check**: Verify target and features are selected correctly
- **Check**: Review error messages in expanders

### SHAP not working
- **Solution**: Install with `pip install shap` (optional dependency)

### Blank pages
- **Solution**: Check browser console (F12) for JavaScript errors
- **Solution**: Ensure you've completed prerequisite steps (e.g., upload data before EDA)

## 📚 Additional Resources

- [Development Workflow Guide](DEVELOPMENT.md)
- [Deployment Guide](DEPLOYMENT.md)

## 👤 Author

Nolan Hedglin (D/Math)

## 📝 License

MIT License
