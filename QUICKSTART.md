# Quick Start Guide

## Installation

```bash
# Navigate to the project directory
cd glucose-mlp-interactive

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Usage Steps

1. **Upload CSV File**
   - Click "Browse files" in the sidebar or drag-and-drop your CSV
   - The app will show a preview of your data

2. **Select Target Variable**
   - Choose the column you want to predict from the dropdown
   - This should be a numeric column

3. **Select Features**
   - Check the boxes for columns to use as predictors
   - You can select multiple features
   - The target column is automatically excluded

4. **Configure Training** (optional)
   - Adjust epochs, batch size, and learning rate if needed
   - Defaults are optimized for most datasets

5. **Train Model**
   - Click "Train Model" button
   - Watch the training progress in real-time
   - See validation RMSE update each epoch

6. **View Results**
   - See performance metrics (RMSE, MAE, R²)
   - View training history plot
   - See predictions vs actual scatter plot
   - Check residual plot
   - Download predictions as CSV

## Example Dataset Format

Your CSV should have:
- Numeric columns for features
- One numeric column for the target variable
- No missing values (or they'll be filled with median)

Example:
```csv
age,bmi,glucose,protein,carb
25,22.5,95,50,200
30,24.1,102,55,220
...
```

## Troubleshooting

**Import errors**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**No numeric columns**: Ensure your CSV has numeric data columns

**Training errors**: Check that:
- Target column is numeric
- Feature columns are numeric
- You have enough data (recommended: >100 rows)

**Memory errors**: Reduce batch size or number of features

## Model Architecture

The app uses a fixed, optimized architecture:
- **Input**: Your selected features (automatically standardized)
- **Hidden Layers**: [32, 32]
- **Output**: Single value prediction
- **Loss Function**: Weighted Huber (optimized for regression)
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout (0.1) and weight decay

This architecture was found to work well across many regression tasks.
