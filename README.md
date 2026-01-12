# Interactive Regression Model Trainer

A production-ready web application for training and comparing multiple regression models on your own datasets. Upload a CSV, select features and target, and train Neural Networks, Random Forest, and Linear Models side-by-side.

## ✨ Features

- 📊 **CSV Upload**: Drag-and-drop or browse to upload your dataset
- 🎯 **Feature Selection**: Choose which columns to use as predictors
- 🎯 **Target Selection**: Select the variable you want to predict
- 🤖 **Multiple Models**: Train Neural Network, Random Forest, GLM OLS, and GLM Huber
- 📈 **Real-time Progress**: Watch training progress with live metrics
- 📉 **Side-by-side Comparison**: Compare all models in one view
- 📥 **Results Download**: Export predictions as CSV
- 🎨 **Interactive Visualizations**: Plotly charts for analysis

## 🚀 Quick Start

### Automated Setup (Recommended)

**Windows:**
```powershell
.\setup.ps1
streamlit run app.py
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
streamlit run app.py
```

### Manual Setup

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

## 📋 Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for first-time dependency installation

## 🎯 Usage

1. **Upload CSV**: Click "Browse files" or drag-and-drop your CSV file
2. **Select Target**: Choose the column you want to predict
3. **Select Features**: Check the boxes for columns to use as predictors
4. **Choose Models**: Select which models to train (all selected by default)
5. **Train Models**: Click "Train Models" and watch the progress
6. **View Results**: Compare models, see metrics, visualizations, and download predictions

## 🤖 Supported Models

### Neural Network
- **Architecture**: 2-layer MLP (32 → 32 → 1)
- **Loss Function**: Weighted Huber (optimized for regression)
- **Features**: Automatic standardization, early stopping, learning rate scheduling
- **Best for**: Complex non-linear relationships

### Random Forest
- **Trees**: 500 (configurable)
- **Features**: Handles non-linear relationships, feature interactions
- **Best for**: Robust predictions, feature importance

### GLM OLS (Ordinary Least Squares)
- **Type**: Linear regression
- **Features**: Fast training, interpretable coefficients
- **Best for**: Linear relationships, baseline comparison

### GLM Huber
- **Type**: Robust linear regression
- **Features**: Outlier-resistant, faster than RF
- **Best for**: Linear relationships with outliers

## 📊 Example Dataset Format

Your CSV should have:
- Numeric columns for features
- One numeric column for the target variable
- At least 100 rows recommended

Example:
```csv
age,bmi,glucose,protein,carb
25,22.5,95,50,200
30,24.1,102,55,220
35,26.3,110,60,240
```

## 🏗️ Architecture

```
regression-model-trainer/
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data loading and preprocessing
├── models.py              # Model implementations (NN, RF, GLM)
├── visualizations.py      # Plotly visualization functions
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup (optional)
├── setup.ps1              # Windows setup script
├── setup.sh               # Unix/macOS setup script
├── README.md              # This file
└── .gitignore            # Git ignore rules
```

## 🔧 Configuration

Models use optimized defaults from research:
- **Neural Network**: lr=0.0015, weight_decay=0.0002, epochs=200
- **Random Forest**: n_estimators=500, min_samples_leaf=10
- **Train/Val/Test Split**: 70/15/15
- **Feature Scaling**: StandardScaler (fit on train only)

## 📋 Usage

1. **Upload CSV**: Click "Browse files" or drag-and-drop your CSV file
2. **Select Target**: Choose the column you want to predict
3. **Select Features**: Check the boxes for columns to use as predictors
4. **Choose Models**: Select which models to train (all selected by default)
5. **Train Models**: Click "Train Models" and watch the progress
6. **View Results**: Compare models, see metrics, visualizations, and download predictions

## 🐛 Troubleshooting

**Import errors**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**No numeric columns**: Ensure your CSV has numeric data columns

**Training errors**: Check that:
- Target column is numeric
- Feature columns are numeric
- You have enough data (recommended: >100 rows)
- No missing values in target column

**Memory errors**: 
- Reduce batch size (for Neural Network)
- Reduce number of trees (for Random Forest)
- Use fewer features

**Port already in use**:
```bash
streamlit run app.py --server.port 8502
```

## 📦 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment options:
- Streamlit Cloud (free)
- Docker
- Heroku
- Local server

## 🧪 Testing

Test with sample data:
1. Create a CSV with numeric columns
2. Upload and select target/features
3. Train all models
4. Compare results

## 👨‍💻 Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for a complete guide on:
- Safe feature development workflow
- Branching strategy
- Best practices for testing changes
- How to use feature branches

**Quick start for new features:**
```powershell
# Windows
.\create-feature-branch.ps1 -FeatureName "your-feature-name"

# macOS/Linux
chmod +x create-feature-branch.sh
./create-feature-branch.sh your-feature-name
```

## 📝 License

MIT License

## 👤 Author

Nolan Hedglin (D/Math)

## 🙏 Acknowledgments

Based on research comparing neural networks vs traditional ML methods for regression tasks.
