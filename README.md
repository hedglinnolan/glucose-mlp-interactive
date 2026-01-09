# Glucose MLP Interactive Predictor

An interactive web application for training glucose prediction models on custom datasets. Upload your CSV, select features and target, and watch your models train in real-time.

## Features

- 📊 **CSV Upload**: Drag-and-drop or browse to upload your dataset
- 🎯 **Feature Selection**: Choose which columns to use as predictors
- 🎯 **Target Selection**: Select the variable you want to predict
- 🤖 **Auto-Training**: Automatically trains the best neural network architecture
- 📈 **Real-time Progress**: Watch training progress with live metrics
- 📉 **Results Visualization**: See performance metrics and predictions

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload CSV**: Click "Browse files" or drag-and-drop your CSV file
2. **Select Target**: Choose the column you want to predict
3. **Select Features**: Check the boxes for columns to use as predictors
4. **Train Model**: Click "Train Model" and watch the progress
5. **View Results**: See metrics, predictions, and visualizations

## Model Details

The application uses the optimized neural network architecture:
- **Architecture**: [32, 32] hidden layers
- **Loss Function**: Weighted Huber (whuber)
- **Optimizer**: Adam with learning rate scheduling
- **Features**: Automatic standardization and optional feature engineering

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- pandas, numpy, scikit-learn
- matplotlib

See `requirements.txt` for full list.

## Architecture

```
glucose-mlp-interactive/
├── app.py                 # Main Streamlit application
├── model_trainer.py        # Training logic wrapper
├── data_processor.py       # Data loading and preprocessing
├── visualizations.py      # Plotting functions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## License

MIT License
