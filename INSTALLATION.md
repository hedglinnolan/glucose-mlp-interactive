# Detailed Installation and Run Instructions

## Prerequisites

- **Python 3.8 or higher** (check with `python --version` or `python3 --version`)
- **Git** (for cloning the repository)
- **Internet connection** (for downloading dependencies)
- **4GB RAM minimum** (8GB recommended for larger datasets)

## Step-by-Step Installation

### Option 1: Automated Setup (Recommended)

#### Windows (PowerShell)

1. **Open PowerShell** in the project directory
2. **Run the setup script:**
   ```powershell
   .\setup.ps1
   ```
   This will:
   - Check Python installation
   - Create a virtual environment (`.venv`)
   - Install all required dependencies
   - Activate the virtual environment

3. **If setup succeeds**, you'll see:
   ```
   ✅ Setup complete!
   ```

4. **Run the app:**
   ```powershell
   streamlit run app.py
   ```

#### macOS/Linux (Bash)

1. **Open Terminal** in the project directory
2. **Make setup script executable:**
   ```bash
   chmod +x setup.sh
   ```

3. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   This will:
   - Check Python installation
   - Create a virtual environment (`.venv`)
   - Install all required dependencies

4. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

### Option 2: Manual Setup

#### Windows

1. **Open PowerShell** in the project directory

2. **Create virtual environment:**
   ```powershell
   python -m venv .venv
   ```

3. **Activate virtual environment:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   You should see `(.venv)` in your prompt.

4. **Upgrade pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

5. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
   This will install:
   - streamlit
   - torch (PyTorch)
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - plotly

6. **Run the app:**
   ```powershell
   streamlit run app.py
   ```

#### macOS/Linux

1. **Open Terminal** in the project directory

2. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```
   You should see `(.venv)` in your prompt.

4. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

5. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Running the Application

### Starting the App

After installation, run:
```bash
streamlit run app.py
```

You should see output like:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

The app will **automatically open** in your default web browser.

### Using the App

1. **Upload CSV File**
   - Click "Browse files" in the sidebar
   - Or drag-and-drop your CSV file
   - The app will show a preview of your data

2. **Select Target Variable**
   - Choose the column you want to predict from the dropdown
   - This must be a numeric column

3. **Select Features**
   - Check the boxes for columns to use as predictors
   - You can select multiple features
   - The target column is automatically excluded

4. **Choose Models**
   - Select which models to train:
     - ✅ Neural Network (default)
     - ✅ Random Forest (default)
     - ✅ GLM OLS (default)
     - ✅ GLM Huber (default)
   - You can uncheck models you don't want to train

5. **Configure Training** (Optional)
   - Adjust epochs, batch size, learning rate if needed
   - Defaults are optimized for most datasets

6. **Train Models**
   - Click "Train Models" button
   - Watch the training progress in real-time
   - See validation RMSE update each epoch

7. **View Results**
   - See performance metrics (RMSE, MAE, R²) for all models
   - Compare models side-by-side
   - View training history plots (Neural Network only)
   - See predictions vs actual scatter plots
   - Check residual plots
   - Download predictions as CSV

## Troubleshooting

### Python Not Found

**Windows:**
- Download Python from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation
- Restart PowerShell after installation

**macOS:**
```bash
# Install via Homebrew
brew install python3
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install python3 python3-pip python3-venv

# Fedora
sudo dnf install python3 python3-pip
```

### Virtual Environment Issues

**Windows - Execution Policy Error:**
```powershell
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Activation not working:**
```powershell
# Windows - try this instead:
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

### Port Already in Use

If port 8501 is already in use:
```bash
streamlit run app.py --server.port 8502
```

### Import Errors

If you see import errors:
```bash
# Make sure virtual environment is activated
# Then reinstall dependencies:
pip install --upgrade -r requirements.txt
```

### Memory Errors

If you run out of memory:
- Reduce batch size (for Neural Network)
- Reduce number of trees (for Random Forest)
- Use fewer features
- Use a smaller dataset

### Data Format Issues

**No numeric columns:**
- Ensure your CSV has numeric data columns
- Check for text/string columns that should be numeric

**Missing values:**
- The app handles missing values automatically
- Missing values in features are filled with median
- Rows with missing target values are removed

## Testing with Example Data

The repository includes `example_data.csv` for testing:

1. Upload `example_data.csv`
2. Select `glucose` as target
3. Select features: `age`, `bmi`, `protein`, `carb`, `fat_total`
4. Train all models
5. Compare results

## Stopping the App

Press `Ctrl+C` in the terminal to stop the Streamlit server.

## Next Steps

- See `README.md` for feature overview
- See `QUICKSTART.md` for quick reference
- See `DEPLOYMENT.md` for deployment options
- See `CONTRIBUTING.md` if you want to contribute

## Getting Help

If you encounter issues:
1. Check the error message in the terminal
2. Check the error message in the Streamlit app
3. Review the troubleshooting section above
4. Open an issue on GitHub with:
   - Your operating system
   - Python version (`python --version`)
   - Error message
   - Steps to reproduce
