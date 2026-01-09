# How to Run - Complete Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Clone or Download

**Option A: Clone from GitHub (after pushing)**
```bash
git clone https://github.com/YOUR_USERNAME/glucose-mlp-interactive.git
cd glucose-mlp-interactive
```

**Option B: Use Local Directory**
```bash
cd glucose-mlp-interactive
```

### Step 2: Run Setup Script

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

### Step 3: Run the App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📋 Detailed Instructions

### Prerequisites Check

1. **Check Python version:**
   ```bash
   python --version
   # Should be 3.8 or higher
   ```

2. **If Python not found:**
   - **Windows**: Download from https://www.python.org/ (check "Add to PATH")
   - **macOS**: `brew install python3`
   - **Linux**: `sudo apt-get install python3 python3-pip python3-venv`

### Installation Methods

#### Method 1: Automated Setup (Easiest)

**Windows:**
```powershell
# Navigate to project directory
cd glucose-mlp-interactive

# Run setup script
.\setup.ps1

# If successful, run app
streamlit run app.py
```

**macOS/Linux:**
```bash
# Navigate to project directory
cd glucose-mlp-interactive

# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh

# Activate virtual environment (if not auto-activated)
source .venv/bin/activate

# Run app
streamlit run app.py
```

#### Method 2: Manual Setup

**Windows:**
```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run app
streamlit run app.py
```

**macOS/Linux:**
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run app
streamlit run app.py
```

---

## 🎯 Using the Application

### 1. Upload Your Data

- Click **"Browse files"** in the sidebar
- Or **drag-and-drop** your CSV file
- The app will show a preview of your data

### 2. Select Target Variable

- Choose the column you want to predict from the dropdown
- Must be a **numeric column**

### 3. Select Features

- Check boxes for columns to use as predictors
- You can select **multiple features**
- Target column is automatically excluded

### 4. Choose Models

Select which models to train:
- ✅ **Neural Network** - 2-layer MLP (32-32)
- ✅ **Random Forest** - 500 trees
- ✅ **GLM OLS** - Linear regression
- ✅ **GLM Huber** - Robust regression

### 5. Configure Training (Optional)

Adjust settings if needed:
- **Epochs**: Number of training iterations (NN only, default: 200)
- **Batch Size**: Samples per batch (NN only, default: 256)
- **Learning Rate**: Step size (NN only, default: 0.0015)
- **RF Trees**: Number of trees (RF only, default: 500)

### 6. Train Models

- Click **"Train Models"** button
- Watch **real-time progress**:
  - Progress bar
  - Current epoch/iteration
  - Validation RMSE updates
- Training time varies:
  - **GLM models**: < 1 second
  - **Random Forest**: 1-10 seconds
  - **Neural Network**: 30 seconds - 5 minutes (depending on data size)

### 7. View Results

After training, you'll see:

**Comparison Table:**
- Side-by-side comparison of all models
- Metrics: RMSE, MAE, R²
- Best model highlighted in green

**Individual Model Tabs:**
- **Metrics**: RMSE, MAE, R²
- **Training History**: Loss curves (NN only)
- **Predictions vs Actual**: Scatter plot
- **Residuals**: Error distribution
- **Download**: Export predictions as CSV

---

## 🧪 Testing with Example Data

The repository includes `example_data.csv`:

1. **Upload** `example_data.csv`
2. **Select target**: `glucose`
3. **Select features**: `age`, `bmi`, `protein`, `carb`, `fat_total`
4. **Train all models**
5. **Compare results**

---

## 🐛 Troubleshooting

### "Python not found"

**Windows:**
- Install Python from https://www.python.org/
- Check "Add Python to PATH" during installation
- Restart PowerShell

**macOS:**
```bash
brew install python3
```

**Linux:**
```bash
sudo apt-get install python3 python3-pip python3-venv
```

### "Execution Policy" Error (Windows)

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Port Already in Use

```bash
streamlit run app.py --server.port 8502
```

### Import Errors

```bash
# Make sure virtual environment is activated
# Then reinstall:
pip install --upgrade -r requirements.txt
```

### Memory Errors

- Reduce batch size (for Neural Network)
- Reduce number of trees (for Random Forest)
- Use fewer features
- Use smaller dataset

### No Numeric Columns

- Ensure CSV has numeric data columns
- Check for text columns that should be numeric
- Remove non-numeric columns or convert them

---

## 📊 Expected Output

When running successfully, you'll see:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

The app opens automatically in your browser.

---

## 🛑 Stopping the App

Press `Ctrl+C` in the terminal to stop the Streamlit server.

---

## 📚 Additional Resources

- **README.md** - Overview and features
- **INSTALLATION.md** - Detailed installation guide
- **QUICKSTART.md** - Quick reference
- **DEPLOYMENT.md** - Deployment options
- **GITHUB_SETUP.md** - GitHub repository setup

---

## ✅ Verification Checklist

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created (`.venv` folder exists)
- [ ] Dependencies installed (`pip list` shows streamlit, torch, etc.)
- [ ] App runs (`streamlit run app.py` opens browser)
- [ ] Can upload CSV file
- [ ] Can select target and features
- [ ] Models train successfully
- [ ] Results display correctly

---

## 💡 Tips

1. **Start with example data** to verify everything works
2. **Use smaller datasets first** to test quickly
3. **Check data format** - ensure numeric columns
4. **Monitor memory usage** for large datasets
5. **Save results** by downloading predictions CSV

---

## 🆘 Getting Help

If you encounter issues:

1. Check error messages in terminal
2. Check error messages in Streamlit app
3. Review troubleshooting section
4. Check GitHub issues
5. Open a new issue with:
   - Operating system
   - Python version
   - Error message
   - Steps to reproduce
