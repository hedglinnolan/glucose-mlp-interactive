# Run script for Windows PowerShell
# This script activates the virtual environment and runs the Streamlit app

Write-Host "🚀 Starting Regression Model Trainer..." -ForegroundColor Cyan

# Check if virtual environment exists
if (-Not (Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run .\setup.ps1 first to set up the environment." -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Check if streamlit is installed
$streamlitCheck = python -c "import streamlit" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Streamlit not found in virtual environment!" -ForegroundColor Red
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Run the app
Write-Host "🌐 Starting Streamlit app..." -ForegroundColor Green
Write-Host "The app will open in your browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

streamlit run app.py
