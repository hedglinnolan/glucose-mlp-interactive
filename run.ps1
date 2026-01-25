# Run script for Windows PowerShell (uses uv)
# Activates the virtual environment and runs the Streamlit app

Write-Host "🚀 Starting Regression Model Trainer..." -ForegroundColor Cyan

# Check if virtual environment exists
if (-Not (Test-Path ".venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run .\setup.ps1 first to set up the environment." -ForegroundColor Yellow
    exit 1
}

# Check for uv
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-Not $uvCmd) {
    Write-Host "❌ uv not found! Install: irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Red
    exit 1
}

# Run the app (uv run uses .venv automatically)
Write-Host "🌐 Starting Streamlit app..." -ForegroundColor Green
Write-Host "The app will open in your browser at http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

uv run streamlit run app.py