# Setup script for Windows PowerShell (uses uv)

Write-Host "🚀 Setting up Regression Model Trainer..." -ForegroundColor Cyan

# Check for uv
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-Not $uvCmd) {
    Write-Host "❌ uv not found!" -ForegroundColor Red
    Write-Host "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
    Write-Host "  powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`"" -ForegroundColor White
    exit 1
}
Write-Host "Using uv: $(uv --version)" -ForegroundColor Green

# Create virtual environment with uv (Python 3.9; llvmlite/numba/shap require <3.10)
if (-Not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment (Python 3.9)..." -ForegroundColor Yellow
    uv venv --python 3.9
} else {
    $py = & .\.venv\Scripts\python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    & .\.venv\Scripts\python -c "import sys; exit(0 if sys.version_info.minor < 10 else 1)" 2>$null
    if ($LASTEXITCODE -ne 0 -and $py) {
        Write-Host "❌ .venv uses Python $py; llvmlite (shap) requires <3.10." -ForegroundColor Red
        Write-Host "   Remove it and re-run setup:  Remove-Item -Recurse -Force .venv; .\setup.ps1" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "✅ Virtual environment already exists (Python $py)" -ForegroundColor Green
}

# Install dependencies with uv
Write-Host "📥 Installing dependencies..." -ForegroundColor Yellow
uv pip install -r requirements.txt

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the app:" -ForegroundColor Cyan
Write-Host "  .\run.ps1" -ForegroundColor White
Write-Host "  # or: uv run streamlit run app.py" -ForegroundColor White