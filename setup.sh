#!/bin/bash
# Setup script for Unix/macOS (uses uv)

set -e  # Exit on error

echo "🚀 Setting up Regression Model Trainer..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found!"
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "Using uv: $(uv --version)"

# Create virtual environment with uv (Python 3.9; llvmlite/numba/shap require <3.10)
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (Python 3.9)..."
    uv venv --python 3.9
else
    _py="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if ! .venv/bin/python -c 'import sys; exit(0 if sys.version_info.minor < 10 else 1)' 2>/dev/null; then
        echo "❌ .venv uses Python $_py; llvmlite (shap) requires <3.10."
        echo "   Remove it and re-run setup:  rm -rf .venv && ./setup.sh"
        exit 1
    fi
    [ -n "$_py" ] && echo "✅ Virtual environment already exists (Python $_py)" || echo "✅ Virtual environment already exists"
fi

# Install dependencies with uv
echo "📥 Installing dependencies..."
uv pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  ./run.sh"
echo "  # or: uv run streamlit run app.py"