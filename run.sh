#!/bin/bash
# Run script for macOS/Linux (uses uv)
# Activates the virtual environment and runs the Streamlit app

echo "🚀 Starting Regression Model Trainer..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to set up the environment."
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found! Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Run the app (uv run uses .venv automatically)
echo "🌐 Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""

uv run streamlit run app.py