#!/bin/bash
# Run script for macOS/Linux
# This script activates the virtual environment and runs the Streamlit app

echo "🚀 Starting Regression Model Trainer..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Check if streamlit is installed
python -c "import streamlit" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Streamlit not found in virtual environment!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the app
echo "🌐 Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""

streamlit run app.py
