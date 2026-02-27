#!/bin/bash
# Tabular ML Lab — Setup Script
set -e

echo "🔬 Setting up Tabular ML Lab..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create cache directory
mkdir -p .cache

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "Optional: For AI-powered interpretation, install an LLM backend:"
echo "  - Ollama (free, local): https://ollama.ai → ollama serve → ollama pull llama3.2"
echo "  - OpenAI: pip install openai (configure API key in app sidebar)"
echo "  - Anthropic: pip install anthropic (configure API key in app sidebar)"
