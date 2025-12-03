#!/bin/bash

echo "🍎 Starting FoodVisor..."
echo ""

# Set library path for zbar (barcode scanning)
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Activate virtual environment
source venv_new/bin/activate

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file from .env.example and add your API keys"
    exit 1
fi

echo "✅ Environment configured"
echo "🚀 Starting Flask server..."
echo ""
echo "Open in browser: http://127.0.0.1:5000"
echo "Press CTRL+C to stop"
echo ""

# Start the app
python food.py
