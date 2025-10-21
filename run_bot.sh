#!/bin/bash

# Enhanced Oracle Bot Runner Script
# This script ensures all dependencies are properly installed and runs the bot

echo "🚀 Starting Enhanced Oracle Bot..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create a .env file with your Kalshi API key:"
    echo "echo 'KALSHI_API_KEY=your_actual_api_key_here' > .env"
    echo ""
fi

# Install/update dependencies
echo "📦 Installing/updating dependencies..."
pip install -r requirements.txt

# Run the bot
echo "🎯 Starting Streamlit dashboard..."
python3 -m streamlit run enhanced_oracle_bot.py
