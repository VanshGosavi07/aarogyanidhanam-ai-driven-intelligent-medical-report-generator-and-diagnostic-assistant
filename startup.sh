#!/bin/bash
echo "🚀 Starting Medical Project on Azure Linux"
echo "📦 Installing dependencies..."

# Upgrade pip first
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Dependencies installed successfully"

# Create necessary directories
mkdir -p /tmp/data
mkdir -p /home/data

echo "🏥 Starting Medical Diagnosis Flask Application..."

# Start the application using gunicorn for production
# Use wsgi:application as the entry point
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 --keep-alive 2 --max-requests 1000 --preload wsgi:application
