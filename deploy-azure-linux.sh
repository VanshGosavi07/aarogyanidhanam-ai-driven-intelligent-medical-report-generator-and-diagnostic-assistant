#!/bin/bash
# Azure Linux Deployment Script for Medical Project
# This script creates a deployment-ready zip file for Azure App Service

echo "🚀 Preparing Medical Project for Azure Linux Deployment"

# Check if all required files exist
echo "📋 Checking required files..."

required_files=("main.py" "wsgi.py" "requirements.txt" "startup.sh")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files: ${missing_files[*]}"
    exit 1
fi

echo "✅ All required files present"

# Create deployment package
echo "📦 Creating deployment package..."

# Get current timestamp for unique filename
timestamp=$(date +%Y%m%d-%H%M%S)
zip_name="medical-app-azure-linux-${timestamp}.zip"

# Files and directories to include in deployment
include_items=(
    "main.py"
    "wsgi.py" 
    "requirements.txt"
    "startup.sh"
    "templates/"
    "static/"
    "Modal/"
    "RAG Data/"
    "instance/"
)

# Create zip file for deployment
if command -v zip >/dev/null 2>&1; then
    zip -r "$zip_name" "${include_items[@]}" \
        -x "*.pyc" "*/__pycache__/*" "*.git*" "venv/*" ".env" "*.log" \
        2>/dev/null || echo "⚠️  Some files may not exist, continuing..."
else
    echo "❌ zip command not found. Please install zip utility."
    exit 1
fi

echo "✅ Deployment package created: $zip_name"

# Display file size
if [ -f "$zip_name" ]; then
    size=$(du -h "$zip_name" | cut -f1)
    echo "📊 Package size: $size"
fi

echo ""
echo "🌐 Next steps for Azure Linux deployment:"
echo "1. Go to Azure Portal: https://portal.azure.com"
echo "2. Create new App Service with these settings:"
echo "   - Runtime: Python 3.10"
echo "   - Operating System: Linux"
echo "   - Region: Your preferred region"
echo "3. Go to Deployment Center"
echo "4. Select 'Zip Deploy'"
echo "5. Upload the file: $zip_name"
echo ""
echo "⚙️  Azure Configuration:"
echo "- Startup Command: bash startup.sh"
echo "- Runtime Stack: Python 3.10 (Linux)"
echo "- Platform: Linux"
echo ""
echo "🔧 After deployment, test these endpoints:"
echo "- /health (for health check)"
echo "- /test (for basic functionality)"
echo "- / (for main application)"
echo ""
echo "✅ Deployment package ready for Azure Linux!"
echo "📁 File: $zip_name"
