#!/bin/bash
# Azure Windows App Service Deployment Script

echo "🚀 Preparing Medical Project for Azure Windows App Service Deployment"

# Check if all required files exist
echo "📋 Checking required files..."

if [ ! -f "main.py" ]; then
    echo "❌ main.py not found!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

if [ ! -f "wsgi.py" ]; then
    echo "❌ wsgi.py not found!"
    exit 1
fi

if [ ! -f "web.config" ]; then
    echo "❌ web.config not found!"
    exit 1
fi

if [ ! -f "startup.cmd" ]; then
    echo "❌ startup.cmd not found!"
    exit 1
fi

echo "✅ All required files present"

# Create deployment package
echo "📦 Creating deployment package..."

# Files to include in deployment
files=(
    "main.py"
    "wsgi.py" 
    "requirements.txt"
    "web.config"
    "startup.cmd"
    "templates/"
    "static/"
    "Modal/"
    "RAG Data/"
    "instance/"
)

# Create zip file for deployment
zip_name="medical-app-azure-$(date +%Y%m%d-%H%M%S).zip"

zip -r "$zip_name" "${files[@]}" -x "*.pyc" "*/__pycache__/*" "*.git*" "venv/*" ".env"

echo "✅ Deployment package created: $zip_name"

echo "🌐 Next steps for Azure deployment:"
echo "1. Go to Azure Portal: https://portal.azure.com"
echo "2. Navigate to your App Service"
echo "3. Go to Deployment Center"
echo "4. Select 'Zip Deploy'"
echo "5. Upload the file: $zip_name"
echo ""
echo "💡 Azure Configuration:"
echo "- Runtime: Python 3.12 (Windows)"
echo "- Startup Command: startup.cmd"
echo "- Platform: Windows"
echo ""
echo "🔧 After deployment, check:"
echo "- App Service Logs for any errors"
echo "- /health endpoint for status"
echo "- /test endpoint for basic functionality"
echo ""
echo "✅ Deployment package ready!"
