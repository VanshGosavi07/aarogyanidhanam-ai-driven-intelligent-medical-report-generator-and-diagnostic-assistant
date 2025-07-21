@echo off
echo "Starting Azure App Service deployment for Medical Project"
echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed successfully"
echo "Starting Flask application..."
python wsgi.py
