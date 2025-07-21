#!/usr/bin/env python
"""
WSGI entry point for production deployment
This file is used by Azure and other WSGI servers to serve the Flask application
"""

import os
import sys
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set production environment variables
os.environ['FLASK_ENV'] = 'production'
os.environ['PYTHONPATH'] = current_dir

logger.info(f"Starting WSGI application from {current_dir}")
logger.info(f"Python path: {sys.path}")

# For Azure deployment
if 'WEBSITE_SITE_NAME' in os.environ:
    logger.info("Running on Azure App Service")
    # Use Azure's managed database or in-memory for demo
    if 'DATABASE_URL' not in os.environ:
        os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
else:
    # For local production testing
    temp_dir = tempfile.gettempdir()
    if 'DATABASE_URL' not in os.environ:
        os.environ['DATABASE_URL'] = f'sqlite:///{os.path.join(temp_dir, "production.db")}'

# Debug: Print current working directory and file structure
logger.info(f"Current working directory: {os.getcwd()}")
if os.path.exists('.'):
    logger.info(f"Files in current directory: {os.listdir('.')}")
if os.path.exists('templates'):
    logger.info(f"Templates directory exists with {len(os.listdir('templates'))} files")
if os.path.exists('static'):
    logger.info(f"Static directory exists with {len(os.listdir('static'))} files")

try:
    # Import the Flask application from main.py
    from main import app
    logger.info("Successfully imported Flask app from main.py")
    
    # Initialize the application for production
    with app.app_context():
        from main import init_app
        init_app()
        logger.info("Application initialized successfully")
        
except ImportError as ie:
    logger.error(f"Import error: {str(ie)}")
    # Create a minimal Flask app as fallback
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def import_error():
        return f"""
        <html>
        <head><title>Import Error</title></head>
        <body>
            <h1>Application Import Error</h1>
            <p>Error: {str(ie)}</p>
            <p>Please check your Python dependencies and file structure.</p>
        </body>
        </html>
        """
    
    logger.warning("Using minimal Flask app due to import error")
        
except Exception as e:
    logger.error(f"General error during app initialization: {str(e)}")
    # Create a fallback app
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def general_error():
        return f"""
        <html>
        <head><title>Application Error</title></head>
        <body>
            <h1>Application Error</h1>
            <p>Error: {str(e)}</p>
            <p>The application failed to start properly.</p>
        </body>
        </html>
        """
    
    logger.warning("Using fallback Flask app due to general error")

# WSGI application object
application = app

# Set production configuration
if hasattr(application, 'config'):
    application.config['ENV'] = 'production'
    application.config['DEBUG'] = False
    application.config['TESTING'] = False

logger.info("WSGI application ready")

if __name__ == "__main__":
    # This is for local testing only
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting application on port {port}")
    application.run(host='0.0.0.0', port=port, debug=False)
