#!/usr/bin/env python
"""
WSGI entry point for production deployment
This file is used by Vercel and other WSGI servers to serve the Flask application
"""

import os
import sys
import tempfile

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Set production environment variables
os.environ['FLASK_ENV'] = 'production'
os.environ['PYTHONPATH'] = os.path.dirname(__file__)

# For Vercel deployment, use in-memory SQLite database
if 'VERCEL' in os.environ or 'VERCEL_ENV' in os.environ:
    # Use in-memory database for serverless deployment
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    # Disable file system operations that might fail in serverless
    os.environ['DISABLE_FILE_UPLOADS'] = 'true'
else:
    # For local production testing
    temp_dir = tempfile.gettempdir()
    os.environ['DATABASE_URL'] = f'sqlite:///{os.path.join(temp_dir, "production.db")}'

# Import the Flask application from main.py
from main import app

# Initialize the application for production
def init_production_app():
    """Initialize app with production configuration"""
    try:
        with app.app_context():
            # Import here to avoid circular imports
            from main import init_app
            init_app()
    except Exception as e:
        # Log error but don't crash the application
        print(f"Error during app initialization: {str(e)}")
        app.logger.error(f"Error during app initialization: {str(e)}")
        # Continue with minimal initialization
        try:
            from main import db, doc_processor
            if doc_processor is None:
                from main import DocumentProcessor
                doc_processor = DocumentProcessor()
        except Exception as init_error:
            print(f"Fallback initialization failed: {str(init_error)}")

# Initialize the application
init_production_app()

# WSGI application object
application = app

# Set production configuration
application.config['ENV'] = 'production'
application.config['DEBUG'] = False
application.config['TESTING'] = False

if __name__ == "__main__":
    # This is for local testing only
    application.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
