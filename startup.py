#!/usr/bin/env python3
"""
Startup script for Azure App Service
This ensures proper initialization in Azure environment
"""

import os
import sys
import logging

# Set up logging for Azure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Main startup function for Azure"""
    try:
        logger.info("Starting Azure App Service...")
        
        # Import and initialize the application
        from wsgi import application
        
        logger.info("Application imported successfully")
        
        # Get port from environment or use default
        port = int(os.environ.get('PORT', 8000))
        
        logger.info(f"Starting application on port {port}")
        
        # For Azure, we don't need to run the server manually
        # Azure handles that through gunicorn/wsgi
        return application
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise

if __name__ == "__main__":
    app = main()
    # This won't be reached in Azure, but useful for local testing
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
