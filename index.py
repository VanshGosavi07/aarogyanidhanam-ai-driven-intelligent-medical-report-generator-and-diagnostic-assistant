from wsgi import app

# Export the WSGI application for Vercel
# This is required for Vercel deployment
application = app

if __name__ == "__main__":
    app.run()