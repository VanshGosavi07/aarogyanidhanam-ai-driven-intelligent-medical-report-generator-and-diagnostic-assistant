@echo off
echo 🚀 Preparing Medical Project for Azure Windows App Service Deployment

echo 📋 Checking required files...

if not exist "main.py" (
    echo ❌ main.py not found!
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)

if not exist "wsgi.py" (
    echo ❌ wsgi.py not found!
    pause
    exit /b 1
)

if not exist "web.config" (
    echo ❌ web.config not found!
    pause
    exit /b 1
)

if not exist "startup.cmd" (
    echo ❌ startup.cmd not found!
    pause
    exit /b 1
)

echo ✅ All required files present

echo 📦 Creating deployment package...

REM Get current date and time for unique filename
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "MIN=%dt:~10,2%" & set "SS=%dt:~12,2%"
set "timestamp=%YYYY%%MM%%DD%-%HH%%MIN%%SS%"

set "zip_name=medical-app-azure-%timestamp%.zip"

REM Create zip file (requires PowerShell)
powershell -command "Compress-Archive -Path 'main.py','wsgi.py','requirements.txt','web.config','startup.cmd','templates','static','Modal','RAG Data','instance' -DestinationPath '%zip_name%' -Force"

if %errorlevel% neq 0 (
    echo ❌ Failed to create zip file. Make sure PowerShell is available.
    pause
    exit /b 1
)

echo ✅ Deployment package created: %zip_name%

echo.
echo 🌐 Next steps for Azure deployment:
echo 1. Go to Azure Portal: https://portal.azure.com
echo 2. Navigate to your App Service
echo 3. Go to Deployment Center
echo 4. Select 'Zip Deploy'
echo 5. Upload the file: %zip_name%
echo.
echo 💡 Azure Configuration:
echo - Runtime: Python 3.12 (Windows)
echo - Startup Command: startup.cmd
echo - Platform: Windows
echo.
echo 🔧 After deployment, check:
echo - App Service Logs for any errors
echo - /health endpoint for status
echo - /test endpoint for basic functionality
echo.
echo ✅ Deployment package ready!
echo.
pause
