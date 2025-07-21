@echo off
REM Azure Linux Deployment Script for Medical Project (Windows version)
echo 🚀 Preparing Medical Project for Azure Linux Deployment

echo 📋 Checking required files...

set "missing_files="
if not exist "main.py" set "missing_files=%missing_files% main.py"
if not exist "wsgi.py" set "missing_files=%missing_files% wsgi.py"
if not exist "requirements.txt" set "missing_files=%missing_files% requirements.txt"
if not exist "startup.sh" set "missing_files=%missing_files% startup.sh"

if not "%missing_files%"=="" (
    echo ❌ Missing required files:%missing_files%
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

set "zip_name=medical-app-azure-linux-%timestamp%.zip"

REM Create zip file using PowerShell
echo Creating deployment package...
powershell -command "try { Compress-Archive -Path 'main.py','wsgi.py','requirements.txt','startup.sh','templates','static','Modal','RAG Data','instance' -DestinationPath '%zip_name%' -Force -ErrorAction SilentlyContinue } catch { Write-Host 'Some files may not exist, continuing...' }"

if not exist "%zip_name%" (
    echo ❌ Failed to create zip file. Make sure PowerShell is available.
    pause
    exit /b 1
)

echo ✅ Deployment package created: %zip_name%

REM Display file size
for %%A in ("%zip_name%") do set "size=%%~zA"
set /a "sizeMB=%size% / 1024 / 1024"
echo 📊 Package size: %sizeMB% MB

echo.
echo 🌐 Next steps for Azure Linux deployment:
echo 1. Go to Azure Portal: https://portal.azure.com
echo 2. Create new App Service with these settings:
echo    - Runtime: Python 3.10
echo    - Operating System: Linux
echo    - Region: Your preferred region
echo 3. Go to Deployment Center
echo 4. Select 'Zip Deploy'
echo 5. Upload the file: %zip_name%
echo.
echo ⚙️  Azure Configuration:
echo - Startup Command: bash startup.sh
echo - Runtime Stack: Python 3.10 (Linux)
echo - Platform: Linux
echo.
echo 🔧 After deployment, test these endpoints:
echo - /health (for health check)
echo - /test (for basic functionality)
echo - / (for main application)
echo.
echo ✅ Deployment package ready for Azure Linux!
echo 📁 File: %zip_name%
echo.
pause
