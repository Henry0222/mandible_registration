@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "BUILD_PYTHON=.venv-integration\Scripts\python.exe"
if not exist "%BUILD_PYTHON%" (
    echo [ERROR] Missing .venv-integration Python environment.
    exit /b 1
)

"%BUILD_PYTHON%" -m pytest
if errorlevel 1 exit /b 1

"%BUILD_PYTHON%" -m PyInstaller --noconfirm --clean --workpath "build\MandibleRegistration-v1.1.0" MandibleRegistration.spec
if errorlevel 1 exit /b 1

"%BUILD_PYTHON%" "scripts\package_windows_release.py"
if errorlevel 1 exit /b 1

echo Build complete: dist\MandibleRegistration-v1.1.0-win64.zip
