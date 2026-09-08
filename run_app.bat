@echo off
setlocal
chcp 65001 >nul
set "APP_DIR=%~dp0"
set "PYTHONPATH=%APP_DIR%src"
set "PYTHONUTF8=1"
if exist "%APP_DIR%.venv-integration\Scripts\python.exe" set "PYTHON_EXE=%APP_DIR%.venv-integration\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%APP_DIR%.venv\Scripts\python.exe" set "PYTHON_EXE=%APP_DIR%.venv\Scripts\python.exe"
if defined PYTHON_EXE goto run
echo Python environment not found. Install the delivered general-model-registration wheel and this project into .venv first.
pause
exit /b 1
:run
"%PYTHON_EXE%" -m mandible_registration %*
if errorlevel 1 pause
endlocal
