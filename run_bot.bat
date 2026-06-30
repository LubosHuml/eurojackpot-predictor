@echo off
title Bybit AI Bot Auto-Setup and Runner
echo ========================================================
echo       BYBIT AI BOT AND LOTTERY FORECASTER AUTO-SETUP
echo ========================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed on this computer!
    echo.
    echo Opening Python download page... Please install Python.
    echo IMPORTANT: Make sure to check the box "Add Python to PATH" during installation!
    start "" "https://www.python.org/downloads/"
    echo.
    echo After installing Python, please close this window and run run_bot.bat again.
    pause
    exit /b
)

echo [OK] Python is installed.
echo.

if not exist ".venv" (
    echo [SETUP] Creating Python Virtual Environment.
    python -m venv .venv
)

echo [SETUP] Activating Virtual Environment...
call .venv\Scripts\activate

echo [SETUP] Upgrading pip...
python -m pip install --upgrade pip

echo [SETUP] Installing required AI and quantitative libraries...
echo This may take 1-2 minutes on first run - please wait...
pip install flask gunicorn tensorflow-cpu scikit-learn joblib pandas numpy requests beautifulsoup4

echo [OK] All dependencies successfully installed.
echo.
echo ========================================================
echo            STARTING AI TRADING BOT SERVER
echo ========================================================
echo.
echo The bot is running the 24/7 background scheduler loop.
echo Do NOT close this command window!
echo.
echo Opening your web browser dashboard...
start "" "http://localhost:5080"

set PORT=5080
python app.py
pause
