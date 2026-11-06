@echo off
setlocal
cd /d "%~dp0"
title Material Recommender
echo.
echo  Data-Driven Machine Learning Recommender for Mechanical Material Selection
echo  ---------------------------------------------------------------------------
where python >nul 2>nul
if %errorlevel%==0 (set PY=python) else (set PY=py)
echo  Installing / checking packages (first time takes a few minutes) ...
%PY% -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo  Package install failed. Check your internet connection and that Python is installed.
  pause
  exit /b 1
)
%PY% run_app.py
echo.
echo  The app has stopped.
pause
