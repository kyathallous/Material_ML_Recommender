@echo off
echo Closing every running Python program (this stops any old copy of the app) ...
taskkill /F /IM python.exe /T >nul 2>nul
taskkill /F /IM pythonw.exe /T >nul 2>nul
taskkill /F /IM py.exe /T >nul 2>nul
echo Done. Now double-click START_APP.bat
pause
