@echo off
setlocal
cd /d "%~dp0\..\.."
set "PART1_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PART1_PYTHON=.venv\Scripts\python.exe"
"%PART1_PYTHON%" main.py --part 1 %*
if errorlevel 1 pause
endlocal
