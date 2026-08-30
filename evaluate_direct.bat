@echo off
setlocal
cd /d "%~dp0"
set "PART2_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PART2_PYTHON=.venv\Scripts\python.exe"
%PART2_PYTHON% -m arena.evaluate_direct
if errorlevel 1 pause
endlocal
