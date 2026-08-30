@echo off
setlocal
cd /d "%~dp0"
set "PART2_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PART2_PYTHON=.venv\Scripts\python.exe"
echo Training tuned direct-control policy...
%PART2_PYTHON% -m arena.train --control-style direct --timesteps 200000 --profile fast_exploration --benchmark-episodes 20 --seed 5200
if errorlevel 1 goto :failed

echo Training tuned rotation-and-thrust policy...
%PART2_PYTHON% -m arena.train --control-style rotation --timesteps 300000 --profile balanced --benchmark-episodes 20 --seed 6200
if errorlevel 1 goto :failed

echo Part II models and evidence logs are ready.
goto :done

:failed
echo Training failed. Review the message above.
pause

:done
endlocal
