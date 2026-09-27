@echo off
REM Waits for the ERA5 download (scripts\download_era5_corrdiff.bat) to finish,
REM then runs the CorrDiff regression spike and writes results\corrdiff_spike.json.
REM Uses the sg-weather conda env (PhysicsNeMo); set PYNEMO to override.
cd /d "%~dp0..\.."
if not defined PYNEMO set PYNEMO=%USERPROFILE%\anaconda3\envs\sg-weather\python.exe
echo === waiting for ERA5 download %DATE% %TIME% === > logs\corrdiff_spike.log
:wait
findstr /C:"=== era5_corrdiff exited" logs\era5_corrdiff.log >nul 2>&1
if errorlevel 1 (
  timeout /t 300 /nobreak >nul
  goto wait
)
echo === running spike %DATE% %TIME% === >> logs\corrdiff_spike.log
"%PYNEMO%" -W ignore -u scripts\corrdiff\spike_regression.py --epochs 60 >> logs\corrdiff_spike.log 2>&1
echo === corrdiff spike exited %DATE% %TIME% === >> logs\corrdiff_spike.log
