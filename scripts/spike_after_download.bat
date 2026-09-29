@echo off
REM Waits for scripts\download_himawari.bat to finish, then runs the satellite spike.
REM Launch:  start "" /B scripts\spike_after_download.bat
cd /d "%~dp0.."
if not defined PY set PY=python
:wait
findstr /C:"download exited" logs\download_himawari.log >nul 2>&1
if errorlevel 1 (
    timeout /t 60 /nobreak >nul
    goto wait
)
echo === spike started %DATE% %TIME% === >> logs\satellite_spike.log
"%PY%" -W ignore -u scripts\satellite_spike.py >> logs\satellite_spike.log 2>&1
echo === spike exited %DATE% %TIME% === >> logs\satellite_spike.log
