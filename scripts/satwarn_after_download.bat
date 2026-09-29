@echo off
REM Waits for the Himawari backfill (scripts\download_himawari.bat) to write its LAST day,
REM then runs Step 2A (tasks/plan_satellite_step2.md). Days are written in order, so the
REM last day's file existing means every earlier day is done.
REM If the laptop shut down mid-download, re-launch BOTH:
REM     start "" /B scripts\download_himawari.bat
REM     start "" /B scripts\satwarn_after_download.bat
cd /d "%~dp0.."
if not defined PY set PY=python
:wait
if not exist data\processed\sat\B08\20260927.npz (
    timeout /t 120 /nobreak >nul
    goto wait
)
echo === satellite warning started %DATE% %TIME% === >> logs\satellite_warning.log
"%PY%" -W ignore -u scripts\satellite_warning.py >> logs\satellite_warning.log 2>&1
echo === satellite warning exited %DATE% %TIME% === >> logs\satellite_warning.log
