@echo off
REM Streams Himawari-9 crops for the satellite spike period (tasks/plan_satellite.md).
REM Detached; finished days are skipped, so re-launching after a shutdown resumes.
REM Launch:  start "" /B scripts\download_himawari.bat
cd /d "%~dp0.."
if not defined PY set PY=python
echo === download started %DATE% %TIME% === >> logs\download_himawari.log
"%PY%" -W ignore -u scripts\download_himawari.py --start 2026-09-02 --end 2026-09-26 >> logs\download_himawari.log 2>&1
echo === download exited %DATE% %TIME% === >> logs\download_himawari.log
