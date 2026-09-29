@echo off
REM Heavy-rain warning calibration, then the hybrid 60-min warning (tasks/plan_hybrid_calibration.md).
REM Detached so no session supervisor can reap it. Validation ensembles are checkpointed
REM every 25 forecasts, so after a shutdown just launch it again and it resumes.
REM Launch:  start "" /B scripts\run_warning_calibration.bat
cd /d "%~dp0.."
if not defined PY set PY=python
echo === calibration started %DATE% %TIME% === >> logs\warning_calibration.log
"%PY%" -W ignore -u scripts\warning_calibration.py >> logs\warning_calibration.log 2>&1
if errorlevel 1 (
    echo === calibration FAILED %DATE% %TIME% === >> logs\warning_calibration.log
    exit /b 1
)
echo === hybrid started %DATE% %TIME% === >> logs\warning_calibration.log
"%PY%" -W ignore -u scripts\hybrid_warning.py >> logs\warning_calibration.log 2>&1
echo === all exited %DATE% %TIME% === >> logs\warning_calibration.log
