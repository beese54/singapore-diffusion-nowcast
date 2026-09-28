@echo off
REM Waits for both motion-input arms (scripts\train_motion_input.bat) to finish, then
REM runs the paired evaluation. Detached so no session supervisor can reap it; the
REM evaluation caches each model's ensembles, so a re-run after a shutdown resumes.
REM Launch:  start "" /B scripts\eval_after_motion.bat
cd /d "%~dp0.."
if not defined PY set PY=python
:wait
findstr /C:"lead60_mctrl exited" logs\train_lead60_mctrl.log >nul 2>&1
if errorlevel 1 (
    timeout /t 60 /nobreak >nul
    goto wait
)
echo === eval started %DATE% %TIME% === >> logs\eval_motion_input.log
"%PY%" -W ignore -u scripts\eval_motion_input.py >> logs\eval_motion_input.log 2>&1
echo === eval exited %DATE% %TIME% === >> logs\eval_motion_input.log
