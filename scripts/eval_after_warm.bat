@echo off
REM Waits for the warm-started 60/90-min runs to finish, then scores them on the
REM pinned test period and regenerates the 22 Sep case-study forecasts.
REM evaluate.py fills the 60/90 entries of results/evaluation_report.json (the
REM from-scratch results are in results/ablations/ and definition_of_done.md).
cd /d "%~dp0.."
REM Python: set PY beforehand to use a specific interpreter, else python on PATH.
if not defined PY set PY=python
set LOG=logs\eval_after_warm.log
echo === waiting for lead90_warm %DATE% %TIME% === > %LOG%
:wait
findstr /C:"=== lead90_warm exited" logs\train_lead90_warm.log >nul 2>&1
if errorlevel 1 (
  timeout /t 300 /nobreak >nul
  goto wait
)
echo === training done, scoring %DATE% %TIME% === >> %LOG%
for %%L in (60 90) do (
  "%PY%" -W ignore -u scripts\case_study_forecasts.py --checkpoint checkpoints/nowcaster/lead%%L_warm/ckpt_step_100000.pt >> %LOG% 2>&1
  "%PY%" -W ignore -u scripts\evaluate.py --checkpoint checkpoints/nowcaster/lead%%L_warm/ckpt_step_100000.pt --n-samples 200 --members 8 >> %LOG% 2>&1
  "%PY%" -W ignore -u scripts\evaluate_probabilistic.py --checkpoint checkpoints/nowcaster/lead%%L_warm/ckpt_step_100000.pt --n-samples 200 --members 8 >> %LOG% 2>&1
)
echo === eval_after_warm exited %DATE% %TIME% === >> %LOG%
