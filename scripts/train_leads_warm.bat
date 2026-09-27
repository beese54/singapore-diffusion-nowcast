@echo off
REM Warm-started 60/90-min models. Re-running this after a shutdown resumes each
REM run from its own latest checkpoint (--init-from only applies to a fresh run).
cd /d "%~dp0.."
REM Python: set PY beforehand to use a specific interpreter, else python on PATH.
if not defined PY set PY=python
set ARGS=--max-steps 100000 --batch-size 4 --num-workers 0 --context-frames 6 --time-channels 0 --residual 0 --intensity-alpha 0 --heavy-weight 0 --parameterization v --resume auto --init-from checkpoints/nowcaster/ckpt_step_300000.pt
"%PY%" -W ignore -u train.py %ARGS% --target-offset 12 --run-name lead60_warm >> logs\train_lead60_warm.log 2>&1
echo === lead60_warm exited %DATE% %TIME% === >> logs\train_lead60_warm.log
"%PY%" -W ignore -u train.py %ARGS% --target-offset 18 --run-name lead90_warm >> logs\train_lead90_warm.log 2>&1
echo === lead90_warm exited %DATE% %TIME% === >> logs\train_lead90_warm.log
