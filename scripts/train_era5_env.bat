@echo off
REM ERA5 conditioning experiment (tasks/plan_era5_conditioning.md): two arms,
REM both warm-started from the 30-min model and fine-tuned for the same steps on
REM the same data, differing ONLY in the ERA5 weather input:
REM   lead30_env   --era5-env 1
REM   lead30_ctrl  --era5-env 0   (control: isolates the effect of the weather input
REM                                 from the effect of extra training)
REM Resumable: re-run after a shutdown and each arm continues from its checkpoint.
cd /d "%~dp0.."
if not defined PY set PY=python
set ARGS=--max-steps 50000 --batch-size 4 --num-workers 0 --context-frames 6 --time-channels 0 --residual 0 --intensity-alpha 0 --heavy-weight 0 --parameterization v --target-offset 6 --resume auto --init-from checkpoints/nowcaster/ckpt_step_300000.pt
"%PY%" -W ignore -u train.py %ARGS% --era5-env 1 --run-name lead30_env >> logs\train_lead30_env.log 2>&1
echo === lead30_env exited %DATE% %TIME% === >> logs\train_lead30_env.log
"%PY%" -W ignore -u train.py %ARGS% --era5-env 0 --run-name lead30_ctrl >> logs\train_lead30_ctrl.log 2>&1
echo === lead30_ctrl exited %DATE% %TIME% === >> logs\train_lead30_ctrl.log
