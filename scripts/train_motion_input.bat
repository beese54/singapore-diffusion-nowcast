@echo off
REM Motion-input experiment (tasks/plan_motion_input.md): two arms, both warm-started
REM from the 60-min model of record and fine-tuned for the same steps on the same
REM data at a LOW learning rate (lesson L032), differing ONLY in the motion channel:
REM   lead60_motion  --motion-input 1
REM   lead60_mctrl   --motion-input 0   (control: isolates the motion channel from
REM                                      the effect of extra training)
REM Resumable: re-run after a shutdown and each arm continues from its checkpoint.
REM Launch detached (see train_detached.bat):  start "" /B scripts\train_motion_input.bat
cd /d "%~dp0.."
if not defined PY set PY=python
set ARGS=--max-steps 50000 --batch-size 4 --num-workers 0 --context-frames 6 --time-channels 0 --residual 0 --intensity-alpha 0 --heavy-weight 0 --parameterization v --target-offset 12 --lr 2e-5 --warmup-steps 500 --resume auto --init-from checkpoints/nowcaster/lead60_warm/ckpt_step_100000.pt
"%PY%" -W ignore -u train.py %ARGS% --motion-input 1 --run-name lead60_motion >> logs\train_lead60_motion.log 2>&1
echo === lead60_motion exited %DATE% %TIME% === >> logs\train_lead60_motion.log
"%PY%" -W ignore -u train.py %ARGS% --motion-input 0 --run-name lead60_mctrl >> logs\train_lead60_mctrl.log 2>&1
echo === lead60_mctrl exited %DATE% %TIME% === >> logs\train_lead60_mctrl.log
