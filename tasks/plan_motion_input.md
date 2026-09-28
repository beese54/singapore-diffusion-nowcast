# Plan: give the nowcaster a motion forecast as an extra input — for approval

*2026-09-28. Follows the Stage 7 spike (`tasks/plan_stage7_heavy_rain.md`). Its side finding: a simple optical-flow
extrapolation beats the 60-min diffusion model on heavy rain (catch +0.072 [+0.014, +0.149], precision +0.089
[+0.010, +0.163]).*

## Idea

The 60-min model places heavy rain badly: it catches 3% of downpours, and when it warns it is right only 7% of the
time. A plain motion forecast does better, so the model is not using the motion it could infer from its six input
frames. Instead of hoping it learns motion, we **hand it the motion forecast**: one extra input channel holding the
last radar frame moved forward by the lead time along the estimated storm motion. The diffusion model can then keep
what extrapolation gets right (where rain moves) and add what it cannot (growth, decay, uncertainty).
This is the standard "extrapolation + learned correction" design used in operational nowcasting.

## Design

| Item | Choice | Why |
|---|---|---|
| Motion field | DIS optical flow between the frames at t−4 and t−1 (15 min apart), smoothed by rain-weighted normalised convolution; the same code as the spike (`advect()`), run on the 70 km frames | Verified on a synthetic known shift (L034); uses only frames the model already sees, so nothing leaks from the future |
| Extra channel | The last frame (t−1) advected by the lead time (65 min for the 60-min model), normalised like the other frames | One channel, and it is on the model's own grid |
| How it enters | The stem convolution gets one more input channel whose weights **start at zero** | At step 0 the model is exactly the current one; any change is learned (as with the ERA5 input) |
| Starting point | Warm start from `lead60_warm` (60-min model of record) | 60 min is where the gap is; 90 min follows only if 60 works |
| Training | **Two arms**, 50k steps each, low learning rate 2e-5 with a short warm-up (L032): `lead60_motion` (with the channel) and `lead60_ctrl` (same fine-tune, no channel) | The control separates the motion input from the effect of extra training, which is what exposed the fine-tuning harm last time |
| Data | The full 70 km archive, same pinned split. Motion is computed on the fly in the data loader (~2 ms per sample) | No waiting for new data; nothing extra stored on disk |
| Cost | About 2 h per arm on the RTX 4060, about 4–5 h in total, detached and resumable. About ½ day of code and tests | — |

## Evaluation (paired, same 200 test forecasts, same seeds)

- Scored on the same 200 test forecasts, with the same seeds, for four forecasters:
  - base (`lead60_warm`);
  - ctrl;
  - motion;
  - extrapolation alone (the new channel used directly as a forecast).
- Metrics:
  - CRPS skill;
  - FSS at 2 and 10 mm/hr (11.9 km);
  - heavy-rain catch rate and precision (2 × 2 km boxes, ≥2 of 8 futures);
  - incoming-heavy-rain catch.
- Paired bootstrap 95% CIs throughout.
- Also re-run the drain-alert benchmark at 60 min, for context only.

## Go / no-go (fixed now)

**KEEP** if, against the control:
1. heavy-rain skill improves: FSS ≥10 **or** catch rate, CI of the difference above 0;
2. and CRPS skill is not worse: CI of the difference not entirely below 0.

**Worth using in practice** only if the kept model also beats extrapolation alone on heavy-rain catch rate or
precision. If it does not, the honest recommendation is "use extrapolation at 60 min", and that is recorded.

## Verification before training (tests in `tasks/repro/motion_input_repro.py`)

1. **Option off:** the current checkpoints load strictly and give bit-identical outputs.
2. **Option on, at step 0:** a warm-started model's output equals the base model's for any motion channel. This proves the zero init.
3. **The zero-initialised stem slice receives a non-zero gradient.**
4. **No leakage:** the motion channel for a sample is unchanged when every frame from t onward is replaced with noise.
5. **Motion code:** it passes the synthetic known-shift check. On a sample of real test anchors, the channel matches the spike's extrapolation.
6. **Guards:**
   - resume and warm start refuse a checkpoint whose motion option differs;
   - `nowcast.py` and `evaluate.py` read the stamp and build the channel.

## Honest limits (stated with any result)

- **The spike's evidence is from about 12 test days and a few storms,** and so will this be.
- **The motion is estimated from 15 minutes of a 35 × 63 km window.** Storms entering from outside are still invisible, but the spike showed that matters little.
- **If only the 60-min model improves,** 30 and 90 min stay as they are until tested.

## Steps

1. `RadarDataset(motion_channel=False)` gets an optional extra context channel, via the shared `advect()` moved into `src/data/motion.py`. `ConditionedUNet` gets `extra_in=0` with zero-init warm start; `train.py` gets the option, stamps, guards and init. Default off everywhere.
2. Tests (above) pass; one short smoke run.
3. Train both arms, detached (`scripts/train_motion_input.bat`), resumable after a shutdown.
4. `scripts/eval_motion_input.py` → `results/motion_input.json`; verdict recorded here; docs, dashboard and post updated only if the verdict changes a published claim.
