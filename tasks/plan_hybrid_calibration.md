# Plan: heavy-rain warning calibration + a hybrid 60-min warning — for approval

*2026-09-29. Follows `tasks/plan_motion_input.md` (L035). At 60 min, plain optical-flow extrapolation catches 17%
of heavy-rain boxes and is right 19% of the time; the diffusion model manages 7% and 16%. The model was handed the
motion forecast and still smoothed intensity away, so the gap is **how the ensemble is turned into a warning**, not
missing information. Neither item below trains a model.*

## Item 2 — calibrate the model's heavy-rain warning

Today a box is warned when ≥2 of 8 futures show ≥10 mm/hr. The model under-forecasts intensity: its heavy-rain area
is about half the observed area. So its futures rarely reach 10 mm/hr, even where heavy rain then falls.

- **Candidates:** a member intensity threshold X ∈ {3, 4, 5, 6, 7, 8, 10} mm/hr × a member count k ∈ {1, 2, 3, 4} of 8.
- **Chosen on validation only** (2–14 Sep, never the test period). The rule is fixed now: **maximise CSI** = hits / (hits + false alarms + misses), which rewards catching without flooding the map with alarms.
- **Needs** 200 validation ensembles per lead (30/60/90; about 25 min of GPU each). The test ensembles are already cached.
- **Scored on the test period:** catch rate, precision, CSI and incoming catch for the calibrated model vs the uncalibrated model, with paired bootstrap 95% CIs.

**Adopt** the calibrated warning at a lead if its test CSI beats the uncalibrated one (CI of the difference above 0).

## Item 1 — hybrid 60-min warning

- **H (hybrid):** warn if **either** plain extrapolation shows ≥10 mm/hr in the box **or** the calibrated model warns (the union), fixed now. The two miss different storms, so the union should catch more; the question is whether precision survives.
- **Compared on the same 200 test forecasts:**
  - H;
  - extrapolation alone;
  - the calibrated model;
  - the uncalibrated model;
  - the naive forecast.
- **Metrics:** catch rate, precision, CSI and incoming catch, with paired bootstrap CIs.
- **Drain-alert benchmark at a 60-min horizon,** reported as context:
  - for the 23 out-of-sample drain alerts, the first warning in the 2 h before each alert, for H, extrapolation and the model;
  - this needs 60-min model forecasts every 5 min over the three storm windows (about 100 forecasts, about 10 min of GPU).

**Adopt H** as the 60-min heavy-rain warning if its CSI beats extrapolation alone (CI above 0). Otherwise **adopt
extrapolation alone** as the 60-min warning, and say so plainly. Either way, the diffusion model stays the
probability and rain-amount forecast.

## Honest limits (stated with any result)

- **Sample size:** about 12 test days and a few storms.
- **The candidate grid is small and chosen in advance.** Choosing on validation protects the test score, but validation is only about 12 days too.
- **Real-time timing:** extrapolation needs the last 15 min of radar, the same inputs as the model, so it is available at forecast time with no extra delay.

## Steps and outputs

1. **`scripts/warning_calibration.py`:**
   - generates and caches validation ensembles per lead;
   - chooses (X, k) by CSI on validation;
   - scores on test → `results/warning_calibration.json`.
2. **`scripts/hybrid_warning.py`:**
   - five-way paired comparison at 60 min, plus the 60-min drain-alert context → `results/hybrid_warning.json`;
   - reuses `src/data/motion.py` and the cached test ensembles (`eval_cache/motioncmp_base_n200_m8.npz`, same seeds).
3. **Verdicts recorded here.** RESULTS, README and `warning_skill`-style dashboard text are updated **only if** a warning is adopted, and the LinkedIn draft only if a published number changes.

Effort: about ½ day of code; about 1.5 h of GPU, detached and cached, so it resumes after a shutdown. No training.

## Verdicts (2026-09-29, run 11:04–11:58; `results/warning_calibration.json`, `results/hybrid_warning.json`)

**Item 2: calibration** (rule chosen on validation, scored on the cached test ensembles, seeds 1000+k)

| lead | chosen rule | test CSI uncal → cal [CI of diff] | catch | precision | verdict |
|---|---|---|---|---|---|
| 30 | ≥4 of 8 ≥ 5 mm/hr | 0.182 → 0.214 [−0.025, +0.079] | 29% → 27% | 32% → 52% | do not adopt (CI spans 0) |
| 60 | ≥2 of 8 ≥ 3 mm/hr | 0.023 → 0.112 [+0.023, +0.173] | 3% → 35% | 7% → 14% | **ADOPT** |
| 90 | ≥2 of 8 ≥ 3 mm/hr | 0.042 → 0.086 [−0.001, +0.097] | 7% → 39% | 10% → 10% | do not adopt (CI touches 0) |

- 60 and 90 min chose X = 3 mm/hr, **the lowest candidate**, so the best rule may lie below the grid. Not searched further: that would be choosing after seeing the result.
- Seed check at 60 min (seeds 2000+k): CSI 0.053 → 0.098, CI [−0.011, +0.118]. The gain is in the same direction but **not significant on the second seed set**, so the adoption is fragile.
- At 30 min the calibrated rule trades catch for precision (52% right); on validation it only ties the naive forecast (val CSI 0.160, naive test 0.160).

**Item 1: hybrid at 60 min** (seeds 2000+k): **ADOPT EXTRAPOLATION ALONE**

- H catches 38% vs extrapolation 17% (+21 pts [+6, +36]) and incoming 33% vs 13%, but precision falls to 13% vs 19% ([−10.5, −1.6] pts). CSI 0.107 vs 0.100, CI [−0.028, +0.050]: **not better**.
- Calibrated model alone: CSI 0.098, catch 30%, precision 13%; the uncalibrated rule: CSI 0.053.
- Drain alerts (23 out of sample, context only): H warned 23/23, median lead 87 min; extrapolation 23/23, 69 min; calibrated model 22/23 (missed 18 Sep), 85 min. Background warning rates are low for all (≤ 5.5% of test forecasts at a site).

**Reading:** calibration fixes most of the 60-min model's heavy-rain under-warning (catch 3% → 35%), but only by warning much more widely, and it still does not beat plain extrapolation on CSI. The pre-set rule therefore keeps extrapolation as the 60-min heavy-rain warning. The hybrid's extra catch (and ~18 min earlier drain warnings) comes at the cost of a third more false alarms per hit. That trade-off is for the user (a flood service may prefer the catch), not a CSI win.
