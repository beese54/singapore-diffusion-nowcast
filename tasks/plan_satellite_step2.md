# Plan: satellite Step 2 — a learned 60-min heavy-rain warning from radar + Himawari — for approval

*2026-09-29. Follows the Step 1 spike (`tasks/plan_satellite.md`: **weak GO**. A cold-cloud rule adds incoming catch
at 60 min only by warning over a wide area: precision 18% → 9%, L036).*

## Why this is not the diffusion model that `plan_satellite.md` sketched

Step 1 sketched "a satellite-conditioned 60-min diffusion model". Since then, the motion-input result
(`tasks/plan_motion_input.md`) changes the odds against that design:

- The 60-min model was **handed** a map of where the heavy rain was heading. It used the map to place light rain
  better (FSS ≥2 +0.057), but it forecast **less** heavy rain (catch −0.014, CI below 0). Its bottleneck at 60 min is
  under-forecasting intensity, not missing information.
- A satellite channel is the same kind of input: "heavy rain is likely near here". The same model would very likely
  smooth it away too. Testing that costs ~6 h of download plus ~2 h of GPU, and the likely answer is already known.
- The spike's problem is also narrower than "the model lacks satellite". The rule could not tell a **growing** storm
  from a **mature** one, so it warned everywhere cold cloud existed. That calls for a better **decision rule**
  over a few satellite and radar features. A generative model is not needed for that.

**So Step 2 builds a small learned warning** (a classifier per 2 × 2 km box) that combines extrapolation with
satellite features. It is judged against extrapolation, and against the same classifier without satellite, at
**matched warning volume**. The diffusion arm becomes an optional Step 2B, run only if 2A shows the satellite
features carry usable skill. The output answers the question that matters: does Himawari improve the 60-min heavy-rain
warning that we would actually issue (today that is extrapolation alone)?

## Step 2A — learned warning (CPU only)

**Data.**
- **Backfill Himawari** bands 13 and 8 for **22 May – 1 Sep** (the training period, ~103 days) and **27 Sep** (the
  out-of-sample flash-flood case). The cache already holds 2–26 Sep.
- Use the existing resumable downloader: `scripts/download_himawari.py --start 2026-05-22 --end 2026-09-01`, run
  detached. At the spike's measured rate (25 days in 83 min), that is **~6 h and ~45 GB streamed**, and only ~0.4 GB is kept.
  C: has 61 GB free and raw strips are deleted as it goes. A shutdown mid-run loses at most one day.
- Radar labels come from the existing zarr. Radar gaps in the training period (2,453 missing frames) are simply
  skipped as issue or target times, as in the spike.

**Unit and label.** These are the same as the spike, so the numbers compare directly. The unit is one of the
17 × 31 boxes of 2 × 2 km, at a 60-min lead, issued every 10 min. The label is observed ≥ 10 mm/hr in the box
at issue + 65 min.

**Features per box** (only data available at issue time; satellite scans ≥ 20 min old, parallax shift
(+3, −8) px from Step 1):

| group | features |
|---|---|
| radar (control set R) | extrapolated max rain in the box at +65 min (the E field); current max rain in the box and within 10 km; rain area ≥ 1 mm/hr within 10 km; hour of day (sin/cos) |
| satellite (added set S) | coldest band-13 temperature within 10 km and within 30 km; its 20-min and 30-min cooling; band 8 − band 13 (overshooting tops, near 0 or positive); cold-cloud (≤ 220 K) area within 30 km and its 20-min change |

The 30 km features are what the radar cannot see. The domain is only 35 × 63 km, so a storm growing just
outside it is invisible to radar but visible to the satellite.

**Models.**
- Two scikit-learn `HistGradientBoostingClassifier` models, identical except for the features: **L_R** (radar only)
  and **L_RS** (radar + satellite). They are trained on 22 May – 1 Sep with class weighting for the 1.2% base rate.
- Hyperparameters come from a small grid (depth, learning rate, iterations), chosen on validation (2–14 Sep) by
  log-loss.
- Test (14–25 Sep) is touched **once**, after everything is frozen.
- **L_R is the control.** A learned combination of radar features could beat plain extrapolation by itself.
  Without L_R, that gain would be credited to the satellite.

**Operating point: matched volume (L036).** Each classifier warns in the top-k boxes by probability. k is set **on
validation** so that the classifier issues the same number of warnings as extrapolation E does on validation. The
resulting probability threshold is then frozen and applied to test. Because warning counts are equal, catch,
precision and CSI all move together, and a wide-area warning cannot pass.

**KEEP rule (fixed now, before any training).** L_RS is kept as the 60-min heavy-rain warning if, on test at
matched volume, with 95% CIs from a bootstrap over days (L031):
1. **L_RS − L_R:** CSI difference CI above 0. This proves the gain comes from the satellite.
2. **L_RS − E:** CSI difference CI above 0. This proves it beats the warning we would issue today.
3. **Precision floor:** L_RS precision on test is ≥ E's precision − 2 points. This guards against the volume
   drifting between validation and test.

Otherwise the verdict is **DO NOT KEEP**. Extrapolation stays the 60-min warning, and satellite work pauses until
the Northeast Monsoon has been collected.

**Also reported (context, not the decision).**
- Incoming and initiation catch (initiation has only ~130 target boxes, so its CIs will be wide).
- Warning counts beside every rate.
- A second operating point at 2× E's volume.
- Reliability of the L_RS probabilities.
- Feature importance, with permutation importance on validation.
- The 27 Sep case: first warning time at the Pasir Panjang site, versus the 30-min model's 10:35 SGT.
- The spike rule S and E ∪ S on the same boxes.

## Step 2B — optional, decided after 2A: satellite-conditioned diffusion model

Step 2B only runs if 2A KEEPs **and** you choose it. It would use the ERA5/motion pattern: zero-initialised extra
input channels (the last 3 usable band-13 scans and the 20-min cooling, resampled onto the radar grid), warm-started
from `lead60_warm`, 50k steps at lr 2e-5.

The control arm already exists: `lead60_mctrl` has the same recipe and budget and no extra input, so only one arm is
trained (~2 h GPU). It would be scored against L_RS and E by the same rule. Given the motion result, **I expect
2B to be DO NOT KEEP**, and 2A may make it unnecessary.

## Step 3 — only if a KEEP: real time

This is unchanged from `plan_satellite.md`. It adds:
- a 10-min Himawari collector (strip 5, crop only),
- the dependencies in **both** Python environments,
- gap-watching in `check_radar_gaps.py --status`.

A learned 2A warning is cheap to run live: CPU only, milliseconds per issue time.

## Honest limits (stated with any result)

- Training covers ~103 days, mostly Southwest-Monsoon afternoon storms. The Northeast Monsoon (Dec–Mar) is absent.
- Test is 12 days and a handful of storm days, so CIs will be wide. A KEEP will rest on a few storms.
- The parallax is corrected by one fixed shift, so the error varies with cloud height. The 10/30 km neighbourhoods absorb part of it.
- The 2 km satellite pixel can say "a storm is growing near here", not which street.
- Hour of day can let the classifier learn "afternoons rain". L_R has it too, so it cannot explain L_RS − L_R.

## Steps and outputs

1. Launch the backfill detached (`scripts/download_himawari.py`, log `logs/download_himawari.log`). The code is written while it runs.
2. `scripts/satellite_warning.py`: builds the features from the spike's `build()`/`sample()` helpers, trains
   L_R and L_RS, calibrates volume on validation, scores test once → `results/satellite_warning.json`.
3. Tests:
   - **no leakage:** features for issue t are unchanged when every radar frame after t and every scan newer than
     t − 20 min is replaced with noise;
   - **matched volume:** validation warning counts equal E's;
   - **reproducibility:** fixed seeds give the same result twice.
4. Record the verdict here and in `tasks/todo.md`. Update docs, dashboard and README only if the 60-min warning changes.

**Effort:** ~6 h of unattended download, ~½ day of code and tests, and minutes of CPU to train and score. No GPU, no cloud
spend.

## Step 2A verdict (2026-09-30): **DO NOT KEEP**. Extrapolation stays the 60-min heavy-rain warning

`scripts/satellite_warning.py` → `results/satellite_warning.json` (the rule above was fixed before training).

**Data used.**
- Train: 27 May – 2 Sep, 12,236 issue times. Radar before 27 May did not form complete 60-min samples.
- Validation: 1,664 issue times. Test: 1,671 issue times over 12 days.
- The satellite image was usable at 100% of train and validation issue times, and 99.5% of test.
- Heavy-rain base rate on test: 1.1% of boxes.
- Both arms chose depth 3, learning rate 0.05, 200 iterations. Validation weighted log-loss: L_R 0.534, L_RS 0.485.

**Test, matched volume** (the threshold is set so each classifier issues as many warnings as E did on validation, 4,767):

| | catch | precision | CSI | incoming catch | warnings |
|---|---|---|---|---|---|
| L_R (radar only) | 0.9% | 2.4% | 0.007 | 0.8% | 3,602 |
| L_RS (radar + satellite) | 2.7% | 8.3% | 0.021 | 2.7% | 3,179 |
| E (extrapolation) | **16.1%** | **18.0%** | **0.093** | **12.1%** | 8,688 |
| naive (persistence) | 12.5% | 12.5% | 0.067 | 0% | 9,735 |

**KEEP criteria** (bootstrap over test days):

| criterion | result | met? |
|---|---|---|
| CSI(L_RS) − CSI(L_R), CI above 0 | +0.014 [−0.000, +0.025] | no |
| CSI(L_RS) − CSI(E), CI above 0 | −0.072 [−0.112, −0.005] | no |
| precision(L_RS) ≥ precision(E) − 2 points | 8.3% vs 18.0% | no |

**Twice the volume (context).** L_RS reaches CSI 0.044 and L_R 0.018; E is 0.093.
- L_RS − L_R: CSI +0.026 [+0.012, +0.031].
- L_RS − E: CSI −0.049 [−0.089, +0.013].

**Feature importance** (permutation importance on validation, L_RS). The top four were:
1. coldest cloud top within 30 km (0.152)
2. hour (cos) (0.148)
3. extrapolated rain within 10 km (0.057)
4. water-vapour minus infrared, the overshooting-top signal (0.040)

Cooling rates were near zero (≤ 0.006). The model used "is deep cold cloud nearby", not "is a storm growing".

**Reliability.** The class-balanced probabilities are not calibrated: the top decile averages p 0.79 while its observed
rate is 4–5%. They are useful for ranking only, which is all the matched-volume rule uses.

**Why the classifiers failed: time of day.** Their top 3,000 test warnings all fall at 03–12 SGT, while test heavy
rain fell mostly at 12–17 SGT. The hour-of-day features let the models learn the training months' timing
(Southwest-Monsoon pre-dawn and morning squalls), which did not hold in mid-September. Class-balanced weights
reward broad separation of the classes (validation AUC 0.81), not precision at the very top. Only 8–9% of the top
warnings fell where E warns.

**Post-hoc check (cannot change the verdict).** `scripts/satellite_warning_posthoc.py` →
`results/satellite_warning_posthoc.json`. The same models were rerun with the hour features removed, with the same hyperparameters and no re-tuning:

| no hour features, matched volume | catch | precision | CSI |
|---|---|---|---|
| L_R | 10.0% | 12.8% | 0.060 |
| L_RS | 9.7% | 13.5% | 0.060 |
| E | 16.1% | 18.0% | 0.093 |

- L_RS − L_R: CSI **+0.000 [−0.023, +0.024]**.
- L_RS − E: CSI −0.033 [−0.081, +0.028].
- Without class weighting the result is the same (L_RS − L_R CSI +0.001 [−0.025, +0.032]).

With the flaw removed, the satellite adds **nothing** a classifier can use at matched volume, and neither
classifier beats plain extrapolation.

**Reading.**
- The spike's gain (incoming catch +28 points) came from warning over a wider area, as L036 warned. At equal
  warning volume, the Himawari features that the spike and 2A tried do not add heavy-rain skill at 60 min over
  radar.
- They do carry information: L_RS beats L_R at 2× volume, and cold cloud is the top feature. But what they know
  (deep convection nearby) overlaps with what extrapolation already knows, and it is too coarse to pick the box.
- A 12-day test with a handful of storm days leaves wide CIs. Nothing here rules out a small effect.

**Decision.**
- Step 2B (the diffusion arm) is not run: its precondition (2A KEEP) failed.
- Step 3 (real-time collection) is not built.
- Satellite work pauses. It is worth revisiting only with a longer and more varied record, such as the Northeast
  Monsoon, and with no clock features, or with the clock features validated across seasons.
- The Himawari cache (~0.5 GB, 21 May – 27 Sep) is kept, so a retry needs no new download.
