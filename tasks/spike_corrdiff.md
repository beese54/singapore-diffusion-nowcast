# Spike: can CorrDiff's regression stage learn Singapore rain from ERA5 on one laptop GPU?

*Pattern N (timeboxed spike). Opened 2026-09-27. Plan context: `tasks/plan_phase2.md` §3 (C1–C2).*

## Question

Trained on our RTX 4060 (8 GB), does a CorrDiff **regression** network (PhysicsNeMo
`CorrDiffRegressionUNet` + `RegressionLoss`), given ERA5 fields, predict the hourly rain field over the
radar domain at ~2 km **better than bilinear-interpolated ERA5 precipitation**?

The regression stage is CorrDiff's first half: it learns the mean high-resolution field; the diffusion
stage then adds the small-scale variability. If the mean is no better than interpolated ERA5, the
diffusion stage has nothing to build on, and full CorrDiff is not worth GPU money yet.

## Design (fixed before looking at results)

| Item | Choice | Why |
|---|---|---|
| Target | NEA 70 km radar, **hourly mean** rain rate over (t−1h, t] — matches ERA5's hourly `tp` accumulation | the only georeferenced radar we have (240 km not yet validated) |
| Target grid | 7 × 7 block mean of the 0.29 km grid → **2.03 km**, cropped to **16 × 32** px (32 × 65 km) | ERA5 ~28 km → ×14, inside NVIDIA's ×11–16 guidance |
| Input | ERA5 single + pressure levels (u10, v10, t2m, msl, tcwv, sp, tp; q/t/z/u/v at 1000/850/500/250 = 27 channels), bilinearly regridded onto the target grid, as CorrDiff does; standardised per channel | CorrDiff convention |
| ERA5 window | N 3.0, W 102.0, S −0.5, E 105.5 (0.25°) | margin for regridding + later larger-context experiments |
| Period | 2026-05-22 → latest ERA5 (≈ 5-day lag): ~3,000 hourly samples | radar and ERA5 must share times (the June brief's error) |
| Split | same pinned dates as the nowcaster (train < 2 Sep, val < 14 Sep, test 14–25 Sep) | comparable, no leakage |
| Model | `CorrDiffRegressionUNet`, SongUNet, small (≈ 2–5 M params), bf16 | fits 8 GB |
| Baseline | ERA5 `tp` (m per hour → mm/hr) bilinearly interpolated to the target grid | "what ERA5 already knows" |

## Go / no-go (decided now)

On the test period, versus the bilinear-ERA5 baseline:
- **GO** if the regression beats the baseline on **both** RMSE of hourly rain and pooled FSS at 1 mm/hr
  (8 km neighbourhood), with bootstrap 95% CIs of the differences excluding 0.
- **NO-GO** otherwise. Also NO-GO if training does not fit in 8 GB or needs more than ~6 h.
- Either way, report skill at ≥5 mm/hr hourly mean (heavy for an hourly average) separately, and what the
  model's fields look like (a regression mean is expected to be smooth).

## Timebox

~1 working day of effort, excluding the ERA5 download wait (Copernicus queue).

## Log

- 2026-09-27: brief written; ERA5 2026 download started.
