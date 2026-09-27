# NVIDIA LaunchPad application brief — Singapore storm nowcasting and CorrDiff downscaling

*Rewritten 2026-09-27. The June 2026 version (kept in `docs/history/launchpad_2026-06/`) had the
radar resolution wrong (1 km; it is 0.29 km), paired 2022–23 ERA5 with 2026 radar (downscaling needs
the same times), and assumed LaunchPad accepts multi-day training jobs (it provides short-term,
pre-configured hands-on labs). Every number below is taken from this repository's results files.*

## What to request

The **LaunchPad for PhysicsNeMo** hands-on lab (and, if offered, the **Earth-2** lab). Purpose: learn
the CorrDiff training recipe and the `DownscalingDataset` interface on NVIDIA-hosted GPUs, before
adapting it to Singapore on our own hardware. This is a learning and feasibility request, not a
request to host our training run.

## Project summary (paste into the application form)

An independent research project building high-resolution, probabilistic flash-flood nowcasting for
Singapore with generative diffusion models. Phase 1 (May–Sep 2026) is complete:

- **Data:** 128 days of NEA 70 km rain-area radar (5-min, 0.29 km pixels, 34.8 × 62.9 km domain,
  losslessly decoded from NEA's 33-level colour scale and validated against geocoded flood reports),
  PUB flood alerts as ground truth (26 out-of-sample geocoded events: 9 in the test period, 17 from the 27 Sep storm), and from Sep 2026 the NEA
  240 km wide-range radar.
- **Model:** a 25.8M-parameter v-prediction diffusion model (DDPM, 8-member ensembles), one model per
  lead time, trained on a single RTX 4060 laptop GPU.
- **Result (held-out 12-day test period):** at 30 min it beats persistence on CRPS skill, +0.38
  (95% CI +0.25 to +0.51), with neighbourhood FSS 0.83 vs 0.74 at 2 mm/hr (difference not yet
  significant). Out of sample, it flagged the 27 Sep 2026 Pasir Panjang flash flood 82 min before
  the flood report, while persistence still showed dry, and the 22 Sep King's Road flood 86 min
  ahead. Inference: 20 s for 8 members × 3 lead times.
- **Limitation:** heavy rain (≥10 mm/hr), which drives flash floods, is not yet predicted better than
  persistence at any lead. Radar alone cannot see storms before they form or reach the domain.

## Why CorrDiff and Earth-2

Phase 2 adds the information radar lacks:

1. **CorrDiff downscaling** (ERA5 → ~2.5 km rainfall over Singapore): the large-scale environment
   that decides where equatorial convection forms. NVIDIA's guidance caps reliable super-resolution
   at ~×11 when inferring a new variable, so from ERA5 (~28 km) the realistic target is ~2.5 km,
   with the radar coarsened to match. The 240 km radar gives a far larger target domain than the
   70 km product (35 × 63 km is only ~2 × 3 ERA5 cells).
2. **Satellite + wide-range radar for storms**, following the approach of NVIDIA's Earth-2
   Nowcasting model (StormScope, GOES satellite + radar, trained over CONUS): Himawari-9 plays the
   role of GOES for Singapore. An equatorial adaptation is novel: near-daily afternoon convection,
   sea-breeze convergence, no extratropical fronts.

## Data (all publicly accessible)

| Data | Coverage | Status |
|---|---|---|
| NEA 70 km radar | 2026-05-22 → ongoing, 5-min | 128 days collected and decoded (27 Sep 2026) |
| NEA 240 km radar | ~2026-08-28 → ongoing, 5-min, 480 × 480 | collection started 2026-09-27 (NEA keeps ~30 days, backfilled) |
| ERA5 (Copernicus) | 2022–2023 downloaded; 2026-05-22 → present to download | pairing with radar needs matching times |
| PUB flood alerts (Telegram) | 2026-05-22 → ongoing | geocoded, radar-matched |
| Himawari-9 (JMA, public archive) | backfillable | planned |

## Plan and compute

1. Download ERA5 for the radar period; build the paired dataset and a `DownscalingDataset`.
2. **Local spike on the RTX 4060 (8 GB):** CorrDiff regression stage only, at reduced size. Go/no-go:
   any skill over bilinear-interpolated ERA5 precipitation.
3. If go: full regression + diffusion training. CorrDiff-Mini-scale training is ~10 A100-hours per
   NVIDIA's documentation; options are a short cloud GPU rental or NVIDIA programme credits.
4. Evaluate against bilinear ERA5 and our radar nowcaster (CRPS, FSS at 2.5 km, spatial spectra),
   and publish the code, results and limitations openly.

## Applicant

Independent researcher/developer in Singapore. NVIDIA developer account active. Code and results:
GitHub repository (link to be added once public).
