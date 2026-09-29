# Plan: Himawari-9 satellite for heavy rain that forms on the spot — for approval

*2026-09-29. Follows the Stage 7 spike (`tasks/plan_stage7_heavy_rain.md`: NO-GO, incoming heavy rain mostly **forms**
rather than drifts in) and the hybrid-warning result (`tasks/plan_hybrid_calibration.md`: at 60 min, plain extrapolation
is the best heavy-rain warning, and it cannot see rain that does not exist yet). Radar sees a storm only once it rains.
A geostationary satellite sees the cloud tops cooling as a storm grows, typically 15–45 min before heavy rain. That is
the information our inputs lack.*

## What was checked today (2026-09-29)

- **Source:** NOAA open data on AWS, `s3://noaa-himawari9` (public, no login, HTTPS). Full-disk Level 1b every 10 min,
  10 horizontal strips per band. **History is complete from before our radar archive began** (22 May 2026: 140 of 144
  slots present), and it does not expire. So no collection race, unlike radar.
- **Singapore (1.15–1.46°N) lies in strip 5 of 10** at 2 km (lines ≈ 2,690–2,715 of 5,500). ±1° of context stays
  inside strip 5 (to be confirmed in Step 1).
- **Size:** infrared band 13 (10.4 µm, cloud-top temperature) is 3.0 MB per strip per scan, band 8 (6.2 µm, water
  vapour) 1.3 MB. So one band is ~0.43 GB per day of download. Only a crop is kept: ~50 kB per scan.
- **Delay:** a scan starting 08:00 UTC was published at 08:11:52. At forecast time t, the newest usable scan
  started ≈ 20 min before t. **Every experiment uses only scans that were available at issue time.**
- **Constraints:** C: has **31 GB free** (97% full), so downloads are streamed: fetch strip → crop → delete. `satpy`
  (Himawari HSD reader) is not installed in either Python environment.

## Step 1 — satellite spike (no training): is the information there?

**Question.** At 60 min, does a simple "growing cold cloud" warning from the satellite catch heavy rain that
extrapolation misses, often enough to be worth building on?

- **Data:** bands 13 and 8, strip 5, for the validation and test periods (2–25 Sep, ≈ 24 days, ≈ 15 GB streamed,
  < 0.3 GB kept). They are reprojected onto a ±1° box around Singapore at 2 km (`src/data/satellite.py`), and a
  georeference check overlays the 22 Sep storm on radar.
- **Parallax:** from 140.7°E, a 12 km cloud top over Singapore appears ~10 km away from where it really is. The
  spike measures the offset on storm cases (the shift that best aligns cold cloud with radar rain) and applies a
  fixed shift. It also scores within a 10 km neighbourhood so that a small residual error does not decide the result.
- **Satellite warning S:** warn a 2 × 2 km box if, within 10 km of it, cloud-top temperature ≤ T **and** it cooled by
  ≥ C over the last 20 min. T ∈ {220, 230, 240} K and C ∈ {4, 8, 12} K are **chosen on validation by CSI**, never on test.
- **Scored on test at 60 min,** every 10 min through the test period (≈ 1,700 forecasts; S and extrapolation need no
  GPU):
  - catch, precision, CSI and incoming catch for S, extrapolation E and E ∪ S;
  - the same on the **initiation subset** (heavy at the target, no radar rain ≥ 1 mm/hr within 10 km at issue);
  - 95% CIs by a **bootstrap over days** (storms are the independent units, L031), not over forecasts.
- **GO (fixed now):** E ∪ S catches more incoming heavy rain than E (CI of the difference above 0), **and** CSI(E ∪ S)
  is not worse than CSI(E) (CI not entirely below 0).
- **NO-GO:** the satellite adds nothing a simple rule can use at 60 min. Record it, and stop satellite work until a
  longer wet season has been collected.
- **Also reported (context, not the decision):** S at 30 and 90 min, the measured parallax, and how often the
  newest scan is missing.

Effort: about ½ day of code, ~2 h of streamed download, minutes of CPU. Nothing touches production code except the
new `src/data/satellite.py` reader.

## Step 2 — only if GO: a satellite-conditioned 60-min model (detail after the spike)

- **Backfill** bands 13 and 8 for the whole radar era (22 May onwards, ≈ 55 GB streamed, ≈ 1 GB kept), resumable.
- **Input:** the last 3 usable scans (30 min, each ≥ 20 min old at issue) as a second context branch at 2 km over
  ±1°, joined through a zero-initialised pathway as in the ERA5 work, so the model starts from the model of record.
- **Training:** warm start from `lead60_warm`, 50k steps at **lr 2e-5** (L032), **with a same-budget control arm
  without satellite**, same seeds (L032).
- **Evaluation:** paired on the pinned test split, with arms base, control, satellite, extrapolation and E ∪ S
  (L035: always the simple baseline). Metrics: CRPS skill, FSS ≥ 2 / ≥ 10, heavy-rain catch/precision/CSI with the
  calibrated rule, initiation-subset catch, drain alerts. **KEEP** if satellite − control on initiation catch or CSI
  has its CI above 0 and CRPS skill is not entirely below 0.
- **Honest limit, stated up front:** ≈ 100 training days (L033), mostly Southwest-Monsoon afternoon storms. The
  Northeast Monsoon (Dec–Mar) will not be in training.

## Step 3 — only if Step 2 KEEPs: real time

A scheduled collector (every 10 min, strip 5, crop only). Its dependencies go into **both** Python environments
(the conda `sg-weather` env for scheduled tasks, system 3.12 for interactive runs). Gap-watching is added to
`check_radar_gaps.py --status`. Real-time use must respect the ~20 min satellite delay.

## Honest limits (stated with any result)

- The 2 km satellite pixel is coarser than the 0.29 km radar and the 2 km warning box. It can say "a storm is growing
  near here", not which street.
- Parallax depends on cloud height, so a single fixed shift is approximate.
- Test ≈ 12 days and a few storms, so the CIs will be wide.
- Only 10-min full-disk scans are used. The 2.5-min "target area" scans move around and do not cover Singapore reliably.

## Steps and outputs

1. `pip install satpy pyresample` (system Python; conda only at Step 3).
2. `src/data/satellite.py`: stream, crop, reproject and cache (`data/processed/sat/*.npz`), resumable, deleting
   raw strips as it goes. `scripts/download_himawari.py --start --end`.
3. `scripts/satellite_spike.py` → `results/satellite_spike.json` + a georeference/parallax figure.
4. Verdict recorded here and in `tasks/todo.md`. Step 2 gets its own detailed plan only on GO.
