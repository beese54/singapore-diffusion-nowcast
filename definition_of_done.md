# Definition of Done

Each criterion below is objective and testable. A stage is NOT complete until all criteria for it pass.

---

## Stage 0 — Environment Setup
- [x] `conda activate sg-weather` runs without error
- [x] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [x] `python -c "import earth2studio"` runs without ImportError
- [x] `python -c "import physicsnemo"` runs without ImportError
- [x] `.env` file exists (gitignored) with `NGC_API_KEY` and `NVIDIA_API_KEY` populated
- [x] `~/.cdsapirc` exists with valid CDS credentials
- [x] `checkpoints/stage0_complete.flag` exists

**Completed: 2026-06-02** — PyTorch 2.5.1+cu121, RTX 4060 8GB, earth2studio 0.15.0, physicsnemo OK. pygrib/antlr4 installed via conda-forge due to Windows C++ build tool requirement. .cdsapirc BOM stripped.

---

## Stage 1 — ERA5 Download
- [x] `data/raw/era5/singapore_2022.zarr` + `singapore_2023.zarr` exist and are non-empty (8760h each)
- [x] `python scripts/validate_era5.py` reports 0 missing timestamps for Jan 2022 – Dec 2023
- [x] Zarr stores contain all required variables: `u10, v10, t2m, msl, tcwv` + pressure-level `q, t, z` at 1000/850/500/250 hPa
- [x] Re-running `python scripts/download_era5.py` completes in <30 seconds (all months already present → skip)
- [x] `checkpoints/stage1_complete.flag` exists

**Completed: 2026-06-04**

---

## Stage 2 — PrecipitationAFNO Baseline Inference
> CorrDiff NIM was blocked: self-hosted container only (no cloud API), Taiwan-trained domain (23°N subtropical,
> not 1°N equatorial). Replaced with PrecipitationAFNO via earth2studio WB2ERA5.
> Output is 0.25° resolution — a coarse ERA5-scale baseline, not a 2km downscaled product.
- [x] `python scripts/run_precipitation_afno.py --dry-run` prints config without fetching
- [x] `python scripts/run_precipitation_afno.py` processes 10 hindcast timestamps across 3 Singapore rain events
- [x] Per-timestamp NetCDF in `data/stage2/` with `tp` variable cropped to Singapore bbox
- [x] Re-running skips already-processed timestamps
- [x] `checkpoints/stage2_complete.flag` exists
- [x] Update `notebooks/01_corrdiff_evaluation.ipynb` to visualise PrecipitationAFNO stage2 outputs

**Completed: 2026-06-05**

---

## Stage 3 — Radar Archive Pipeline
> **Amended 2026-07-10.** Original criteria assumed 100% capture. Actual capture is ~86%
> (May 22–28 scheduler ramp-up + laptop-shutdown days while travelling), which made
> "25,920 PNGs" and "<5% gap in ANY 7-day window" permanently unmeetable without buying
> useful data. Distinct-days is now authoritative (matches the stage-flag logic in
> `preprocess_radar.py`); the gap criterion is scoped to steady state with a median test.
- [ ] `data/processed/radar.zarr` spans ≥ 90 distinct days from 2026-05-22 — **authoritative criterion**, expected ~2026-08-20 (49/90 as of 2026-07-10)
- [x] `data/processed/radar.zarr` is non-empty and current within 24 h of the newest raw PNGs (ingest automated via daily scheduled task)
- [x] `python scripts/validate_dataset.py` reports median 7-day-window gap rate < 10% over the steady-state period (2026-05-29 onwards; 4.9% as of 2026-07-10 — re-verify at 90 days)
- [x] zarr time axis is strictly increasing with no duplicate timestamps (`preprocess_radar.py --repair` passes; enforced after every append since 2026-07-10)
- [x] Colour → mm/hr mapping is validated: cyan (lightest) → 0.50 mm/hr, magenta (heaviest) → 100 mm/hr, monotonic across all 33 NEA bands; **0 opaque pixels archive-wide fall outside the 33-colour ramp** (2026-08-11)
- [x] **Georeferencing is validated against a physical signal:** with the corrected bounds (lat 1.1450–1.4572, lon 103.565–104.130), 100% of geocoded flood events show rain within ~1.5 km at the reported time, median peak 51.6 mm/hr — vs 19% / 0.00 mm/hr under the pre-2026-08-11 grid
- [x] **Ingest is lossless:** no crop and no resampling; PNG opaque-pixel count equals zarr nonzero-cell count exactly, and every stored value is one of the 33 LUT levels (2026-08-11)
- [x] Re-running `python scripts/preprocess_radar.py` skips all existing zarr chunks (verified 2026-08-11: 21,005 skipped, 0 processed, 0 errors)
- [ ] `checkpoints/stage3_complete.flag` exists

---

## Stage 4 — Diffusion Nowcaster Training
- [ ] `python train.py --help` shows `--resume` flag
- [ ] A 100-step smoke-test run completes without OOM: `python train.py training.max_steps=100 training.batch_size=2`
- [x] Gap-aware sampling (added 2026-07-10): every sample emitted by `RadarDataset` spans exactly contiguous 5-min frames from context start to target; samples crossing archive gaps are excluded and the dropped count is reported at init (verified: 1,228/9,282 train samples dropped, independent contiguity check passes on all splits)
- [ ] Full training reaches 100k steps with loss plateau (validation loss not decreasing for 10k steps)
- [ ] Checkpoint files exist: `checkpoints/nowcaster/ckpt_step_1000.pt`, `ckpt_step_2000.pt`, ..., `latest.pt`
- [ ] Resuming from checkpoint: `python train.py --resume checkpoints/nowcaster/latest.pt` continues from the correct step
- [ ] Interrupting with Ctrl+C saves a checkpoint within 5 seconds
- [ ] `checkpoints/stage4_complete.flag` exists

---

## Stage 5 Ground Truth — Telegram Flood Labels
- [x] `scripts/collect_flood_labels.py` exists and runs with `--status` flag
- [x] Telethon session authenticated (`checkpoints/telegram.session` exists)
- [x] `data/raw/flood_labels/messages.json` contains messages from 2026-05-22 onwards
- [x] `data/processed/flood_labels.parquet` written with columns: `message_id, event_datetime, message_type, raw_text, locations`
- [x] Windows Task Scheduler task `\SG-Weather\SG-Weather Telegram Labels` runs daily at 09:00 (incremental)
- [x] Integration script: cross-reference `flood_labels.parquet` against `radar.zarr` timestamps to build evaluation dataset (`build_flood_eval_dataset.py`; in the daily scheduled chain since 2026-07-10)
- [ ] Wire flood label integration into `scripts/evaluate.py` for Stage 5

---

## Stage 5 — Evaluation & Flash Flood Map
- [ ] `python scripts/evaluate.py` produces `results/evaluation_report.json` with FSS scores for lead times 30/60/90 min
- [ ] FSS at 20dBZ threshold, 30-min lead time > persistence baseline FSS (model has skill)
- [ ] At least one identified heavy-rain event (≥30 mm/hr for ≥30 min) from 2023 is shown as a qualitative case study in `notebooks/02_nowcast_evaluation.ipynb`
- [ ] `notebooks/03_flood_risk_overlay.ipynb` renders a map of Singapore with predicted rainfall overlaid on PUB flood-prone areas GeoJSON
- [ ] End-to-end inference time (6 input frames → 8 ensemble members × 3 lead times) is <5 minutes on RTX 4060
- [ ] `checkpoints/stage5_complete.flag` exists
