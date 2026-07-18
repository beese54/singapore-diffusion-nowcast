# Task Checklist

## Task 4.8 — NaN-aware sample filtering in RadarDataset (2026-07-18)

> Approved scope: update tracker for the blank-frame audit, add task 4.8, implement the filter.
> Defect: radar.zarr holds all-NaN frames at *valid* timestamps (174 in the last 7 days alone;
> multi-hour runs Jul 11/13/17). The 4.7 gap filter only checks time-step contiguity, so windows
> containing blank frames pass into training and would poison the loss. `compute_stats` would
> also return NaN today (plain mean/std; current stats file predates the blank frames).

- [x] Phase 0 — repro `tasks/repro/nan_samples_repro.py`: archive is perfectly bimodal —
      862 all-NaN frames, 0 partial-NaN → validity rule = any-NaN. Pre-fix poisoned samples:
      train 659/9344, val 130/1020, test 107/1045 (896 total passing the 4.7 gap filter)
- [x] Phase 3 — fix in `src/data/radar_dataset.py`: `frame_bad` mask + `bad_cumsum` window
      check over full span [t-ctx, t+offset] (symmetric with `_contiguous`); `compute_stats`
      → nanmean/nanstd/nanmax. radar_stats.json (Jun 5, pre-blanks) deliberately untouched
- [x] Phase 4 — repro post-fix: 0 NaN samples in all splits; drops train 667 / val 130 /
      test 109 (extras vs Phase 0 = conservative full-span rule catching unread mid-window blanks)
- [x] Phase 5 — regression sweep passed: default train init (oversample=3, len 11641),
      __getitem__ finite (6,190,220)/(1,190,220), denormalise round-trip ok, compute_stats
      finite to scratch path; diff touches only Phase-3-justified lines
- [x] Tracker: task 4.8 Complete, Stage 3 QC escalation noted (862 blanks, coverage
      overstatement, open root-cause item), Stage 5 collector verified, last_updated 2026-07-18
- [x] Commits: (1) fix(data) NaN-aware filter, (2) chore tracker sync

### Review
- Mechanism: blank scrapes → all-NaN frames at valid timestamps → pass 4.7 contiguity check
  → log1p(NaN) flows into loss. Fix removes poisoned windows from the index list (O(1)/sample
  via prefix sums); proven by exhaustive post-fix scan of every surviving sample, not masking.
- Open follow-ups (tracked in Stage 3 notes): diagnose multi-hour blank-scrape runs
  (NEA outage vs scraper saving placeholder); consider ingest-side skip + --repair so blanks
  become time gaps at the source. No formal tests/ suite yet — repro script is the durable check.

---

## ▶ RESUME HERE  (saved 2026-06-17)

**Project state:** Stages 0–2 done. Stage 3 (radar archive) is the gate — **26/90 days**, target ~Aug 20 2026. Stages 4 (training) & 6 (LaunchPad) blocked on it. Stage 5 (eval) plumbing largely built.

**Everything runs itself — nothing to babysit.** 4 Windows scheduled tasks under `\SG-Weather\` are all Ready/healthy:
- Radar Continuous (scrape PNGs, 30 min) · Radar Scraper (daily backfill) · **Radar Preprocess (daily 08:00, PNG→zarr)** · Telegram Labels (daily 09:00)
- As of save: 5,722 PNGs, `radar.zarr` current to 2026-06-17 07:06, 12 flood labels (3 FLOOD_RISK / 9 RAIN_WARNING).

**When you come back, pick any of these (all unblocked):**
1. Quick health check: `python scripts/status.py` (or per-task: `--status` on collect_flood_labels / build_flood_eval_dataset / geocode_flood_labels).
2. Geocode the flood-label *location strings* if new flood events arrived: `python scripts/geocode_flood_labels.py` then `python scripts/build_flood_eval_dataset.py`.
3. Optional de-risking before Aug training: re-run the Stage 4 smoke test (`python train.py --smoke training.max_steps=100 training.batch_size=2 training.num_workers=0`).

**Open decisions (deferred, not forgotten):**
- Task 5.5 — wire `flood_eval_dataset.parquet` into `evaluate.py`. HELD until Stage 5: spatial signal is weak today (only FLOOD_RISK warnings, 0 FLASH_FLOOD; rain not at flood cells — see L008). Needs real flood events + lead-time-aware design.

**Nothing is in a broken/half-done state.** See `lessons.md` L004–L008 for gotchas (smoke-test guard, lstrip prefix bug, dual python env, radar=UTC, FLOOD_RISK≠rain).

---

## Stage 0 — Environment Setup ✅
- [x] Run `init.sh` to create conda environment `sg-weather`
- [x] Copy `.env.example` → `.env` and fill in `NGC_API_KEY`, `NVIDIA_API_KEY`
- [x] Configure `~/.cdsapirc` with CDS credentials
- [x] Verify: `python -c "import torch; print(torch.cuda.is_available())"` → True
- [x] Verify: `python -c "import earth2studio; import physicsnemo"` → no error
- [x] `checkpoints/stage0_complete.flag` exists

## Stage 1 — ERA5 Data Download ✅
- [x] `python scripts/download_era5.py`  → singapore_2022.zarr + singapore_2023.zarr complete
- [x] `python scripts/validate_era5.py`  → 0 missing timestamps
- [x] `checkpoints/stage1_complete.flag` exists

## Stage 2 — PrecipitationAFNO Baseline Inference ✅
> Note: CorrDiff NIM was blocked (self-hosted only, Taiwan domain). Replaced with PrecipitationAFNO
> via WB2ERA5 (global ERA5 from WeatherBench2 / Google Cloud). Output is 0.25° — a coarse baseline,
> not the primary deliverable.
- [x] `python scripts/run_precipitation_afno.py --dry-run`  → prints config, no fetch
- [x] `python scripts/run_precipitation_afno.py`  → 10 hindcast timestamps over 3 rain events
- [x] NetCDF outputs in `data/stage2/` (max tp 0.11–0.24 mm / 6h per SG grid cell)
- [x] `checkpoints/stage2_complete.flag` exists
- [x] Run `notebooks/01_corrdiff_evaluation.ipynb` (updated to use stage2 NetCDFs — PrecipitationAFNO, 10 timestamps)

## Stage 3 — Radar Archive (Auto-collecting)
- [x] Windows Task Scheduler running `scrape_radar.py` every 5 min
- [x] Archive started 2026-05-22; 2,546 PNGs as of 2026-06-05 (15/90 days)
- [x] `python scripts/preprocess_radar.py` — run 2026-06-05; 2,546 timestamps in radar.zarr (first: 2026-05-22 01:55, last: 2026-06-05 07:15); bug fixed (encoding= on append mode)
- [x] `python scripts/preprocess_radar.py` — run 2026-06-10; 4,009 timestamps in radar.zarr (first: 2026-05-22 01:55, last: 2026-06-10 12:00, 20/90 days); fixed safe_chunks=False on append path
- [x] `python scripts/preprocess_radar.py` — run 2026-06-16; +844 frames → 5,605 timestamps (last: 2026-06-16 12:25, 26/90 days); run periodically to append new PNGs (idempotent)
- [x] AUTOMATED 2026-06-17: scheduled task `\SG-Weather\SG-Weather Radar Preprocess` runs preprocess_radar.py daily 08:00 SGT (conda sg-weather python; battery-allowed + StartWhenAvailable for travel). Test-triggered → result 0, zarr updated. radar.zarr now stays current without manual runs.
- [x] `python scripts/validate_dataset.py` — run 2026-06-16; 23.5% overall gap rate, but ~1,500/1,722 missing slots are the May 22–28 scheduler ramp-up + in-progress current day. Steady state (May 29–Jun 14) has ≥240 frames/day → healthy. Fixed cp1252 UnicodeEncodeError (≥ → >=).
- [ ] Archive ≥ 90 days (target: late August 2026) before starting Stage 4
- [ ] `checkpoints/stage3_complete.flag`

## Stage 4 — Nowcaster Training
> Blocked until Stage 3 reaches 90-day target (~late August 2026)
> Smoke-test command is now: `python train.py --smoke training.max_steps=100 training.batch_size=2 training.num_workers=0`
- [x] Smoke-test — run 2026-06-16. PASSED: no OOM on RTX 4060 (7GB), 25.8M params, loss 1.12→0.52 over 100 steps. Fixes made during testing:
      - `pip install tensorboard` (in requirements.txt but missing from system Python3.12 env)
      - train.py: added `--smoke` flag → isolates ckpts to `nowcaster/smoke/` + skips `stage4_complete.flag` (was falsely self-certifying on `step==max_steps`)
      - train.py: fixed `lstrip("training.")` prefix bug that silently dropped `num_workers`/`target_offset`/`inference_steps` overrides (caused Windows DataLoader spawn OSError)
      - benign fp16 `nan` at step ~20 (GradScaler recovers) — expected, not blocking
- [ ] Full run: `python train.py`  (resumes with `--resume checkpoints/nowcaster/latest.pt`)
- [ ] Confirm loss plateau; `checkpoints/stage4_complete.flag`

## Stage 5 — Evaluation
- [ ] `python scripts/evaluate.py`
- [ ] Run `notebooks/02_nowcast_evaluation.ipynb`
- [ ] Run `notebooks/03_flood_risk_overlay.ipynb`
- [ ] FSS > persistence baseline → confirm in `results/evaluation_report.json`
- [ ] `checkpoints/stage5_complete.flag`

## Stage 5 Ground Truth Integration
> Telegram flood alerts collected incrementally from 2026-05-22 (radar archive start)
- [x] `scripts/collect_flood_labels.py` — self-contained incremental Telegram collector + rule-based parser
      - **First run (interactive auth):** `python scripts/collect_flood_labels.py`
      - **Status check:** `python scripts/collect_flood_labels.py --status`
      - **Outputs:** `data/raw/flood_labels/messages.json` + `data/processed/flood_labels.parquet`
      - **Task Scheduler:** `\SG-Weather\SG-Weather Telegram Labels` — daily 09:00 (register manually, see below)
- [x] Add Telegram creds to `.env` (`TELEGRAM_API_ID` + `TELEGRAM_API_HASH` present)
- [x] Run once interactively to authenticate Telegram session (`checkpoints/telegram.session` exists; headless runs confirmed 2026-06-16 — "No new messages", 12 rows)
- [x] Register Task Scheduler task — `\SG-Weather\SG-Weather Telegram Labels` exists, Ready, daily 09:00 SGT, LastTaskResult=0, next 2026-06-17 09:00. NOTE: scheduled tasks run the **conda `sg-weather`** python; interactive/manual runs in this repo use **system Python 3.12** (two separate envs — keep deps in sync across both).
- [x] Write integration script: `scripts/build_flood_eval_dataset.py` — temporal cross-reference of flood_labels.parquet against radar.zarr → `data/processed/flood_eval_dataset.parquet`. Run 2026-06-16: 12 labels, all 3 flood events matched to radar within ±1 min (8.6/1.8/8.5 mm/hr); 4 unmatched are May 22–28 archive-gap RAIN_WARNINGs. Has `--status` and `--tolerance-min`. **Clocks: both UTC — radar.zarr names PNGs by UTC instant; the +8h in scrape_radar.py only builds NEA's SGT URL. Join is direct, NO offset.**
- [x] Geocode flood-label locations: `scripts/geocode_flood_labels.py` — simplifies PUB strings, resolves via curated gazetteer -> OneMap API -> editable cache (`data/processed/geocode_cache.json`), maps to radar cells. Run 2026-06-16: 3/3 flood locations resolved (Jalan Boon Lay, Upper Paya Lebar Rd, Thomson Rd). `--offline/--refresh/--status`. Fixed "A off B" simplifier bug (anchor on B). `build_flood_eval_dataset.py` now joins the cache -> adds location lat/lon/idx + radar_rain_at_cell + radar_rain_near_cell (neighbourhood max, `--cell-radius`, default 5).
      - FINDING: all 3 geocoded flood cells show ~0 mm/hr even at r=5/r=10 neighbourhoods, while grid-max is 8.5–18 elsewhere (north SG). Not an artifact (checked: no persistent hotspot). Cause: all 3 are FLOOD_RISK *warnings* (rain approaching / flood lags rain), so point-in-time+space matching is weak. Reinforces deferring evaluate.py wiring until real FLASH_FLOOD events exist + a lead-time-aware design.
- [ ] Wire into `scripts/evaluate.py` for Stage 5 (NOT done — additive change to Stage 5 semantics, awaiting approval. evaluate.py currently does radar-vs-radar FSS/CRPS only and never reads flood labels. Spatial signal currently weak — see finding above; revisit at Stage 5 with more events.)

## LaunchPad (Parallel track — CorrDiff fine-tuning for SG)
- [ ] Apply at https://developer.nvidia.com/launchpad
- [ ] On approval: submit `scripts/launchpad/submit_job.sh`
