# Task Checklist

## Phase 2 — publish, dashboard, CorrDiff, heavy-rain data (started 2026-09-27)

> Plan: `tasks/plan_phase2.md` (approved 2026-09-27). **Resume here after a shutdown:**
> the first unchecked item is the next step. Background jobs are listed under "Running".

### Running / scheduled
- 240 km backfill (one-off, ~3 h from 11:52 SGT 27 Sep): `scripts/backfill_240km.bat` -> `logs/backfill_240km.log`. Safe to re-run after a shutdown (skips existing); only the oldest day is lost per day missed.
- 240 km ongoing: added to `SG-Weather Radar Continuous` (every 30 min, --hours 2) and `SG-Weather Radar Scraper` (daily 08:00, --hours 120). Task XML backups: `logs/task_backup_20260927/`.

### New event — 27 Sep 2026 flash flood (out of sample: after the pinned test period)
- [x] Collected: Neo Pee Teck Lane / Pasir Panjang Rd junction, FLASH_FLOOD 11:57 SGT (subsided 12:08), amid 29 flood-risk warnings from 11:05. Site is PUB flood-prone #26. Geocode hand-fixed to the layer's point, cell (63,78)
- [x] Observed (7x7 box): dry to 11:05, 61 mm/hr 11:20, 100 mm/hr 11:30-11:40
- [x] Forecasts `data/processed/eval_cache/case_27sep_lead{30,60,90}.npz`. **30 min: first flag issued 10:35 SGT (P>=10 0.38, persistence dry) = 35 min before rain reached the site, 30 min before PUB's first warning anywhere (11:05), 82 min before the flood report.** Median intensity ~10x low (8-10 vs 61-100). 60 min: one flag (P 0.50) issued 10:35 for 11:40. 90 min: none
- [ ] Add as second case study (notebook 02 section, dashboard, docs/RESULTS.md). No retraining: one event does not change the training data materially; retrain when months more radar exist (e.g. NE monsoon)

### D. Heavy-rain data
- [x] D1a Probe NEA 240 km product: 5-min, 480x480 RGBA PNG ~45 KB, **served ~30 days back** (70 km: ~7 d). ~13 MB/day.
- [x] D1b `scrape_radar.py --product 240km` -> `data/raw/radar_240km/` (70 km default unchanged)
- [x] D1c Scheduled collection + verified a task run under the conda env (result 0, 12 files for its 2 h window)
- [ ] D1d Backfill complete — check `logs/backfill_240km.log` for "backfill exited"; expect ~8.6k images from ~2026-08-28
- [ ] D1e 240 km gap alert (extend `check_radar_gaps.py` or a `--status` check) + colour-legend/georeference check of the 240 km PNG before any modelling
- [ ] D2 Rain-gauge API history depth (data.gov.sg)
- [ ] D3 Stage 7 heavy-rain plan after 2-4 weeks of 240 km data

### A. Repo hygiene & docs
- [x] A1 cleanup: portable launchers, results/ablations/ (+README), docs/history/, spec wording, per-lead eval output (5c1a431). ERA5 tmp files KEPT: they are download_era5.py's cache (17 MB, gitignored)
- [x] C0 LaunchPad brief rewritten with verified facts; June package -> docs/history/launchpad_2026-06/. **User action: apply for the PhysicsNeMo lab**
- [x] A2 README.md (results table, mermaid diagram, repo map, NEA notice, roadmap)
- [x] A3 docs/ METHODS, RESULTS, LIMITATIONS, LESSONS, REPRODUCE, DATA + docs/img. Every command in REPRODUCE was run; all numbers read from results/*.json; links checked
- [x] A4 LICENSE (MIT, holder "beese54" — user may change to real name) + CC BY 4.0 docs note + CITATION.cff

### B. Dashboard (static, GitHub Pages)
- [x] B0 `scripts/build_dashboard.py` data export (greyscale sprites, 4.7 MB; results read from results/*.json)
- [x] B1-B7 hero loop, denoising player, explorer (27 & 22 Sep, 30/60/90, flood-prone overlay, site chart), evidence, live timeline, limits; verified in Chrome (dark/light, 390 px, controls, no console errors) — see dashboard/README.md
- [x] B8 240 km loop added (19 frames, 22 Sep 14:30-17:30 SGT; caption states it is not yet georeferenced); failed sprite loads now retry
- [x] Pages workflow `.github/workflows/pages.yml` (activates at publish)

### E. LinkedIn
- [x] E1 screenshots via the Chrome extension -> docs/img/linkedin/ (hero 22 Sep peak, explorer 27 Sep first warning, noise-to-rain); GIF `27sep_nowcast.gif` built in Python (`scripts/make_social_gif.py`) — a browser GIF export would need a download
- [x] E1b screenshots re-taken after the plain-language rewrite -> docs/img/dashboard/1..6 (README gallery + post); outdated docs/img/linkedin/*.jpg removed; post rewritten as prose
- [x] E2 post draft `content/linkedin_post_phase1.md` (gitignored): V1 short, V2 full, first comment, alt text, claim-by-claim truth check. **User posts it**; URLs filled in after publishing

### Publish
- [x] A5 published 2026-09-27: https://github.com/beese54/singapore-diffusion-nowcast (public), dashboard https://beese54.github.io/singapore-diffusion-nowcast/ via GitHub Actions Pages.
  Gate: 0/7 secret values and 0 key-format hits in 79 commits; history rewritten (user choice) to the GitHub no-reply email with the Windows username scrubbed (backup: checkpoints/pre-rewrite-2026-09-27.bundle); re-scanned the published clone: clean. Live site rendered headless: 14 canvases, correct verdict. Needed `gh auth refresh -s workflow` (done by user).
- [ ] USER: post on LinkedIn (content/linkedin_post_phase1.md, links filled; images docs/img/linkedin/)
- [x] ~~USER: apply for NVIDIA LaunchPad~~ dropped 2026-09-27: console needs an enterprise sign-in; hosted CorrDiff NIM deprecated. Spike runs locally.

### F. Flood-response layer (from the user, 2026-09-27)
- [ ] F1 "minutes ahead of PUB's drain sensor" benchmark over all out-of-sample drain alerts
- [ ] F2 per-location P(drain alert | forecast rain) once enough alerts accumulate (months)
- [ ] F3 check whether PUB's public water-level page can be used for continuous levels

### C. CorrDiff
- [ ] C1 ERA5 2026-05..09 download RUNNING (started 2026-09-27 ~20:30 SGT): `scripts/download_era5_corrdiff.bat` -> `logs/era5_corrdiff.log`, store `data/raw/era5/era5_corrdiff_2026.zarr`; resumable (re-run the .bat after a shutdown)
- [x] C2 CorrDiff regression spike DONE 2026-09-27: **NO-GO** (RMSE tie with ERA5 and with always-zero; never reaches 1 mm/hr). Real signal: island-rain timing correlation 0.64 vs ERA5 0.22 (diff CI [+0.02, +0.66]). No cloud spend. See tasks/spike_corrdiff.md
- [x] C3 ERA5 conditioning DONE 2026-09-28: **DO NOT KEEP** (env vs control: FSS≥10 −0.076, catch −0.029, CIs below 0). Control itself worse than base (fine-tune LR restart, L032). Model of record unchanged; option kept default-off. See plan_era5_conditioning.md, L032/L033.
- [x] Recovered 141 missing slots for 22 Sep (laptop off 06:15–17:55 SGT); coverage now 241/241
- [x] Measured NEA retention: ~6.5–7 d (-156h served, -168h 404) — this is the real safe laptop-off window
- [x] Verified no colorbar/legend leak in the decode (4 independent checks; `UNKNOWN_COLOURS` empty over 937k px + 400 random frames). The sustained 84.7–100 mm/hr was real: 22 Sep exceeded 99.98% of archive frames by area ≥60 mm/hr
- [x] Rebuilt `flood_eval_dataset.parquet` after the backfill — matched flood events 46/48 → **48/48** (lesson L014)

### Gap alerting — `scripts/check_radar_gaps.py`
- [x] Severity keyed to **verified** recoverability, not gap existence: CRITICAL (recoverable+aging, or overlaps a flood label) / WARN (recent; zarr behind PNGs) / INFO (past retention, or never published upstream — suppressed forever)
- [x] `probe_upstream()` confirms NEA actually serves a gap before paging — added after the first live run printed CRITICAL for 4 frames that all 404'd (lesson L015)
- [x] Mirrors `check_flash_flood.py`: durable flag (`checkpoints/RADAR_GAP_ALERT.flag`) + best-effort toast + JSON dedup state; CRITICAL re-pages at most every 24h
- [x] `tasks/repro/gap_alert_tiers_repro.py` — **7/7 tiers verified**
- [x] Live end-to-end: real WARN (zarr 9 frames behind) fired with flag+toast → printed runbook cleared it → next run silent
- [x] Registered as 5th action on `\SG-Weather\SG-Weather Telegram Labels` (daily 09:00 SGT, conda `sg-weather`; deps confirmed present). Task backed up before edit; verified Ready, 1 trigger, 5 actions
- [ ] Consider collapsing the 332 known-unrecoverable gaps into a single suppressed record (state file is currently one id per gap)

### Stage 4 unblocked + launched
- [x] Found the blocker stale: "until 90-day archive (~late Aug 2026)" had been satisfied ~1 month (archive 123 d / 33,102 frames) — lesson L013
- [x] Fixed `num_workers` default 2 → 0: `RadarDataset` is 3.45 GB in RAM and Windows spawn cannot pickle it (crashed `OSError 22`/truncated pickle; worked in June at 1.21 GB). `num_workers>0` now raises with the real reason
- [x] Measured throughput: **132 ms/step (7.56 steps/s)** → 300k steps ≈ 11 h on RTX 4060
- [x] Confirmed today's floods sit at index 33,058 → **TEST** split (clean holdout), 139 usable 22-Sep samples; contiguity passes only because of the backfill
- [x] Launched real run: `python train.py --max-steps 300000 --batch-size 4 --num-workers 0` → `logs/nowcaster_train.log`
- [ ] Score the run against the 139 usable 22-Sep test samples (needs Stage 5 tasks 5.1/5.2/5.3 — `evaluate.py` does radar-vs-radar FSS/CRPS only and never reads flood labels)
- [ ] Proper fix: lazy/memmap reads in `RadarDataset` so workers become usable again as the archive grows


## Georeference + colour-mapping correction, full re-ingest (2026-08-11) — tasks 3.7 / 3.8

> Approved scope: tasks 3.7 and 3.8. Diagnosis found 3.7 understated, 3.8's premise wrong,
> and a third defect. Ran under Pattern L (Data Migration): expand → migrate → contract, with
> validation queries declared before execution. Full ledger in `docs/history/migration_plan.md`.

**What was actually wrong** (all four verified by reproduction, not inspection):
1. Crop constants written for a 480×480 composite; images are 217×120, so the guard clamped
   to the **bottom-right corner** and bilinear-upsampled it. lat/lon coords were fiction.
2. Bilinear resampling of a **banded** field manufactured rain (9 → 7,302 distinct values;
   nonzero 68.5% → 95.8%). **This, not alpha, was the "phantom drizzle ≤4 mm/hr floor."**
3. Nearest-RGB matching against 11 anchors vs NEA's **33 real colours** → 8/33 non-monotonic.
4. Alpha drop: harmless in practice (binary alpha, transparent RGB = black = 0 mm/hr).
   **Task 3.8's stated cause was wrong** — these PNGs contain no basemap at all.

- [x] Colour ramp ordering derived from the archive by three independent, agreeing methods
      (distance-transform depth, spatial-adjacency graph walk, RGB continuity); values
      log-uniform 0.5–100 mm/hr = 0.719 dBR/band, justified by the product being dBR
- [x] `preprocess_radar.py`: exact-match 33-colour LUT (unknown colours counted, never
      snapped), explicit alpha mask, crop + resize deleted, native 120×217 grid from verified
      bounds, `--rebuild PATH`; `ensure_sorted_archive()` parameterised by store
- [x] `geocode_flood_labels.py --reindex`: recomputes cell indices from cached lat/lon
      **without** re-resolving (protects the 4 hand-curated entries, incl. KPE from 2c4957d)
- [x] Dry run: `radar_v2.zarr` built from 21,005 PNGs in 13 s; **9/9 validation checks passed**
- [x] Cutover gated on explicit approval; writers paused for the rename, old store retained
- [x] Derived artifacts rebuilt: `radar_stats.json` (n=21,005 vs stale n=2,357 from Jun 5),
      geocode cache re-indexed (18 entries), `flood_eval_dataset.parquet`
- [x] Verified: 0 NaN · strictly increasing · PNG/zarr in exact sync · preprocess idempotent
      (21,005 skipped, 0 errors) · scheduled chain appended 5 frames, LastTaskResult 0 ·
      100-step training smoke test passed on 120×217 (loss 0.4313, stage flag withheld)
- [x] DoD Stage 3 colour + georeference criteria now checkable and checked; tracker synced;
      lessons L011 (silent clamp) and L012 (banded fields) written

### Review
- **The decisive result is check 9.** Rain within ~1.5 km of a geocoded flood cell at event
  time went from **19% of events (median peak 0.00 mm/hr) to 100% (median peak 51.6 mm/hr)**.
  That is the physical proof the new georeferencing is right, and it is the acceptance gate
  L009 demands — no bounds claim was trusted on documentation alone.
- **This closes L008.** "Flood cells show 0 mm/hr" was never warning semantics. L009 fixed the
  timezone half in July; this crop bug was the other half. L008's caveat can now be resolved:
  Stage 5 has real spatial signal, which unblocks the design of task 5.5.
- **Two approved tasks, and one of them was aimed at the wrong thing.** 3.8 had a plausible,
  documented, month-old diagnosis that turned out to be false — the fix it asked for would
  have changed nothing. Reproducing the symptom before implementing the approved fix is what
  caught it. Same failure mode as L008→L009: a plausible story that explains the evidence is
  not the same as the cause.
- Stage 3 day count is unaffected (81/90, ETA 2026-08-20) — PNGs are the source of truth and
  were never touched. Stage 4 will now train on correctly georeferenced, lossless data.
- **Open:** `radar.zarr.old` (74 MB) is retained pending a soak-period contract gate — delete
  it only after Stage 4 has trained successfully on the new archive.
- **Note:** the "▶ RESUME HERE (saved 2026-06-17)" block below is stale (references 26/90 days
  and the Preprocess task deleted in July). Left as-is, superseded by newer sections.

---

## Blank-frame root cause + 24h-off collection guarantee (2026-07-18, session 2)

> Approved scope: the two Stage 3 follow-ups + ensure collection survives laptop-off up to 1 day.
> DIAGNOSIS OVERTURNED THE JUL-18-MORNING THEORY: blanks are NOT NEA outages and NOT tiny PNGs.
> Evidence: (E1) 66 NaN frames have full-size rainy PNGs, all 862 have PNGs on disk; (E2) 10,900
> tiny PNGs are normal no-rain basemap frames, 9,975 of them valid in zarr; (E3) NaN runs align
> with zarr chunk boundaries and shutdown/travel windows. Mechanism: interrupted/racing appends
> commit timestamps to the time axis without writing their rain_rate rows -> zarr serves
> fill_value=NaN; skip-logic then treats them as done forever. (E4) wake race confirmed: PNG
> downloaded 08:50:55, preprocess globbed at 08:50:12. NEA retention probed: >=120h (all 200s).

- [x] `heal_blank_frames()` in preprocess_radar.py: in-place row recovery from PNGs, grouped
      region writes; auto-runs every preprocess; `--heal` flag. Verified in conda env too
      (zarr 3.1.6 open r+ ok)
- [x] Heal run: all 862 recovered, 0 no-PNG, 0 failed. 4.8 repro re-run: 0 all-NaN frames
      archive-wide, 0 NaN samples in any split; train 8677 -> 9344 samples (+667 recovered)
- [x] Scheduled tasks rewired: Scraper --hours 120 + preprocess chained as action 2, PT2H;
      standalone Preprocess task deleted. scrape_radar.py: PNG-magic validation replaces
      '>280 bytes' (275b no-rain frames were silently rejected)
- [x] Bonus: retention edge probed (~Jul 11-12 = 6-7 days; Jul 8 404) -> one-time
      `--hours 168` backfill launched to recover Jul 12-13 frames lost to the old 25h window
- [x] One-time 168h backfill: 274 frames recovered, 1740 skipped, 3 errors (retention edge,
      early Jul 11). Jul 12/13 now 288/288 on disk
- [x] E2E verified: triggered rewired Scraper task -> chain ran scrape+preprocess, LastResult 0;
      zarr 13,556 -> 14,009 frames, current through 2026-07-18 02:05 UTC, strictly increasing,
      0 NaN; zarr day counts Jul 11/12/13/17 = 286/288/288/287
- [x] Tracker + memory sync (laptop-off constraint now "up to 1 day" — memory updated); commits
- [x] REPORTED to user + tracked as tasks 3.7/3.8 (Pending, awaiting approval): (a) NEA images
      are 217x120 px (uniform since May 22) but crop constants assume 480x480 -> grid
      georeferencing + geocoded cells need calibration; (b) conversion drops RGBA alpha ->
      basemap colours map to phantom drizzle (<=~4 mm/hr floor archive-wide); fix = alpha
      masking + full re-ingest (lossless, all PNGs on disk)

### Review
- Diagnosis discipline paid off: the morning's "NEA outage" theory (already written into the
  tracker) was falsified by forensics before any code was built on it. Blank frames were
  write-path corruption; the data was never lost, only unread.
- Recovery totals: 862 zarr frames healed in place + 274 PNGs re-fetched from NEA; archive
  14,009 frames, 0 NaN, all low travel-days (Jun 21/28/30, Jul 7/9/12/13) partially or fully
  restored where retention allowed.
- Collection now survives laptop-off up to ~5 days (user needs 1): 120h daily backfill window
  vs ~6-7 day NEA retention, StartWhenAvailable wake catch-up, sequential scrape->preprocess
  chain (no wake race), self-healing ingest, PNG-magic response validation.
- Follow-ups live in tracker tasks 3.7 (georeference calibration) and 3.8 (alpha-aware
  conversion) — both awaiting user approval since they change data semantics project-wide.

---

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
- [x] Archive ≥ 90 days — reached ~2026-08-20; **123 days** as of 2026-09-22
- [x] `checkpoints/stage3_complete.flag` — written 2026-09-22

## Stage 4 — Nowcaster Training
> UNBLOCKED 2026-09-22 (gate had been clear ~1 month — see lesson L013). Full run in progress.
> Smoke-test command is now: `python train.py --smoke training.max_steps=100 training.batch_size=2 training.num_workers=0`
- [x] Smoke-test — run 2026-06-16. PASSED: no OOM on RTX 4060 (7GB), 25.8M params, loss 1.12→0.52 over 100 steps. Fixes made during testing:
      - `pip install tensorboard` (in requirements.txt but missing from system Python3.12 env)
      - train.py: added `--smoke` flag → isolates ckpts to `nowcaster/smoke/` + skips `stage4_complete.flag` (was falsely self-certifying on `step==max_steps`)
      - train.py: fixed `lstrip("training.")` prefix bug that silently dropped `num_workers`/`target_offset`/`inference_steps` overrides (caused Windows DataLoader spawn OSError)
      - benign fp16 `nan` at step ~20 (GradScaler recovers) — expected, not blocking
- [~] Full run launched 2026-09-22 ~20:50 SGT: `python train.py --max-steps 300000 --batch-size 4 --num-workers 0` (~11 h; resumes with `--resume auto`)
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
