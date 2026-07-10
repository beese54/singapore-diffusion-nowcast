# Lessons Learned

This file is updated after every correction or unexpected finding. Read at session start.

---

## Session 2026-06-01

### L001 — StormCast is a HRRR emulator, not a general downscaler
**Mistake pattern:** Assuming StormCast can be adapted to any region by replacing the dataset.  
**Root cause:** StormCast uses HRRR as both training target and autoregressive initialization. HRRR covers North America only.  
**Rule:** For non-US regions, always use CorrDiff (which only requires ERA5 as input) rather than StormCast.  
**How to apply:** Any time someone says "adapt StormCast to region X", check whether HRRR covers that region first. If not, redirect to CorrDiff.

### L002 — RTX 4060 Laptop GPU: 8GB VRAM (WMI reports wrong value)
**Finding:** `Win32_VideoController.AdapterRAM` reports ~4GB for RTX 4060 Laptop GPU. Actual VRAM is 8GB (confirmed via `nvidia-smi`).  
**Rule:** Always use `nvidia-smi --query-gpu=name,memory.total` for accurate VRAM; never trust WMI for NVIDIA VRAM.

### L003 — All stages must be resumable; user travels with laptop
**User preference:** Laptop is shut down frequently mid-session. Every script must be idempotent and resume from where it left off.  
**Rule:** Every stage must produce a checkpoint artifact. Scripts check for existing artifacts and skip completed work. Training uses `--resume` flag and saves on Ctrl+C.

---

## Session 2026-06-16

### L004 — Smoke tests must not self-certify a stage
**Mistake pattern:** `train.py` wrote `stage4_complete.flag` and real checkpoints whenever `step == max_steps`, so a 100-step smoke test falsely marked Stage 4 done and dropped toy checkpoints that `--resume auto` would later load.  
**Fix:** Added a `--smoke` flag that routes checkpoints to an isolated `checkpoints/nowcaster/smoke/` subdir (invisible to `find_latest_checkpoint`) and skips the completion flag.  
**Rule:** Any "done" side effect (stage flags, promoted checkpoints, published artifacts) must be gated so short/test runs can never trigger it. Always run test invocations through a clearly-isolated path.

### L005 — Never use `str.lstrip("prefix")` to remove a prefix
**Mistake pattern:** `train.py` did `k.lstrip("training.")` to strip the `training.` prefix from Hydra-style overrides. `lstrip` strips any leading char in the SET `{t,r,a,i,n,g,.}`, so `num_workers`→`um_workers`, `target_offset`→`et_offset`, `inference_steps`→`ference_steps` — silently dropped. On Windows this left `num_workers` at its default of 2, so the DataLoader spawned workers and tried to pickle the multi-GB in-memory radar array → `OSError [Errno 22]`.  
**Fix:** `k[len("training."):] if k.startswith("training.") else k`.  
**Rule:** To strip a literal prefix use `removeprefix()` (3.9+) or a `startswith`+slice; never `lstrip`/`rstrip` for prefixes/suffixes.

### L008 — FLOOD_RISK labels are warnings, not rain observations — don't point-match to radar
**Finding:** Geocoded all 3 current flood events to radar cells; every one showed ~0 mm/hr at the cell AND in a ~1.5–3 km neighbourhood at the event time, while the grid-wide rain max (8.5–18) was elsewhere (north SG). Verified it is NOT a radar artifact (no persistent hotspot across 5,605 frames).  
**Root cause:** All are `FLOOD_RISK` (PUB *warnings* — rain approaching, or flooding lagging the rainfall), so the rain peak is spatially/temporally offset from the reported flood point at the message timestamp.  
**Rule:** Do not evaluate the nowcaster by point-in-time+point-in-space matching of FLOOD_RISK to radar. A meaningful Stage 5 spatial eval needs (a) real `FLASH_FLOOD` occurrences (we have 0 so far), (b) a lead-time window, and (c) a neighbourhood, not a single 0.3 km pixel. This is why evaluate.py wiring is deferred to Stage 5.

### L007 — radar.zarr time is UTC, despite the SGT-looking scraper code
**Finding:** `scrape_radar.py` computes `sgt_dt = dt + 8h` and a comment says filenames are "SGT" — but `image_path(dt)` writes the filename from `dt` (which is `datetime.now(timezone.utc)`), and the +8h is used ONLY to build NEA's SGT-labelled URL. `preprocess_radar.py` parses the filename into the zarr `time` coord, so **radar.zarr.time is UTC (tz-naive)**. flood_labels.parquet is tz-aware UTC.  
**Rule:** When joining flood labels to radar, match directly with NO offset (just drop the label tz). Do not add/subtract 8h. Verified empirically: the 2026-06-11 09:49 UTC FLOOD_RISK = 17:49 SGT (classic afternoon flood) lands on real rain under the direct match.  
**How to apply:** Any new radar↔X temporal join: treat radar time as UTC. If tempted to shift for SGT, re-read this — the shift is already baked out at archive time.

### L006 — Windows + system Python: don't trust `tee` exit codes; verify env from requirements
**Findings:** (1) `python ... | tee log` returns `tee`'s exit code (0) even when Python crashes — a crashed smoke test looked "successful." Use `> log 2>&1` and check `$?`. (2) The active runtime is the **system Python 3.12**, not the conda `sg-weather` env; deps listed in `requirements.txt` (e.g. `tensorboard`) were missing. (3) `rich` console crashes with `UnicodeEncodeError` on non-cp1252 glyphs (e.g. `≥`) in the legacy Windows terminal — stick to ASCII in console output.
