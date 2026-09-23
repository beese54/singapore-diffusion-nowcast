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
> **⚠ PARTIALLY SUPERSEDED by L009 (2026-07-10):** the all-zero cells described below were primarily an 8-hour timezone bug in the labels, not warning semantics. After the fix, several geocoded cells show real rain at/near warning time. The rule (lead-time window + neighbourhood, wait for FLASH_FLOOD events) still stands, but the evidence here was contaminated.
**Finding:** Geocoded all 3 current flood events to radar cells; every one showed ~0 mm/hr at the cell AND in a ~1.5–3 km neighbourhood at the event time, while the grid-wide rain max (8.5–18) was elsewhere (north SG). Verified it is NOT a radar artifact (no persistent hotspot across 5,605 frames).  
**Root cause:** All are `FLOOD_RISK` (PUB *warnings* — rain approaching, or flooding lagging the rainfall), so the rain peak is spatially/temporally offset from the reported flood point at the message timestamp.  
**Rule:** Do not evaluate the nowcaster by point-in-time+point-in-space matching of FLOOD_RISK to radar. A meaningful Stage 5 spatial eval needs (a) real `FLASH_FLOOD` occurrences (we have 0 so far), (b) a lead-time window, and (c) a neighbourhood, not a single 0.3 km pixel. This is why evaluate.py wiring is deferred to Stage 5.

### L007 — radar.zarr time is UTC, despite the SGT-looking scraper code
> **⚠ CORRECTED by L009 (2026-07-10):** radar.zarr time IS UTC (that part holds), but the "verified empirically" claim below was wrong — the label side of that join was SGT mislabelled as UTC, so the 09:49 event was 09:49 **SGT** (morning), not "17:49 SGT afternoon flood". The no-offset join rule remains valid only because collect_flood_labels.py now converts text times SGT→UTC at parse time.
**Finding:** `scrape_radar.py` computes `sgt_dt = dt + 8h` and a comment says filenames are "SGT" — but `image_path(dt)` writes the filename from `dt` (which is `datetime.now(timezone.utc)`), and the +8h is used ONLY to build NEA's SGT-labelled URL. `preprocess_radar.py` parses the filename into the zarr `time` coord, so **radar.zarr.time is UTC (tz-naive)**. flood_labels.parquet is tz-aware UTC.  
**Rule:** When joining flood labels to radar, match directly with NO offset (just drop the label tz). Do not add/subtract 8h. Verified empirically: the 2026-06-11 09:49 UTC FLOOD_RISK = 17:49 SGT (classic afternoon flood) lands on real rain under the direct match.  
**How to apply:** Any new radar↔X temporal join: treat radar time as UTC. If tempted to shift for SGT, re-read this — the shift is already baked out at archive time.

### L006 — Windows + system Python: don't trust `tee` exit codes; verify env from requirements
**Findings:** (1) `python ... | tee log` returns `tee`'s exit code (0) even when Python crashes — a crashed smoke test looked "successful." Use `> log 2>&1` and check `$?`. (2) The active runtime is the **system Python 3.12**, not the conda `sg-weather` env; deps listed in `requirements.txt` (e.g. `tensorboard`) were missing. (3) `rich` console crashes with `UnicodeEncodeError` on non-cp1252 glyphs (e.g. `≥`) in the legacy Windows terminal — stick to ASCII in console output.

---

## Session 2026-07-10

### L009 — Times embedded in message text are local wall-clock; validate every timestamp against a physical signal
**Mistake pattern:** `collect_flood_labels.py` parsed "[HH:MM hours]" tags from PUB Telegram messages and stamped them onto the (UTC) message date with `.replace(hour=...)` — treating SGT wall-clock as UTC. Every flood label was 8 h late; every radar match hit the wrong frames; geocoded flood cells all read 0.0 mm/hr, which L008 then *rationalized* as "warnings precede rain" instead of catching the bug.
**How it was caught:** Rain at the Jalan Boon Lay cell peaked at 10 mm/hr at 01:55 UTC — 6 min after the 09:49 **SGT** warning (= 01:49 UTC). Three separate events showing zero rain ±90 min was too consistent to be meteorology.
**Fix:** Parse the tag as SGT, resolve to the nearest SGT day (warnings may reference a near-future window start), convert to UTC. Rebuilt parquet from raw messages.json.
**Rule:** (1) Any timestamp extracted from human-facing text is local time until proven otherwise — the transport-layer timestamp (Telegram `msg.date`) is the only trustworthy UTC. (2) Before trusting a temporal join, validate it against a physical signal (did rain actually fall where/when the ground truth says?). A plausible narrative that explains away a null result is a red flag — test the alternative hypothesis first.

### L010 — Zarr appends don't preserve global time order; positional consumers corrupt silently
**Mistake pattern:** `preprocess_radar.py` sorted each batch but appended blindly; a backfill batch (late-syncing PNGs) landed after newer frames → 13 monotonicity violations. Two silent consumers: (a) `build_flood_eval_dataset.py` searched a *sorted* copy of the time axis but `isel()`-ed the *unsorted* store — wrong frames for every index past the first violation; (b) `RadarDataset` builds context/target windows by positional adjacency, so training pairs would have spanned 8-hour backwards jumps.
**Fix:** `ensure_sorted_archive()` (temp store + atomic rename, verify before deleting backup) runs after every append; `--repair` flag for manual runs; build script now maps sorted positions back to zarr indices via argsort.
**Rule:** Never index a store positionally through a sorted *view* of its coordinate — carry the argsort mapping. Any append-only archive feeding positional consumers needs an order invariant enforced at write time, not assumed at read time.

---

## Session 2026-08-11

### L011 — A defensive clamp turned a loud crash into 81 days of silently wrong data
**Mistake pattern:** `preprocess_radar.py` cropped rows 80:270, cols 130:350 — constants written for a 480×480 NEA composite. The real images are 217×120. A "guard against images smaller than expected" (`r1 = min(SG_PIXEL_ROW_MAX, arr.shape[0])`) clamped the crop to rows 80:120, cols 130:217 — **the bottom-right corner** — then bilinear-upsampled 40×87 to 190×220 and labelled it with lat/lon coordinates that described a different place entirely. Every frame from 2026-05-22 onward was wrong. Without the guard, the mismatch would have raised on the first frame in May.
**How it was caught:** Not by the guard, and not by any check that ran for 81 days. It surfaced only when the crop constants were read against a measured image size. The archive had passed NaN audits, monotonicity checks, coverage audits, and a training smoke test — none of which look at *what region* the pixels cover.
**Rule:** A guard that substitutes a plausible value for an invalid one converts a crash into silent corruption. Guards on *shape and units* must reject, not clamp: assert the expected shape and raise. Reserve fallbacks for things that are genuinely optional. If a constant encodes an assumption about external data (image size, projection, field order), verify it against the data at load time and fail loudly when it drifts.
**How to apply:** Any time you write `min(CONSTANT, actual)` or `clip`/`resize`-to-fit against external input, ask what the code does when the input is a *different thing* rather than a smaller version of the same thing. Prefer `if shape != EXPECTED: return None` with a counter, so rejects show up in the run summary.

### L012 — Banded (categorical) fields must not be resampled or nearest-matched
**Findings, all from the same archive:**
1. **Resampling invents data.** The NEA radar overlay is *banded* — 33 discrete colours, each a rain-rate class. Bilinear upsampling turned 9 distinct values in one frame into 7,302 and pushed nonzero cells from 68.5% to 95.8%: interpolation between a 4.30 mm/hr cell and a dry cell manufactured rain that exists nowhere in the source. This — not the dropped alpha channel — was the real cause of the "phantom drizzle ≤4 mm/hr floor" that task 3.8 had been blaming on the basemap for a month.
2. **Nearest-neighbour in RGB space is not order-preserving.** Matching 33 real colours to 11 hand-guessed anchors by Euclidean RGB distance produced **8 non-monotonic mappings**: the lightest teal read 2.0 mm/hr while five *brighter* cyans read 1.0, and the heaviest magenta read 80 while a lighter one read 100. RGB proximity has nothing to do with physical intensity ordering.
3. **The wrong diagnosis survived because it was plausible.** "Basemap colours bleed into the rain field" explained the symptom well enough that it sat in the tracker as an approved-pending task. The PNGs turned out to contain no basemap at all — they are pure transparent overlays.
**Rule:** For categorical/banded raster data: match colours **exactly** (packed-RGB LUT), never by nearest distance; count and report unmatched colours instead of snapping them; and never interpolate — if you must change resolution, use nearest-neighbour or don't resample at all. A validation that "every stored value is one of the N legal levels" catches all three failures at once, and is cheap.
**How to apply, and the meta-lesson:** Derive the class ordering *from the data* rather than from intuition about the palette. Here, three independent methods agreed exactly — distance-transform depth (outer bands sit shallower in rain cells), a spatial-adjacency graph walk (each band touches only its neighbours in intensity), and RGB ramp continuity. Agreement across independent methods is what makes an ordering claim safe to build on. Same discipline as L009: when the fix depends on an assumption about external data, find a physical signal that tests it.

---

## Session 2026-09-22

### L013 — A blocker phrased as a future date keeps blocking after it comes true
**Mistake pattern:** Stage 4 read `"Blocked until Stage 3 reaches 90-day archive (~late August 2026)"`. The archive passed 90 days on ~2026-08-20 and stood at **123 days / 33,093 frames** by 2026-09-22 — the gate had been clear for a month, but nothing re-evaluated it, so training never started. The tracker's own `last_updated` was 2026-08-11, nine days *before* the condition was met.
**How it was caught:** Only by measuring the archive span while answering an unrelated question ("can we predict today's flood?"). No check existed whose job was to notice.
**Rule:** A status of `Blocked` must record the *condition*, the *measurement* that tests it, and be re-tested whenever it is read — not a prose ETA. A blocker whose condition is satisfiable by the passage of time will silently outlive its own reason.
**How to apply:** Write blockers as an assertion someone can run (`radar.zarr span >= 90d` → `preprocess_radar.py --status`). When reading any `Blocked` item, evaluate the condition before trusting the status. Prefer a flag file written *by the check* (`stage3_complete.flag`) over a human note about when the check will probably pass.

### L014 — A point-in-time derived dataset does not re-derive itself when its inputs are backfilled
**Mistake pattern:** `flood_eval_dataset.parquet` is a join of flood labels against `radar.zarr`. The labels for today's two flash floods were collected at 20:02 while the radar for 06:15–17:55 SGT was still missing (laptop off), so both events were written with `matched=False`, `radar_frame_index=-1`, NaN rain. Backfilling the 141 missing frames did **not** re-join them; they stayed unmatched until the builder was re-run by hand (46/48 → **48/48** matched flood events).
**Why it mattered:** Stage 5 scores from this parquet, so the two most valuable events in the archive — the first real FLASH_FLOOD labels since collection began — would have been silently absent from evaluation. `RadarDataset._contiguous` compounds it: any sample whose context+lead span crosses a hole is dropped, so an unbackfilled gap deletes the event from *training* too.
**Rule:** Any artifact derived from two sources that arrive at different times must either re-derive on input change or be re-run explicitly after any backfill. Order is always scrape → preprocess → build, and "no error" is not evidence the join succeeded.
**How to apply:** After any gap heal, re-run `build_flood_eval_dataset.py` and confirm `--status` reads `flood events: N/N`. A partial match count is the signal — it never raises. `check_radar_gaps.py` now also warns when the zarr is behind the PNGs, which is the upstream half of this same failure.

### L015 — An alert that assumes it is actionable will teach you to ignore it
**Mistake pattern:** The first version of `check_radar_gaps.py` treated "gap inside NEA's ~6.5d retention window" as "gap is recoverable" and paged CRITICAL. Its first live run raised two such gaps (2026-09-16 03:15, 2026-09-17 11:35; 4 frames). Running the runbook it printed returned **404 for all four** — in-window, but NEA never published those slots. Nothing could ever clear them, so they would have re-paged every 24h until they aged out: the exact alert-fatigue failure the severity design was built to prevent.
**How it was caught:** Only by *executing the runbook the alert printed*, rather than trusting the classification. The `--status` report looked entirely correct.
**Rule:** Severity must be keyed to *verified* actionability, not inferred actionability. If an alert claims something can be fixed, the check should confirm the fix is available — and an alert nobody can act on is INFO and suppressed, never a page.
**How to apply:** `probe_upstream()` samples up to 4 slots per in-window gap against NEA before classifying. Generalise: for any alert, ask "what happens if the remedy fails?" — if the answer is "it fires again forever", the remedy's availability belongs *inside* the check. Verify an alert by running its own runbook (Pattern AP Phase 4), not by reading its output.

### L016 — A GradScaler that only halves is a one-way ratchet, and it fails silently
**Mistake pattern:** The Stage 4 run was configured for fp16 + `GradScaler`. The forward pass overflowed on roughly **1 step in 10**, and GradScaler halves its scale on every inf/nan. Its `growth_interval` needs **2,000 consecutive** clean steps to double the scale back — impossible at a 10% failure rate. So the scale only ever went down: **2¹⁹ (524,288) at step 10,000 → 8.9e-16 at 15,000 → 3.8e-37 at 17,000**, a one-way ratchet of ~140 halvings. Below fp32's denormal floor (~1.2e-38) the collapse became self-sustaining: scaling the loss underflows, and unscaling amplifies denormal noise back into inf.
**Why it hid for 7,000 steps:** Nothing crashed and nothing was corrupt on disk. **Model weights stayed finite in every checkpoint (0/127 non-finite tensors, max|w| a stable 31.62)** precisely *because* the scaler was doing its job and skipping those steps. The only visible symptom was `val_loss nan` at step 15,000 — and that signal was itself destroyed by `sum(val_losses)/len(val_losses)`, which turns one non-finite batch out of ~800 into a NaN for the whole epoch. A healthy-looking loss curve and clean checkpoints masked a model that had **stopped learning entirely**: steps 15,000, 16,000 and 17,000 scored *identical* val_loss to five decimal places (0.06826), and step 10,000 scored **0.01467 — 4.7× better**. The collapse first degraded the model, then froze it.
**Rule:** Never judge mixed-precision health by loss values or by checking weights for NaN. Monitor the **scaler scale itself** — a scale trending down over thousands of steps means the run is dying, and it is the only early signal. Prefer **bf16** where the GPU supports it: it has fp32's exponent range, so there is nothing to overflow and no scaler to collapse (measured 0 non-finite events over 40 steps vs fp16's 1-in-10). And never restore a collapsed scaler from a checkpoint — resuming step 17,000 inherited scale=3.8e-37 and produced nan on *every* step.
**How to apply:** `amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16`, with `GradScaler(enabled=...)` only for fp16. Log the scale to TensorBoard alongside the loss. Aggregate validation over **finite batches only** and report the non-finite count, so one overflow costs you a number instead of the whole metric. When a metric can be destroyed by a single bad sample, it is not a metric. Same family as L011: a mechanism that quietly substitutes a plausible value for a broken one converts a loud failure into silent uselessness.

### L017 — A health check that inspects its own runner will always find itself failing
**Mistake pattern:** `check_radar_gaps.py` flagged any scheduled task whose `LastTaskResult != 0` as broken. It runs as the **last action of** the `SG-Weather Telegram Labels` task, so on its first real scheduled run it read its own parent's result as `267009` (`0x41301` = "task is currently running") and reported that task as failed — guaranteed, on every single scheduled run, forever.
**How it was caught:** By reading the alert flag after the 09:00 production run, not by any test. The `--status` output had looked correct every time I ran it **manually**, because run by hand the parent task is idle and reports 0. The bug existed only in the one context that matters.
**Rule:** Any check that observes the system it runs inside must account for its own footprint. And a monitoring script must be verified *in its deployed context*, not just invoked directly — manual invocation is a different environment from the scheduler, and the difference is exactly where self-referential bugs live.
**How to apply:** Whitelist benign states rather than treating "not success" as failure: `BENIGN_TASK_RESULTS = {"0", "267009" (running), "267011" (never run), "267014" (user-terminated)}`. Third false-positive class in this one alert (see also L015 and the zarr-cadence threshold) — which is the real lesson: for alerting code, *every* branch needs a case that proves it does not fire when it shouldn't. Precision is the hard part, not detection.

### L018 — Ask what the data *is* before optimising how it is stored
**Mistake pattern:** Two training runs were killed for host memory pressure. The obvious reading was "the archive is big, 33k frames won't fit" — and the obvious fixes (memmap, lazy zarr reads, float16) all trade something away: extra disk in a OneDrive-synced folder, per-sample decompression, or precision. None were necessary. Two cheaper facts were sitting there unexamined:
1. **The product is banded.** Every pixel is one of NEA's 33 rain levels or zero — exactly **34 distinct values across all 33,246 frames**. float32 was spending 4 bytes to store 6 bits, so uint8 codes into a float32 value table are **lossless** and 4× smaller (3.46 GB → 866 MB).
2. **It was loaded twice.** `train.py` builds a `"train"` and a `"val"` dataset over the same archive, and each `__init__` did its own `.values` — 2 × 3.46 GB = **6.93 GB**, essentially the entire ~7 GB process. The arrays are read-only after load; one shared copy serves every split.
**How it was caught:** Only by measuring `ds.codes.nbytes` against the process RSS and noticing the factor-of-two discrepancy. The duplicate load had been there since the class was written and never showed up as anything but "training uses a lot of RAM".
**Rule:** Before optimising a representation, check what the values actually are and how many times you are holding them. A quantised field stored as float, or any structure loaded once per consumer, is free money — and unlike a precision or I/O trade it costs nothing, so it does not need to be justified against a deadline. Count distinct values; count copies.
**How to apply:** `np.unique` over a sample of any large numeric array before deciding its dtype — if the count is small the field is categorical and belongs in a codebook. Cache a shared read-only array at module level, keyed on something that invalidates (here: path + frame count, so a grown archive is never served stale). And when the change touches training inputs, prove bit-exactness against the old path rather than asserting it — `decode(encode(x)) == x` over the whole archive, plus `__getitem__` compared sample-by-sample — because a silent encoding error poisons every batch while the loss curve still looks fine. Same discipline as L016: the dangerous failures here are the ones that keep running.
