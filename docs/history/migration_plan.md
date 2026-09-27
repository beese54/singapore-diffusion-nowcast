# Migration: radar.zarr georeference + colour-mapping correction

**Date:** 2026-08-11 · **Tasks:** 3.7, 3.8 (+ two defects found during diagnosis)
**Protocol:** Pattern L — Data Migration & Schema Change

## Phase 0 — Classify

- **Change class:** TRANSFORMATIONAL (full re-ingest / reshape). No source data is
  destroyed — all 20,993 PNGs stay on disk and are the source of truth.
- **Volume:** 20,993 PNGs → ~20,993 frames. Grid 190×220 (fabricated) → 120×217 (native).
  Measured conversion ~0.3 ms/frame; full rebuild well under 2 minutes. 172 GB free.
- **Downtime tolerance:** none needed. The new store is built alongside the live one;
  the cutover is a directory rename.
- **Rollback:** rename `radar.zarr.old` back. The old store is retained until a separate
  CONTRACT gate, so rollback stays available after the swap.
- **Concurrent writers:** `\SG-Weather\SG-Weather Radar Continuous` (every 30 min) and
  `\SG-Weather\SG-Weather Radar Scraper` (daily 08:00, chains preprocess). Both are paused
  for the rename and resumed after; the rebuild itself only *reads* PNGs.

## Phase 1 — Defects being corrected

| # | Defect | Evidence |
|---|---|---|
| 1 | Crop constants sized for a 480×480 image; NEA PNGs are 217×120, so the clamp kept the **bottom-right corner** (rows 67–100%, cols 60–100%) and bilinear-upsampled it 4.75×/2.53×. lat/lon coords were fiction. | `preprocess_radar.py:67-70` (pre-fix) |
| 2 | Bilinear resampling of a **banded** field: 9 discrete values → 7,302; nonzero cells 68.5% → 95.8% on a rainy frame. Real source of the "phantom drizzle ≤4 mm/hr floor" that task 3.8 blamed on alpha. | reproduced pre-fix |
| 3 | Nearest-RGB matching against 11 anchors while NEA uses **33 discrete colours** → **8 of 33 non-monotonic** (lightest teal → 2.0 mm/hr while 5 brighter cyans → 1.0). | reproduced pre-fix |
| 4 | Alpha dropped by `.convert("RGB")`. Alpha is binary archive-wide and transparent RGB is always (0,0,0) → mapped to 0 anyway. **Harmless today**; now handled explicitly. | task 3.8 premise **disproved** |

**Colour ramp ordering** was derived from the archive, not assumed — three independent
methods agree exactly: distance-transform depth (monotonic 1.43→12.36 px on outer bands),
spatial-adjacency graph walk (adjacency 287k→473, monotonic decay), and RGB ramp continuity.
Absolute mm/hr assumes a dBR-linear scale (0.719 dBR/band, 0.5→100 mm/hr) — approximate, and
documented as such in `preprocess_radar.py`.

**Georeferencing:** lat 1.1450–1.4572, lon 103.565–104.130 (weather.gov.sg page source,
corroborated by cheeaun/checkweather-sg). Cross-check: 0.002602°/px lat vs 0.002604°/px lon
→ square pixels ~0.29 km, matching the NEA product spec.

## Phase 2 — Validation queries (defined before execution)

Run against `radar_v2.zarr` **before** any swap. Any failure stops the migration.

Executed 2026-08-11 against `radar_v2.zarr` (21,005 frames, built in 13 s).

| # | Check | Expected | Actual | Pass? |
|---|---|---|---|---|
| 1 | Frame count | ≥ 20,947 | 21,005 | **PASS** |
| 2 | All-NaN / partial-NaN frames | 0 / 0 | 0 / 0 | **PASS** |
| 3 | Time axis | strictly increasing, no duplicates | 0 violations | **PASS** |
| 4 | Distinct rain values ⊆ LUT ∪ {0} | 0 outside | 0 outside (34 distinct total) | **PASS** |
| 5 | Unknown-colour opaque pixels | 0 | 0 px, 0 colours (full archive at rebuild + every 20th PNG re-checked) | **PASS** |
| 6 | Grid | (T,120,217); lat desc, lon asc, ~0.0026° | (120,217); lat 1.4559→1.1463, lon 103.5663→104.1287; d=0.002602/0.002604 | **PASS** |
| 7 | Losslessness (50 frames) | PNG opaque px == zarr nonzero | 0 mismatches | **PASS** |
| 8 | Day coverage | 81 days, no per-day regression | 81 days, 0 regressions | **PASS** |
| 9 | **Physical test:** rain within ~1.5 km of geocoded flood cell at event time | new materially > old | **NEW: 100% of 21 events have rain, median max 51.57 mm/hr — OLD: 19%, median max 0.00 mm/hr** | **PASS** |

**9/9 passed.** Check 9 is the decisive result: under the corrected georeferencing every
geocoded flood event has heavy rain at its location at the reported time; under the old
corner-crop, 81% had none. This also resolves the open question behind lesson **L008**
("flood cells show 0 mm/hr") — L009 fixed the timezone half in July, and the remaining half
was this crop bug, not warning semantics. Stage 5 now has real spatial signal.

## Phase 3 — Execute

Approved by the user after the Phase 2 table was presented. Executed 2026-08-11:

```powershell
Disable-ScheduledTask -TaskPath '\SG-Weather\' -TaskName 'SG-Weather Radar Continuous'   # OK
Disable-ScheduledTask -TaskPath '\SG-Weather\' -TaskName 'SG-Weather Radar Scraper'      # OK
Rename-Item 'data\processed\radar.zarr'    'radar.zarr.old'                              # OK
Rename-Item 'data\processed\radar_v2.zarr' 'radar.zarr'                                  # OK
Enable-ScheduledTask  -TaskPath '\SG-Weather\' -TaskName 'SG-Weather Radar Continuous'   # OK
Enable-ScheduledTask  -TaskPath '\SG-Weather\' -TaskName 'SG-Weather Radar Scraper'      # OK
```

No python process was running at cutover; both writers confirmed `Disabled` before the
renames and `Ready` after. Backup at cutover: `data/raw/radar/` (21,005 PNGs, 19 MB,
untouched — the archive rebuilds from it in 13 s) plus `radar.zarr.old` on disk.

## Phase 4 — Verify & Contract

Derived artifacts rebuilt (in this order):

| Artifact | Result |
|---|---|
| `radar_stats.json` | recomputed: n=21,005 (was 2,357 from Jun 5); log_mean 0.0279 (was 0.0660), log_std 0.2148 (was 0.3254) — the old values were inflated by interpolation-smeared drizzle |
| `geocode_cache.json` | 18 entries re-indexed via new `--reindex` (not `--refresh`); all 4 hand-curated entries preserved, coordinates and methods untouched |
| `flood_eval_dataset.parquet` | rebuilt; every geocoded event now shows real rain (e.g. Cambridge Rd `0.0 | 0.0` → `11.62 | 84.74 mm/hr`). 3 unmatched, all in known radar gaps (May 24 ramp-up, Jun 29 zero-data day) |

Post-migration verification, all passed:

- 0 NaN · time axis strictly increasing · 0 duplicates
- PNG and zarr counts in exact sync (21,005/21,005 at swap)
- `preprocess_radar.py` idempotent: 21,005 skipped, 0 processed, 0 errors
- Scheduled `scrape → preprocess` chain triggered live: appended 5 frames (21,005 → 21,010)
  on the new grid, 0 NaN, 0 values outside the LUT, `LastTaskResult 0`
- `train.py --smoke` 100 steps on 120×217: exit 0, loss 0.4313, checkpoint to the isolated
  smoke path, stage flag correctly withheld (L004 guard intact)
- `status.py`: Stage 3 unchanged at 81/90 days, ETA 2026-08-20

### CONTRACT — still open

`data/processed/radar.zarr.old` (74 MB) is **retained**. Delete it only after Stage 4 has
trained successfully on the new archive, as its own approval gate. Until then, rollback is:
pause writers → `radar.zarr` → `radar_v2.zarr`, `radar.zarr.old` → `radar.zarr` → resume.
