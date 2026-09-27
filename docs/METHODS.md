# Methods

## 1. Data

### Radar (model input and target)
- **Source:** NEA / Meteorological Service Singapore rain-area images, 70 km product
  (`weather.gov.sg/files/rainarea/50km/v2/dpsri_70km_<SGT timestamp>0000dBR.dpsri.png`), every 5 min.
  NEA serves roughly the last 7 days, so the archive exists only because it was collected
  continuously from **2026-05-22** (scheduled task every 30 min + a daily 5-day catch-up).
- **Decoding** (`scripts/preprocess_radar.py`): each opaque pixel colour is mapped to one of NEA's
  33 rain-rate levels (0.5 → 100 mm/hr). The mapping is exact — no interpolation, no resampling,
  no crop — because the field is categorical (lesson L012). Verified archive-wide: every opaque
  pixel matches a legend colour, and the PNG opaque-pixel count equals the zarr non-zero count.
- **Georeferencing:** lat 1.1450–1.4572°N, lon 103.565–104.130°E; 120 × 217 pixels of **0.290 km**
  (measured from the coordinates, lesson L025). Validated physically: with these bounds, 100% of
  geocoded flood reports show rain within ~1.5 km at the reported time (19% under an earlier grid).
- **Store:** `data/processed/radar.zarr`, UTC time axis, strictly increasing (enforced after every
  append, lesson L010). In memory the dataset is held as uint8 codes into the 34-value table
  (lossless, 4× smaller; lesson L018).

### Flood ground truth
- **Source:** PUB's public Telegram channel (@pubfloodalerts), collected daily
  (`scripts/collect_flood_labels.py`). Message types: rain warnings, flood-risk warnings, flash
  floods. Times in the message text are Singapore wall-clock and are converted to UTC (lesson L009).
- **Geocoding** (`scripts/geocode_flood_labels.py`): OneMap search with a local cache
  (`data/processed/geocode_cache.json`, versioned); junctions are placed on the first road named;
  unresolvable names are fixed by hand in the cache and marked `method: manual`.
- **Matching** (`scripts/build_flood_eval_dataset.py`): each report is attached to the nearest radar
  frame and its grid cell.

### PUB flood-prone areas
`scripts/build_flood_prone_layer.py` parses PUB's published list (36 locations, Nov 2025) and
geocodes each to a point (`data/processed/flood_prone_areas.geojson`). PUB publishes names, not
polygons, so this layer is points.

## 2. Model

- **Task:** given the last 6 radar frames (30 min), sample the rain field at a fixed lead.
  One model per lead: 30 min (target t+6), 60 min (t+12), 90 min (t+18) after the anchor; the last
  input frame is t−1, so the forecasts are 35/65/95 min ahead of the latest data.
- **Transform:** rain → log1p(rain) / log1p(100) → [−1, 1], with no clipping (a clipped z-score
  earlier collapsed 30 of 35 levels into one value, lesson in `normalise_rain`).
- **Network:** conditioned U-Net (`src/model/unet.py`), 25.8 M parameters: channel widths
  64/128/256/512, residual blocks with sinusoidal timestep embedding, GroupNorm, self-attention at the
  bottleneck; the context frames are concatenated to the noisy target on the channel axis.
- **Diffusion** (`src/model/diffusion.py`): DDPM, 1000 steps, cosine noise schedule,
  **v-prediction** (ε-prediction failed on this 97%-dry field: samples drifted off the data manifold,
  lesson L021). Sampling: DDIM, 50 steps, η = 1, 8 independent members per forecast.
- **Training** (`train.py`): batch 4, AdamW lr 2e-4 (weight decay 1e-4), 2k warm-up then cosine
  decay, bf16 autocast (fp16 + GradScaler collapsed a run, lesson L016), gradient checkpointing,
  heavy-rain frames (≥10 mm/hr) oversampled 3× in the training split. 30-min model: 300k steps
  (~11 h on an RTX 4060 Laptop GPU). 60/90-min models: **warm-started** from the 30-min weights
  (`--init-from`), 100k steps each — from scratch they learned *whether* it rains but not *where*
  (lesson L030).
- **Safety stamps:** every checkpoint records parameterization, input layout and lead; loading
  refuses mismatches instead of guessing (lesson L023).

## 3. Splits

Pinned by date (`SPLIT_*` in `src/data/radar_dataset.py`) — an earlier "last 10% of a growing
archive" split moved every day (lesson L028):

| Split | Period (UTC) | Samples (30-min lead, pinned split) |
|---|---|---|
| Train | 2026-05-22 → 2026-09-02 06:50 | 25,177 distinct (49,881 with 3× heavy-rain oversampling) |
| Validation | 2026-09-02 06:50 → 2026-09-14 03:20 | 3,356 |
| Test | 2026-09-14 03:20 → 2026-09-25 23:55 | 3,367 |
| After test | 2026-09-26 → | never used for training or selection; new events are scored separately |

**Provenance of the 30-min model.** It was trained (24–25 Sep, from step 0) under the earlier
fraction-based split. With the archive at ≤34,100 frames at the time, that split's training data
ended by 2026-09-02 06:45 UTC; the pinned boundaries were set to match (validation from 06:50), so
the model has never seen the validation or test periods. The 60/90-min models were trained on the
pinned split.

A sample is kept only if its 6 context frames and its target are contiguous 5-min frames with no
blank (failed-scrape) frame, and its target lies inside its own split.

## 4. Evaluation

Implemented in `scripts/evaluate.py` (headline) and `scripts/evaluate_probabilistic.py` (families
A–D). Forecasts are cached with the exact times they cover, and a cache that no longer matches is
refused (lesson L028).

- **Baseline:** persistence — the last observed frame, unchanged.
- **CRPS** (primary): per-pixel ensemble CRPS, pooled over pixels that are wet (≥0.5 mm/hr) in the
  observation, the persistence forecast or any member; skill = 1 − CRPS_model / MAE_persistence.
- **FSS** (Roberts & Lean 2008), **pooled** over the evaluation set (numerator and denominator
  summed, not per-sample FSS averaged — lesson L026), on the ensemble **probability** field, at
  0.5 / 2 / 10 mm/hr and 6.1 / 11.9 / 23.5 km neighbourhoods.
- **Catchment boxes:** box-mean CRPS, Brier score and CSI for "≥0.5 mm/hr over the box".
- **Flood events:** out-of-sample only, with an ordinary-time alarm-rate control (see
  [RESULTS.md §3](RESULTS.md)).
- **Uncertainty:** bootstrap 95% CIs over forecast times.
- **Guard:** CRPS against persistence is always read together with FSS (lesson L029).

## 5. Inference

`src/inference/nowcast.py` builds the input with the same `build_context()` function as training
(no second copy of the transform), refuses gappy or blank history, runs each lead's own model and
writes one zarr per lead plus a summary PNG.
