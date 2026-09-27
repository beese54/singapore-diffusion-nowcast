# Data sources, licences and what is published

## Copyright notice for radar imagery

> **Radar imagery © National Environment Agency (NEA) / Meteorological Service Singapore,
> [weather.gov.sg](https://www.weather.gov.sg). All rights reserved by NEA.** Radar images and
> animations in this repository, its dashboard and related posts are derived from NEA's rain-area
> images and are shown for **personal, non-commercial and informational purposes only**, as
> permitted by the [weather.gov.sg Terms of Use](https://www.weather.gov.sg/terms-of-use/). This
> project is independent and is not affiliated with or endorsed by NEA, MSS or PUB.

## Sources

| Data | Provider | Used for | In this repo? |
|---|---|---|---|
| Rain-area radar, 70 km product (5-min PNG) | NEA / MSS, weather.gov.sg | model input and target | **No** — derived figures only |
| Rain-area radar, 240 km product (5-min PNG) | NEA / MSS, weather.gov.sg | heavy-rain stage (collected from Sep 2026) | **No** |
| Flash-flood alerts (@pubfloodalerts, Telegram) | PUB | evaluation ground truth | **No** raw messages; event times and geocoded locations appear in results and figures |
| List of Flood Prone Areas (Nov 2025, PDF) | PUB | flood-risk map | **No** PDF; the derived point layer `data/processed/flood_prone_areas.geojson` is generated locally |
| Address search | OneMap (Singapore Land Authority) | geocoding | `data/processed/geocode_cache.json` (place names → coordinates) |
| ERA5 reanalysis | ECMWF / Copernicus Climate Change Service | baseline (Stage 2) and CorrDiff (planned) | **No** |

## What is published

- All code, configuration and documentation.
- Evaluation results (`results/*.json`) and the geocode cache.
- Figures and animations derived from a few hours of radar for the case studies, with the notice above.

## What is not published, and why

- The radar archive (`data/raw/radar/`, `data/processed/radar.zarr`): NEA content, not licensed for
  redistribution.
- Raw Telegram messages and the Telegram session.
- Model checkpoints (`checkpoints/`): large; trained on NEA data. Available on request for research.
- API keys (`.env`), which have never been committed (checked across the full git history).

## Rebuilding the data yourself

NEA keeps about **7 days** of the 70 km product and about **30 days** of the 240 km product. The
scripts can only collect what NEA still serves, so a new archive starts from the day you begin:

```bash
python scripts/scrape_radar.py --hours 168                     # last ~7 days, 70 km
python scripts/scrape_radar.py --product 240km --hours 720     # last ~30 days, 240 km
python scripts/preprocess_radar.py                             # PNG -> data/processed/radar.zarr
```

Flood labels need your own Telegram API credentials (`.env`, see `.env.example`).
