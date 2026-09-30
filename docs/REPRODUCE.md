# Reproducing this work

Every command below was run in this project on 2026-09-26/27 on Windows 11 with an NVIDIA
RTX 4060 Laptop GPU (8 GB). Paths are relative to the repository root. Radar data can only be
collected for dates NEA still serves (see [DATA.md](DATA.md)), so your archive — and therefore your
exact numbers — will differ; the method and code are identical.

## 1. Environment

```bash
bash init.sh                  # conda env "sg-weather" (Python 3.11) + requirements + CUDA PyTorch
cp .env.example .env          # then fill in the keys you need (Telegram for flood labels, CDS for ERA5)
```

Tested with PyTorch 2.5.1 + CUDA 12.1. Training and evaluation were also run from a plain
Python 3.12 environment with `pip install -r requirements.txt`; the Windows launchers use `python`
on `PATH` (set `PY` to override).

## 2. Collect data (continuous)

```bash
python scripts/scrape_radar.py --hours 2                      # 70 km radar, recent 2 h
python scripts/scrape_radar.py --product 240km --hours 2      # 240 km radar
python scripts/scrape_radar.py --status                       # coverage report
python scripts/preprocess_radar.py                            # decode new PNGs into radar.zarr
python scripts/collect_flood_labels.py                        # PUB Telegram alerts (needs .env)
python scripts/geocode_flood_labels.py                        # OneMap, cached
python scripts/build_flood_eval_dataset.py                    # match alerts to radar frames
python scripts/check_radar_gaps.py                            # alert on missing frames
python scripts/status.py                                      # whole-pipeline status
```

This project runs them as Windows Scheduled Tasks under `\SG-Weather\`: *Radar Continuous* (both
radar products, every 30 min), *Radar Scraper* (daily 08:00: 5-day catch-up, then preprocess) and
*Telegram Labels* (daily 09:00: collect → geocode → build → flash-flood check → gap check).

## 3. Train

```bash
# 30-min model (the model of record: 300k steps, ~11 h on the RTX 4060)
python train.py --max-steps 300000 --batch-size 4 --num-workers 0 --resume auto

# 60/90-min models, warm-started from the 30-min model (100k steps each)
python train.py --max-steps 100000 --batch-size 4 --num-workers 0 --target-offset 12 \
    --run-name lead60_warm --init-from checkpoints/nowcaster/ckpt_step_300000.pt --resume auto
python train.py --max-steps 100000 --batch-size 4 --num-workers 0 --target-offset 18 \
    --run-name lead90_warm --init-from checkpoints/nowcaster/ckpt_step_300000.pt --resume auto
```

Checkpoints are saved every 1,000 steps and `--resume auto` continues after an interruption.
`scripts/train_detached.bat` and `scripts/train_leads_warm.bat` launch these detached on Windows.
`python train.py --smoke --max-steps 100` is a quick check that writes to a throwaway folder.

## 4. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt --n-samples 200 --members 8
python scripts/evaluate_probabilistic.py --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt
#   ... repeat both for lead60_warm/ckpt_step_100000.pt and lead90_warm/ckpt_step_100000.pt
python scripts/evaluate_probabilistic.py --checkpoint <ckpt> --from-cache   # recompute metrics, no GPU
```

`evaluate.py` fills one section per lead in `results/evaluation_report.json`;
`evaluate_probabilistic.py` writes `results/probabilistic_eval_lead<N>.json`. Ensembles are cached
in `data/processed/eval_cache/` with the exact times they cover; a stale cache is refused.

## 5. Case studies and notebooks

```bash
python scripts/case_study_forecasts.py --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt   # 22 Sep
python scripts/case_study_forecasts.py --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt \
    --start 2026-09-27T02:30 --end 2026-09-27T04:30 --name 27sep                               # 27 Sep
python scripts/case_study_forecasts.py --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt     --start 2026-09-30T05:00 --end 2026-09-30T07:40 --name 30sep                               # 30 Sep
# repeat each case with lead60_warm/ and lead90_warm/ ckpt_step_100000.pt for the 60/90-min caches;
# targets whose radar history has a gap are skipped and listed
python scripts/build_flood_prone_layer.py        # PUB flood-prone list -> GeoJSON points
jupyter nbconvert --to notebook --execute --inplace notebooks/02_nowcast_evaluation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_flood_risk_overlay.ipynb
```

## 6. Live nowcast

```bash
python src/inference/nowcast.py --time "2026-09-22 16:15" \
    --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt \
                 checkpoints/nowcaster/lead60_warm/ckpt_step_100000.pt \
                 checkpoints/nowcaster/lead90_warm/ckpt_step_100000.pt
```

`--time` is the last observed frame in SGT (`--utc` for UTC; default: newest frame). Output:
`results/nowcast_<time>/lead<N>.zarr` and `summary.png`. Measured: 19.6 s for 8 members × 3 leads.

## 7. Tests

```bash
for f in tasks/repro/*_repro.py; do python "$f"; done     # 9 suites, all passing on 2026-09-27
```

They pin down past failures: codebook round-trip, context layout, NaN handling, normalisation and
collapse guard, v-prediction, residual target, heavy-rain weighting, gap-alert tiers.
