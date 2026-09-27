# Singapore flash-flood nowcasting with generative diffusion

**A diffusion model that looks at the last 30 minutes of Singapore's rain radar and generates an
ensemble of possible rain maps for the next 35–95 minutes — built to give neighbourhood-level
warning of the sudden downpours that cause flash floods.**

**Interactive dashboard: [https://beese54.github.io/singapore-diffusion-nowcast/](https://beese54.github.io/singapore-diffusion-nowcast/)** — two real flash floods, forecast by forecast, and the evidence.

An independent research project, run end-to-end on one laptop GPU (RTX 4060, 8 GB): radar and
flood-report collection since May 2026, a 25.8M-parameter diffusion nowcaster, and an evaluation
that reports what works *and what does not*. Phase 1 is complete; Phase 2 (wide-range radar,
satellite and NVIDIA CorrDiff downscaling) is under way.

![Rain at the King's Road flood site on 22 Sep 2026: observed (black), forecasts at 30/60/90 min, and the ensemble's probability of heavy rain](docs/img/case22sep_flood_cell_all_leads.png)

*22 Sep 2026, King's Road flash flood (red lines: flood reports). Top: observed rain near the site
(black) and ensemble forecasts; bottom: forecast probability of ≥10 mm/hr. Radar-derived figure —
radar imagery © NEA / Meteorological Service Singapore (weather.gov.sg), shown for personal,
non-commercial, informational use only.*

## The dashboard at a glance

Screenshots from the [live dashboard](https://beese54.github.io/singapore-diffusion-nowcast/). Radar imagery
© NEA / Meteorological Service Singapore (weather.gov.sg), shown for personal, non-commercial, informational use only.

| | |
|---|---|
| ![Overview: headline numbers and why flash-flood nowcasting matters](docs/img/dashboard/1_overview.jpg) | ![Flood explorer: at 10:35 on 27 Sep the AI model warns of heavy rain, the naive forecast does not, and heavy rain fell](docs/img/dashboard/2_flood_explorer_27sep.jpg) |
| **Overview.** What the model does, the headline numbers, and why minutes matter. | **Would it have warned us?** The 10:35 forecast for the 27 Sep flood: the model warns (3 of 8 futures), the naive forecast doesn't, heavy rain came. |
| ![Chart: the model's warning bars rise before the observed rain at the flood site](docs/img/dashboard/3_flood_site_chart_27sep.jpg) | ![Scorecard: catch rate and precision of heavy-rain warnings by lead time](docs/img/dashboard/4_when_it_warns_is_it_right.jpg) |
| **Warnings before the rain.** Bars (model warnings) rise before the bold line (observed rain); red dashes mark the flood report. | **The honest scorecard.** 30 min ahead: catches 29% of downpours, right 1 time in 3 (27× chance); 60/90 min: no better than the naive forecast. |
| ![Noise to rain: the diffusion model's 50-step denoising](docs/img/dashboard/5_noise_to_rain.jpg) | ![What comes next: NEA's 240 km radar and the Phase 2 plan](docs/img/dashboard/6_whats_next_240km_radar.jpg) |
| **How it draws a forecast.** From random noise to a rain map in 50 steps, guided by the last 30 minutes of radar. | **What comes next.** The wider 240 km radar, satellite, NVIDIA CorrDiff, and PUB's drain-sensor alerts as flood ground truth. |

## Results in one table

Held-out test period (14–25 Sep 2026), 200 forecast times × 8 members, against persistence
(repeat the last radar frame). Details and all other scores: [docs/RESULTS.md](docs/RESULTS.md).

| Lead (time after last radar frame) | CRPS skill [95% CI] | Rain placement, FSS ≥2 mm/hr (model / persistence) | Heavy rain, FSS ≥10 mm/hr (model / persistence) |
|---|---|---|---|
| 30 min (35) | **+0.38** [+0.25, +0.51] | **0.83** / 0.74 | 0.34 / **0.45** |
| 60 min (65) | +0.42 [+0.27, +0.58] | 0.56 / 0.56 | 0.06 / **0.24** |
| 90 min (95) | +0.41 [+0.28, +0.56] | 0.57 / 0.51 | 0.09 / **0.16** |

- **Works:** at 30 minutes the ensemble is a better probabilistic forecast than persistence.
  Out of sample, it flagged the **27 Sep 2026 Pasir Panjang flash flood 82 minutes before the flood
  report**, while the radar still showed the site dry, and the 22 Sep King's Road flood 86 minutes
  ahead.
- **Does not work yet:** **heavy rain** (≥10 mm/hr, what actually floods roads) is not predicted
  better than persistence at any lead, and intensities are 5–10× too weak. Longer leads only help
  with light rain. **This is a research prototype, not a warning system.** See
  [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
- **Fast:** 20 seconds for 8 members × 3 lead times on a laptop GPU.

## How it works

```mermaid
flowchart LR
    A[NEA rain radar<br/>every 5 min] --> B[Decode 33 colour levels<br/>to mm/hr, 0.29 km grid]
    B --> C[Last 6 frames<br/>= 30 min of history]
    C --> D[Diffusion model<br/>noise → rain map, 50 steps]
    D --> E[8 ensemble members<br/>per lead: 30 / 60 / 90 min]
    E --> F[Probability of heavy rain<br/>near flood-prone locations]
    G[PUB flood alerts<br/>Telegram] --> H[Geocoded events] --> I[Evaluation vs persistence]
    E --> I
```

The model is a conditioned U-Net trained as a v-prediction DDPM (one model per lead; the 60/90-min
models are warm-started from the 30-min one). Each forecast starts from random noise and is
denoised in 50 steps, conditioned on the recent radar; repeating that 8 times gives 8 different,
plausible futures, whose agreement is the forecast probability.
Full method: [docs/METHODS.md](docs/METHODS.md).

## Repository map

| Path | What |
|---|---|
| `src/data/radar_dataset.py` | dataset, pinned train/val/test split, the model-input transform |
| `src/model/` | U-Net and diffusion (v-prediction, DDIM sampling) |
| `train.py` | training, resumable, `--init-from` warm start |
| `src/inference/nowcast.py` | live forecast from the latest radar frames |
| `scripts/` | data collection, preprocessing, geocoding, evaluation, case studies |
| `notebooks/02_nowcast_evaluation.ipynb` | 22 Sep flash-flood case study |
| `notebooks/03_flood_risk_overlay.ipynb` | forecast probability over PUB flood-prone areas |
| `results/` | evaluation results of record; `results/ablations/` for superseded runs |
| `docs/` | methods, results, limitations, data, reproduction, lessons; `docs/history/` for past plans |
| `tasks/` | working plan (`plan_phase2.md`), checklist (`todo.md`), full lessons log |
| `definition_of_done.md`, `progress_tracking.json` | stage-by-stage acceptance criteria and status |

## Quick start

```bash
bash init.sh && cp .env.example .env
python scripts/scrape_radar.py --hours 168        # NEA keeps ~7 days: your archive starts today
python scripts/preprocess_radar.py
python train.py --max-steps 300000 --resume auto  # ~11 h on an RTX 4060
python scripts/evaluate.py --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt
```

Step-by-step instructions, including the scheduled collection tasks: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## What we learned

The most useful part for other researchers may be what failed: ε-prediction, rain-weighted losses,
residual targets, crops, and a flood-event evaluation that turned out to be partly in-sample. Each
is written up with the evidence in [docs/LESSONS.md](docs/LESSONS.md).

## Roadmap (Phase 2)

- **Wide-range radar (240 km)** — collected from Sep 2026 — to see storms before they reach the
  domain, the information the heavy-rain limitation points to.
- **Satellite (Himawari-9)**, following the design of NVIDIA's Earth-2 Nowcasting model (StormScope:
  satellite + radar).
- **NVIDIA CorrDiff** (PhysicsNeMo): ERA5 → ~2.5 km rainfall downscaling over Singapore.
- **More out-of-sample flood events**, and retraining once the archive covers the Northeast Monsoon
  (Dec–Mar).

Plan: [tasks/plan_phase2.md](tasks/plan_phase2.md).

## Data and licences

- **Code:** MIT ([LICENSE](LICENSE)). **Documentation and figures:** CC BY 4.0.
- **Radar imagery © National Environment Agency (NEA) / Meteorological Service Singapore
  ([weather.gov.sg](https://www.weather.gov.sg)).** Radar-derived figures are shown for personal,
  non-commercial and informational purposes only, per the
  [weather.gov.sg Terms of Use](https://www.weather.gov.sg/terms-of-use/). The radar archive itself is
  not redistributed.
- Flood alerts: PUB (@pubfloodalerts). Geocoding: OneMap (SLA). ERA5: Copernicus / ECMWF.
- This is an independent project, not affiliated with or endorsed by NEA, MSS, PUB or NVIDIA.
  Details: [docs/DATA.md](docs/DATA.md).

## Citation

See [CITATION.cff](CITATION.cff).
