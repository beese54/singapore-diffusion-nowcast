# Results

Every number here is read from a file in `results/`; the command that produced it is given with
each table. Scores are against **persistence** (repeat the last observed radar frame), the standard
nowcasting baseline. "Lead 30" means the model forecasts frame *t+6* from frames *t−6 … t−1*, i.e.
**35 min after the last observed frame** (65 and 95 min for leads 60 and 90).

- **Test period:** 2026-09-14 03:20 → 2026-09-25 23:55 UTC, pinned by date (`src/data/radar_dataset.py`),
  never used for training or model selection. 200 forecast times spread evenly across it, 8 ensemble
  members each.
- **Grid:** 0.29 km pixels, 120 × 217 (34.8 × 62.9 km). Neighbourhood sizes below are in km.
- **Confidence intervals:** 95%, bootstrap over the 200 forecast times (2,000 resamples).
- **Models of record:** 30 min `checkpoints/nowcaster/ckpt_step_300000.pt` (300k steps);
  60/90 min `lead60_warm`, `lead90_warm` (warm-started from the 30-min model, 100k steps each).

## 1. Headline

`python scripts/evaluate.py --checkpoint <ckpt> --n-samples 200 --members 8` → `results/evaluation_report.json`

| Lead | CRPS skill vs persistence [95% CI] | FSS ≥2 mm/hr @ 11.9 km: model / persistence [CI of diff] | FSS ≥10 mm/hr @ 11.9 km: model / persistence [CI of diff] |
|---|---|---|---|
| 30 min | **+0.384** [+0.251, +0.505] | 0.827 / 0.742 [−0.006, +0.132] | 0.337 / 0.446 [−0.159, +0.035] |
| 60 min | +0.422 [+0.266, +0.579] | 0.558 / 0.558 [−0.162, +0.094] | 0.056 / **0.236** [−0.279, −0.033] |
| 90 min | +0.413 [+0.279, +0.555] | 0.571 / 0.510 [−0.021, +0.125] | 0.091 / 0.162 [−0.289, +0.103] |

How to read it:

- **30 min** is the result of record: CRPS skill is clearly positive, and the rain-placement score
  (FSS) is higher than persistence at 2 mm/hr, although that difference is not yet significant.
- **60/90 min** also show positive CRPS skill, but **CRPS against persistence is not enough on its
  own** at longer leads (lesson L029): persistence is penalised twice when a storm moves, so a
  forecast of smooth, weak rain can "win" on CRPS while placing rain badly. The from-scratch 60/90
  models did exactly that. The warm-started models of record pass the placement check for light rain
  (tie at 60, slightly ahead at 90) — see `results/ablations/` for the failed from-scratch runs.
- **Heavy rain (≥10 mm/hr) is not beaten at any lead.** This is the project's main open problem
  (see [LIMITATIONS.md](LIMITATIONS.md)).

## 2. Rain placement by scale

`python scripts/evaluate_probabilistic.py --checkpoint <ckpt>` → `results/probabilistic_eval_lead{30,60,90}.json`
(family A). Pooled FSS of the ensemble probability field, model / persistence.

| Lead | Threshold | 6.1 km | 11.9 km | 23.5 km |
|---|---|---|---|---|
| 30 | 0.5 mm/hr | 0.82 / 0.78 | 0.88 / 0.86 | 0.93 / 0.91 |
| 30 | 2 mm/hr | 0.74 / 0.65 | 0.83 / 0.74 | 0.90 / 0.82 |
| 30 | 10 mm/hr | 0.28 / 0.36 | 0.34 / 0.45 | 0.38 / 0.52 |
| 60 | 0.5 mm/hr | 0.64 / 0.61 | 0.70 / 0.70 | 0.78 / 0.79 |
| 60 | 2 mm/hr | 0.47 / 0.46 | 0.56 / 0.56 | 0.68 / 0.68 |
| 60 | 10 mm/hr | 0.04 / 0.17 | 0.06 / 0.24 | 0.10 / 0.33 |
| 90 | 0.5 mm/hr | 0.63 / 0.56 | 0.68 / 0.64 | 0.75 / 0.73 |
| 90 | 2 mm/hr | 0.49 / 0.43 | 0.57 / 0.51 | 0.69 / 0.61 |
| 90 | 10 mm/hr | 0.07 / 0.11 | 0.09 / 0.16 | 0.14 / 0.27 |

Other diagnostics in the same files: CRPS over all pixels (family B), catchment-box CRPS/Brier/CSI
(family C). Rain-area frequency bias (model area / observed area), computed from the cached
ensembles: 30 min 0.91 at ≥2 mm/hr and **0.48 at ≥10 mm/hr** (heavy rain under-forecast);
60 min 0.89 / 0.50; 90 min 1.51 / 0.93.

## 2b. When it warns of heavy rain, is it right? (plain-language skill)

`python scripts/warning_skill.py` → `results/warning_skill.json`. The domain is cut into 527 boxes of 2 × 2 km;
for each of the 200 test forecasts, *heavy rain* = ≥10 mm/hr somewhere in the box at the target time; the model
*warns* when ≥2 of 8 futures show it; the naive forecast warns when the last radar map shows it.
95% ranges by bootstrap over forecast times.

| Lead | Catches (share of heavy-rain boxes warned): model [95%] · naive | Right when it warns: model [95%] · naive | vs chance |
|---|---|---|---|
| 30 | **29%** [15, 43] · 25% | **32%** [19, 43] · 31% | 27× |
| 60 | 3% [1, 5] · **13%** | 7% [2, 10] · **13%** | 6× |
| 90 | 7% [2, 11] · 9% | 10% [4, 16] · 9% | 9× |

Heavy rain occupies ~1.2% of boxes, so 1-in-3 precision is ~27× chance. At 30 min the model roughly matches the
naive forecast overall; its distinctive contribution is warnings where the radar shows no heavy rain yet: 849
such warnings, 200 verified (24%) — 16% of all heavy-rain boxes flagged before arrival, at the cost of 76% false
alarms among those. At 60/90 min it is no better than persistence for heavy rain.

## 3. Recorded flash-flood events (out of sample only)

Family D of `evaluate_probabilistic.py`. Ground truth: PUB flood alerts (Telegram), geocoded to the
radar grid. A "hit" is ≥4 of 8 members putting rain above the threshold within the radius of the
reported location, in the forecast issued 35/65/95 min before the event's radar frame. The control
is the alarm rate at the same locations at **ordinary** test times — a forecaster that simply rains
more often would score hits without skill.

Only events the model never trained on are scored: the test period (9 events) and events after it
(17, all from the 27 Sep storm). 29 earlier events are excluded as in-sample (lesson L031 — an
earlier version of this table included them).

≥2 mm/hr within 1.4 km, model / persistence (ordinary-time alarm rate, model / persistence):

| Lead | Test period: 9 events (18 & 22 Sep) | After test: 17 events (27 Sep) |
|---|---|---|
| 30 | 9/9 vs 9/9 (3.9% / 13.8%) | **17/17** vs 12/17 (3.6% / 13.2%) |
| 60 | 6/9 vs 5/9 (2.3% / 12.8%) | **10/17** vs 3/17 (2.2% / 12.6%) |
| 90 | 0/9 vs 2/9 (3.3% / 13.8%) | 3/17 vs 3/17 (3.2% / 11.8%) |

**These are three storms, not 26 independent trials** — treat them as indicative. They do show the
30-min model hitting as many or more events than persistence while raising alarms about 4× less
often at ordinary times.

## 4. Case studies

Reproduce with `python scripts/case_study_forecasts.py --checkpoint <ckpt> [--start --end --name]`,
then `notebooks/02_nowcast_evaluation.ipynb`.

**22 Sep 2026, King's Road / Coronation Road (test period).** ≥30 mm/hr for 100 min over the flood
sites (peak 85 mm/hr). 30-min model: missed the first minutes of a rapidly growing cell; first flag
(P≥10 mm/hr ≥ 0.25) issued 15:45 SGT, 86 min before the flash-flood report (17:11); intensity ~5×
too low; handled the decay far better than persistence. 60/90 min: no flag.

**27 Sep 2026, Neo Pee Teck Lane / Pasir Panjang Road (after the test period — fully out of
sample).** Dry until 11:05 SGT, 100 mm/hr (top of NEA's scale) 11:30–11:40, flash flood 11:57.
30-min model: first flag issued **10:35 SGT (P≥10 = 0.38) while persistence was still dry** —
35 min before rain reached the site, 30 min before PUB's first warning anywhere, 82 min before the
flood report. Intensity ~10× too low. 60 min: one flag (P = 0.50) issued 10:35; 90 min: none.

## 5. Inference speed

`python src/inference/nowcast.py --checkpoint <30> <60> <90> --time "2026-09-22 16:15"`:
**19.6 s** end to end for 8 members × 3 leads on an RTX 4060 Laptop GPU (model loading 1.7 s,
~5.5 s per lead, saving included).

## 6. What was tried and did not work

Recorded in `results/ablations/` and `tasks/lessons.md`:

| Idea | Outcome | Lesson |
|---|---|---|
| ε-prediction diffusion | Samples collapsed on a 97%-dry field | L021 |
| Intensity-weighted / heavy-rain-weighted loss | Rain everywhere; CRPS skill −0.136 | L020, L027 |
| Residual (change-from-last-frame) target | Worse than no change at every scale | L024 |
| Rain-centred training crops | Train/inference mismatch, wrong prior | L022 |
| 12 frames of history + time of day | No effect | — |
| Deterministic sampler, calibration | Fragile or no gain | — |
| 60/90-min models from scratch | Right amount of rain, wrong places | L029 |
| Warm start from the 30-min model | **Fixed** light-rain placement at 60/90 | L030 |
