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

**Recalibrated warning rule and the recommended 60-min warning** (`scripts/warning_calibration.py`,
`scripts/hybrid_warning.py` → `results/warning_calibration.json`, `results/hybrid_warning.json`). The model
under-forecasts intensity, so its futures rarely reach 10 mm/hr. The rule "≥ k of 8 futures ≥ X mm/hr" was re-chosen
by CSI (hits / (hits + false alarms + misses)) on validation (2–14 Sep) only, from X ∈ {3…10}, k ∈ {1…4}, then
scored on the same test forecasts as the table above:

| Lead | Rule chosen | CSI old → new [95% of diff] | Catches | Right | Adopted? |
|---|---|---|---|---|---|
| 30 | ≥4 of 8 ≥ 5 mm/hr | 0.18 → 0.21 [−0.03, +0.08] | 29% → 27% | 32% → 52% | no |
| 60 | ≥2 of 8 ≥ 3 mm/hr | 0.02 → 0.11 [+0.02, +0.17] | 3% → **35%** | 7% → 14% | **yes** |
| 90 | ≥2 of 8 ≥ 3 mm/hr | 0.04 → 0.09 [−0.00, +0.10] | 7% → 39% | 10% → 10% | no |

The 60-min gain is **fragile**: on a second set of ensemble seeds it is in the same direction (CSI 0.05 → 0.10) but
its CI includes 0, and 3 mm/hr is the lowest threshold tried. Even recalibrated, the model does not beat **plain
optical-flow extrapolation**, which is the **recommended 60-min heavy-rain warning** (catches 17%, right 19%, CSI 0.10).
A hybrid (warn if either extrapolation or the recalibrated model warns) catches 38% and flags 33% of incoming
downpours vs 13%, but is right only 13% of the time; its CSI (0.107 vs 0.100, diff CI [−0.03, +0.05]) is not
better, so under the rule fixed in advance it was not adopted. Before the 23 out-of-sample drain alerts (§3b) the
hybrid warned for all 23 (median 87 min ahead) and extrapolation for all 23 (69 min); three storms, context only.

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

## 3b. Minutes ahead of PUB's drain sensors (out of sample only)

`python scripts/drain_alert_benchmark.py [--seed N]` → `results/drain_alert_benchmark.json` (seed 7000;
`_seed8000.json` is an independent re-draw). PUB's "Risk of Flash Floods" alert fires when a drain reaches
90% of its depth, so each alert is a drain-sensor reading with a time and place. For each out-of-sample
alert (first per site per storm), 30-min forecasts were issued every 5 min over the 2 hours before it.
**Model warns** = ≥2 of 8 futures show ≥10 mm/hr in the ~2×2 km box around the site. **Radar shows** =
the observed radar already has ≥10 mm/hr there (what someone watching the radar would see). Lead =
minutes from the first warning to the alert. Definitions were fixed before the results were seen.

| Drain alerts (FLOOD_RISK) | 18 Sep (1 site) | 22 Sep (6 sites) | 27 Sep (16 sites) |
|---|---|---|---|
| Sites the model warned for | 1/1 | 6/6 | 16/16 |
| Median model lead, seed 7000 / 8000 | 53 / 63 min | 78 / 68 min | 58 / 69 min |
| Median radar lead (heavy rain already visible) | 43 min | 53 min | 28 min |
| **Median head start over the radar**, seed 7000 / 8000 | **10 / 20 min** | **20 / 13 min** | **28 / 43 min** |

- **Over all 23 drain alerts:** the model warned before every one, with a median lead of 63–68 min. Heavy rain was already visible on radar a median of 32 min before the alert. The model's median head start over the radar was positive in all 3 storms and in both draws.
- **The warnings mostly held.** Once the model first warned, it kept warning at a median of 81% of the later issue times (minimum 50%).
- **Warnings are rare at other times.** At the same sites, it warned in only 0.5–2.5% of 200 random test forecasts, so these warnings are not "always on".
- **The same pattern holds for flash-flood reports:** 3/3 warned, with a median lead of 86–96 min.

**Limits.**
- **This is 3 storms, not 23 independent trials.**
- **The exact first-warning time depends on the random draw.** The median difference between the two draws is 10 min and the worst case is 40 min. For example, Neo Pee Teck Lane gets a first warning at 10:35 in one draw and 10:45 in the other, i.e. 59 or 49 min before its 11:34 drain alert.
- **Hits do not show that the warnings are specific.** This benchmark counts warnings before alerts. It does not count warnings on storm days at sites that never flooded; §2b does, and there the model is right about 1 in 3 times.
- **Location is approximate.** The site is the geocoded road name, not the sensor itself.
- **Real leads are a few minutes shorter,** because NEA publishes radar frames a few minutes late.

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
35 min before rain reached the site, 30 min before PUB's first alert anywhere, **59 min before the
drain-sensor flood-risk alert at that junction (11:34)**, 82 min before the flash-flood report (an
independent re-draw in §3b first warns at 10:45, 49 min before the alert: the exact minute depends on the
random draw). Intensity ~10× too low. 60 min: flags issued 10:35 (P = 0.50) and 10:45; 90 min: no
advance warning (its one flag, issued 10:55 at exactly P = 0.25, is for 12:30, after the flood).

**30 Sep 2026, Riverside Road and Neo Tiew Road (after the test period — a late warning).** Flash
floods reported 15:13 and 15:38 SGT. Both sites went from ~1 to 100 mm/hr within 15 min (≥10 mm/hr from
14:40 and 14:35). The storm grew in place: the nearest ≥10 mm/hr pixel stayed 5–30 km away until 14:20,
then reached the sites within 15 min. Riverside Road is ~1.5 km from the domain's northern edge.
30-min model: first storm-related flags issued **14:25** (P≥10 = 0.38 at both sites), 10–15 min before
the heavy rain arrived and 48 / 73 min before the flash-flood reports, while persistence showed no
heavy rain. A 2-of-8 flag issued 12:35 at both sites was a false alarm. NEA never published the 13:50
radar frame, so forecasts issued 13:55–14:15 could not be made; an earlier warning cannot be ruled out
or in. 60/90 min: no flag (max P = 0.12). These events are **not yet in the §3/§3b tables**; they will
be added in the next scripted re-run.

On all three days the first 30-min flag came 10–35 min before heavy rain reached the site, and 48–86 min
before the flash-flood report; the warning is short when a storm grows within a few km of the site.
Notebook 02 §7 has the 27 and 30 Sep charts.

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
| NVIDIA CorrDiff (regression stage) ERA5 → 2 km hourly rain | NO-GO: never predicts ≥1 mm/hr; RMSE 1.085 vs always-zero 1.096. It does track *when* rain falls (timing correlation 0.64 vs 0.22) | `tasks/spike_corrdiff.md`, `results/corrdiff_spike.json` |
| ERA5 weather vector fed to the 30-min model | Did not help: FSS ≥10 −0.076 [−0.095, −0.031] vs a same-budget control; the control itself was worse than the base (fine-tuning LR restart) | L032, L033, `results/era5_conditioning.json` |
| Wide 240 km radar, extrapolated, as the missing heavy-rain information | NO-GO: catches no more incoming heavy rain than the same method on the 70 km footprint (60 min: −0.010 [−0.027, +0.008]). Incoming heavy rain mostly forms, not arrives. Side finding: that simple extrapolation beats the 60-min model on heavy rain | `tasks/plan_stage7_heavy_rain.md`, `results/spike_240km.json`, L034 |
| Motion forecast as an extra input to the 60-min model | Better light-rain placement (FSS ≥2 +0.057 [+0.005, +0.090] vs control) but worse heavy rain (catch −0.014 [−0.026, −0.004]). Plain extrapolation catches 17% of heavy-rain boxes vs the model's 7% (catch difference −0.106 [−0.178, −0.029]): at 60 min the bottleneck is intensity, not information | `tasks/plan_motion_input.md`, `results/motion_input.json`, L035 |
| Recalibrated heavy-rain warning rule; hybrid (extrapolation OR model) 60-min warning | 60-min rule adopted (catch 3% → 35%, CSI 0.02 → 0.11; fragile across seeds); 30/90 not adopted. The hybrid catches more (38% vs 17%) but is right less often (13% vs 19%): CSI not better than extrapolation, which stays the 60-min heavy-rain warning (§2b) | `tasks/plan_hybrid_calibration.md`, `results/warning_calibration.json`, `results/hybrid_warning.json` |
| Himawari-9 satellite: cold-cloud rule added to extrapolation (spike) | Weak GO: 60-min incoming heavy-rain catch +28 points [+7, +45], CSI −0.013 [−0.031, +0.018], but about 5× the warnings and precision 18% → 9%. The gain came from warning over a wider area; growing storms (initiation) were barely caught | `tasks/plan_satellite.md`, `results/satellite_spike.json`, L036 |
| Learned 60-min heavy-rain warning, radar vs radar + satellite, at matched warning volume | Did not help: radar + satellite CSI 0.021 vs extrapolation 0.093 (diff −0.072 [−0.112, −0.005]). The hour-of-day inputs had learned the training months' storm timing, which did not hold in the test month (L037). Without them the satellite adds +0.000 [−0.023, +0.024] over radar only, and neither beats extrapolation | `tasks/plan_satellite_step2.md`, `results/satellite_warning.json`, `results/satellite_warning_posthoc.json`, L037 |

**Reading the negative results.** Every idea above was trained on 103 days of one season (22 May – 2 Sep 2026,
Southwest Monsoon and inter-monsoon), chosen on 12 days and tested on 12 days with a handful of storm days, so the
confidence intervals are wide. "Did not help" means **no measurable gain on this record**, not "cannot help". The
comparisons are scripted and will be re-run once the archive covers the Northeast Monsoon (Dec–Mar).
