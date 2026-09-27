# Plan: give the radar nowcaster the large-scale weather (ERA5 conditioning) — for approval

*2026-09-27. Follows the CorrDiff spike (`tasks/spike_corrdiff.md`): its only real signal was that ERA5 tracks
**when** rain falls over the island (island-rain timing correlation 0.64 vs 0.22 for ERA5's own rain, CI
[+0.02, +0.66]). The radar nowcaster's main weakness is storms that form or arrive before they show on radar.
Question: does telling the nowcaster "what kind of weather day it is" improve its heavy-rain forecasts?*

## Design

| Item | Choice | Why |
|---|---|---|
| What is added | A small vector of **domain-averaged ERA5 predictors** at the hour of the last radar frame: total column water vapour, 2 m temperature, mean sea-level pressure, humidity and temperature at 850/500 hPa (instability proxies), winds at 850 and 500 hPa (steering flow) — ~12 numbers, averaged over the 3°×3° window | At 28 km, ERA5 is nearly constant across our 35×63 km radar domain, so full maps as extra image channels would add nothing but parameters; a few numbers carry the same information and cannot overfit spatial detail |
| How it enters the model | A small MLP embeds the vector and **adds it to the diffusion timestep embedding** (every U-Net block sees it), with the last layer **initialised to zero** | Zero-init means the model starts exactly as the current 30-min model; any change is learned, never imposed |
| Starting point | Warm-start from the 30-min model (`--init-from`, lesson L030) | Keeps what already works; only the new pathway and fine-tuning are trained |
| Training | 30-min lead, 50–100k steps (~2–4 h on the RTX 4060), same pinned split | Comparable with the model of record |
| Evaluation | **Paired** against the current 30-min model on the same test forecasts: CRPS, FSS at 2 and 10 mm/hr, and the plain-language warning skill (catch rate, precision, warnings before rain arrives), bootstrap 95% CIs | Same forecasts, same seeds — only the conditioning differs |

## Go / no-go (fixed now)

**Worth keeping** if, versus the current model on the same test forecasts: heavy-rain skill improves (FSS ≥10 mm/hr
**or** warning catch rate, 95% CI of the difference above 0) **and** overall CRPS skill does not get worse (CI of
the difference not below 0). Otherwise the environment signal is not usable this way, and we record why.

## Honest limits (will be stated with any result)

1. **ERA5 is not real-time** (≈ 6-day lag) and is a reanalysis that uses observations from hours around each time,
   including slightly after. A positive result is therefore an **upper bound** on the value of large-scale
   weather; an operational version must use data available at forecast time (e.g. ECMWF open-data forecasts),
   which is a follow-up if this works.
2. **Test coverage:** ERA5 currently reaches 21 Sep, so the test period is 14–21 Sep (≈ 2/3 of it) until the next
   download; the 27 Sep flood case needs ERA5 from ~3 Oct.
3. **Few independent weather days:** ~94 training days of environment. The model could learn "storm days" from very
   few examples, which is why the conditioning is tiny and zero-initialised, and why the test is paired with CIs.

## Steps

1. `RadarDataset`: optional ERA5 predictor vector per sample (the hour at/before the last radar frame), standardised
   on training days; samples without ERA5 are dropped when the option is on. Default off — current models unaffected.
2. `ConditionedUNet` / `GaussianDiffusion`: optional environment embedding (zero-initialised). Default off; checkpoints
   stamp the option so loading refuses a mismatch (lesson L023).
3. Tests in `tasks/repro/`: option off = bit-identical output to today; option on at step 0 = identical to the
   warm-start model (proves the zero-init).
4. Train (detached, resumable), then paired evaluation → `results/era5_conditioning.json`, verdict recorded here.

Estimated effort: ~½ day of code and tests, ~3 h training, ~½ h evaluation. No cloud spend.
