# Spike: does optical-flow extrapolation beat persistence? (2026-09-25)

**Pattern N.** Spike code was disposable (scratchpad), not merged.

## Question
Can advecting the last radar frame along estimated storm motion beat
persistence on FSS at 2 mm/hr for the ~35-min lead? If so, motion is the
missing ingredient and the diffusion model should refine an advected field.

## Correction found on the way: the grid is 0.29 km, not 1 km
From the zarr lat/lon: pixel = 0.290 km x 0.290 km, domain = **34.8 x 62.9 km**,
not 120 x 217 km. `specification.json` states `resolution_km: 1` -- wrong.
Every FSS window reported earlier this session was 3.4x smaller than labelled
("41 km" was 11.9 km). Model-vs-persistence comparisons at a given window are
unaffected; absolute scales and the crop-size reasoning (L020's "64 km of
context" was 18.6 km) were wrong.

Also: samples use context t-6..t-1 and target t+6, so the "30-min" lead is
**35 min** from the last observed frame.

## Results (FSS @ 2 mm/hr)
Full domain, 60 test samples:

| | 0.9 km | 3.2 km | 6.1 km | 11.9 km |
|---|---|---|---|---|
| persistence | 0.068 | 0.156 | 0.274 | **0.451** |
| global shift (phase corr.) | 0.044 | 0.056 | 0.075 | 0.105 |
| DIS dense flow | 0.058 | 0.096 | 0.148 | 0.253 |
| Farneback dense flow | 0.071 | 0.147 | 0.249 | 0.406 |

True storm motion (39 samples with rain in last frame and target): median
**19 km/h** (IQR 5-34), ~11 km over the lead. Only a median **4%** of target
rain had no rain within ~6 km in the last frame -> inflow is NOT dominant.

**Oracle** (shift by the TRUE displacement, measured from the target itself),
scored on the interior only to remove edge artifacts:

| | 6 km | 12 km | 18 km |
|---|---|---|---|
| persistence | 0.306 | **0.384** | 0.439 |
| oracle global shift, edge-filled | 0.280 | 0.337 | 0.382 |

Phase-correlation sign convention was verified on a synthetic field (exact).

## Answer
**No.** Even perfect knowledge of the bulk displacement loses to persistence,
away from domain edges. At this lead and scale, Singapore rain changes shape
and intensity (convective growth and decay) more than it translates. Motion is
not the main missing ingredient -- the previous framing ("the network learns
statistics but not motion") was incomplete.

## Gotchas
- Global phase correlation on sparse fields gave implausible speeds when
  estimated from 10-min pairs (median 71 km/h after scale correction) vs 19 km/h
  measured over the full lead.
- Farneback with heavy smoothing came closest (0.406 vs 0.451) -- dense motion
  roughly reproduces persistence, it does not beat it.

## Recommendation
Stop treating "beat persistence at 35 min, 2 mm/hr" as a model-debugging
target. Check how persistence skill decays with lead time; if it drops sharply
at 60-90 min, that is where a generative model can add value (and the DoD asks
for 60/90 min anyway).

## Follow-up A/B: time-of-day + 60-min history (2026-09-25)
Same 12k-step schedule, same 60 anchors and seeds, FSS @ 2 mm/hr:

| | 6.1 km | 11.9 km | 23.5 km |
|---|---|---|---|
| persistence | 0.270 | 0.441 | 0.617 |
| control: 6 frames, no clock | 0.061 | 0.107 | 0.182 |
| 12 frames + hour-of-day | 0.058 | 0.109 | 0.185 |
| full model, 6 frames, 300k steps | 0.082 | 0.142 | 0.214 |

No measurable effect from the extra inputs. Defaults reverted to 6 / no clock
(flags kept). More training helped slightly (0.107 -> 0.142 at 25x the steps).
