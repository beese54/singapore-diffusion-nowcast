# Re-plan: Stages 4–5 (nowcaster skill) — for approval

*2026-09-25. No further experiments until this is approved.*

## 1. What is established

**The pipeline works.** Radar archive (124+ days, gap-alerted), lossless codebook
dataset, bf16 v-prediction diffusion, checkpoints stamped with their full input
layout, evaluation rebuilt from the stamp, eight verification suites passing.

**The model is physically realistic but has little placement skill.**

| | FSS @ 2 mm/hr, 11.9 km, 35-min lead |
|---|---|
| persistence (repeat the last frame) | **0.441** |
| best model (v-pred, full frames, 300k steps) | 0.142 |
| + residual forecasting | 0.091 |
| + 60-min history + hour of day (same-length A/B) | no change (0.109 vs 0.107) |
| optical flow, best estimator (Farneback) | 0.406 |
| optical flow with the **true** displacement | still below persistence |

Its rain *amount* is right (1.25–1.80% wet vs 1.83% observed); its rain
*placement* is not.

**Why it's stuck.** At this scale (0.29 km pixels, 35 × 63 km domain) and lead,
Singapore rain **changes shape and intensity more than it moves** — even perfect
knowledge of motion loses to persistence. Inflow from outside the box is minor
(~4% of target rain). So the remaining skill is in predicting **where storms form
and decay**, which recent radar frames barely encode.

**Persistence is beatable in principle.** At 12 km it falls below the "useful
skill" line (FSS ≈ 0.52) after ~20 min, and to 0.25 by 90 min. There is room
above it; nothing we have built reaches it.

## 2. Options

| | What | Effort | Expected payoff | Main risk |
|---|---|---|---|---|
| **A. Re-score** | Score the existing 300k model on metrics suited to a *probabilistic* flood tool: ensemble-probability FSS, CRPS vs persistence, rain accumulated over catchment-sized boxes, flood-cell hit rate on the 48 events | ~½ day, **no training** | Reveals whether useful skill is hidden by pixel-placement FSS | May confirm there is none |
| **B. Deterministic baseline** | Train the same UNet as a plain regressor (MSE) and score vs persistence | ~5 h (mostly unattended) | A **predictability test**: tells us whether *any* network can beat persistence here from radar alone | Deterministic forecasts blur; FSS at high thresholds may suffer |
| **C. ERA5 context** | Condition on instability / moisture / winds for May–Sep 2026 | 1–2 days + 10 h training | Only untested information source about storm initiation | ERA5 is 0.25° ≈ 28 km — only ~2 × 3 cells over this domain, so it informs the *regime* (will storms form?), not placement |
| **D. Shorter lead** | Target 10–20 min, where persistence is still "useful" | small | Easier to show skill | Little flood-warning value; floods on 22 Sep came ~40 min after peak rain |
| **E. Larger domain** | Add NEA's 240 km radar for upstream context | new scraper, weeks of history | Addresses inflow | Inflow measured at only ~4%: low priority |

## 3. Proposed sequence (with gates)

**Step 1 — A: re-score what we have.** ~½ day, no GPU training.
- *Gate:* if the 300k model beats persistence on ensemble-probability FSS or
  catchment accumulation, the Stage 5 criterion is measuring the wrong thing for
  this tool → go to the DoD decision below with evidence.

**Step 2 — B: deterministic predictability test.** ~5 h.
- *Gate:* if a deterministic regressor **also** cannot beat persistence, radar-only
  placement at 35 min is at the predictability limit for this domain; stop
  architecture work and decide between C and a criterion change.
- If it **can**, adopt a hybrid: deterministic mean for placement + diffusion for
  realistic texture/uncertainty (the pattern used by stronger published nowcasters).

**Step 3 — C: ERA5, only if Step 2 shows radar alone is capped** and the goal still
needs better initiation forecasts.

Steps 1 and 2 are cheap and each decides the next; together ~1 day.

## 4. Decision needed on the Stage 5 criterion

The DoD says: *"FSS at 20 dBZ, 30-min lead > persistence."* Evidence so far suggests
this may be unreachable from radar alone on this domain. Options, to decide
**after Step 1**, not now:

1. Keep it as is (and accept Stage 5 may not close without ERA5 or more).
2. Add a probabilistic criterion (e.g. CRPS or ensemble-probability FSS beats
   persistence) — matches what a diffusion model is for.
3. Add a flood-relevant criterion (e.g. accumulated rain over catchment boxes, or
   hit rate on the recorded flood events) — matches what the project is for.

## 5. Constraints to plan around

- **Memory:** host sits at 77–93% load; monitoring shells get reaped. Long runs
  must use the detached launcher; scoring cannot run alongside training.
- **Compute:** full-frame training ≈ 129 ms/step (300k ≈ 10.75 h); probes ≈ 30 min.
- **Flood ground truth:** only 2 observed FLASH_FLOOD events; any flood-hit metric
  is indicative, not statistically meaningful, for months.
- **Scale:** pixel = 0.29 km (spec corrected); always label windows in km.

## 6. What I need from you

1. Approve (or change) the sequence **A → B → (C)**.
2. Confirm the Stage 5 criterion decision is deferred until Step 1's results.

---

## Step 1 results (2026-09-25) — gate triggered

`scripts/evaluate_probabilistic.py`, 300k model, 200 test times x 8 members +
38 flood events; 95% CIs by bootstrap over test times (2,000 resamples).

| | model vs persistence |
|---|---|
| **CRPS skill**, rain-relevant pixels | **+0.44** [+0.34, +0.56] |
| CRPS skill vs *lagged-persistence ensemble* (fair probabilistic baseline) | **+0.33** |
| Rain over 2.9 / 6.1 / 11.9 km boxes — CRPS skill vs lagged persistence | **+0.36 / +0.36 / +0.36** |
| same — Brier skill, "box mean >= 0.5 mm/hr" | **+0.31 / +0.35 / +0.31** |
| Ensemble-prob. FSS, 2 mm/hr, 11.9 km (pooled) | 0.820 vs 0.801 — **tie**, CI [-0.064, +0.063] |
| Ensemble-prob. FSS, 10 mm/hr, 11.9 km | 0.530 vs 0.612 — worse, not significant |
| Flood events, >=2 mm/hr within 1.4 km | 87% vs 68% hits; ordinary-time alarms 3.1% vs 10.7% |
| Flood events, >=10 mm/hr within 1.4 km | 47% vs 58% hits |

**Reading.** The earlier "FSS 0.14 vs 0.44" came mostly from *how* evaluate.py
scores: single members, and FSS averaged per sample rather than pooled (the
standard definition, Roberts & Lean 2008). Scored as an ensemble with pooled
FSS, placement is a statistical **tie** with persistence; probabilistic skill is
**clearly positive and robust**. The real weakness is **heavy rain (>= 10 mm/hr)** —
which is what drives flash floods.

Per section 3, this triggers the Stage 5 criterion decision.

## Decisions (2026-09-25)

1. **Stage 5 criterion amended** (definition_of_done.md, specification.json):
   PRIMARY = CRPS skill vs persistence > 0 with 95% CI lower bound > 0;
   TRACKED = pooled ensemble-probability FSS at 2 mm/hr; TARGET = heavy rain
   (>= 10 mm/hr). `scripts/evaluate.py` now scores exactly this.
2. **Step 2 (deterministic predictability test) skipped** — Step 1 answered the
   main question. Next effort goes to heavy-rain skill.

## Heavy-rain work (2026-09-25)

**Diagnosis** (300k model, cached 200 x 8 test forecasts):
- *Intensity bias grows with intensity* — frequency bias 0.90 / 0.85 / 0.54 / 0.40 / 0.12
  at >= 0.5 / 2 / 10 / 20 / 50 mm/hr; median member peak 6.5 vs observed 22.5 mm/hr
  on heavy-rain cases (best-of-8 member: 30.6).
- *Dilution* — members put heavy cores in different places: some member >= 10 mm/hr on
  1.37% of pixels (3x observed), >= half the members on 0.056% (8x too little).
- *Decay* — of pixels heavy in the last frame, 27% are still heavy 35 min later;
  the model keeps 12% (66% vs 43% allowing 3 km of movement).

**Cheap fixes — rejected:**
- *Deterministic sampling (eta 0)*: fixes heavy intensity bias (0.54 -> 1.09) but
  over-forecasts light rain (bias 1.47), lowers CRPS skill (+0.437 -> +0.370) and does
  not improve heavy-rain FSS on the full 200 (0.530 -> 0.506). A 40-sample sweep on
  heavy-rain cases only had suggested otherwise -- confirmed on the full set before
  deciding.
- *250 DDIM steps*: 5x compute, no clear gain on the full picture.
- *Quantile calibration* (fitted on validation): modest, safe gain on the current
  sampler (heavy FSS 0.530 -> 0.564, light 0.820 -> 0.830, CRPS unchanged, bias partly
  fixed) but fragile — fitted on ~11 days, and the same method on the eta-0 sampler
  collapsed heavy FSS to 0.269 through a val/test climate mismatch. Not adopted.

**Conclusion:** intensity is fixable at the margins but is not what holds heavy rain
back; the anchoring/decay of storm cores is. Next: heavy-rain weighted loss, clean A/B.

**Heavy-rain weighted loss — rejected** (clean A/B, 12k steps each, 200 test times x 8):

| | control | heavy-weighted (beta 20, 2 km) |
|---|---|---|
| training guard | clean | DEGENERATE "rain everywhere" at 8k and 12k |
| frequency bias >=2 / >=10 mm/hr | 0.69 / 0.60 | 4.90 / 4.97 |
| storm cores kept (observed 27.0%) | 14.6% | 5.0% |
| pooled FSS 2 mm/hr, 11.9 km | 0.746 | 0.521 (paired CI [-0.312, -0.008]) |
| pooled FSS 10 mm/hr, 11.9 km | 0.556 | 0.312 |
| CRPS skill vs persistence | +0.415 | -0.136 |
| rain flagged within 4.9 km of flood sites at ordinary times | 5.9% | 84.8% |

Mechanism (see L027): with example-dependent weights the optimal denoiser is the
posterior mean under a *reweighted* distribution, so the model samples a world
where rain is more common. This is the second loss weighting to produce "rain
everywhere" (the first, intensity_alpha, was attributed to confounded crops in
L022). Loss reweighting is not a usable lever for rare-event skill here.

**Status of heavy rain:** open. Cheap fixes (sampler, calibration) and loss
weighting are exhausted; what remains is information or architecture.
