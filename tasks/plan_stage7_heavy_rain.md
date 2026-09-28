# Plan: Stage 7 — does the wide 240 km radar hold the missing heavy-rain information? — for approval

*2026-09-28. Workstream D3 of `tasks/plan_phase2.md`. The 240 km product is now collected, gap-watched and
georeferenced (`results/georef_240km.json`: 480 × 480 km, 1 km pixels, same colour legend).*

## Where we are

- The 30-min model beats the naive forecast on rain overall, but not on heavy rain:
  - FSS ≥10 mm/hr at 11.9 km is 0.34 vs 0.45;
  - it catches 29% of downpours.
- At 60 and 90 min, its heavy-rain skill is no better than the naive forecast.
- Adding the large-scale weather did not help: neither CorrDiff nor ERA5 (L033).
- The remaining hypothesis is **arrival from outside**. The 70 km domain is only 35 × 63 km, so at 60–90 min most of the rain that will fall on it is still outside the domain, where the model cannot see it. The 240 km view does see it.

## The constraint that shapes this plan

The 240 km archive starts on **27 Aug 2026**, while the model's training period ends on 2 Sep. Only about 6 training
days overlap, so a model conditioned on the wide view **cannot be trained yet**. It needs a new split with months of
overlap, i.e. the Dec–Mar monsoon. Before spending those months, we should check that the wide view actually carries
the information. That check needs no training.

## Step 1 — spike (no training, CPU, ~½ day)

**Question.** At 30/60/90 min, does simply moving the wide-view rain forward catch heavy rain arriving on the
70 km domain that a 70 km-only forecast cannot see?

0. **Time alignment.** Correlate 240 km and 70 km rain at shifts of −10…+10 min, and use the best shift. The two products may be timestamped differently.
1. **Wide-view extrapolation forecaster (W).**
   - At each test issue time, estimate motion from the last 30 min of 240 km frames using optical flow (planned: Farneback; **changed to DIS** before any results, because Farneback failed a synthetic known-shift check, L034).
   - Advect the rain forward by the lead time.
   - Resample onto the 70 km grid using the measured bounds.
2. **Same method, 70 km only (S).** This isolates the value of the *wider view* from the value of extrapolation itself. Rain that starts outside the 70 km box is invisible to S.
3. **Scoring.**
   - Test period 14–25 Sep plus 26–28 Sep, reusing the 200 cached test forecasts per lead.
   - Metrics, as in `warning_skill.py` (2 × 2 km boxes, ≥10 mm/hr): catch rate, precision, and FSS ≥10 at 11.9 km.
   - Main metric: **incoming heavy rain**, i.e. boxes that are heavy at the target time but were not heavy at issue time. These are exactly the cases the naive forecast misses by construction.
   - Comparisons: W vs S vs persistence vs our model at the same lead.
   - Paired bootstrap 95% CIs; the 3 flood storms are also shown as cases.

**Go / no-go (fixed now).**
- **GO** for a Stage 7 model: at 60 or 90 min, W catches more incoming heavy rain than S, with the paired CI of the difference above 0, while its precision is not lower than S's (CI of the difference not entirely below 0). That would show the wide view contains information our model lacks.
- **NO-GO:** the wide view adds nothing that simple motion can use. We record this and move the heavy-rain effort to satellite data (convective initiation, which radar sees only once rain has formed).
- **Also reported, not part of the decision:** whether W on its own beats our model at 60/90 min. If it does, a simple "wide extrapolation" warning could be useful before any retraining.

**Honest limits, stated with any result.**
- Optical flow is a weak forecaster: it lost to persistence on the 70 km domain, because rain grows and decays more than it moves.
- The spike tests whether the information is there, not the best way to use it.
- The test period covers about 15 days and a few storms.

## Step 2 — only if GO: the Stage 7 model (plan detail after the spike)

- **Input:** a second context branch holding the last 30 min of 240 km frames, cropped to about 200 × 200 km around Singapore at 2 km (100 × 100 px). It is encoded and added to the U-Net at the matching resolution. The zero-initialised pathway is reused as in the ERA5 work, so the model starts from the current one.
- **Training:** a new pinned split inside the 240 km era, started once there are about 3 months of overlap (around Dec 2026). Fine-tuning must avoid the learning-rate-restart harm found in the ERA5 control (L032): low peak learning rate, and the control arm is kept.
- **Evaluation:** the same Stage 5 metrics plus the drain-alert benchmark, paired against the model of record.

## Meanwhile (automatic)

- Collection of both radar products continues, with the gap alert watching both.
- ERA5 downloads continue.
- D2 (rain-gauge history depth) remains open.

## Result, Step 1 (2026-09-28): **NO-GO** for the wide view as the missing information

`scripts/spike_240km_extrapolation.py` → `results/spike_240km.json`. 200 test forecasts per lead. The 240 km and
70 km timestamps are aligned (correlation 0.94 at zero shift).

| Incoming heavy rain (≥10 mm/hr, 2 × 2 km boxes) | 30 min | 60 min | 90 min |
|---|---|---|---|
| Share of heavy boxes that are "incoming" (not heavy at issue time) | 75% | 87% | 91% |
| Caught by W (wide view extrapolated) | 12.1% | 7.5% | 9.3% |
| Caught by S (same method, 70 km footprint only) | 13.1% | 8.5% | 8.7% |
| **W − S** | −0.011 [−0.046, +0.019] | −0.010 [−0.027, +0.008] | +0.006 [−0.033, +0.038] |
| New-warning precision, W − S | −0.037 [−0.099, +0.029] | −0.081 [−0.172, −0.002] | −0.079 [−0.200, +0.009] |

**Answer.** Seeing 480 km instead of 63 km adds nothing a motion forecast can use: W catches no more incoming heavy
rain than S, and its extra warnings are less often right. The heavy rain that appears on the domain 1–1.5 h later is
mostly **not** rain that already exists upwind and moves in. It forms or intensifies, which is convective initiation.
Per the rule, the Stage 7 wide-radar model is not built on this evidence, and heavy-rain effort moves to satellite
data, which can see clouds growing before they rain.

**Two further findings.**
1. **The 240 km product sees only part of the heavy rain.** Even at the target time itself (the "ceiling" row), the 1 km product
   marks only 51–59% of the boxes the 70 km product calls heavy (93% precision). Any wide-radar model inherits this.
2. **Simple motion extrapolation beats our diffusion model at 60 min for heavy rain.** W − model: catch
   +0.072 [+0.014, +0.149], precision +0.089 [+0.010, +0.163], incoming catch +0.039 [+0.003, +0.077]. At 90 min it
   ties, and at 30 min it is more precise but catches less incoming rain. This matches L034: the earlier
   optical-flow spike used a method that under-read motion. **Motion is a missing ingredient at 60 min after all**, and
   it can be tested with the 70 km archive we already have (months of data), for example by giving the model the
   extrapolated field as an extra input. That is the next candidate, and it needs its own plan.

**Limits.** The test covers about 12 days and a few storms. Extrapolation is one simple method; a learned model might
still extract something from the wide view (storm organisation, outflow boundaries) that advection cannot. That idea
can be revisited once months of 240 km data exist, but this spike gives no evidence for it.
