# Lessons for other researchers

The full log, with the evidence behind each entry, is [`tasks/lessons.md`](../tasks/lessons.md)
(L001–L031; there is no L019, a numbering gap). This page groups the ones most likely to save
someone else time.

## Evaluating probabilistic nowcasts
- **Score an ensemble as an ensemble** (L026). Single-member, per-sample-averaged FSS made the model
  look 3× worse than persistence; pooled ensemble-probability FSS showed a tie.
- **CRPS against persistence is not enough** (L029). Persistence is double-penalised when storms
  move, so smooth weak rain in the wrong place can still "win". Always pair it with a placement score.
- **Every metric family must respect the split** (L031). Our flood-event scores silently included
  training-period events until a cache check exposed it. Count storms, not reports.
- **Pin splits by date, and make caches record what they cover** (L028). A "last 10%" split of a
  growing archive changed daily and once turned CRPS skill +0.44 into +0.13 with no error.

## Diffusion models on sparse rain fields
- **v-prediction, not ε-prediction, for a near-binary field** (L021). With 97% of pixels at the dry
  value, ε-prediction sampling drifted off the data manifold.
- **Data-dependent loss weights change what a diffusion model samples** (L027). Up-weighting rain
  teaches the model that rain is more common — it rained everywhere. Weighting is fine in a regressor,
  not in a generative model.
- **A residual target does not give a persistence floor to a generative model** (L024). It samples a
  plausibly sized change in the wrong places.
- **Warm-start harder variants from the model that solves the easier one** (L030). From scratch, the
  60/90-min models learned whether it rains but not where.
- **A falling loss on a sparse target proves nothing** (L020); check samples.
- **fp16 + GradScaler can ratchet into silent collapse; prefer bf16** (L016).

## Radar data engineering
- **Categorical fields must never be resampled or interpolated** (L012) — decode colours exactly.
- **Measure the grid's physical scale from its coordinates** (L025) — we carried a wrong 1 km pixel
  size for weeks; the real value is 0.29 km.
- **Validate georeferencing and timestamps against a physical signal** (L009) — every flood label
  was 8 h late until rain at the reported place was checked.
- **Defensive clamps can turn a loud crash into months of silently wrong data** (L011).
- **Store categorical data as codes** (L018): lossless and 4× smaller.

## Process
- **Smoke tests must not certify a stage** (L004); **dated blockers keep blocking after they come
  true** (L013); **alerts must verify they are actionable** (L015).
