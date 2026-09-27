# Superseded and ablation results

Kept for the record, because the reasoning in `definition_of_done.md`,
`tasks/lessons.md` and `docs/history/` depends on them. **Not results of record**:
the current numbers are in `results/evaluation_report.json` and
`results/probabilistic_eval_lead{30,60,90}.json`.

| File | What it is | Why it is here |
|---|---|---|
| `probabilistic_eval_lead30_prepin_2026-09-25.json` | 30-min model (300k steps), scored 2026-09-25 | Scored before the test split was pinned by date; its test period differs from the final one (lesson L028). Re-scored on the pinned split in `probabilistic_eval_lead30.json` |
| `probabilistic_eval_lead60_scratch.json` | 60-min model trained from scratch, 100k steps | Superseded by the warm-started model. Passed CRPS but lost to persistence on every placement measure (lesson L029) |
| `probabilistic_eval_lead90_scratch.json` | 90-min model trained from scratch, 100k steps | Same as above |
| `heavy_weight_control_12k.json` | 12k-step control run for the heavy-rain loss-weighting A/B | A/B control |
| `heavy_weight_beta20_12k.json` | 12k-step run with heavy-rain loss weight β=20 | Rejected: over-forecast rain area ~5x, CRPS skill -0.136 (lesson L027) |

The checkpoint each file came from is recorded in its `checkpoint` field.
