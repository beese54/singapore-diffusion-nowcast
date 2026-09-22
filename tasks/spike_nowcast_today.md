# Spike: Can we predict today's (2026-09-22) flash flood with current data?

**Pattern N — Timeboxed Spike.** Timebox: ~20 tool-call rounds. Spent: ~15.
Spike code: none retained. Timing runs used `train.py --smoke` (isolated
`checkpoints/nowcaster/smoke/`, stage flag guard verified working); the three
checkpoints those runs produced were deleted.

## THE QUESTION
With the radar archive as it stands on 2026-09-22, can we produce a 30-minute
nowcast that would have flagged the King's Road / Coronation Road flash floods
(17:11 / 17:15 SGT)?

**Success criteria**
- YES: a trained nowcaster exists (or can be trained today) AND today's event
  sits in a clean holdout AND the samples covering it are usable.
- NO: no weights exist and/or the data cannot support the sample.

## OUT OF SCOPE
Flood-specific skill scoring (FSS/CRPS/lead-time curves — that is Stage 5,
tasks 5.1/5.2/5.3/5.5), CorrDiff fine-tuning (Stage 6), ERA5 conditioning.

## EXPERIMENTS

| Tried | Result | Implication |
|---|---|---|
| `ls checkpoints/nowcaster/` | only `smoke/` (June toy ckpts) | **No trained model exists.** Task 4.6 still Pending. |
| Read Stage 4 blocker | "Blocked until Stage 3 reaches 90-day archive (~late Aug 2026)" | Blocker is **stale**. |
| Measure archive span | 33,093 frames, 2026-05-22 → 2026-09-22 = **123 days** | 90-day gate passed ~2026-08-20, a month ago. `stage3_complete.flag` written today. |
| Locate today's flood in dataset splits | index 33,058 → **TEST** split (test starts 2026-09-10 23:50) | Clean holdout. No leakage if trained now. |
| Contiguity check on flood sample (ctx 6, lead 6) | **True** | Sample is usable — *because* today's 06:15–17:55 SGT gap was backfilled. Pre-backfill it would have been dropped by `_contiguous`. |
| Usable TEST samples | 3,266 total; **139 dated 22 Sep** | Enough to score today specifically. |
| `train.py --smoke` with default `num_workers=2` | **CRASH** — `OSError: [Errno 22]` / truncated pickle | **New blocker.** `RadarDataset` holds the whole array in RAM: 33,093×120×217 fp32 = **3.45 GB**, which Windows spawn cannot pickle to workers. Worked in June at 11,610 frames (1.21 GB). Archive growth broke it. |
| Same with `--num-workers 0` | Runs clean. 48,192 train / 3,249 val samples; 25.8M params; loss 1.33→0.09 over 450 steps | Workaround confirmed; no OOM on 7 GB VRAM. |
| Timed throughput (2-point, 150 vs 450 steps) | **132 ms/step ≈ 7.56 steps/s** | 50k steps ≈ **1.8 h**; 100k ≈ **3.7 h**; full 300k ≈ **11.0 h**. |

## ANSWER
**No — not today, but only because nothing is trained yet; the data is ready and
a real run is ~11 hours, not weeks.**

The archive cleared its 90-day gate a month ago and today's flood lands in the
held-out test split with usable samples, so the experiment is properly set up.
What is missing is purely the training run (task 4.6).

## GOTCHAS
1. **`num_workers=2` is now a hard crash**, not a slowdown. Any real run must
   pass `--num-workers 0` until `RadarDataset` is changed to lazy/memmap reads.
   This will keep getting worse as the archive grows.
2. **The Stage 4 blocker text is stale** and would have kept the run deferred
   indefinitely if taken at face value.
3. **The backfill was load-bearing.** `_contiguous` drops any sample whose
   6-frame context + 6-frame lead span crosses a hole, so an unbackfilled gap
   deletes the event from *both* training and evaluation silently.
4. Lead time is fixed at `target_offset=6` = **30 min**. Today's floods were
   reported ~40 min after peak rain (peak 16:30–16:50, floods 17:11/17:15), so
   a 30-min nowcast is operationally meaningful for this event.
5. 300k steps at batch 4 is the default but is not justified by evidence; loss
   was already at ~0.09 by step 450. Treat 50k as the first checkpoint to score.

## PRODUCTION DELTA
- `--num-workers 0` is a workaround; the real fix is lazy loading in
  `RadarDataset` (moderate: one class, but touches the index/filter logic).
- Scoring today's event needs Stage 5 tasks 5.1/5.2/5.3 (`evaluate.py` currently
  does radar-vs-radar FSS/CRPS only and never reads flood labels).
- Checkpoint/resume already works (`--resume auto`), so an 11-hour run survives
  the laptop being shut for travel.

## RECOMMENDATION
**Proceed.** Unblock Stage 4 and launch the real training run with
`--num-workers 0`, scoring the first checkpoint at 50k steps (~1.8 h) against
the 139 usable 22-Sep test samples before committing to the full 300k. Fix the
stale blocker note and the `num_workers` default regardless of whether the run
starts today.
