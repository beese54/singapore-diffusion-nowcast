# Observability Plan — Radar Ingest Gap Alerting

**Pattern AP.** Scope: the data-collection pipeline (scrape → preprocess →
label join). Not the model.

## Phase 0 — The Questions

The pipeline has no SLO in the usual sense; its product is *archive
completeness*. Every signal below answers a question that decides an action.

| # | Question | Signal | Why it matters |
|---|---|---|---|
| Q1 | Are frames missing? | count of absent 5-min slots vs expected | baseline health |
| Q2 | **Are missing frames still recoverable?** | gap age vs NEA retention (~6.5–7 d, measured) | **the only window in which action is possible** |
| Q3 | Did a gap overlap a flood event? | gap ∩ FLASH_FLOOD/FLOOD_RISK label times | a gap here silently deletes ground truth from train *and* eval (`_contiguous` drops the sample) |
| Q4 | Is the zarr behind the PNGs? | `radar.zarr` frame count / last time vs PNG count | eval joins read the zarr, not the PNGs |
| Q5 | Are the scheduled tasks actually running? | last result / last run time per task | a dead task looks identical to clear weather |

## Phase 1 — Instrument

Existing: `scrape_radar.py --status` (passive day counts, <240/day) and
`preprocess_radar.py --status` (zarr span). Neither distinguishes
**recoverable** from **permanently lost**, and neither alerts. That distinction
is the whole point of this work.

New: `scripts/check_radar_gaps.py`, mirroring the proven shape of
`check_flash_flood.py` — durable flag file + best-effort Windows toast +
JSON state for dedup, `--status` for read-only reporting.

## Phase 2 — Alerts (symptom-based, runbook-linked)

Symptom in user terms: *"training/eval data is incomplete, and part of it is
about to become unrecoverable."*

| Severity | Fires when | Runbook | Route |
|---|---|---|---|
| **CRITICAL** | gap inside retention **and** older than 4 d (≤~2.5 d left to act) | `python scripts/scrape_radar.py --hours <age+6>` then `preprocess_radar.py` then `build_flood_eval_dataset.py` | toast + flag |
| **CRITICAL** | gap overlaps a FLASH_FLOOD / FLOOD_RISK label | same, then verify `build_flood_eval_dataset.py --status` reads `N/N` | toast + flag |
| **WARN** | gap < 4 d old (daily catch-up task should self-heal) | none — observe; escalates on its own if it survives to 4 d | flag only |
| **WARN** | zarr count < PNG count, or zarr last time > 1 d behind | `python scripts/preprocess_radar.py` | flag only |
| **INFO** | gap older than retention | **none — unrecoverable.** Recorded once, then permanently suppressed. | state file only |

**Alert-budget decisions (explicit, because they are what keep this trusted):**
- The three permanently-lost events (2026-05-24, 2026-06-29 ×2) must **never**
  page again. A naive gap checker would fire on them every single run and train
  the user to ignore all gap alerts. They are recorded as INFO and suppressed by
  id in the state file.
- Today is *not* a gap alert. It self-healed. Only gaps that survive to 4 days
  page — that is the threshold that separates "the catch-up task has it" from
  "act now or lose it forever."
- No alert fires on a cause (laptop was off). Being off is expected and normal
  for this project; only the *consequence* pages.

## Phase 3 — Dashboard
`--status` is the dashboard: per-gap table (start, end, slots, age, recoverable
y/n, overlaps-flood y/n), plus Q4/Q5 lines and the current retention estimate.
Answers "what's wrong and can I still fix it" in one screen.

## Phase 4 — Verify by Simulating (done)
`tasks/repro/gap_alert_tiers_repro.py` — injects synthetic gaps, runs the real
`find_gaps()` + `classify()`, asserts the tier. **7/7 pass.**

Live end-to-end also verified 2026-09-22: a real WARN (zarr 9 frames behind
PNGs) fired with flag + toast, the printed runbook cleared it, and the next run
went silent.

### What verification changed — the 404 discovery
The first live run raised two CRITICAL gaps (2026-09-16 03:15, 2026-09-17 11:35;
4 frames) as "recoverable, aging out". Running the runbook returned **404 for
all four**: they are inside the retention window but NEA never published them.

Assumed recoverability was therefore wrong, and the consequence was the precise
failure this design exists to prevent — those gaps would have re-paged as
CRITICAL every 24h until they aged out, with no action able to clear them.

Fix: `probe_upstream()` samples up to 4 slots per in-window gap against NEA and
**verifies** availability. A gap NEA cannot serve is INFO/suppressed, not a page.
This added a second INFO kind ("never published upstream"), reported separately
from "past retention" because the two are different facts.

Corollary now enforced: *inside retention ≠ available.* Only upstream-confirmed
gaps page.

## Non-goals
Email/Telegram push (toast + durable flag already survives travel, per the
checkpointing rule); alerting on rain intensity (that is the model's job).
