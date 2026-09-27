# Plan: Phase 2 — publish, dashboard, CorrDiff, heavy-rain data (for approval)

*Written 2026-09-27, after Stage 5 closed. Nothing below is built yet (Gatekeeper rule).
Protocols: Pattern X (GitHub curation, first-push gate) for publication; others named per workstream.*

## 0. Where we start (verified today)

| Item | State |
|---|---|
| Stages 0–5 | Complete. 30-min nowcaster beats persistence (CRPS +0.384 [+0.25, +0.51], FSS 2 mm/hr 0.83 vs 0.74); 60/90-min on par for light rain; heavy rain not beaten at any lead |
| Secret scan | **Clean.** 64 commits, all tracked files, full history: no `.env` value (NGC, NVIDIA, CDS, data.gov, Telegram id/hash) and no key-format string (`nvapi-`, `ghp_`, AWS, private keys) ever committed. `.env`, `telegram.session`, `data/raw/`, `checkpoints/` ignored since the first commit. No email/phone tracked |
| Scan findings to fix | 3 `.bat` launchers hard-code `C:\Users\<username>\...` (Windows username; also non-portable) |
| Publication blockers | **No README, no LICENSE.** No git remote yet. `gh` logged in as `beese54` |
| Stale docs | `tasks/todo.md` (last 22 Sep), `specification.json` goal still says "1–2 km", `scripts/launchpad/project_brief.md` (June: wrong resolution, and pairs 2022–23 ERA5 with 2026 radar — unusable, see §3) |
| Clutter | `data/raw/era5/tmp_pressure_*.nc` leftovers; superseded results (`probabilistic_eval.json` pre-pin, from-scratch lead60/90, probe_hw_*) mixed with results of record |
| Repo size | 32 MB `.git`; largest blobs are notebook outputs (≤2.7 MB). No history rewrite needed |

## 1. Workstream A — Repo hygiene & documentation for researchers  *(Pattern X + O)*

Goal: a stranger can understand, reproduce and critique the work from GitHub alone.

A1. **Cleanup** (each a small commit)
- Launchers: replace hard-coded paths with `%~dp0` / `python` on PATH (removes the username).
- Move superseded results to `results/ablations/` with a README table (what each was, why superseded); keep results of record at top level.
- Move historical plans/spikes (`migration_plan`, `observability_plan`, `replan_stage4_5`, `spike_*`) to `docs/history/` — kept, not deleted: they are the audit trail.
- Delete local leftovers `data/raw/era5/tmp_*.nc` (untracked; confirm list before deleting).
- Refresh `tasks/todo.md` to the Phase 2 checklist; fix `specification.json` goal/resolution wording.

A2. **README.md** (front door) — what it is (one sentence) → dashboard GIF → results table with CIs → how it works (diagram) → quick start → **limitations** (prominent) → data sources & licences → citation → roadmap.

A3. **`docs/`** for researchers
- `docs/METHODS.md` — data pipeline (PNG→33-level LUT→codebook zarr, georeferencing validation), model (v-pred DDPM, UNet, one model per lead, warm start), evaluation (pinned split, pooled FSS, CRPS, bootstrap CIs, flood-event scoring with ordinary-time control).
- `docs/RESULTS.md` — every number of record, with the command that produced it and the JSON it lives in.
- `docs/LIMITATIONS.md` — heavy rain; one case study; 4-month archive (one monsoon transition, no NE monsoon peak); radar-only; CRPS-vs-persistence pitfall (L029); 0.29 km pixels but skill only at ~12 km neighbourhoods; flood labels are warnings, not gauge truth; geocoding to points.
- `docs/LESSONS.md` — the 30 lessons, lightly edited (this is the most useful part for other researchers: what failed and why).
- `docs/REPRODUCE.md` — environment, scheduled tasks, rebuild archive, train, evaluate — every command run once in a clean checkout before it is written (Pattern O rule).
- `docs/DATA.md` — sources, licences, what is and is not redistributed (see decision D2).

A4. **LICENSE** (decision D1), `CITATION.cff`, repo description + topics.

A5. **First-push gate** (Pattern X): re-run the secret scan on the final tree + history, `git ls-files` review, then push. Post-push verify with `git grep` on `origin/main`.

## 2. Workstream B — Dashboard web app  *(Pattern BB for the frame player; Pattern O for content)*

Picture: one page that explains the project to a non-technical reader in 30 s and lets a researcher dig in.

**Architecture (recommended): static site** — plain HTML/JS + precomputed data, hosted free on GitHub Pages, no server. Everything heavy (sampling) is precomputed by one script `scripts/build_dashboard.py` from the caches we already have. A GPU backend cannot run on Pages; "live" is shown as a recorded run plus the exact command (and optionally a local live mode, B6).

Sections:
- B1. **Hero** — animated radar loop of the 22 Sep storm with flood reports popping up at 17:11/17:15 SGT; one-line result.
- B2. **How it works** — 6 frames in → diffusion denoising (show a sample going from noise to rain field over the 50 DDIM steps — striking and honest) → 8 members → probability map.
- B3. **Time-series explorer (the key visual)** — scrub through the afternoon: observed | persistence | ensemble mean | P(≥10 mm/hr), with a lead selector 30/60/90 and the flood-cell time series underneath (the notebook 02 figure, made interactive). Shows *why* 60/90 degrade.
- B4. **Evidence** — results table with CIs; FSS by scale; flood-event hits; reliability diagram; the ablation story (what we tried, what failed) — sourced from `results/*.json`, not retyped.
- B5. **Live forecast in 20 s** — timeline of one real `nowcast.py` run (load 1.7 s → 3 leads × ~5.5 s → save), GPU/model facts, the command.
- B6. *(optional)* local live mode: `python scripts/serve_dashboard.py` runs `nowcast.py` on the latest frame and refreshes B3.
- B7. **Flood-risk map** — PUB flood-prone points over P(≥10 mm/hr) (notebook 03).
- B8. **Roadmap** — CorrDiff + heavy-rain stage (§3, §4).

Verification: open in Chrome, check desktop + phone width, no console errors, reduced-motion fallback; screenshots captured for the post.

## 3. Workstream C — NVIDIA CorrDiff  *(Pattern N spike first, then Pattern C)*

**What I found (27 Sep):**
- **LaunchPad is free, temporary, pre-configured hands-on labs** (curated datasets, notebooks, SDKs) — not a place to submit a multi-day custom training job. Our June `submit_job.sh` assumes the latter. Worth applying for exposure and a guided PhysicsNeMo lab, but it is not the training plan.
- **CorrDiff** (PhysicsNeMo) trains a regression U-Net then a diffusion corrector; custom data plugs in through a `DownscalingDataset` class. NVIDIA's guidance: reliable super-resolution **≤ ×16, or ≤ ×11 when inferring a new variable** (ERA5 fields → radar rain rate is that case). ERA5 is ~28 km → realistic target **~2.5 km, not 0.29 km**. CorrDiff-Mini trains in ~10 A100-hours; a single GPU works but is slow.
- **Data pairing error in the June brief:** ERA5 2022–23 was paired with 2026 radar. Downscaling needs the same times → **download ERA5 for 22 May 2026 → now** (ERA5 lags ~5 days).
- **Domain is tiny for ERA5:** our radar domain (35 × 63 km) is only ~2 × 3 ERA5 cells. The input must be a wider ERA5 window (e.g. 3°×3° around Singapore) with the target the radar domain coarsened to ~2.5 km (~14 × 25 px). The 240 km NEA radar (see §4) would make a far better target domain.

**Plan:**
- C1. Download ERA5 (same variables as 2022–23) for 2026-05-22 → present, and keep it updating weekly.
- C2. Spike (Pattern N, 1 day): build the paired dataset + `DownscalingDataset`, train CorrDiff-Mini-style **regression stage only** on the RTX 4060 to see if 8 GB suffices and whether there is any skill over bilinear ERA5 `tp`. Go/no-go.
- C3. If go: full regression + diffusion. Compute options, in order: (a) local 4060 at reduced size; (b) short cloud A100 rental (NVIDIA Brev or similar, ~10 GPU-hours for a Mini-scale run); (c) LaunchPad/Inception if they fit.
- C4. Evaluate vs bilinear ERA5 and vs our nowcaster's domain statistics (spatial spectrum, CRPS, FSS at 2.5 km), write up honestly.

**What you can do to facilitate (only you can):**
1. Apply at <https://www.nvidia.com/en-us/launchpad/> with your NVIDIA developer account — I will rewrite `project_brief.md` with correct facts first (C0) so you submit an accurate application.
2. Decide a small cloud-GPU budget if the spike says go (D5).
3. Optionally NVIDIA Inception (if you register a company) for credits.

## 4. Workstream D — Heavy-rain stage: collect the right data now  *(Pattern N, then Stage 7 plan)*

Why: every radar-only fix failed; the diagnosis is missing information — storms that are not on the 35 × 63 km radar yet, and growth/decay. Two kinds of data: things we must **collect forward** (not archived upstream), and things we can **backfill** later.

| Source | Why for heavy rain | Archive upstream? | Action |
|---|---|---|---|
| **NEA 240 km radar** (`rainarea/240km/dpsri_240km_*.png`, 480×480, verified 200 today) | sees storms 100+ km out, 1–2 h before they arrive | **~7 days only** | **Start collecting now** (extend the scraper + scheduled task); cheapest, highest-value |
| NEA 70 km radar (current) | nowcaster input | ~7 days | already collecting |
| **Rain gauges** (data.gov.sg real-time rainfall, 5-min, ~60 stations) | ground truth for heavy rain; radar calibration | historical API — verify depth | verify backfill; else collect forward |
| **Himawari-9** (JMA, NOAA open-data bucket, 10-min, 2 km IR) | cloud-top cooling = storm growth *before* rain on radar | yes (public archive) | backfill later; spike reader now |
| ERA5 | large-scale environment (CAPE, moisture, wind) + CorrDiff input | yes (5-day lag) | C1 |
| NWP forecasts (e.g. ECMWF open data) | forecast environment | **recent days only** | decide in Stage 7 plan |
| PUB Telegram flood labels | evaluation truth | ours | already collecting |

- D1. Add 240 km radar collection (scraper + gap alert) — this week, because every day not collected is lost.
- D2. Verify gauge API history depth; set up collection accordingly.
- D3. Write `tasks/plan_stage7_heavy_rain.md` (for approval) after 2–4 weeks of 240 km data: candidate designs (240 km context as extra conditioning; satellite channels; gauge-calibrated targets), evaluation on heavy-rain FSS + flood hits, with an honest go/no-go.
- Data longevity: the archive is the project's most valuable asset → document backup (OneDrive already syncs `data/`? verify) and storage growth (~X GB/month, measure).

## 5. Workstream E — LinkedIn post  *(Pattern V)*

- Framing: **continuation**, not a wrap-up — "Phase 1 done, here's what worked, what didn't, and what's next (CorrDiff + heavy rain)".
- Visuals (captured from the dashboard with the Chrome extension): (1) radar time-series animation/GIF of 22 Sep with flood reports; (2) observed vs forecast panel strip; (3) the results/limitations card; (4) roadmap graphic with NVIDIA CorrDiff.
- Honesty rules: say 30-min works, heavy rain doesn't yet; no "predicts floods" overclaim; credit NEA/PUB data; NVIDIA mention factual (PhysicsNeMo/CorrDiff planned, LaunchPad applied — only if true at posting time).
- Deliverable: `docs/linkedin_post.md` + `docs/img/` screenshots. **You post it** — I will not post on your behalf.

## 6. Sequencing

| Order | Work | Depends on | Est. |
|---|---|---|---|
| 1 | D1 start 240 km collection | — | 0.5 d (time-critical) |
| 2 | A1 cleanup, C0 fix LaunchPad brief | — | 0.5 d |
| 3 | A2–A4 README, docs, licence | A1 | 1–1.5 d |
| 4 | B dashboard | A (results docs) | 1.5–2 d |
| 5 | E screenshots + post draft | B | 0.5 d |
| 6 | A5 first-push gate → **publish** (your go-ahead at that moment) | A, B | 0.5 d |
| 7 | C1 ERA5 2026 download (background), C2 spike | — | 1–2 d |
| 8 | D2, D3 heavy-rain plan | 2–4 weeks of data | later |

The GitHub push and the LinkedIn post are the two outward-facing steps; each gets an explicit confirmation from you when we reach it.

## 7. Decisions (answered 2026-09-27)

- D1 **MIT** for code, **CC-BY-4.0** for docs/figures.
- D2 **Code + results + small derived figures** with NEA/PUB attribution; no radar archive, no raw messages.
- D3 repo name: default `beese54/singapore-diffusion-nowcast` unless changed at push time.
- D4 **Static site on GitHub Pages.**
- D5 CorrDiff: **local spike first**, then a go/no-go with a cost estimate before any spend.

### Original options

- **D1 Licence:** MIT (simplest, permissive) vs Apache-2.0 (permissive + patent grant). Recommend **MIT** for code; docs CC-BY-4.0.
- **D2 Data redistribution:** NEA radar images are © NEA (weather.gov.sg terms); Telegram messages are PUB's. Recommend: **publish code, derived results and small figures/animations with attribution; do not publish the radar archive or raw messages** — researchers rebuild with our scripts (and NEA's 7-day retention means they would need their own collection; say so). The dashboard's radar images are derived figures of a few hours — include with attribution.
- **D3 Repo name / visibility:** e.g. `beese54/singapore-diffusion-nowcast`, public.
- **D4 Dashboard hosting:** static on GitHub Pages (recommended) vs local only.
- **D5 CorrDiff compute budget** if the spike says go (e.g. up to ~S$50 cloud GPU), or local-only.
