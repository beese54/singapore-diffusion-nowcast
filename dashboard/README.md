# Dashboard

A static site (plain HTML/CSS/JS, no build step, no external libraries). Published to GitHub
Pages by `.github/workflows/pages.yml`.

- **Run locally:** `cd dashboard && python -m http.server 8765` then open http://127.0.0.1:8765
  (it must be served over HTTP; `file://` blocks reading the sprite pixels).
- **Rebuild the data:** `python scripts/build_dashboard.py` (reads the evaluation caches, radar
  archive and `results/`; the denoising demo uses the GPU if available). Output: `dashboard/data/`.

## Design notes (Pattern BB: frame runtime)

**Raster budget.** Every radar panel is a `<canvas>` whose backing store is exactly the radar's
native **217 × 120 px**; a frame is one `putImageData` with no resampling, and CSS scales it on the
GPU (so there is deliberately no resize handler). Rain fields ship as **greyscale PNG sprite sheets**
(1 byte/pixel: `round(255·log1p(mm/hr)/log1p(100))` for rain, `round(255·p)` for probability) and are
coloured in the browser through a 256-entry lookup table. Total data: **~4.7 MB** for 2 cases ×
3 leads × 3 fields, 126 observed frames and the 50-step denoising sequence (the noisy x_t sheet is
the largest single file, 1.1 MB, because noise does not compress).

**Idle cost.** Players run `requestAnimationFrame` only while playing; pause, scrolling a player out
of view (IntersectionObserver) or hiding the tab cancels the loop. Panels keep a dirty key and skip
redrawing an unchanged frame.

**Loading.** Sprite sheets resolve on `onload`; `img.decode()` is never awaited (it can stall in a
background tab).

**Reduced motion.** Under `prefers-reduced-motion: reduce` nothing autoplays: the hero shows one
static frame (16:50 SGT, peak of the 22 Sep storm) and its play button is hidden; the denoising panel
opens on the final step; sliders remain user-driven.

## Verification (2026-09-27)

| Check | How | Result |
|---|---|---|
| Renders, no console errors (desktop, dark and light themes) | Chrome, hard reload | pass |
| Every radar canvas is 217 × 120 | `querySelectorAll('canvas')` in the page | pass (14 canvases) |
| No horizontal scroll at phone width | page in a 390 px iframe: scrollWidth = clientWidth | pass |
| Controls: lead and case switch, denoise slider to step 50, pause stops the hero | scripted clicks in Chrome | pass |
| Timing / idle cost | not measured: the automation tab reported `visibilityState: hidden`, which suspends rAF, so a browser timing reading there would be meaningless | code-level guarantee only (`stop()` cancels rAF) |
| Numbers match `results/` | exporter reads the JSON directly; spot-checked against docs/RESULTS.md | pass |

## Attribution

Radar imagery © National Environment Agency (NEA) / Meteorological Service Singapore
(weather.gov.sg), shown for personal, non-commercial, informational purposes only
(see the page footer and `docs/DATA.md`).
