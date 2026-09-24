"""
Verification for the normalisation fix and the collapse guard (2026-09-24).

Background. `_normalise` used to z-score log1p(rain) then clamp to [-3, 3]. The
field is 97.5% zeros so log_std was 0.215, putting the clip at 0.96 mm/hr:
2 mm/hr is 5 sigma, 20 is 14, 100 is 21.4. Thirty of thirty-five rain levels
mapped to exactly +3.0 and the ramp collapsed to 6 distinct values. A 300k-step
run on that target collapsed to emitting the constant dry value (max 0.006
mm/hr) and scored FSS 0.000 vs persistence 0.070 -- while val_loss plateaued at
a healthy-looking 0.0084, because diffusion noise-prediction MSE rewards a model
that predicts the mean. train.py then wrote stage4_complete.flag.

So there are two things to prove:
  1. the transform preserves every rain level and round-trips
  2. the guard would actually have caught the collapse

Note on (2): collapse is REPRESENTATION-RELATIVE. The dead checkpoints output
~-0.130, which was dry under the old z-score but decodes to ~7 mm/hr under the
new scale -- so they cannot be used to test the guard, and are in fact
misleading if loaded. The guard is therefore tested against synthetic forecasts
expressed in the CURRENT normalised space.

Expectation: all PASS.
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.radar_dataset import RadarDataset  # noqa: E402

RAIN_MIN_MMHR = 0.1  # must match train.py

results = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<54} {detail}")
    results.append(bool(ok))


def main() -> None:
    ds = RadarDataset("val")
    print()

    # ── 1. every archive rain level survives the transform ───────────────────
    levels = np.asarray(ds.values, dtype=np.float32)
    norm = ds._normalise(levels)
    distinct = len(set(np.round(norm, 6)))
    check("all rain levels remain distinct", distinct == len(levels),
          f"{len(levels)} levels -> {distinct} distinct")
    check("nothing saturates except the true maximum",
          int((norm >= 0.999).sum()) == 1, f"{int((norm >= 0.999).sum())} at +1.0 (was 30)")
    check("range is exactly [-1, 1]",
          abs(norm.min() + 1) < 1e-6 and abs(norm.max() - 1) < 1e-6,
          f"[{norm.min():.4f}, {norm.max():.4f}]")

    back = ds.denormalise(torch.from_numpy(norm)).numpy()
    err = float(np.abs(back - levels).max())
    check("round-trips to mm/hr", err < 1e-3, f"max error {err:.2e} mm/hr")

    # the thresholds the DoD actually scores at must be representable
    for thr in (0.5, 2.0, 20.0):
        n = float(ds._normalise(np.array([thr], dtype=np.float32))[0])
        r = float(ds.denormalise(torch.tensor([n])))
        check(f"{thr} mm/hr is representable", abs(r - thr) / thr < 0.01,
              f"-> {n:+.4f} -> {r:.3f} mm/hr")

    # ── 2. the guard's predicate ─────────────────────────────────────────────
    # A collapsed forecast is the constant DRY value in the current space.
    dry = float(ds._normalise(np.array([0.0], dtype=np.float32))[0])
    collapsed = torch.full((1, 2, 1, 120, 217), dry)
    mx = float(ds.denormalise(collapsed).max())
    check("guard flags an all-dry forecast", mx < RAIN_MIN_MMHR,
          f"max {mx:.5f} mm/hr < {RAIN_MIN_MMHR}")

    # A forecast carrying real rain must NOT be flagged, or the guard is useless.
    real = torch.from_numpy(ds[0]["target"].numpy()).unsqueeze(0).unsqueeze(0)
    mx_real = float(ds.denormalise(real).max())
    check("guard passes a forecast with real rain", mx_real >= RAIN_MIN_MMHR,
          f"max {mx_real:.3f} mm/hr")

    # Near-dry but not exactly dry: a model nudging just above the floor is
    # still collapsed for our purposes (lowest real NEA level is 0.5 mm/hr).
    nearly = torch.full((1, 1, 1, 120, 217), dry + 0.01)
    mx_n = float(ds.denormalise(nearly).max())
    check("guard flags a near-dry forecast", mx_n < RAIN_MIN_MMHR,
          f"max {mx_n:.5f} mm/hr")

    # ── 3. the old transform would have failed these ─────────────────────────
    import json
    s = json.load(open(ROOT / "data" / "processed" / "radar_stats.json"))
    lm, ls = s["log_mean"], max(s["log_std"], 1e-6)
    old = np.clip((np.log1p(levels) - lm) / ls, -3.0, 3.0)
    old_distinct = len(set(np.round(old, 6)))
    check("old transform did collapse the ramp (regression witness)",
          old_distinct < len(levels) / 3,
          f"{len(levels)} levels -> {old_distinct} distinct, "
          f"{int((old >= 2.999).sum())} pinned at +3.0")
    old_ceiling = float(np.expm1(np.clip(3.0 * ls + lm, 0, None)))
    check("old ceiling was below the 2 mm/hr scoring threshold",
          old_ceiling < 2.0, f"ceiling {old_ceiling:.3f} mm/hr")

    print(f"\n{sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
