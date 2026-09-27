#!/usr/bin/env python3
"""
warning_skill.py -- "When it warns of heavy rain in my area, how often is it right,
and how many real downpours does it catch?" -- for each lead, on the test period.

Plain-language companion to the CRPS/FSS scores. The domain is cut into 2 x 2 km
boxes (7 x 7 radar pixels; 17 x 31 = 527 boxes). For each of the 200 cached test
forecasts and each box:
    heavy rain happened  = observed >= 10 mm/hr somewhere in the box at the target time
    model warns          = at least 2 of the 8 ensemble futures show >= 10 mm/hr there
    naive forecast warns = the last radar map the model saw shows >= 10 mm/hr there
Reported: catch rate (share of heavy-rain boxes warned), precision (share of
warnings that came true), how that compares with chance (the base rate), and the
warnings the naive forecast cannot give (box not raining heavily at issue time).
95% ranges: bootstrap over the 200 forecast times (2,000 resamples).

Inputs: the ensemble caches written by scripts/evaluate_probabilistic.py
Output: results/warning_skill.json
Usage:  python scripts/warning_skill.py
"""

import json
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "processed" / "eval_cache"
MODELS = {  # lead: (cache file, target offset in 5-min steps, checkpoint)
    "30": ("ens_nowcaster_ckpt_step_300000_s300000_n200_m8.npz", 6, "checkpoints/nowcaster/ckpt_step_300000.pt"),
    "60": ("ens_lead60_warm_ckpt_step_100000_s100000_n200_m8.npz", 12, "checkpoints/nowcaster/lead60_warm/ckpt_step_100000.pt"),
    "90": ("ens_lead90_warm_ckpt_step_100000_s100000_n200_m8.npz", 18, "checkpoints/nowcaster/lead90_warm/ckpt_step_100000.pt"),
}
B, HEAVY, WARN = 7, 10.0, 0.25          # box size (px), mm/hr, >= 2 of 8 futures
REPS = 2000


def box_max(a: np.ndarray) -> np.ndarray:
    a = a[..., :119, :217 - 217 % B]
    *lead, h, w = a.shape
    return a.reshape(*lead, h // B, B, w // B, B).max(axis=(-3, -1))


def counts(W, O):
    """Per-forecast counts (n_forecasts, 3): hits, false alarms, misses."""
    ax = (-2, -1)
    return np.stack([(W & O).sum(ax), (W & ~O).sum(ax), (~W & O).sum(ax)], axis=1)


def rates(c):
    h, f, m = c.sum(0)
    return {"catch_rate": h / max(h + m, 1), "precision": h / max(h + f, 1),
            "hits": int(h), "false_alarms": int(f), "misses": int(m), "warnings": int(h + f)}


def ci(c, key, rng):
    idx = rng.integers(0, len(c), (REPS, len(c)))
    vals = [rates(c[i])[key] for i in idx]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    da = xr.open_zarr(ROOT / "data" / "processed" / "radar.zarr", consolidated=True)["rain_rate"]
    T = da.time.values.astype("datetime64[m]")
    out = {"definition": {"box_km": 2.03, "boxes": 527, "heavy_mm_hr": HEAVY,
                          "model_warns": "at least 2 of 8 futures show heavy rain in the box",
                          "naive_warns": "the last radar map shows heavy rain in the box",
                          "ci": "95% bootstrap over forecast times"}, "by_lead": {}}
    rng = np.random.default_rng(0)
    for lead, (f, off, ckpt) in MODELS.items():
        z = np.load(CACHE / f)
        ens = z["ens"].astype(np.float32)
        a = np.searchsorted(T, z["anchor_times"].astype("datetime64[m]"))
        obs = np.stack([da.isel(time=int(i + off)).values for i in a])
        per = np.stack([da.isel(time=int(i - 1)).values for i in a])
        O = box_max(obs) >= HEAVY
        naive = box_max(per) >= HEAVY
        model = (box_max(ens) >= HEAVY).mean(1) >= WARN
        cm, cn = counts(model, O), counts(naive, O)
        cnew = counts(model & ~naive, O)                     # warnings naive cannot give
        base = float(O.mean())
        m, n, new = rates(cm), rates(cn), rates(cnew)
        out["by_lead"][lead] = {
            "checkpoint": ckpt, "forecasts": int(len(a)), "test_period": [str(z["anchor_times"][0]), str(z["anchor_times"][-1])],
            "heavy_rain_boxes": int(O.sum()), "base_rate": base,
            "model": m | {"catch_rate_ci": ci(cm, "catch_rate", rng), "precision_ci": ci(cm, "precision", rng),
                          "times_chance": m["precision"] / base},
            "naive": n | {"catch_rate_ci": ci(cn, "catch_rate", rng), "precision_ci": ci(cn, "precision", rng)},
            "model_only_warnings": {"warnings": new["warnings"], "came_true": new["hits"],
                                    "precision": new["precision"],
                                    "share_of_all_heavy_rain": new["hits"] / max(int(O.sum()), 1)},
        }
        r = out["by_lead"][lead]
        print(f"lead {lead}: model catches {m['catch_rate']:.0%} {np.round(r['model']['catch_rate_ci'], 2)}, "
              f"right {m['precision']:.0%} {np.round(r['model']['precision_ci'], 2)} "
              f"({m['precision'] / base:.0f}x chance) | naive catches {n['catch_rate']:.0%}, right {n['precision']:.0%} | "
              f"model-only warnings {new['warnings']}, {new['hits']} true")
    (ROOT / "results" / "warning_skill.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("-> results/warning_skill.json")


if __name__ == "__main__":
    main()
