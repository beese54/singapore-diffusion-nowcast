#!/usr/bin/env python3
"""
warning_calibration.py -- Calibrate the heavy-rain warning (tasks/plan_hybrid_calibration.md, item 2).

The model under-forecasts intensity (its heavy-rain area is about half the observed
area), so its futures rarely reach 10 mm/hr even where heavy rain then falls. Instead
of retraining, the warning rule is re-chosen:

    a 2 x 2 km box is warned when >= k of the 8 futures show >= X mm/hr in it
    candidates   X in {3, 4, 5, 6, 7, 8, 10} mm/hr  x  k in {1, 2, 3, 4}
    target       heavy rain OBSERVED in the box: >= 10 mm/hr (unchanged)
    chosen by    max CSI = hits / (hits + false alarms + misses), on VALIDATION only
                 (2-14 Sep; 200 forecasts per lead, seeds 3000+k); ties -> the rule
                 closest to the current one (X = 10, k = 2)
    scored on    the cached TEST ensembles of scripts/evaluate_probabilistic.py
                 (the same forecasts behind results/warning_skill.json)
    compared     calibrated vs uncalibrated (X = 10, k = 2): catch rate, precision, CSI,
                 incoming catch (heavy at the target, not at the last frame the model saw);
                 paired bootstrap over forecasts, 2,000 resamples, 95%
    adopt        at a lead if the test CSI difference CI is entirely above 0

Validation ensembles are cached per lead and checkpointed every 25 forecasts, so a
run interrupted by a shutdown resumes where it stopped.

Usage:  python scripts/warning_calibration.py [--leads 30 60 90]
Output: results/warning_calibration.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import load_model  # noqa: E402
from src.data.radar_dataset import RadarDataset  # noqa: E402

CACHE = ROOT / "data" / "processed" / "eval_cache"
MODELS = {  # lead: (checkpoint, test cache) -- the models of record, as in warning_skill.py
    "30": ("checkpoints/nowcaster/ckpt_step_300000.pt", "ens_nowcaster_ckpt_step_300000_s300000_n200_m8.npz"),
    "60": ("checkpoints/nowcaster/lead60_warm/ckpt_step_100000.pt", "ens_lead60_warm_ckpt_step_100000_s100000_n200_m8.npz"),
    "90": ("checkpoints/nowcaster/lead90_warm/ckpt_step_100000.pt", "ens_lead90_warm_ckpt_step_100000_s100000_n200_m8.npz"),
}
B, HEAVY, REPS, N, MEMBERS, VAL_SEED = 7, 10.0, 2000, 200, 8, 3000
XS, KS = (3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0), (1, 2, 3, 4)
UNCAL = (10.0, 2)


def box_max(a):
    """Max over 7 x 7 px boxes (17 x 31 boxes), over the last two axes."""
    a = a[..., :119, :217 - 217 % B]
    *lead, h, w = a.shape
    return a.reshape(*lead, h // B, B, w // B, B).max(axis=(-3, -1))


def model_warns(bm_ens, x, k):
    """bm_ens: (n, members, 17, 31) box maxima -> (n, 17, 31) warned boxes."""
    return (bm_ens >= x).sum(1) >= k


def counts(W, O, P):
    """Per-forecast (n, 5): hits, false alarms, misses, incoming hits, incoming boxes.
    O = heavy rain observed at the target, P = heavy rain in the last frame seen."""
    inc = O & ~P
    ax = (-2, -1)
    return np.stack([(W & O).sum(ax), (W & ~O).sum(ax), (~W & O).sum(ax),
                     (W & inc).sum(ax), inc.sum(ax)], 1)


def scores(c):
    h, f, m, hi, ni = c.sum(0)
    return {"catch_rate": h / max(h + m, 1), "precision": h / max(h + f, 1),
            "csi": h / max(h + f + m, 1), "incoming_catch": hi / max(ni, 1),
            "hits": int(h), "false_alarms": int(f), "misses": int(m), "warnings": int(h + f)}


METRICS = ("catch_rate", "precision", "csi", "incoming_catch")


def paired(ca, cb, idx):
    """Difference a - b for each metric with a paired bootstrap 95% CI."""
    sa, sb = scores(ca), scores(cb)
    boot = np.array([[scores(ca[i])[m] - scores(cb[i])[m] for m in METRICS] for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {m: {"diff": sa[m] - sb[m], "ci": [float(lo[j]), float(hi[j])]} for j, m in enumerate(METRICS)}


def val_ensembles(lead, ckpt, dev):
    """200 validation forecasts x 8 members, cached; resumes from a partial cache."""
    ck = ROOT / ckpt
    model = load_model(ck, dev)
    off = model.target_offset
    if off != {"30": 6, "60": 12, "90": 18}[lead]:
        sys.exit(f"{ck}: target offset {off} does not match lead {lead}")
    ds = RadarDataset("val", target_offset=off, **model.data_cfg)
    anchors = [ds.indices[i] for i in np.linspace(0, len(ds) - 1, N).astype(int)]
    a_times = ds.times[np.array(anchors)].astype("datetime64[m]")
    dec = lambda i: ds._decode(ds.codes[i])  # noqa: E731
    obs = np.stack([dec(t + off) for t in anchors]).astype(np.float32)
    per = np.stack([dec(t - 1) for t in anchors]).astype(np.float32)

    cache = CACHE / f"calib_val_lead{lead}_n{N}_m{MEMBERS}.npz"
    part = cache.with_suffix(".partial.npz")
    if cache.exists():
        z = np.load(cache)
        if not np.array_equal(z["anchor_times"], a_times):
            sys.exit(f"{cache.name} covers different validation anchors; delete it and regenerate")
        print(f"  lead {lead}: validation ensembles cached")
        del model
        return z["ens"].astype(np.float32), obs, per, a_times
    ens = np.zeros((N, MEMBERS, *obs.shape[1:]), np.float16)
    done = 0
    if part.exists():
        z = np.load(part)
        if np.array_equal(z["anchor_times"], a_times):
            done = int(z["done"])
            ens[:done] = z["ens"][:done]
            print(f"  lead {lead}: resuming at {done}/{N}")
    for k in range(done, N):
        torch.manual_seed(VAL_SEED + k)
        with torch.no_grad():
            e = model.ensemble_sample(ds.context_at(anchors[k]).unsqueeze(0).to(dev),
                                      n_members=MEMBERS, eta=1.0)
        ens[k] = ds.denormalise(e.cpu().float()[0, :, 0]).numpy()
        if (k + 1) % 25 == 0 and k + 1 < N:
            np.savez_compressed(part, ens=ens, anchor_times=a_times, done=k + 1)
            print(f"  lead {lead}: {k + 1}/{N}", flush=True)
    np.savez_compressed(cache, ens=ens, anchor_times=a_times)
    part.unlink(missing_ok=True)
    del model
    torch.cuda.empty_cache()
    return ens.astype(np.float32), obs, per, a_times


def test_data(lead, test_cache):
    """The cached test ensembles and their observations (anchors from the cache's own times)."""
    z = np.load(CACHE / test_cache)
    off = {"30": 6, "60": 12, "90": 18}[lead]
    ds = RadarDataset("test", target_offset=off)
    a_times = z["anchor_times"].astype("datetime64[m]")
    anchors = np.searchsorted(ds.times.astype("datetime64[m]"), a_times)
    if not np.array_equal(ds.times[anchors].astype("datetime64[m]"), a_times):
        sys.exit(f"{test_cache}: anchor times missing from the archive")
    if not set(anchors.tolist()) <= set(ds.indices):
        sys.exit(f"{test_cache}: anchors outside the pinned test split")
    dec = lambda i: ds._decode(ds.codes[i])  # noqa: E731
    obs = np.stack([dec(t + off) for t in anchors]).astype(np.float32)
    per = np.stack([dec(t - 1) for t in anchors]).astype(np.float32)
    return z["ens"].astype(np.float32), obs, per, a_times


def main():
    global N
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", nargs="+", default=list(MODELS))
    ap.add_argument("--n", type=int, default=N, help="validation forecasts per lead (smoke: e.g. 4)")
    ap.add_argument("--out", default="results/warning_calibration.json")
    args = ap.parse_args()
    N = args.n
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    out = {"definition": {
        "box_km": 2.03, "boxes": 527, "heavy_mm_hr": HEAVY,
        "rule": "warn a box when >= k of 8 futures show >= X mm/hr in it",
        "candidates": {"X_mm_hr": list(XS), "k": list(KS)}, "uncalibrated": {"X": UNCAL[0], "k": UNCAL[1]},
        "chosen_by": "max CSI on validation (2-14 Sep); ties -> closest to X=10, k=2",
        "adopt_if": "test CSI (calibrated - uncalibrated) 95% paired-bootstrap CI entirely above 0",
        "validation_seeds": f"{VAL_SEED}+k"}, "by_lead": {}}

    for lead in args.leads:
        ckpt, test_cache = MODELS[lead]
        ens_v, obs_v, per_v, t_v = val_ensembles(lead, ckpt, dev)
        bm_v, O_v, P_v = box_max(ens_v), box_max(obs_v) >= HEAVY, box_max(per_v) >= HEAVY
        grid = []
        for x in XS:
            for k in KS:
                s = scores(counts(model_warns(bm_v, x, k), O_v, P_v))
                grid.append({"X": x, "k": k} | s)
        best = max(grid, key=lambda g: (round(g["csi"], 6), -abs(g["X"] - UNCAL[0]) - abs(g["k"] - UNCAL[1])))
        X, K = best["X"], best["k"]

        ens_t, obs_t, per_t, t_t = test_data(lead, test_cache)
        bm_t, O_t, P_t = box_max(ens_t), box_max(obs_t) >= HEAVY, box_max(per_t) >= HEAVY
        c_cal = counts(model_warns(bm_t, X, K), O_t, P_t)
        c_unc = counts(model_warns(bm_t, *UNCAL), O_t, P_t)
        c_nai = counts(P_t, O_t, P_t)
        idx = rng.integers(0, len(c_cal), (REPS, len(c_cal)))
        d = paired(c_cal, c_unc, idx)
        adopt = bool(d["csi"]["ci"][0] > 0)
        out["by_lead"][lead] = {
            "checkpoint": ckpt, "test_cache": test_cache,
            "validation": {"forecasts": len(t_v), "period": [str(t_v[0]), str(t_v[-1])],
                           "heavy_rain_boxes": int(O_v.sum()), "grid": grid},
            "chosen": {"X": X, "k": K, "validation_csi": best["csi"]},
            "test": {"forecasts": len(t_t), "period": [str(t_t[0]), str(t_t[-1])],
                     "heavy_rain_boxes": int(O_t.sum()),
                     "calibrated": scores(c_cal), "uncalibrated": scores(c_unc), "naive": scores(c_nai)},
            "calibrated_minus_uncalibrated": d,
            "verdict": "ADOPT" if adopt else "DO NOT ADOPT",
        }
        u, c = scores(c_unc), scores(c_cal)
        print(f"lead {lead}: chosen X={X:g} k={K} (val CSI {best['csi']:.3f}) | test CSI "
              f"{u['csi']:.3f} -> {c['csi']:.3f} {np.round(d['csi']['ci'], 3)}, catch {u['catch_rate']:.1%} -> "
              f"{c['catch_rate']:.1%}, precision {u['precision']:.1%} -> {c['precision']:.1%} | "
              f"{out['by_lead'][lead]['verdict']}", flush=True)

    path = ROOT / args.out
    if path.exists() and set(args.leads) != set(MODELS):            # partial run: keep other leads
        old = json.loads(path.read_text(encoding="utf-8"))
        out["by_lead"] = old.get("by_lead", {}) | out["by_lead"]
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("->", path)


if __name__ == "__main__":
    main()
