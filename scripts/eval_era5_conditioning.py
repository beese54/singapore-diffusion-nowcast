#!/usr/bin/env python3
"""
eval_era5_conditioning.py -- Paired test of the ERA5 weather input
(tasks/plan_era5_conditioning.md).

Three 30-min models, scored on the SAME test forecasts with the SAME random seeds:
    base  checkpoints/nowcaster/ckpt_step_300000.pt        (model of record)
    ctrl  checkpoints/nowcaster/lead30_ctrl/ckpt_step_50000.pt  (+50k steps, no weather)
    env   checkpoints/nowcaster/lead30_env/ckpt_step_50000.pt   (+50k steps, with weather)
env vs ctrl isolates the weather input (same extra training); env vs base is the
practical question. Test anchors are limited to hours that have ERA5 (it lags ~6 days).

Scores per model: CRPS skill vs persistence; pooled ensemble FSS at 2 and 10 mm/hr
(11.9 km); heavy-rain warning skill in 2 x 2 km boxes (>= 2 of 8 futures; catch rate,
precision). Differences: paired bootstrap over forecasts (2,000 resamples, 95%).

Verdict (fixed in the plan): KEEP if env beats ctrl on heavy-rain skill (FSS >=10 or
catch rate, CI > 0) and CRPS skill is not worse (CI of the difference not below 0).

Usage:  python scripts/eval_era5_conditioning.py [--n 200]
Output: results/era5_conditioning.json (ensembles cached in data/processed/eval_cache/)
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
from evaluate import crps_pixelwise, fss_parts, load_model  # noqa: E402
from src.data.radar_dataset import RadarDataset  # noqa: E402

MODELS = {
    "base": ROOT / "checkpoints/nowcaster/ckpt_step_300000.pt",
    "ctrl": ROOT / "checkpoints/nowcaster/lead30_ctrl/ckpt_step_50000.pt",
    "env": ROOT / "checkpoints/nowcaster/lead30_env/ckpt_step_50000.pt",
}
CACHE = ROOT / "data" / "processed" / "eval_cache"
WIN, B, HEAVY, WARN, REPS = 41, 7, 10.0, 0.25, 2000


def box_max(a):
    a = a[..., :119, :217 - 217 % B]
    *lead, h, w = a.shape
    return a.reshape(*lead, h // B, B, w // B, B).max(axis=(-3, -1))


def per_forecast(ens, obs, per):
    """Per-forecast pieces, so every score can be bootstrapped over forecasts."""
    n = len(obs)
    rel = (obs >= 0.5) | (per >= 0.5) | (ens.max(1) >= 0.5)
    crps_m = np.array([crps_pixelwise(ens[k], obs[k])[rel[k]].sum() for k in range(n)])
    crps_p = np.array([np.abs(per[k] - obs[k])[rel[k]].sum() for k in range(n)])
    fss = {}
    for thr in (2.0, 10.0):
        parts = np.array([fss_parts((ens[k] >= thr).mean(0), obs[k] >= thr, WIN) for k in range(n)])
        fss[thr] = parts                                                   # (n, 2): num, den
    O = box_max(obs) >= HEAVY
    W = (box_max(ens) >= HEAVY).mean(1) >= WARN
    warn = np.stack([(W & O).sum((1, 2)), (W & ~O).sum((1, 2)), (~W & O).sum((1, 2))], 1)   # hits, fa, misses
    return {"crps_m": crps_m, "crps_p": crps_p, "fss": fss, "warn": warn}


def stats(P, i):
    s = {"crps_skill": 1 - P["crps_m"][i].sum() / max(P["crps_p"][i].sum(), 1e-12)}
    for thr, parts in P["fss"].items():
        s[f"fss{thr:g}"] = 1 - parts[i, 0].sum() / max(parts[i, 1].sum(), 1e-12)
    h, f, m = P["warn"][i].sum(0)
    s["catch_rate"] = h / max(h + m, 1)
    s["precision"] = h / max(h + f, 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--env-ckpt", help="override the env checkpoint (e.g. for a smoke test)")
    ap.add_argument("--ctrl-ckpt", help="override the control checkpoint")
    ap.add_argument("--out", default="results/era5_conditioning.json")
    args = ap.parse_args()
    if args.env_ckpt:
        MODELS["env"] = ROOT / args.env_ckpt
    if args.ctrl_ckpt:
        MODELS["ctrl"] = ROOT / args.ctrl_ckpt
    tag = "" if not (args.env_ckpt or args.ctrl_ckpt) else "_override"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = RadarDataset("test", target_offset=6, era5_env=True)        # anchors with ERA5 only
    anchors = [ds.indices[i] for i in np.linspace(0, len(ds) - 1, args.n).astype(int)]
    a_times = ds.times[np.array(anchors)].astype("datetime64[m]")
    dec = lambda i: ds._decode(ds.codes[i])  # noqa: E731
    obs = np.stack([dec(t + 6) for t in anchors]).astype(np.float32)
    per = np.stack([dec(t - 1) for t in anchors]).astype(np.float32)
    print(f"{len(anchors)} paired test forecasts, {str(a_times[0])} .. {str(a_times[-1])}")

    P = {}
    for name, ckpt in MODELS.items():
        cache = CACHE / f"era5cond_{name}{tag}_n{args.n}_m{args.members}.npz"
        if cache.exists() and np.array_equal(np.load(cache)["anchor_times"], a_times):
            ens = np.load(cache)["ens"].astype(np.float32)
            print(f"  {name}: cached")
        else:
            model = load_model(ckpt, dev)
            if model.target_offset != 6:
                sys.exit(f"{ckpt} is not a 30-min model")
            use_env = model.data_cfg["era5_env"]
            ens = np.empty((len(anchors), args.members, *obs.shape[1:]), np.float16)
            for k, t in enumerate(anchors):
                torch.manual_seed(1000 + k)                       # same noise for every model
                env = ds.env_at(t).unsqueeze(0).to(dev) if use_env else None
                with torch.no_grad():
                    e = model.ensemble_sample(ds.context_at(t).unsqueeze(0).to(dev),
                                              n_members=args.members, eta=1.0, env=env)
                ens[k] = ds.denormalise(e.cpu().float()[0, :, 0]).numpy()
                if k % 50 == 0:
                    print(f"  {name}: {k}/{len(anchors)}", flush=True)
            np.savez_compressed(cache, ens=ens, anchor_times=a_times)
            ens = ens.astype(np.float32)
            del model
            torch.cuda.empty_cache()
        P[name] = per_forecast(ens, obs, per)

    n = len(anchors)
    idx = np.random.default_rng(0).integers(0, n, (REPS, n))
    all_i = np.arange(n)
    report = {"test_forecasts": n, "test_period": [str(a_times[0]), str(a_times[-1])],
              "models": {k: str(v.relative_to(ROOT)) for k, v in MODELS.items()},
              "scores": {k: stats(v, all_i) for k, v in P.items()}, "differences": {}}
    for a, b in (("env", "ctrl"), ("env", "base"), ("ctrl", "base")):
        d = {}
        for key in report["scores"]["base"]:
            boot = [stats(P[a], i)[key] - stats(P[b], i)[key] for i in idx]
            d[key] = {"diff": report["scores"][a][key] - report["scores"][b][key],
                      "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        report["differences"][f"{a}_minus_{b}"] = d
    ec = report["differences"]["env_minus_ctrl"]
    heavy_better = ec["fss10"]["ci"][0] > 0 or ec["catch_rate"]["ci"][0] > 0
    crps_not_worse = ec["crps_skill"]["ci"][1] >= 0          # CI of the difference not entirely below 0
    report["verdict"] = "KEEP" if (heavy_better and crps_not_worse) else "DO NOT KEEP"
    report["verdict_rule"] = ("env vs ctrl: heavy-rain skill better (FSS>=10 or catch rate, CI>0) "
                              "and CRPS skill not worse (CI of difference not entirely below 0)")
    report["caveat"] = "ERA5 is a reanalysis with a ~6-day lag: an upper bound on real-time value"
    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for k, s in report["scores"].items():
        print(f"{k:>4}: " + ", ".join(f"{m} {v:.3f}" for m, v in s.items()))
    for k, d in report["differences"].items():
        print(f"{k}: " + ", ".join(f"{m} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for m, v in d.items()))
    print("VERDICT:", report["verdict"], "->", out)


if __name__ == "__main__":
    main()
