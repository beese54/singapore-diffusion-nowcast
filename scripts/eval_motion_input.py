#!/usr/bin/env python3
"""
eval_motion_input.py -- Paired test of the motion-forecast input (tasks/plan_motion_input.md).

Four 60-min forecasters on the SAME test forecasts; the three models use the SAME seeds:
    base    checkpoints/nowcaster/lead60_warm/ckpt_step_100000.pt   (model of record)
    ctrl    checkpoints/nowcaster/lead60_mctrl/ckpt_step_50000.pt   (+50k steps, no motion)
    motion  checkpoints/nowcaster/lead60_motion/ckpt_step_50000.pt  (+50k steps, motion input)
    extrap  the motion channel itself used as a forecast (deterministic)
motion vs ctrl isolates the motion input (same extra training); motion vs extrap asks
whether the model adds anything over plain extrapolation.

Scores: CRPS skill vs persistence (MAE for extrap); pooled FSS at 2 and 10 mm/hr (11.9 km);
heavy-rain warnings in 2 x 2 km boxes (models: >= 2 of 8 futures; extrap: >= 10 mm/hr):
catch rate, precision, and incoming catch (boxes heavy at the target but not at the last
frame). Differences: paired bootstrap over forecasts (2,000 resamples, 95%).

Verdict (fixed in the plan):
  KEEP      motion - ctrl: FSS >=10 or catch rate CI > 0, and CRPS skill CI not entirely < 0
  PRACTICAL (only if KEEP) motion - extrap: catch rate or precision CI > 0; otherwise the
            honest recommendation at 60 min is plain extrapolation

Usage:  python scripts/eval_motion_input.py [--n 200] [--motion-ckpt ... --ctrl-ckpt ...]
Output: results/motion_input.json (ensembles cached in data/processed/eval_cache/)
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
from src.data.motion import motion_forecast  # noqa: E402
from src.data.radar_dataset import RadarDataset  # noqa: E402

MODELS = {
    "base": ROOT / "checkpoints/nowcaster/lead60_warm/ckpt_step_100000.pt",
    "ctrl": ROOT / "checkpoints/nowcaster/lead60_mctrl/ckpt_step_50000.pt",
    "motion": ROOT / "checkpoints/nowcaster/lead60_motion/ckpt_step_50000.pt",
}
CACHE = ROOT / "data" / "processed" / "eval_cache"
OFF, MINUTES = 12, 65
WIN, B, HEAVY, WARN, REPS = 41, 7, 10.0, 0.25, 2000


def box_max(a):
    a = a[..., :119, :217 - 217 % B]
    *lead, h, w = a.shape
    return a.reshape(*lead, h // B, B, w // B, B).max(axis=(-3, -1))


def per_forecast(ens, obs, per):
    """Per-forecast pieces, so every score can be bootstrapped. ens: (n, m, H, W)."""
    n = len(obs)
    rel = (obs >= 0.5) | (per >= 0.5) | (ens.max(1) >= 0.5)
    crps_m = np.array([crps_pixelwise(ens[k], obs[k])[rel[k]].sum() for k in range(n)])
    crps_p = np.array([np.abs(per[k] - obs[k])[rel[k]].sum() for k in range(n)])
    fss = {thr: np.array([fss_parts((ens[k] >= thr).mean(0), obs[k] >= thr, WIN) for k in range(n)])
           for thr in (2.0, 10.0)}
    O, P = box_max(obs) >= HEAVY, box_max(per) >= HEAVY
    W = (box_max(ens) >= HEAVY).mean(1) >= WARN
    inc = O & ~P
    warn = np.stack([(W & O).sum((1, 2)), (W & ~O).sum((1, 2)), (~W & O).sum((1, 2)),
                     (W & inc).sum((1, 2)), inc.sum((1, 2))], 1)
    return {"crps_m": crps_m, "crps_p": crps_p, "fss": fss, "warn": warn}


def stats(P, i):
    s = {"crps_skill": 1 - P["crps_m"][i].sum() / max(P["crps_p"][i].sum(), 1e-12)}
    for thr, parts in P["fss"].items():
        s[f"fss{thr:g}"] = 1 - parts[i, 0].sum() / max(parts[i, 1].sum(), 1e-12)
    h, f, m, hi, ni = P["warn"][i].sum(0)
    s["catch_rate"] = h / max(h + m, 1)
    s["precision"] = h / max(h + f, 1)
    s["incoming_catch"] = hi / max(ni, 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--motion-ckpt", help="override the motion checkpoint (e.g. a smoke test)")
    ap.add_argument("--ctrl-ckpt", help="override the control checkpoint")
    ap.add_argument("--out", default="results/motion_input.json")
    args = ap.parse_args()
    if args.motion_ckpt:
        MODELS["motion"] = ROOT / args.motion_ckpt
    if args.ctrl_ckpt:
        MODELS["ctrl"] = ROOT / args.ctrl_ckpt
    tag = "" if not (args.motion_ckpt or args.ctrl_ckpt) else "_override"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = RadarDataset("test", target_offset=OFF)
    ds_m = RadarDataset("test", target_offset=OFF, motion_minutes=MINUTES)
    assert ds.indices == ds_m.indices, "motion option must not change the test anchors"
    anchors = [ds.indices[i] for i in np.linspace(0, len(ds) - 1, args.n).astype(int)]
    a_times = ds.times[np.array(anchors)].astype("datetime64[m]")
    dec = lambda i: ds._decode(ds.codes[i])  # noqa: E731
    obs = np.stack([dec(t + OFF) for t in anchors]).astype(np.float32)
    per = np.stack([dec(t - 1) for t in anchors]).astype(np.float32)
    print(f"{len(anchors)} paired test forecasts, {str(a_times[0])} .. {str(a_times[-1])}")

    P = {}
    for name, ckpt in MODELS.items():
        cache = CACHE / f"motioncmp_{name}{tag}_n{args.n}_m{args.members}.npz"
        if cache.exists() and np.array_equal(np.load(cache)["anchor_times"], a_times):
            ens = np.load(cache)["ens"].astype(np.float32)
            print(f"  {name}: cached")
        else:
            model = load_model(ckpt, dev)
            if model.target_offset != OFF:
                sys.exit(f"{ckpt} is not a 60-min model")
            mm = model.data_cfg.get("motion_minutes", 0)
            if mm not in (0, MINUTES):
                sys.exit(f"{ckpt}: motion_minutes {mm}, expected 0 or {MINUTES}")
            src = ds_m if mm else ds
            ens = np.empty((len(anchors), args.members, *obs.shape[1:]), np.float16)
            for k, t in enumerate(anchors):
                torch.manual_seed(2000 + k)                        # same noise for every model
                with torch.no_grad():
                    e = model.ensemble_sample(src.context_at(t).unsqueeze(0).to(dev),
                                              n_members=args.members, eta=1.0)
                ens[k] = ds.denormalise(e.cpu().float()[0, :, 0]).numpy()
                if k % 50 == 0:
                    print(f"  {name}: {k}/{len(anchors)}", flush=True)
            np.savez_compressed(cache, ens=ens, anchor_times=a_times)
            ens = ens.astype(np.float32)
            del model
            torch.cuda.empty_cache()
        P[name] = per_forecast(ens, obs, per)
    # plain extrapolation: the motion channel as a one-member "ensemble"
    ex = np.stack([motion_forecast(ds._decode(ds.codes[t - 6:t]), MINUTES) for t in anchors])[:, None]
    P["extrap"] = per_forecast(ex.astype(np.float32), obs, per)

    n = len(anchors)
    idx = np.random.default_rng(0).integers(0, n, (REPS, n))
    all_i = np.arange(n)
    report = {"test_forecasts": n, "test_period": [str(a_times[0]), str(a_times[-1])],
              "models": {k: str(v.relative_to(ROOT)) for k, v in MODELS.items()},
              "extrap": f"DIS optical-flow extrapolation, {MINUTES} min (src/data/motion.py)",
              "scores": {k: stats(v, all_i) for k, v in P.items()}, "differences": {}}
    for a, b in (("motion", "ctrl"), ("motion", "base"), ("ctrl", "base"),
                 ("motion", "extrap"), ("base", "extrap")):
        d = {}
        for key in report["scores"]["base"]:
            boot = [stats(P[a], i)[key] - stats(P[b], i)[key] for i in idx]
            d[key] = {"diff": report["scores"][a][key] - report["scores"][b][key],
                      "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        report["differences"][f"{a}_minus_{b}"] = d
    mc, mx = report["differences"]["motion_minus_ctrl"], report["differences"]["motion_minus_extrap"]
    keep = ((mc["fss10"]["ci"][0] > 0 or mc["catch_rate"]["ci"][0] > 0)
            and mc["crps_skill"]["ci"][1] >= 0)
    practical = keep and (mx["catch_rate"]["ci"][0] > 0 or mx["precision"]["ci"][0] > 0)
    report["verdict"] = "KEEP" if keep else "DO NOT KEEP"
    report["practical"] = ("model beats plain extrapolation" if practical
                           else "at 60 min, plain extrapolation is at least as good")
    report["verdict_rule"] = ("KEEP: motion - ctrl heavy-rain skill (FSS>=10 or catch rate) CI > 0 and "
                              "CRPS skill CI not entirely below 0; PRACTICAL: also motion - extrap "
                              "catch rate or precision CI > 0")
    out = ROOT / args.out
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for k, s in report["scores"].items():
        print(f"{k:>6}: " + ", ".join(f"{m} {v:.3f}" for m, v in s.items()))
    for k, d in report["differences"].items():
        print(f"{k}: " + ", ".join(f"{m} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                                   for m, v in d.items()))
    print("VERDICT:", report["verdict"], "|", report["practical"], "->", out)


if __name__ == "__main__":
    main()
