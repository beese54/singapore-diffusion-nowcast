#!/usr/bin/env python3
"""
evaluate_probabilistic.py -- Re-score a nowcaster with metrics suited to a
probabilistic flood tool (re-plan Step 1, tasks/replan_stage4_5.md).

The headline Stage 5 metric (evaluate.py) is FSS of individual forecasts
against persistence, which rewards exact rain placement. The model is a
generative ensemble and the use case is flood warning, so this asks four other
questions, each against persistence (repeat the last observed frame):

  A. Ensemble-probability FSS -- score the fraction of members exceeding a
     threshold as a probability field, instead of each member separately.
  B. CRPS -- the standard probabilistic accuracy score, per pixel.
  C. Area-aggregated rain -- mean rain rate over catchment-sized boxes, and
     detection of "heavy rain over this box" as a probability.
  D. Recorded flood events -- did the forecast issued ~35 min earlier put rain
     near the reported flood location? (Indicative only: few events.)

Ensembles are generated once and cached under data/processed/eval_cache/, so
every metric reads identical forecasts and metrics can be recomputed without the
GPU (--from-cache).

Usage:
    python scripts/evaluate_probabilistic.py --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt
    python scripts/evaluate_probabilistic.py --checkpoint ... --from-cache
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter, uniform_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_model  # noqa: E402
from src.data.radar_dataset import RadarDataset  # noqa: E402

PX_KM = 0.290            # measured from radar.zarr lat/lon (see lesson L025)
LEAD = 6                 # target_offset; lead is 35 min from the last frame
CACHE_DIR = ROOT / "data" / "processed" / "eval_cache"
RESULTS = ROOT / "results" / "probabilistic_eval.json"


def km(px: int) -> str:
    return f"{px * PX_KM:.1f} km"


# ── ensemble generation (cached) ──────────────────────────────────────────────

def generate(model, ds, anchors, members, device, seed0) -> np.ndarray:
    """(N, M, H, W) float16 mm/hr. Fixed seeds so reruns are identical."""
    out = np.empty((len(anchors), members, *ds.codes.shape[1:]), np.float16)
    t0 = time.time()
    for k, t in enumerate(anchors):
        torch.manual_seed(seed0 + k)
        with torch.no_grad():
            e = model.ensemble_sample(ds.context_at(t).unsqueeze(0).to(device),
                                      n_members=members, eta=1.0)
        out[k] = ds.denormalise(e.cpu().float()[0, :, 0]).numpy().astype(np.float16)
        if k % 25 == 0:
            el = time.time() - t0
            eta = el / (k + 1) * (len(anchors) - k - 1) / 60
            print(f"  {k}/{len(anchors)}  ~{eta:.0f} min left", flush=True)
    return out


# ── metrics ───────────────────────────────────────────────────────────────────

def fss_pooled(pred_prob: np.ndarray, obs_bin: np.ndarray, win: int) -> float:
    """FSS pooled over samples (Roberts & Lean 2008). pred_prob in [0, 1];
    a deterministic forecast is passed as its 0/1 exceedance field."""
    num = den = 0.0
    for p, o in zip(pred_prob, obs_bin):
        pf = uniform_filter(p.astype(np.float64), size=win, mode="constant")
        of = uniform_filter(o.astype(np.float64), size=win, mode="constant")
        num += np.sum((pf - of) ** 2)
        den += np.sum(pf ** 2) + np.sum(of ** 2)
    return float(1.0 - num / den) if den > 0 else float("nan")


def crps_ens(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-element CRPS of an ensemble (M, ...) against obs (...)."""
    ens = ens.astype(np.float32)
    t1 = np.abs(ens - obs[None]).mean(0)
    M = ens.shape[0]
    t2 = np.zeros_like(obs, dtype=np.float32)
    for i in range(M):
        t2 += np.abs(ens[i][None] - ens).sum(0)
    return t1 - 0.5 * t2 / (M * M)


def box_means(field: np.ndarray, b: int) -> np.ndarray:
    """Mean over non-overlapping b x b boxes on the trailing two axes."""
    H, W = field.shape[-2:]
    h, w = H // b, W // b
    f = field[..., : h * b, : w * b]
    return f.reshape(*f.shape[:-2], h, b, w, b).mean(axis=(-3, -1))


def contingency(pred: np.ndarray, obs: np.ndarray) -> dict:
    h = int((pred & obs).sum()); m = int((~pred & obs).sum()); f = int((pred & ~obs).sum())
    return {"hits": h, "misses": m, "false_alarms": f,
            "POD": h / max(h + m, 1), "FAR": f / max(h + f, 1),
            "CSI": h / max(h + m + f, 1)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parameterization", choices=["v", "eps"], default=None)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--from-cache", action="store_true",
                    help="Skip generation; fail if the cache is missing")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = ROOT / args.checkpoint
    model = load_model(ckpt, device, args.parameterization)
    ds = RadarDataset("test", target_offset=LEAD, **model.data_cfg)

    step = torch.load(ckpt, map_location="cpu", weights_only=True).get("step", "x")
    cache = CACHE_DIR / f"ens_{ckpt.stem}_s{step}_n{args.n_samples}_m{args.members}.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Anchors spread evenly over the whole test period, not its first hours.
    anchors = [ds.indices[i] for i in
               np.linspace(0, len(ds) - 1, args.n_samples).astype(int)]
    dec = lambda i: ds._decode(ds.codes[i])
    obs = np.stack([dec(t + LEAD) for t in anchors]).astype(np.float32)
    per = np.stack([dec(t - 1) for t in anchors]).astype(np.float32)

    # Flood events: anchor so the TARGET frame is the event's radar frame.
    import pandas as pd
    ev = pd.read_parquet(ROOT / "data" / "processed" / "flood_eval_dataset.parquet")
    ev = ev[(ev["geocoded"] == True) & (ev["is_flood_event"] == True)  # noqa: E712
            & ev["matched_radar_time"].notna()]
    step_ok = np.diff(ds.times) == np.timedelta64(5, "m")
    okc = np.concatenate(([0], np.cumsum(step_ok)))
    cf = ds.context_frames
    f_anchor, f_cell, f_type = [], [], []
    for _, r in ev.iterrows():
        idx = int(np.searchsorted(ds.times, np.datetime64(r["matched_radar_time"])))
        t = idx - LEAD
        if t - cf < 0 or idx >= len(ds.times):
            continue
        if okc[idx] - okc[t - cf] != idx - (t - cf):      # needs a contiguous span
            continue
        f_anchor.append(t)
        f_cell.append((int(r["location_lat_idx"]), int(r["location_lon_idx"])))
        f_type.append(r["message_type"])

    if cache.exists():
        z = np.load(cache)
        ens, fens = z["ens"], z["fens"]
        print(f"Loaded cached ensembles: {cache.name}")
    elif args.from_cache:
        sys.exit(f"--from-cache given but {cache.name} does not exist")
    else:
        print(f"Generating {len(anchors)} test forecasts x {args.members} members...")
        ens = generate(model, ds, anchors, args.members, device, seed0=1000)
        print(f"Generating {len(f_anchor)} flood-event forecasts...")
        fens = generate(model, ds, f_anchor, args.members, device, seed0=5000)
        np.savez_compressed(cache, ens=ens, fens=fens)
        print(f"Cached -> {cache}")
    ens = ens.astype(np.float32)
    fens = fens.astype(np.float32)

    report = {"checkpoint": str(args.checkpoint), "step": step,
              "n_samples": len(anchors), "members": args.members,
              "lead_min_from_last_frame": 35, "pixel_km": PX_KM}

    # ── A. ensemble-probability FSS ───────────────────────────────────────
    print("\nA. FSS -- ensemble probability vs persistence (pooled over samples)")
    A = {}
    wins = [21, 41, 81]
    print(f"{'threshold':>11}  {'forecast':<30}" + "".join(f"{km(w):>11}" for w in wins))
    for thr in (0.5, 2.0, 10.0):
        ob = obs >= thr
        rows = {
            "persistence": (per >= thr).astype(np.float32),
            "model: ensemble probability": (ens >= thr).mean(1),
            "model: single member": (ens[:, 0] >= thr).astype(np.float32),
        }
        A[str(thr)] = {}
        for name, p in rows.items():
            vals = {km(w): fss_pooled(p, ob, w) for w in wins}
            A[str(thr)][name] = vals
            print(f"{thr:>8} mm  {name:<30}" + "".join(f"{v:>11.3f}" for v in vals.values()))
        f0 = ob.mean()
        A[str(thr)]["useful_threshold"] = 0.5 + f0 / 2
    report["A_fss"] = A

    # ── B. CRPS ─────────────────────────────────────────────────────────────
    print("\nB. CRPS (mm/hr, lower is better); persistence CRPS = its absolute error")
    c_model = np.stack([crps_ens(e, o) for e, o in zip(ens, obs)])
    c_pers = np.abs(per - obs)
    relevant = (obs >= 0.5) | (per >= 0.5) | (ens.max(1) >= 0.5)
    B = {}
    for name, mask in (("all pixels", np.ones_like(relevant)),
                       ("rain-relevant pixels", relevant)):
        cm, cp = float(c_model[mask].mean()), float(c_pers[mask].mean())
        B[name] = {"model": cm, "persistence": cp, "skill": 1 - cm / cp}
        print(f"  {name:<22} model {cm:.4f}   persistence {cp:.4f}   "
              f"skill {1 - cm / cp:+.3f}")
    report["B_crps"] = B

    # ── C. area-aggregated rain ─────────────────────────────────────────────
    print("\nC. Rain averaged over boxes (catchment scale)")
    C = {}
    for b in (10, 21, 41):
        eb, ob_, pb = box_means(ens, b), box_means(obs, b), box_means(per, b)
        cr_m = np.stack([crps_ens(e, o) for e, o in zip(eb, ob_)])
        cr_p = np.abs(pb - ob_)
        rel = (ob_ > 0.05) | (pb > 0.05) | (eb.max(1) > 0.05)
        crps_m, crps_p = float(cr_m[rel].mean()), float(cr_p[rel].mean())
        # "moderate rain over this box": box mean >= 0.5 mm/hr. (>= 1 mm/hr
        # averaged over a 6-12 km box proved too rare to score.)
        EV = 0.5
        o_ev = ob_ >= EV
        p_mod = (eb >= EV).mean(1)
        p_per = (pb >= EV).astype(np.float32)
        brier_m = float(np.mean((p_mod - o_ev) ** 2))
        brier_p = float(np.mean((p_per - o_ev) ** 2))
        C[km(b)] = {
            "crps_model": crps_m, "crps_persistence": crps_p,
            "crps_skill": 1 - crps_m / crps_p,
            "brier_model": brier_m, "brier_persistence": brier_p,
            "brier_skill": 1 - brier_m / brier_p if brier_p > 0 else float("nan"),
            "event_rate": float(o_ev.mean()),
            "model_p>=0.5": contingency(p_mod >= 0.5, o_ev),
            "persistence": contingency(p_per >= 0.5, o_ev),
        }
        cm, cp = C[km(b)]["model_p>=0.5"], C[km(b)]["persistence"]
        print(f"  box {km(b):>8}: CRPS skill {1 - crps_m / crps_p:+.3f} | "
              f"Brier skill {C[km(b)]['brier_skill']:+.3f} | "
              f"CSI model {cm['CSI']:.3f} vs persistence {cp['CSI']:.3f} "
              f"(event rate {o_ev.mean() * 100:.2f}%)")
    report["C_boxes"] = C

    # ── D. recorded flood events ────────────────────────────────────────────
    print(f"\nD. Recorded flood events ({len(f_anchor)} usable; "
          f"forecast issued 35 min before the event frame)")
    D = {"n_events": len(f_anchor),
         "types": {t: f_type.count(t) for t in set(f_type)}}
    # Hits alone reward a forecaster that simply rains more often. The control
    # is the ALARM RATE at the same flood locations at ordinary times (the
    # regular test anchors): skill = hit rate on events well above that rate.
    cells = sorted(set(f_cell))
    for thr in (2.0, 10.0):
        for rad in (5, 17):
            am = ap_ = ao = n_ctrl = 0
            for k in range(len(anchors)):
                for (y, x) in cells:
                    sl = (slice(max(y - rad, 0), y + rad + 1), slice(max(x - rad, 0), x + rad + 1))
                    am += int(np.mean([(m[sl] >= thr).any() for m in ens[k]]) >= 0.5)
                    ap_ += int((per[k][sl] >= thr).any())
                    ao += int((obs[k][sl] >= thr).any())
                    n_ctrl += 1
            hm = hp = ho = 0
            for k, (t, (y, x)) in enumerate(zip(f_anchor, f_cell)):
                sl = (slice(max(y - rad, 0), y + rad + 1), slice(max(x - rad, 0), x + rad + 1))
                p_mod = np.mean([(m[sl] >= thr).any() for m in fens[k]])
                hm += int(p_mod >= 0.5)
                hp += int((dec(t - 1)[sl] >= thr).any())
                ho += int((dec(t + LEAD)[sl] >= thr).any())
            key = f">={thr} mm/hr within {rad * PX_KM:.1f} km"
            n = max(len(f_anchor), 1)
            D[key] = {"model_hits": hm, "persistence_hits": hp, "observed": ho,
                      "n_events": len(f_anchor),
                      "model_alarm_rate_ordinary": am / max(n_ctrl, 1),
                      "persistence_alarm_rate_ordinary": ap_ / max(n_ctrl, 1),
                      "observed_rate_ordinary": ao / max(n_ctrl, 1)}
            print(f"  {key:<28} events: model {hm / n:>5.0%}  persistence {hp / n:>5.0%}  "
                  f"observed {ho / n:>5.0%}  |  ordinary times: model {am / max(n_ctrl,1):>5.1%}  "
                  f"persistence {ap_ / max(n_ctrl,1):>5.1%}  observed {ao / max(n_ctrl,1):>5.1%}")
    report["D_flood_events"] = D

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved -> {RESULTS}")


if __name__ == "__main__":
    main()
