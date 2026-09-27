#!/usr/bin/env python3
"""
evaluate_probabilistic.py -- Re-score a nowcaster with metrics suited to a
probabilistic flood tool (re-plan Step 1, docs/history/replan_stage4_5.md).

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

from evaluate import crps_pixelwise, fss_parts, load_model  # noqa: E402
from src.data.radar_dataset import RadarDataset, SPLIT_TEST_END, SPLIT_TEST_START  # noqa: E402

PX_KM = 0.290            # measured from radar.zarr lat/lon (see lesson L025)
LEAD = 6                 # set from the checkpoint in main(); one model per lead
CACHE_DIR = ROOT / "data" / "processed" / "eval_cache"


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
    """Pooled FSS over samples; the per-sample building block is in evaluate.py."""
    num = den = 0.0
    for p, o in zip(pred_prob, obs_bin):
        a, b = fss_parts(p, o, win)
        num += a
        den += b
    return float(1.0 - num / den) if den > 0 else float("nan")


crps_ens = crps_pixelwise    # single implementation, in evaluate.py


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
    ap.add_argument("--out", default=None,  # default: results/probabilistic_eval_lead<N>.json
                    help="Results JSON, relative to the repo root")
    ap.add_argument("--from-cache", action="store_true",
                    help="Skip generation; fail if the cache is missing")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = ROOT / args.checkpoint
    model = load_model(ckpt, device, args.parameterization)
    global LEAD
    LEAD = model.target_offset
    ds = RadarDataset("test", target_offset=LEAD, **model.data_cfg)

    step = torch.load(ckpt, map_location="cpu", weights_only=True).get("step", "x")
    # Key includes the checkpoint's folder: probes all save as ckpt_step_<N>.pt,
    # so filename + step alone would let one model silently reuse another's
    # cached forecasts.
    cache = CACHE_DIR / (f"ens_{ckpt.parent.name}_{ckpt.stem}_s{step}"
                         f"_n{args.n_samples}_m{args.members}.npz")
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
    # Only events the model never trained on. Scoring used to take every
    # geocoded event in the archive: 29 of 38 fell in the train/val periods,
    # whose radar frames the model was fitted to, which flattered it (lesson
    # L031). "test" = inside the pinned test period; "after_test" = after it
    # (never used for training or model selection -- a fresh holdout that grows
    # as new floods are reported).
    f_anchor, f_cell, f_type, f_group = [], [], [], []
    n_in_sample = 0
    for _, r in ev.iterrows():
        et = np.datetime64(r["matched_radar_time"], "m")
        if SPLIT_TEST_START <= et <= SPLIT_TEST_END:
            group = "test"
        elif et > SPLIT_TEST_END:
            group = "after_test"
        else:
            n_in_sample += 1
            continue
        idx = int(np.searchsorted(ds.times, np.datetime64(r["matched_radar_time"])))
        t = idx - LEAD
        if t - cf < 0 or idx >= len(ds.times):
            continue
        if okc[idx] - okc[t - cf] != idx - (t - cf):      # needs a contiguous span
            continue
        f_anchor.append(t)
        f_cell.append((int(r["location_lat_idx"]), int(r["location_lon_idx"])))
        f_type.append(r["message_type"])
        f_group.append(group)

    # A cache that stores only forecasts can be scored against the wrong
    # observations -- this happened once, when the test split was still "the
    # last 10% of a growing archive": cached forecasts from 25 Sep, re-scored on
    # 26 Sep, read CRPS skill +0.129 instead of +0.437 (L028). The split is now
    # pinned, but the cache still stores the exact times it covers: a change in
    # the TEST anchors is refused; a change in the flood-event list (new
    # reports arrive daily) only regenerates the flood-event forecasts.
    a_times = ds.times[np.array(anchors)].astype("datetime64[m]")
    f_times = ds.times[np.array(f_anchor, dtype=int)].astype("datetime64[m]") \
        if f_anchor else np.array([], dtype="datetime64[m]")
    if cache.exists():
        z = np.load(cache)
        if "anchor_times" not in z.files:
            sys.exit(f"{cache.name} predates time-stamped caches, so the test "
                     f"times it covers are unknown. Delete it and regenerate.")
        if not np.array_equal(z["anchor_times"], a_times):
            sys.exit(f"{cache.name} covers different TEST anchors than the current "
                     f"split. Delete it and regenerate; do not reuse it.")
        ens, fens = z["ens"], z["fens"]
        print(f"Loaded cached ensembles: {cache.name} "
              f"(test times {str(a_times[0])} .. {str(a_times[-1])})")
        if not np.array_equal(z["flood_times"], f_times):
            if args.from_cache:
                sys.exit(f"{cache.name}: flood-event list changed and --from-cache "
                         f"forbids generating the new flood forecasts")
            print(f"Flood-event list changed ({len(z['flood_times'])} -> {len(f_times)}); "
                  f"regenerating flood-event forecasts only")
            fens = generate(model, ds, f_anchor, args.members, device, seed0=5000)
            np.savez_compressed(cache, ens=ens, fens=fens,
                                anchor_times=a_times, flood_times=f_times)
    elif args.from_cache:
        sys.exit(f"--from-cache given but {cache.name} does not exist")
    else:
        print(f"Generating {len(anchors)} test forecasts x {args.members} members...")
        ens = generate(model, ds, anchors, args.members, device, seed0=1000)
        print(f"Generating {len(f_anchor)} flood-event forecasts...")
        fens = generate(model, ds, f_anchor, args.members, device, seed0=5000)
        np.savez_compressed(cache, ens=ens, fens=fens,
                            anchor_times=a_times, flood_times=f_times)
        print(f"Cached -> {cache}")
    ens = ens.astype(np.float32)
    fens = fens.astype(np.float32)

    report = {"checkpoint": str(args.checkpoint), "step": step,
              "test_period": [str(a_times[0]), str(a_times[-1])],
              "n_samples": len(anchors), "members": args.members,
              "nominal_lead_min": LEAD * 5,
              "lead_min_from_last_frame": (LEAD + 1) * 5, "pixel_km": PX_KM}

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
    print(f"\nD. Recorded flood events, out of sample only "
          f"(forecast issued {(LEAD + 1) * 5} min before the event frame; "
          f"{n_in_sample} train/val-period events excluded)")
    D = {"excluded_in_sample": n_in_sample}
    for group in ("test", "after_test"):
        ks = [k for k, g in enumerate(f_group) if g == group]
        types = [f_type[k] for k in ks]
        G = {"n_events": len(ks), "types": {t: types.count(t) for t in set(types)},
             "event_times": [str(f_times[k]) for k in ks]}
        # Hits alone reward a forecaster that simply rains more often. The
        # control is the ALARM RATE at the same flood locations at ordinary
        # times (the regular test anchors): skill = hit rate on events well
        # above that rate.
        cells = sorted({f_cell[k] for k in ks})
        print(f"  [{group}] {len(ks)} events")
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
                for k in ks:
                    t, (y, x) = f_anchor[k], f_cell[k]
                    sl = (slice(max(y - rad, 0), y + rad + 1), slice(max(x - rad, 0), x + rad + 1))
                    p_mod = np.mean([(m[sl] >= thr).any() for m in fens[k]])
                    hm += int(p_mod >= 0.5)
                    hp += int((dec(t - 1)[sl] >= thr).any())
                    ho += int((dec(t + LEAD)[sl] >= thr).any())
                key = f">={thr} mm/hr within {rad * PX_KM:.1f} km"
                G[key] = {"model_hits": hm, "persistence_hits": hp, "observed": ho,
                          "n_events": len(ks),
                          "model_alarm_rate_ordinary": am / max(n_ctrl, 1),
                          "persistence_alarm_rate_ordinary": ap_ / max(n_ctrl, 1),
                          "observed_rate_ordinary": ao / max(n_ctrl, 1)}
                print(f"    {key:<28} events: model {hm}/{len(ks)}  persistence {hp}/{len(ks)}  "
                      f"observed {ho}/{len(ks)}  |  ordinary times: model {am / max(n_ctrl, 1):>5.1%}  "
                      f"persistence {ap_ / max(n_ctrl, 1):>5.1%}  observed {ao / max(n_ctrl, 1):>5.1%}")
        D[group] = G
    report["D_flood_events"] = D

    # Per-lead default: one shared file let the three lead models overwrite
    # each other's results.
    out = ROOT / (args.out or f"results/probabilistic_eval_lead{LEAD * 5}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
