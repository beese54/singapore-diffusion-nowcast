#!/usr/bin/env python3
"""
hybrid_warning.py -- A hybrid 60-min heavy-rain warning (tasks/plan_hybrid_calibration.md, item 1).

Five warnings for the same 2 x 2 km box, on the SAME 200 test forecasts:
    H       hybrid: extrapolation shows >= 10 mm/hr in the box OR the calibrated model warns
    extrap  DIS optical-flow extrapolation of the last frame, 65 min (src/data/motion.py)
    cal     the model with the rule chosen on validation (results/warning_calibration.json)
    uncal   the model with the old rule (>= 2 of 8 futures >= 10 mm/hr)
    naive   the last radar frame the model saw shows >= 10 mm/hr in the box
Model ensembles: eval_cache/motioncmp_base_n200_m8.npz (seeds 2000+k, from eval_motion_input.py).
The calibrated rule is also scored on the warning_skill cache (seeds 1000+k) as a seed check.
Metrics: catch rate, precision, CSI, incoming catch; paired bootstrap, 2,000 resamples, 95%.

Verdict (fixed in the plan): ADOPT H as the 60-min heavy-rain warning if CSI(H) - CSI(extrap)
has a CI entirely above 0; otherwise ADOPT EXTRAPOLATION ALONE. Either way the diffusion
model stays the probability and rain-amount forecast.

Drain-alert context (reported, not part of the verdict): for each out-of-sample FLOOD_RISK
alert (first per site per storm day), the earliest issue time in the 120 min before it at
which H / extrap / cal warned in the 7 x 7 px site box -- forecasts issued every 5 min,
valid 65 min after issue. As in drain_alert_benchmark.py, "background" is the share of the
200 test forecasts that warned at the same site.

Usage:  python scripts/hybrid_warning.py
Output: results/hybrid_warning.json (60-min drain forecasts cached as drain_bench_<day>_s9000.npz)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from drain_alert_benchmark import (SGT, STEP, WINDOW, first_lead,  # noqa: E402
                                   forecasts_for_day, load_alerts)
from drain_alert_benchmark import box_max as site_max  # noqa: E402
from evaluate import load_model  # noqa: E402
from src.data.motion import motion_forecast  # noqa: E402
from src.data.radar_dataset import ZARR_PATH, RadarDataset  # noqa: E402
from warning_calibration import CACHE, HEAVY, REPS, box_max, counts, paired, scores  # noqa: E402

CKPT = ROOT / "checkpoints/nowcaster/lead60_warm/ckpt_step_100000.pt"
ENS = CACHE / "motioncmp_base_n200_m8.npz"
ENS_SEEDCHECK = CACHE / "ens_lead60_warm_ckpt_step_100000_s100000_n200_m8.npz"
OFF, MINUTES, DRAIN_SEED = 12, 65, 9000
UNCAL = (10.0, 2)


def test_set(cache, ds):
    z = np.load(cache)
    a_times = z["anchor_times"].astype("datetime64[m]")
    anchors = np.searchsorted(ds.times.astype("datetime64[m]"), a_times)
    if not np.array_equal(ds.times[anchors].astype("datetime64[m]"), a_times) \
            or not set(anchors.tolist()) <= set(ds.indices):
        sys.exit(f"{cache.name}: anchors not in the pinned test split")
    return z["ens"].astype(np.float32), anchors, a_times


def main():
    cal = json.loads((ROOT / "results/warning_calibration.json").read_text(encoding="utf-8"))
    X, K = cal["by_lead"]["60"]["chosen"]["X"], cal["by_lead"]["60"]["chosen"]["k"]
    print(f"calibrated 60-min rule: >= {K} of 8 futures >= {X:g} mm/hr")

    ds = RadarDataset("test", target_offset=OFF)
    dec = lambda i: ds._decode(ds.codes[i])  # noqa: E731
    ens, anchors, a_times = test_set(ENS, ds)
    obs = np.stack([dec(t + OFF) for t in anchors]).astype(np.float32)
    per = np.stack([dec(t - 1) for t in anchors]).astype(np.float32)
    ex = np.stack([motion_forecast(dec(np.arange(t - 6, t)), MINUTES) for t in anchors])
    O, P = box_max(obs) >= HEAVY, box_max(per) >= HEAVY
    bm = box_max(ens)
    E = box_max(ex) >= HEAVY
    Mc = (bm >= X).sum(1) >= K
    W = {"H": E | Mc, "extrap": E, "cal": Mc, "uncal": (bm >= UNCAL[0]).sum(1) >= UNCAL[1], "naive": P}
    C = {k: counts(w, O, P) for k, w in W.items()}
    idx = np.random.default_rng(0).integers(0, len(anchors), (REPS, len(anchors)))
    diffs = {f"{a}_minus_{b}": paired(C[a], C[b], idx)
             for a, b in (("H", "extrap"), ("H", "cal"), ("cal", "uncal"), ("extrap", "naive"), ("H", "naive"))}
    adopt_h = bool(diffs["H_minus_extrap"]["csi"]["ci"][0] > 0)

    # seed check: same checkpoint, the other cached ensemble (seeds 1000+k)
    ens2, anchors2, _ = test_set(ENS_SEEDCHECK, ds)
    same = np.array_equal(anchors2, anchors)
    seedcheck = None
    if same:
        bm2 = box_max(ens2)
        seedcheck = {"cal": scores(counts((bm2 >= X).sum(1) >= K, O, P)),
                     "uncal": scores(counts((bm2 >= UNCAL[0]).sum(1) >= UNCAL[1], O, P)),
                     "H": scores(counts(E | ((bm2 >= X).sum(1) >= K), O, P))}
    del ens2

    out = {"lead_min": 60, "test_forecasts": len(anchors), "test_period": [str(a_times[0]), str(a_times[-1])],
           "heavy_rain_boxes": int(O.sum()), "base_rate": float(O.mean()),
           "checkpoint": str(CKPT.relative_to(ROOT)), "ensembles": ENS.name,
           "calibrated_rule": {"X": X, "k": K},
           "definitions": __doc__.split("Five warnings")[1].split("Usage")[0].strip(),
           "scores": {k: scores(c) for k, c in C.items()}, "differences": diffs,
           "seed_check": {"ensembles": ENS_SEEDCHECK.name, "same_anchors": same, "scores": seedcheck},
           "verdict": "ADOPT H" if adopt_h else "ADOPT EXTRAPOLATION ALONE",
           "verdict_rule": "ADOPT H if CSI(H) - CSI(extrap) CI entirely above 0; otherwise extrapolation alone"}
    for k, s in out["scores"].items():
        print(f"{k:>6}: catch {s['catch_rate']:.1%}  precision {s['precision']:.1%}  CSI {s['csi']:.3f}  "
              f"incoming {s['incoming_catch']:.1%}  ({s['warnings']} warnings)")
    for k, d in diffs.items():
        print(f"{k}: " + ", ".join(f"{m} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]" for m, v in d.items()))
    if seedcheck:
        print("seed check (seeds 1000+k): " + ", ".join(f"{k} CSI {s['csi']:.3f} catch {s['catch_rate']:.1%}"
                                                       for k, s in seedcheck.items()))
    print("VERDICT:", out["verdict"], flush=True)

    # ── drain-alert context at a 60-min horizon ──────────────────────────────
    alerts = load_alerts()
    alerts = alerts[alerts.message_type == "FLOOD_RISK"]
    da = xr.open_zarr(ZARR_PATH, consolidated=True)["rain_rate"]
    times = da.time.values.astype("datetime64[m]")
    model = load_model(CKPT, "cuda" if torch.cuda.is_available() else "cpu")
    if model.target_offset != OFF:
        sys.exit("expected the 60-min model")
    bg = {"H": W["H"], "extrap": E, "cal": Mc}                     # test-period warnings, for background
    rows = []
    for day, g in alerts.groupby("day"):
        t0 = np.datetime64(g.t.min(), "m") - np.timedelta64(WINDOW, "m")
        t0 -= np.timedelta64(int(t0.astype("int64") % 5), "m")
        issue = np.arange(t0, np.datetime64(g.t.max(), "m") + np.timedelta64(1, "m"), STEP)
        issue = issue[np.isin(issue, times)]
        dens = forecasts_for_day(model, da, times, day, issue, DRAIN_SEED)
        ii = np.searchsorted(times, issue)
        if np.any(times[ii - 3] != issue - 3 * STEP):
            sys.exit(f"{day}: a frame 15 min before an issue time is missing")
        lo = ii.min() - 3
        block = da.isel(time=slice(lo, ii.max() + 1)).values.astype(np.float32)
        dex = np.stack([motion_forecast(block[a - 3 - lo:a + 1 - lo], MINUTES) for a in ii])
        for _, r in g.iterrows():
            i, j = r.site
            ta = np.datetime64(r.t, "m")
            win = (issue >= ta - np.timedelta64(WINDOW, "m")) & (issue <= ta)
            fe = site_max(dex[win], i, j) >= HEAVY
            fm = (site_max(dens[win], i, j) >= X).sum(1) >= K
            row = {"day": day, "site": r["name"], "alert_sgt": str(pd.Timestamp(r.t) + SGT)[11:16]}
            for name, flags in (("H", fe | fm), ("extrap", fe), ("cal", fm)):
                lead, held = first_lead(flags, issue[win], ta)
                row[f"{name}_lead_min"], row[f"{name}_held"] = lead, held
                bi, bj = min(i // 7, bg[name].shape[1] - 1), min(j // 7, bg[name].shape[2] - 1)
                row[f"{name}_background"] = float(bg[name][:, bi, bj].mean())
            rows.append(row)
        del dens
    A = pd.DataFrame(rows)
    out["drain_alerts_60min"] = {
        "alerts": rows, "model_seed": DRAIN_SEED,
        "summary": {name: {"warned": int(A[f"{name}_lead_min"].notna().sum()), "of": len(A),
                           "median_lead_min": (float(A[f"{name}_lead_min"].median())
                                               if A[f"{name}_lead_min"].notna().any() else None),
                           "storms_warned": int(A[A[f"{name}_lead_min"].notna()].day.nunique())}
                    for name in ("H", "extrap", "cal")},
        "note": "context only: 3 storm days; alerts within a storm are not independent; background uses "
                "the grid box containing the site (the 7 x 7 px site box is centred on the site)"}
    print(A.to_string(index=False))
    print(json.dumps(out["drain_alerts_60min"]["summary"]))
    caveats = ["About 12 test days and a few storms; per-box results are not independent.",
               "The calibration grid is small and fixed in advance; validation is also only about 12 days.",
               "Extrapolation uses the last 15 min of radar, the same inputs as the model, so it is available "
               "at forecast time with no extra delay."]
    out["caveats"] = caveats
    path = ROOT / "results" / "hybrid_warning.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("->", path)


if __name__ == "__main__":
    main()
