#!/usr/bin/env python3
"""
satellite_spike.py -- Step 1 of tasks/plan_satellite.md: does Himawari-9 see heavy rain coming
that extrapolation cannot? No training; no GPU.

Boxes: the 17 x 31 grid of 2 x 2 km boxes (7 x 7 radar px) used by every warning score.
Forecasts are issued every 10 min (issue time = the last radar frame the forecast sees).

    heavy        observed >= 10 mm/hr in the box at the target time (issue + lead + 5 min)
    incoming     heavy at the target, not heavy at issue
    initiation   heavy at the target, and NO radar rain >= 1 mm/hr within ~10 km of the box at issue
    E            DIS optical-flow extrapolation shows >= 10 mm/hr in the box (src/data/motion.py)
    S            satellite: somewhere within ~10 km of the box, band-13 cloud-top temperature <= T K
                 AND it cooled by >= C K over the last 20 min, using only the newest scan that was
                 available at issue (scan start <= issue - 20 min; src/data/satellite.py)
    E u S        either warns

Parallax: cloud tops appear displaced away from the satellite (140.7 E). The shift (whole
satellite pixels) that best aligns cold cloud (<= 235 K) with heavy radar rain is measured on
VALIDATION storms and applied to every satellite sample.
T in {220, 230, 240} K and C in {4, 8, 12} K are chosen per lead by CSI of S on VALIDATION only.

Scored on TEST: catch rate, precision, CSI, incoming catch, initiation catch; 95% CIs by a
bootstrap over DAYS (storms are the independent units, L031), 2,000 resamples.

GO (fixed in the plan), at 60 min: (E u S) - E incoming catch CI entirely above 0, AND
CSI(E u S) - CSI(E) CI not entirely below 0. 30 and 90 min are reported as context.

Usage:  python scripts/satellite_spike.py
Output: results/satellite_spike.json, results/satellite_spike_georef.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.motion import motion_forecast  # noqa: E402
from src.data.radar_dataset import ZARR_PATH, RadarDataset  # noqa: E402
from src.data.satellite import LAT1, LON0, N, SCAN, load_days, usable_scan  # noqa: E402
from warning_calibration import box_max  # noqa: E402

B, HEAVY, WET, REPS = 7, 10.0, 1.0, 2000
TS, CS = (220.0, 230.0, 240.0), (4.0, 8.0, 12.0)
LEADS = {"30": 6, "60": 12, "90": 18}
STEP_DEG = (104.85 - 102.85) / (N - 1)
R_SAT = 5              # ~10 km in 2 km satellite pixels
R_BOX = 5              # ~10 km in 2 km boxes
COLD = 235.0
SHIFTS = range(-10, 11)          # +-20 km; expected ~11 km west for a 12 km cloud top
DISK = (lambda r: (np.add.outer(np.arange(-r, r + 1) ** 2, np.arange(-r, r + 1) ** 2) <= r * r))


def box_centres():
    """Satellite (row, col) of each box centre, (17, 31) each."""
    da = xr.open_zarr(ZARR_PATH, consolidated=True)["rain_rate"]
    lat, lon = da.lat.values, da.lon.values
    blat = lat[:119].reshape(-1, B).mean(1)
    blon = lon[:217 - 217 % B].reshape(-1, B).mean(1)
    rows = np.rint((LAT1 - blat) / STEP_DEG).astype(int)
    cols = np.rint((blon - LON0) / STEP_DEG).astype(int)
    return np.meshgrid(rows, cols, indexing="ij")


def sample(field, rc, dy, dx):
    """field (..., N, N) at the box centres shifted by (dy, dx) satellite px -> (..., 17, 31)."""
    r, c = rc
    return field[..., np.clip(r + dy, 0, N - 1), np.clip(c + dx, 0, N - 1)]


def days_of(times):
    d = np.unique(times.astype("datetime64[D]"))
    return [str(x).replace("-", "") for x in np.arange(d.min() - 1, d.max() + 2)]


def build(split, rc):
    """Everything per issue time for one split: radar truth, E per lead, satellite at issue."""
    ds = RadarDataset(split, target_offset=max(LEADS.values()))
    dec = lambda i: ds._decode(ds.codes[i])  # noqa: E731
    t_all = ds.times.astype("datetime64[m]")
    anchors = [a for a in ds.indices if int(t_all[a - 1].astype("int64")) % 10 == 0]
    issue = t_all[np.array(anchors) - 1]
    st, sbt = load_days("B13", days_of(issue))
    per = np.stack([dec(a - 1) for a in anchors]).astype(np.float32)
    P_heavy = box_max(per) >= HEAVY
    wet_near = maximum_filter((box_max(per) >= WET).astype(np.uint8), footprint=DISK(R_BOX), axes=(1, 2))
    out = {"issue": issue, "day": issue.astype("datetime64[D]"), "P": P_heavy, "wet_near": wet_near > 0, "lead": {}}
    for L, off in LEADS.items():
        obs = np.stack([dec(a - 1 + off + 1) for a in anchors]).astype(np.float32)
        ex = np.stack([motion_forecast(dec(np.arange(a - 6, a)), (off + 1) * 5) for a in anchors])
        out["lead"][L] = {"O": box_max(obs) >= HEAVY, "E": box_max(ex) >= HEAVY}
    # satellite at issue: newest usable scan and the one 20 min before it
    now, before, ok = [], [], []
    for t in issue:
        k = usable_scan(st, t)
        k2 = int(np.searchsorted(st, st[k] - 2 * SCAN)) if k >= 0 else -1
        good = k >= 0 and k2 < len(st) and st[k2] == st[k] - 2 * SCAN and t - st[k] <= np.timedelta64(40, "m")
        ok.append(good)
        now.append(sbt[k] if good else np.full((N, N), np.nan, np.float32))
        before.append(sbt[k2] if good else np.full((N, N), np.nan, np.float32))
    out["bt"], out["cool"], out["sat_ok"] = np.stack(now), np.stack(before) - np.stack(now), np.array(ok)
    out["sat_frames"] = (st, sbt)
    out["radar_frames"] = (t_all, dec)
    print(f"{split}: {len(anchors)} issue times {issue[0]} .. {issue[-1]}, satellite usable {np.mean(ok):.1%}", flush=True)
    return out


def measure_parallax(D, rc):
    """Shift maximising CSI(cold cloud, heavy radar) on validation frames with heavy rain."""
    st, sbt = D["sat_frames"]
    t_all, dec = D["radar_frames"]
    pairs = []
    for i, t in enumerate(D["issue"]):
        if not D["P"][i].any():
            continue
        k = int(np.searchsorted(st, t - np.timedelta64(5, "m")))       # scan whose Singapore line ~ t
        if k < len(st) and abs(st[k] + np.timedelta64(5, "m") - t) <= np.timedelta64(5, "m"):
            pairs.append((sbt[k] <= COLD, D["P"][i]))
    cold = np.stack([p[0] for p in pairs])
    heavy = np.stack([p[1] for p in pairs])
    best, grid = None, {}
    for dy in SHIFTS:
        for dx in SHIFTS:
            c = sample(cold, rc, dy, dx)
            h, f, m = (c & heavy).sum(), (c & ~heavy).sum(), (~c & heavy).sum()
            grid[(dy, dx)] = h / max(h + f + m, 1)
            if best is None or grid[(dy, dx)] > grid[best]:
                best = (dy, dx)
    return best, grid[best], grid[(0, 0)], len(pairs)


def sat_warn(D, rc, shift, T, C):
    cond = (D["bt"] <= T) & (D["cool"] >= C)
    near = maximum_filter(cond.astype(np.uint8), footprint=DISK(R_SAT), axes=(1, 2))
    return (sample(near, rc, *shift) > 0) & D["sat_ok"][:, None, None]


def counts(W, O, P, wet_near):
    inc, ini = O & ~P, O & ~wet_near
    ax = (1, 2)
    return np.stack([(W & O).sum(ax), (W & ~O).sum(ax), (~W & O).sum(ax),
                     (W & inc).sum(ax), inc.sum(ax), (W & ini).sum(ax), ini.sum(ax)], 1)


def scores(c):
    h, f, m, hi, ni, hn, nn = c.sum(0)
    return {"catch_rate": h / max(h + m, 1), "precision": h / max(h + f, 1), "csi": h / max(h + f + m, 1),
            "incoming_catch": hi / max(ni, 1), "initiation_catch": hn / max(nn, 1),
            "hits": int(h), "false_alarms": int(f), "misses": int(m), "warnings": int(h + f),
            "incoming_boxes": int(ni), "initiation_boxes": int(nn)}


METRICS = ("catch_rate", "precision", "csi", "incoming_catch", "initiation_catch")


def day_boot(days, rng):
    """Bootstrap over days: list of index arrays."""
    u = np.unique(days)
    groups = [np.flatnonzero(days == d) for d in u]
    return [np.concatenate([groups[j] for j in rng.integers(0, len(u), len(u))]) for _ in range(REPS)]


def paired(ca, cb, idx):
    sa, sb = scores(ca), scores(cb)
    boot = np.array([[scores(ca[i])[m] - scores(cb[i])[m] for m in METRICS] for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {m: {"diff": sa[m] - sb[m], "ci": [float(lo[j]), float(hi[j])]} for j, m in enumerate(METRICS)}


def figure(V, rc, shift, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i = int(np.argmax(V["P"].sum((1, 2))))                         # the heaviest validation frame
    st, sbt = V["sat_frames"]
    t = V["issue"][i]
    k = int(np.searchsorted(st, t - np.timedelta64(5, "m")))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, s, title in ((ax[0], (0, 0), "no shift"), (ax[1], shift, f"parallax shift {shift} px")):
        a.imshow(sample(sbt[k], rc, *s), cmap="gray_r", vmin=190, vmax=300)
        a.contour(V["P"][i], levels=[0.5], colors="red", linewidths=1.2)
        a.set_title(f"B13 {str(st[k])[:16]} UTC vs radar >= 10 mm/hr (red) {str(t)[11:16]}\n{title}", fontsize=9)
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    rc = box_centres()
    V = build("val", rc)
    T_ = build("test", rc)
    shift, csi_best, csi_zero, n_pairs = measure_parallax(V, rc)
    print(f"parallax: best shift (dy, dx) = {shift} satellite px, cold/heavy CSI {csi_zero:.3f} -> {csi_best:.3f} "
          f"({n_pairs} validation frames)", flush=True)
    figure(V, rc, shift, ROOT / "results" / "satellite_spike_georef.png")

    rng = np.random.default_rng(0)
    idx = day_boot(T_["day"], rng)
    out = {"definitions": __doc__.split("Boxes:")[1].split("Usage")[0].strip(),
           "validation_period": [str(V["issue"][0]), str(V["issue"][-1])],
           "test_period": [str(T_["issue"][0]), str(T_["issue"][-1])],
           "test_days": int(len(np.unique(T_["day"]))),
           "issue_times": {"val": int(len(V["issue"])), "test": int(len(T_["issue"]))},
           "satellite_usable": {"val": float(V["sat_ok"].mean()), "test": float(T_["sat_ok"].mean())},
           "parallax": {"shift_px_dy_dx": list(shift), "shift_km": [2 * shift[0], 2 * shift[1]],
                        "csi_cold_vs_heavy": {"no_shift": csi_zero, "shifted": csi_best}, "frames": n_pairs},
           "by_lead": {}}
    for L in LEADS:
        grid = []
        for T in TS:
            for C in CS:
                s = scores(counts(sat_warn(V, rc, shift, T, C), V["lead"][L]["O"], V["P"], V["wet_near"]))
                grid.append({"T": T, "C": C} | s)
        best = max(grid, key=lambda g: g["csi"])
        S = sat_warn(T_, rc, shift, best["T"], best["C"])
        O, E = T_["lead"][L]["O"], T_["lead"][L]["E"]
        C_ = {k: counts(w, O, T_["P"], T_["wet_near"]) for k, w in
              (("S", S), ("E", E), ("EuS", E | S), ("naive", T_["P"]))}
        r = {"chosen": {"T": best["T"], "C": best["C"], "validation_csi": best["csi"]}, "validation_grid": grid,
             "scores": {k: scores(c) for k, c in C_.items()},
             "differences": {"EuS_minus_E": paired(C_["EuS"], C_["E"], idx),
                             "S_minus_E": paired(C_["S"], C_["E"], idx)}}
        d = r["differences"]["EuS_minus_E"]
        r["go_criteria"] = {"incoming_catch_ci_above_0": bool(d["incoming_catch"]["ci"][0] > 0),
                            "csi_not_entirely_below_0": bool(d["csi"]["ci"][1] > 0)}
        out["by_lead"][L] = r
        print(f"lead {L}: S rule T<={best['T']:g} K, cooling>={best['C']:g} K (val CSI {best['csi']:.3f})")
        for k, s in r["scores"].items():
            print(f"   {k:>5}: catch {s['catch_rate']:.1%} precision {s['precision']:.1%} CSI {s['csi']:.3f} "
                  f"incoming {s['incoming_catch']:.1%} initiation {s['initiation_catch']:.1%} ({s['warnings']} warnings)")
        print("   EuS - E: " + ", ".join(f"{m} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                                         for m, v in d.items()), flush=True)
    g = out["by_lead"]["60"]["go_criteria"]
    out["verdict"] = "GO" if all(g.values()) else "NO-GO"
    out["verdict_rule"] = ("GO at 60 min: (E u S) - E incoming catch CI entirely above 0 AND CSI difference CI "
                           "not entirely below 0 (bootstrap over days)")
    (ROOT / "results" / "satellite_spike.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("VERDICT:", out["verdict"], "-> results/satellite_spike.json")


if __name__ == "__main__":
    main()
