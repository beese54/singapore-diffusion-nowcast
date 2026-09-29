#!/usr/bin/env python3
"""
satellite_warning.py -- Step 2A of tasks/plan_satellite_step2.md: a learned 60-min heavy-rain
warning per 2 x 2 km box, from radar alone (L_R, the control) and radar + Himawari (L_RS).

Boxes, labels, E, incoming and initiation are exactly those of scripts/satellite_spike.py:
    heavy        observed >= 10 mm/hr in the box at issue + 65 min
    E            DIS optical-flow extrapolation shows >= 10 mm/hr in the box
Features use only data available at issue t: radar frames <= t, satellite scans that started
<= t - 20 min (src/data/satellite.py usable_scan), parallax shift from results/satellite_spike.json.

Models: HistGradientBoostingClassifier, class-balanced weights, trained on the TRAIN split with
negatives subsampled to 25% (seeded). Hyperparameters from a small grid, chosen on VALIDATION by
weighted log-loss. TEST is scored once, after everything is frozen.

Matched volume (L036): each classifier warns where p >= a threshold set on VALIDATION so that it
issues exactly as many warnings as E does on validation; that threshold is frozen for test.

KEEP (fixed in the plan, before training), on test, 95% CIs by bootstrap over days (L031):
    1. CSI(L_RS) - CSI(L_R) CI above 0      (the gain is the satellite's)
    2. CSI(L_RS) - CSI(E)   CI above 0      (it beats today's 60-min warning)
    3. precision(L_RS) >= precision(E) - 0.02

Feature tables are cached per split in data/processed/eval_cache/satwarn_<split>.npz (delete to
rebuild), so an interrupted run resumes at the first split not yet built.

Usage:  python scripts/satellite_warning.py              (full run -> results/satellite_warning.json)
        python scripts/satellite_warning.py --selftest   (leakage / matched-volume / reproducibility)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, maximum_filter, minimum_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data.motion import motion_forecast  # noqa: E402
from src.data.radar_dataset import RadarDataset  # noqa: E402
from src.data.satellite import N, SCAN, load_days, usable_scan  # noqa: E402
from satellite_spike import (DISK, R_BOX, box_centres, counts, day_boot, days_of, paired,  # noqa: E402
                             sample, scores)
from warning_calibration import box_max  # noqa: E402

HEAVY, WET, COLD = 10.0, 1.0, 220.0
OFF = 12                            # target index = anchor + 12 = issue + 65 min
R10, R30 = 5, 15                    # ~10 km and ~30 km in 2 km satellite pixels
FRESH = np.timedelta64(40, "m")     # newest usable scan must be at most this old at issue
NEG_FRAC, SEED = 0.25, 0
GRID = [{"max_depth": d, "learning_rate": lr, "max_iter": it}
        for d in (3, 6) for lr in (0.05, 0.1) for it in (200, 400)]
CACHE = ROOT / "data" / "processed" / "eval_cache"
R_NAMES = ["ex_box", "ex_near10", "now_box", "now_near10", "wet_frac10", "hour_sin", "hour_cos"]
S_NAMES = ["bt_min10", "bt_min30", "cool20_min10", "cool30_min10", "wv_minus_ir_max10",
           "cold_frac30", "cold_frac30_change20"]
NAMES = R_NAMES + S_NAMES
N_R = len(R_NAMES)
_DISK_MEAN30 = DISK(R30).astype(np.float32) / DISK(R30).sum()


# ---------------------------------------------------------------- features

class Sat:
    """Both bands for a set of days, with per-scan derived fields computed once and cached."""

    def __init__(self, t13, b13, t08, b08):
        self.t, self.b13 = t13, b13
        self.t08, self.b08 = t08, b08
        self._cache = {}

    @classmethod
    def load(cls, days):
        return cls(*load_days("B13", days), *load_days("B08", days))

    def fields(self, k):
        """Derived fields for B13 scan index k, or None if the scan is unusable."""
        if k in self._cache:
            return self._cache[k]
        bt = self.b13[k]
        f = None
        if np.isfinite(bt).mean() > 0.99:
            bt = np.where(np.isfinite(bt), bt, 330.0)              # rare holes read as warm
            f = {"min10": minimum_filter(bt, footprint=DISK(R10)),
                 "min30": minimum_filter(bt, footprint=DISK(R30)),
                 "cold30": convolve((bt <= COLD).astype(np.float32), _DISK_MEAN30, mode="nearest")}
            j = int(np.searchsorted(self.t08, self.t[k]))
            if j < len(self.t08) and self.t08[j] == self.t[k] and np.isfinite(self.b08[j]).mean() > 0.99:
                d = np.where(np.isfinite(self.b08[j]), self.b08[j], 0.0) - bt
                f["wv10"] = maximum_filter(d, footprint=DISK(R10))
        self._cache[k] = f
        return f

    def back(self, k, minutes):
        """Index of the scan exactly `minutes` before scan k, or -1."""
        want = self.t[k] - np.timedelta64(minutes, "m")
        j = int(np.searchsorted(self.t, want))
        return j if j < len(self.t) and self.t[j] == want else -1


def radar_features(ctx, t):
    """ctx: (6, H, W) mm/hr, frames issue-25 min .. issue. -> (17, 31, N_R), E field (17, 31)."""
    ex = box_max(motion_forecast(ctx, (OFF + 1) * 5).astype(np.float32))
    now = box_max(ctx[-1].astype(np.float32))
    disk = DISK(R_BOX)
    wet = convolve((now >= WET).astype(np.float32), disk / disk.sum(), mode="constant")
    h = ((t.astype("datetime64[m]").astype("int64") / 60.0 + 8.0) % 24.0) * (2 * np.pi / 24.0)   # SGT hour
    f = np.stack([np.log1p(ex), np.log1p(maximum_filter(ex, footprint=disk)),
                  np.log1p(now), np.log1p(maximum_filter(now, footprint=disk)), wet,
                  np.full(ex.shape, np.sin(h)), np.full(ex.shape, np.cos(h))], -1)
    return f.astype(np.float32), ex >= HEAVY


def satellite_features(sat, t, rc, shift):
    """(17, 31, len(S_NAMES)); NaN where a scan is missing, too old or unusable."""
    out = np.full(rc[0].shape + (len(S_NAMES),), np.nan, np.float32)
    k = usable_scan(sat.t, t)
    if k < 0 or t - sat.t[k] > FRESH:
        return out
    f = sat.fields(k)
    if f is None:
        return out
    s = lambda a: sample(a, rc, *shift)  # noqa: E731
    out[..., 0], out[..., 1], out[..., 5] = s(f["min10"]), s(f["min30"]), s(f["cold30"])
    if "wv10" in f:
        out[..., 4] = s(f["wv10"])
    for col, mins in ((2, 20), (3, 30)):
        j = sat.back(k, mins)
        g = sat.fields(j) if j >= 0 else None
        if g is not None:
            out[..., col] = s(g["min10"]) - s(f["min10"])           # positive = cooling
            if mins == 20:
                out[..., 6] = s(f["cold30"]) - s(g["cold30"])
    return out


def build(split, rc, shift):
    path = CACHE / f"satwarn_{split}.npz"
    if path.exists():
        z = np.load(path)
        print(f"{split}: cached {path.name} ({len(z['issue'])} issue times)", flush=True)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    ds = RadarDataset(split, target_offset=OFF, heavy_rain_oversample=1)
    dec = lambda i: ds._decode(ds.codes[i]).astype(np.float32)  # noqa: E731
    t_all = ds.times.astype("datetime64[m]")
    anchors = [a for a in ds.indices if int(t_all[a - 1].astype("int64")) % 10 == 0]
    issue = t_all[np.array(anchors) - 1]
    sat = Sat.load(days_of(issue))
    X, E, O, P, WN = [], [], [], [], []
    for n, a in enumerate(anchors):
        ctx = np.stack([dec(i) for i in range(a - 6, a)])
        fr, e = radar_features(ctx, issue[n])
        X.append(np.concatenate([fr, satellite_features(sat, issue[n], rc, shift)], -1))
        E.append(e)
        per = box_max(ctx[-1])
        P.append(per >= HEAVY)
        WN.append(maximum_filter((per >= WET).astype(np.uint8), footprint=DISK(R_BOX)) > 0)
        O.append(box_max(dec(a + OFF)) >= HEAVY)
        if n % 1000 == 0:
            print(f"  {split}: {n}/{len(anchors)} ({time.time() - t0:.0f} s)", flush=True)
    out = {"issue": issue, "X": np.stack(X), "E": np.stack(E), "O": np.stack(O),
           "P": np.stack(P), "wet_near": np.stack(WN)}
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)
    sat_ok = np.isfinite(out["X"][:, 0, 0, N_R]).mean()
    print(f"{split}: {len(issue)} issue times {issue[0]} .. {issue[-1]}, satellite usable {sat_ok:.1%}, "
          f"{time.time() - t0:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------- models

def rows(D, cols):
    return D["X"][..., cols].reshape(-1, len(cols)), D["O"].reshape(-1)


def weights(y):
    w1 = 0.5 / max(y.mean(), 1e-9)
    w0 = 0.5 / max(1 - y.mean(), 1e-9)
    return np.where(y, w1, w0)


def fit(X, y, params, seed=SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    keep = y | (rng.random(len(y)) < NEG_FRAC)
    Xk, yk = X[keep], y[keep]
    w = weights(y)[keep] * np.where(yk, 1.0, 1.0 / NEG_FRAC)        # undo the subsampling
    m = HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)
    return m.fit(Xk, yk, sample_weight=w)


def volume_threshold(p, n):
    """Threshold so that exactly n of p are >= it (ties broken by lowering n to the tie edge)."""
    if n <= 0:
        return np.inf
    s = np.sort(p)[::-1]
    return float(s[min(n, len(s)) - 1])


def reliability(p, y, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    return [{"p_mean": float(p[idx == b].mean()), "observed": float(y[idx == b].mean()), "n": int((idx == b).sum())}
            for b in range(bins) if (idx == b).any()]


# ---------------------------------------------------------------- main

def main():
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import log_loss
    spike = json.loads((ROOT / "results" / "satellite_spike.json").read_text(encoding="utf-8"))
    shift = tuple(spike["parallax"]["shift_px_dy_dx"])
    rc = box_centres()
    Tr, V, Te = build("train", rc, shift), build("val", rc, shift), build("test", rc, shift)
    shape = Te["O"].shape

    arms = {"L_R": list(range(N_R)), "L_RS": list(range(len(NAMES)))}
    models, chosen = {}, {}
    for arm, cols in arms.items():
        X, y = rows(Tr, cols)
        Xv, yv = rows(V, cols)
        best = None
        for g in GRID:
            t0 = time.time()
            m = fit(X, y, g)
            ll = log_loss(yv, m.predict_proba(Xv)[:, 1], sample_weight=weights(yv))
            print(f"{arm} {g}: val weighted log-loss {ll:.4f} ({time.time() - t0:.0f} s)", flush=True)
            if best is None or ll < best[0]:
                best = (ll, g, m)
        chosen[arm] = {"params": best[1], "val_weighted_log_loss": best[0]}
        models[arm] = best[2]

    n_E_val = int(V["E"].sum())
    out = {"plan": "tasks/plan_satellite_step2.md",
           "definitions": __doc__.split("Boxes,")[1].split("Feature tables")[0].strip(),
           "features": {"radar": R_NAMES, "satellite": S_NAMES}, "parallax_shift_px": list(shift),
           "periods": {s: [str(D["issue"][0]), str(D["issue"][-1]), int(len(D["issue"]))]
                       for s, D in (("train", Tr), ("val", V), ("test", Te))},
           "base_rate_test": float(Te["O"].mean()),
           "satellite_usable": {s: float(np.isfinite(D["X"][:, 0, 0, N_R]).mean())
                                for s, D in (("train", Tr), ("val", V), ("test", Te))},
           "chosen": chosen, "E_warnings_val": n_E_val, "operating_points": {}}

    idx = day_boot(Te["issue"].astype("datetime64[D]"), np.random.default_rng(0))
    p_val = {a: models[a].predict_proba(rows(V, c)[0])[:, 1] for a, c in arms.items()}
    p_te = {a: models[a].predict_proba(rows(Te, c)[0])[:, 1] for a, c in arms.items()}
    for name, mult in (("matched", 1), ("2x", 2)):
        thr = {a: volume_threshold(p_val[a], mult * n_E_val) for a in arms}
        W = {a: (p_te[a] >= thr[a]).reshape(shape) for a in arms} | {"E": Te["E"]}
        C = {k: counts(w, Te["O"], Te["P"], Te["wet_near"]) for k, w in W.items()}
        C["naive"] = counts(Te["P"], Te["O"], Te["P"], Te["wet_near"])
        op = {"thresholds": thr, "val_warnings": {a: int((p_val[a] >= thr[a]).sum()) for a in arms},
              "scores": {k: scores(c) for k, c in C.items()},
              "differences": {"L_RS_minus_L_R": paired(C["L_RS"], C["L_R"], idx),
                              "L_RS_minus_E": paired(C["L_RS"], C["E"], idx),
                              "L_R_minus_E": paired(C["L_R"], C["E"], idx)}}
        out["operating_points"][name] = op
        print(f"--- test, {name} volume (E val warnings x{mult} = {mult * n_E_val})")
        for k, s in op["scores"].items():
            print(f"   {k:>5}: catch {s['catch_rate']:.1%} precision {s['precision']:.1%} CSI {s['csi']:.3f} "
                  f"incoming {s['incoming_catch']:.1%} initiation {s['initiation_catch']:.1%} ({s['warnings']} warnings)")
        for k, d in op["differences"].items():
            print(f"   {k}: " + ", ".join(f"{m} {v['diff']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
                                          for m, v in d.items()), flush=True)

    op = out["operating_points"]["matched"]
    s, d = op["scores"], op["differences"]
    crit = {"csi_L_RS_minus_L_R_ci_above_0": bool(d["L_RS_minus_L_R"]["csi"]["ci"][0] > 0),
            "csi_L_RS_minus_E_ci_above_0": bool(d["L_RS_minus_E"]["csi"]["ci"][0] > 0),
            "precision_floor": bool(s["L_RS"]["precision"] >= s["E"]["precision"] - 0.02)}
    out["keep_criteria"] = crit
    out["verdict"] = "KEEP" if all(crit.values()) else "DO NOT KEEP"
    out["verdict_rule"] = ("KEEP at matched volume if CSI(L_RS)-CSI(L_R) CI > 0 AND CSI(L_RS)-CSI(E) CI > 0 "
                           "AND precision(L_RS) >= precision(E) - 0.02 (bootstrap over test days)")

    yte = Te["O"].reshape(-1)
    out["reliability_test"] = {a: reliability(p_te[a], yte) for a in arms}
    rng = np.random.default_rng(1)
    Xv, yv = rows(V, arms["L_RS"])
    sub = np.concatenate([np.flatnonzero(yv), rng.choice(np.flatnonzero(~yv), 200_000, replace=False)])
    pi = permutation_importance(models["L_RS"], Xv[sub], yv[sub], scoring="neg_log_loss", n_repeats=3,
                                random_state=0, sample_weight=weights(yv)[sub])
    out["permutation_importance_val"] = dict(sorted(
        {n: float(v) for n, v in zip(NAMES, pi.importances_mean)}.items(), key=lambda kv: -kv[1]))

    (ROOT / "results" / "satellite_warning.json").write_text(json.dumps(out, indent=2, default=float),
                                                            encoding="utf-8")
    print("importance:", out["permutation_importance_val"])
    print("KEEP criteria:", crit)
    print("VERDICT:", out["verdict"], "-> results/satellite_warning.json")


# ---------------------------------------------------------------- self-tests

def selftest():
    rng = np.random.default_rng(0)
    # 1. no leakage: features at t ignore radar after t and scans newer than t - 20 min
    rc = box_centres()
    t = np.datetime64("2026-09-10T08:00")
    st = np.arange(t - np.timedelta64(120, "m"), t + np.timedelta64(60, "m"), SCAN)
    b = (230 + 40 * rng.random((len(st), N, N))).astype(np.float32)
    ctx = (20 * rng.random((6, 120, 217)) * (rng.random((6, 120, 217)) > 0.8)).astype(np.float32)
    f1 = np.concatenate([radar_features(ctx, t)[0], satellite_features(Sat(st, b, st, b - 10), t, rc, (3, -8))], -1)
    b2 = b.copy()
    b2[st > t - np.timedelta64(20, "m")] = rng.random((int((st > t - np.timedelta64(20, "m")).sum()), N, N)) * 300
    f2 = np.concatenate([radar_features(ctx, t)[0], satellite_features(Sat(st, b2, st, b2 - 10), t, rc, (3, -8))], -1)
    assert np.array_equal(f1, f2, equal_nan=True), "leakage: features changed with future scans"
    assert np.isfinite(f1).all(), "features unexpectedly missing on a complete synthetic archive"
    # a radar frame after t is not an input at all (radar_features takes ctx only); check E uses ctx[-1]
    print("ok  no leakage (future scans replaced with noise: features identical)")
    # 2. matched volume
    p = rng.random(100_000)
    for n in (1, 17, 5_000):
        assert (p >= volume_threshold(p, n)).sum() == n
    print("ok  matched volume (exact warning counts)")
    # 3. reproducibility
    X = rng.normal(size=(20_000, 5)).astype(np.float32)
    y = X[:, 0] + rng.normal(size=20_000) > 2.5
    g = GRID[0] | {"max_iter": 50}
    a, c = fit(X, y, g).predict_proba(X)[:, 1], fit(X, y, g).predict_proba(X)[:, 1]
    assert np.array_equal(a, c), "not reproducible"
    print("ok  reproducible (same seed -> identical probabilities)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    (selftest if ap.parse_args().selftest else main)()
