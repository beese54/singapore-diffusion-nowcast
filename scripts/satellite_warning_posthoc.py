#!/usr/bin/env python3
"""
satellite_warning_posthoc.py -- POST-HOC diagnostic for Step 2A (tasks/plan_satellite_step2.md).
It cannot change the pre-registered verdict in results/satellite_warning.json.

Step 2A's classifiers put their top warnings in the 03-12 SGT hours while the test period's heavy
rain fell mostly 12-17 SGT: the hour-of-day features let them learn the training months' diurnal
timing. This re-runs the same matched-volume comparison with variants:
    no_hour            hour features dropped, class-balanced weights (as in Step 2A)
    no_hour_unweighted hour features dropped, no class weighting
Hyperparameters: the Step 2A choice (depth 3, lr 0.05, 200 iterations) for every variant, no re-tuning.

Usage:  python scripts/satellite_warning_posthoc.py  -> results/satellite_warning_posthoc.json
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import satellite_warning as w  # noqa: E402
from satellite_spike import counts, day_boot, paired, scores  # noqa: E402

PARAMS = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200}
HOUR = [w.NAMES.index("hour_sin"), w.NAMES.index("hour_cos")]


def fit_unweighted(X, y, params, seed=w.SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    keep = y | (rng.random(len(y)) < w.NEG_FRAC)
    wt = np.where(y[keep], 1.0, 1.0 / w.NEG_FRAC)                  # undo the subsampling only
    m = HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)
    return m.fit(X[keep], y[keep], sample_weight=wt)


def main():
    D = {s: dict(np.load(w.CACHE / f"satwarn_{s}.npz")) for s in ("train", "val", "test")}
    Te, V = D["test"], D["val"]
    shape = Te["O"].shape
    idx = day_boot(Te["issue"].astype("datetime64[D]"), np.random.default_rng(0))
    n_E_val = int(V["E"].sum())
    r_cols = [c for c in range(w.N_R) if c not in HOUR]
    arms = {"L_R": r_cols, "L_RS": r_cols + list(range(w.N_R, len(w.NAMES)))}
    out = {"note": "POST-HOC diagnostic; the pre-registered Step 2A verdict is in results/satellite_warning.json",
           "params": PARAMS, "variants": {}}
    for variant, fitter in (("no_hour", w.fit), ("no_hour_unweighted", fit_unweighted)):
        W = {"E": Te["E"]}
        for arm, cols in arms.items():
            m = fitter(*w.rows(D["train"], cols), PARAMS)
            thr = w.volume_threshold(m.predict_proba(w.rows(V, cols)[0])[:, 1], n_E_val)
            W[arm] = (m.predict_proba(w.rows(Te, cols)[0])[:, 1] >= thr).reshape(shape)
        C = {k: counts(x, Te["O"], Te["P"], Te["wet_near"]) for k, x in W.items()}
        v = {"scores": {k: scores(c) for k, c in C.items()},
             "differences": {"L_RS_minus_L_R": paired(C["L_RS"], C["L_R"], idx),
                             "L_RS_minus_E": paired(C["L_RS"], C["E"], idx)}}
        out["variants"][variant] = v
        print(f"--- {variant} (matched volume, E val warnings = {n_E_val})")
        for k, s in v["scores"].items():
            print(f"   {k:>5}: catch {s['catch_rate']:.1%} precision {s['precision']:.1%} CSI {s['csi']:.3f} "
                  f"incoming {s['incoming_catch']:.1%} ({s['warnings']} warnings)")
        for k, d in v["differences"].items():
            print(f"   {k}: " + ", ".join(f"{m} {x['diff']:+.3f} [{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]"
                                          for m, x in d.items() if m != "initiation_catch"), flush=True)
    (ROOT / "results" / "satellite_warning_posthoc.json").write_text(json.dumps(out, indent=2, default=float),
                                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
