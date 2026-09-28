#!/usr/bin/env python3
"""
spike_240km_extrapolation.py -- Does the wide 240 km radar hold heavy-rain information
that the 70 km domain lacks? (tasks/plan_stage7_heavy_rain.md, Step 1; no training)

Forecasters, all scored on the 70 km grid at the SAME test issue times as the cached
model ensembles (200 per lead; lead = 35/65/95 min from the last observed frame):
  W      wide view: DIS optical-flow motion from the last 15 min of 240 km frames, the latest
         240 km frame advected forward by the lead, sampled onto the 70 km grid
         (bounds from results/georef_240km.json)
  S      identical pipeline, but the 240 km input is blanked outside the 70 km
         footprint -- rain that starts outside the domain is invisible, as it is to
         anything that only sees the 70 km radar. W - S isolates the wider VIEW from
         the extrapolation method and the 1 km resolution.
  naive  the last 70 km frame (persistence)
  model  the diffusion model of record for that lead (>= 2 of 8 futures)
  (ceiling: the 240 km frame AT the target time, i.e. how well the 1 km product can
  represent 70 km heavy rain at all)

Scoring as in warning_skill.py: ~2 x 2 km boxes (7 x 7 px), heavy = >= 10 mm/hr.
  incoming  boxes heavy at the target that were NOT heavy at the last observed frame
            -- what persistence misses by construction
  catch     share of incoming boxes warned; new-warning precision = share of warnings
            in not-currently-heavy boxes that came true
Paired bootstrap over forecasts (2,000, 95%).

Step 0 measures the 240 km vs 70 km time alignment (correlation at -10..+10 min)
and uses the best shift.

GO (fixed in the plan): at 60 or 90 min, W catches more incoming heavy rain than S
(CI of the difference > 0) and W's new-warning precision is not lower than S's (CI of
the difference not entirely below 0).

Usage:  python scripts/spike_240km_extrapolation.py
Output: results/spike_240km.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import xarray as xr
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from src.data import motion  # noqa: E402
from preprocess_radar import (RADAR_LAT_BOTTOM, RADAR_LAT_TOP,  # noqa: E402
                              RADAR_LON_LEFT, RADAR_LON_RIGHT, ZARR_PATH, rgba_to_rain_rate)

DIR240 = ROOT / "data" / "raw" / "radar_240km"
CACHE = ROOT / "data" / "processed" / "eval_cache"
ENS = {30: ("ens_nowcaster_ckpt_step_300000_s300000_n200_m8.npz", 6),
       60: ("ens_lead60_warm_ckpt_step_100000_s100000_n200_m8.npz", 12),
       90: ("ens_lead90_warm_ckpt_step_100000_s100000_n200_m8.npz", 18)}
B, HEAVY, WARN, REPS, FLOW_MIN, SMOOTH_KM = 7, 10.0, 0.25, 2000, 15, 20
M5 = np.timedelta64(5, "m")

geo = json.loads((ROOT / "results" / "georef_240km.json").read_text())
X0, Y0, KMPX = geo["x0_px"], geo["y0_px"], geo["km_per_px"]
W70 = (RADAR_LON_RIGHT - RADAR_LON_LEFT) * 111.320 * KMPX
H70 = (RADAR_LAT_TOP - RADAR_LAT_BOTTOM) * 110.574 * KMPX
# 70 km pixel centres in 240 km pixel coordinates (OpenCV: integer = pixel centre)
_jj, _ii = np.meshgrid(np.arange(217), np.arange(120))
MAP_X = (X0 + (_jj + 0.5) / 217 * W70 - 0.5).astype(np.float32)
MAP_Y = (Y0 + (_ii + 0.5) / 120 * H70 - 0.5).astype(np.float32)
FOOT = np.zeros((480, 480), bool)
FOOT[int(np.floor(Y0)):int(np.ceil(Y0 + H70)), int(np.floor(X0)):int(np.ceil(X0 + W70))] = True


def load240(t):
    p = DIR240 / (str(t).replace("-", "").replace("T", "_").replace(":", "")[:13] + ".png")
    return rgba_to_rain_rate(np.array(Image.open(p).convert("RGBA"))) if p.exists() else None


def to70(f):
    return cv2.remap(f, MAP_X, MAP_Y, cv2.INTER_NEAREST, borderValue=0)


def advect(prev, now, minutes):
    """Shared implementation (src/data/motion.py): DIS flow, smoothed ~20 km."""
    return motion.advect(prev, now, minutes / FLOW_MIN, SMOOTH_KM)


def boxes(a):
    a = a[..., :119, :217 - 217 % B]
    *lead, h, w = a.shape
    return a.reshape(*lead, h // B, B, w // B, B).max(axis=(-3, -1))


def alignment(da, t70):
    """Correlation of 240 km (on the 70 km grid) with 70 km at time shifts."""
    rng = np.random.default_rng(1)
    cand = [t for t in rng.permutation(t70[t70 >= np.datetime64("2026-08-28")])[:600]]
    out = {}
    for s in (-10, -5, 0, 5, 10):
        xs, ys = [], []
        for t in cand:
            f70 = da.sel(time=t).values
            if np.isnan(f70).any() or (f70 > 0.5).mean() < 0.08:
                continue
            f = load240(t + np.timedelta64(s, "m"))
            if f is None:
                continue
            xs.append(np.log1p(to70(f)).ravel()); ys.append(np.log1p(f70).ravel())
            if len(xs) == 80:
                break
        x, y = np.concatenate(xs), np.concatenate(ys)
        out[s] = float(np.corrcoef(x, y)[0, 1])
    return out


def main():
    da = xr.open_zarr(ZARR_PATH)["rain_rate"]
    t70 = da.time.values.astype("datetime64[m]")
    align = alignment(da, t70)
    shift = np.timedelta64(max(align, key=align.get), "m")
    print("alignment (shift min -> corr):", align, "-> using", shift)

    report = {"alignment_corr": {str(k): v for k, v in align.items()}, "shift_min": int(shift.astype(int)),
              "by_lead": {}}
    rng = np.random.default_rng(0)
    for lead, (fname, off) in ENS.items():
        z = np.load(CACHE / fname)
        anchors = z["anchor_times"].astype("datetime64[m]")
        ens = z["ens"]
        minutes = (off + 1) * 5
        rows = []
        for k, a in enumerate(anchors):
            last, target = a - M5, a + off * M5
            now240 = load240(last + shift)
            prev240 = load240(last + shift - np.timedelta64(FLOW_MIN, "m"))
            tgt240 = load240(target + shift)
            if now240 is None or prev240 is None:
                continue
            obs = da.sel(time=target).values
            per = da.sel(time=last).values
            if np.isnan(obs).any() or np.isnan(per).any():
                continue
            fW = to70(advect(prev240, now240, minutes))
            fS = to70(advect(np.where(FOOT, prev240, 0), np.where(FOOT, now240, 0), minutes))
            O, P = boxes(obs) >= HEAVY, boxes(per) >= HEAVY
            warn = {"W": boxes(fW) >= HEAVY, "S": boxes(fS) >= HEAVY, "naive": P,
                    "model": (boxes(ens[k].astype(np.float32)) >= HEAVY).mean(0) >= WARN}
            if tgt240 is not None:
                warn["ceiling"] = boxes(to70(tgt240)) >= HEAVY
            inc = O & ~P
            r = {"inc": int(inc.sum())}
            for n, w in warn.items():
                r[n] = (int((w & inc).sum()), int((w & ~P).sum()), int((w & ~P & O).sum()),
                        int((w & O).sum()), int(w.sum()), int(O.sum()))
            rows.append(r)
        names = ["W", "S", "naive", "model"] + (["ceiling"] if all("ceiling" in r for r in rows) else [])
        inc = np.array([r["inc"] for r in rows])
        arr = {n: np.array([r[n] for r in rows]) for n in names}   # (n, 6)

        def stats(ix):
            s = {}
            for n in names:
                a = arr[n][ix].sum(0)
                s[n] = {"incoming_catch": a[0] / max(inc[ix].sum(), 1),
                        "new_warning_precision": a[2] / max(a[1], 1),
                        "catch": a[3] / max(a[5], 1), "precision": a[3] / max(a[4], 1)}
            return s

        all_i = np.arange(len(rows))
        sc = stats(all_i)
        boot = [stats(rng.integers(0, len(rows), len(rows))) for _ in range(REPS)]

        def diff(a, b, key):
            d = [x[a][key] - x[b][key] for x in boot]
            return {"diff": sc[a][key] - sc[b][key], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

        report["by_lead"][str(lead)] = {
            "forecasts": len(rows), "minutes_after_last_frame": minutes,
            "incoming_heavy_boxes": int(inc.sum()),
            "incoming_share_of_heavy": float(inc.sum() / max(arr["naive"][:, 5].sum(), 1)),
            "scores": sc,
            "W_minus_S": {k: diff("W", "S", k) for k in ("incoming_catch", "new_warning_precision")},
            "W_minus_model": {k: diff("W", "model", k) for k in ("incoming_catch", "new_warning_precision",
                                                                 "catch", "precision")},
        }
        L = report["by_lead"][str(lead)]
        print(f"\nlead {lead} ({minutes} min): {len(rows)} forecasts, {inc.sum()} incoming heavy boxes "
              f"({L['incoming_share_of_heavy']:.0%} of heavy)")
        for n in names:
            s = sc[n]
            print(f"  {n:>8}: incoming catch {s['incoming_catch']:.3f}  new-warn prec {s['new_warning_precision']:.3f}"
                  f"  catch {s['catch']:.3f}  prec {s['precision']:.3f}")
        for k, v in L["W_minus_S"].items():
            print(f"  W-S {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")

    go = any(report["by_lead"][l]["W_minus_S"]["incoming_catch"]["ci"][0] > 0
             and report["by_lead"][l]["W_minus_S"]["new_warning_precision"]["ci"][1] >= 0
             for l in ("60", "90"))
    report["verdict"] = "GO" if go else "NO-GO"
    report["verdict_rule"] = ("at 60 or 90 min: W - S incoming catch CI > 0 and W - S new-warning "
                              "precision CI not entirely below 0")
    out = ROOT / "results" / "spike_240km.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("\nVERDICT:", report["verdict"], "->", out)


if __name__ == "__main__":
    main()
