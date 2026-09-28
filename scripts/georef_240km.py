#!/usr/bin/env python3
"""
georef_240km.py -- Where on Earth is the 480 x 480 NEA "240 km range" radar image?
(tasks/plan_phase2.md, D1e)

NEA publishes no bounds for this product, so they are measured. The 70 km product is
already georeferenced (preprocess_radar.py, from weather.gov.sg's page source) and both
products are made from the same radar at the same 5-min times, so the 70 km domain must
appear inside the 240 km image as the same rain. The script finds where:

  1. pairs up to 150 rainy frames present in both archives (seeded choice),
  2. for a candidate placement (top-left corner x0, y0 in 240 km pixels, and a scale s
     where s = 1 means 1 km per pixel), samples the 240 km image at every 70 km pixel
     centre and correlates log(1 + rain) over all pairs,
  3. searches coarse (2 px, s = 1) then fine (0.25 px, s in 0.96..1.04).

Result (2026-09-28): correlation 0.95 at x0 = 194.5, y0 = 226.5, s = 1.00 (flat to
+/-0.01): square 1 km pixels, so the image is 480 x 480 km centred on Singapore --
"240 km" is the radius. Cross-check: fitting NEA's own coastline basemap for this page
(240km-v2.jpg) to Natural Earth coastlines independently gives the same scale to 1%
and the same position to ~3 km. Colour legend: the same 33 colours as the 70 km product
(checked over every 50th frame; no other opaque colour occurs), so
preprocess_radar.rgba_to_rain_rate() applies unchanged.

Usage:  python scripts/georef_240km.py            (~15 min, CPU)
Output: results/georef_240km.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from preprocess_radar import (RADAR_LAT_BOTTOM, RADAR_LAT_TOP,  # noqa: E402
                              RADAR_LON_LEFT, RADAR_LON_RIGHT, ZARR_PATH, rgba_to_rain_rate)

DIR240 = ROOT / "data" / "raw" / "radar_240km"
N_PAIRS, SIZE = 150, 480
KM_LAT, KM_LON = 110.574, 111.320          # km per degree at the equator (lat 1.3 N: <0.03% off)


def pairs():
    files = {p.stem: p for p in DIR240.glob("*.png")}
    da = xr.open_zarr(ZARR_PATH)["rain_rate"]
    t70 = da.time.values.astype("datetime64[m]")
    keys = np.array([np.datetime64(f"{k[:4]}-{k[4:6]}-{k[6:8]}T{k[9:11]}:{k[11:13]}") for k in files])
    common = np.intersect1d(keys, t70)
    a70, a240 = [], []
    for c in np.random.default_rng(0).permutation(len(common)):
        f = da.sel(time=common[c]).values
        if np.isnan(f).any() or (f > 0.5).mean() < 0.08:          # rainy, complete frames only
            continue
        stem = str(common[c]).replace("-", "").replace("T", "_").replace(":", "")[:13]
        a70.append(f)
        a240.append(rgba_to_rain_rate(np.array(Image.open(files[stem]).convert("RGBA"))))
        if len(a70) == N_PAIRS:
            break
    return np.log1p(np.stack(a70)).astype(np.float32), np.log1p(np.stack(a240)).astype(np.float32)


def main():
    a70, a240 = pairs()
    h70, w70 = a70.shape[1:]
    # size of the 70 km domain in 1 km units
    W = (RADAR_LON_RIGHT - RADAR_LON_LEFT) * KM_LON
    H = (RADAR_LAT_TOP - RADAR_LAT_BOTTOM) * KM_LAT

    def score(x0, y0, s, sub):
        ii, jj = sub
        u = np.floor(x0 + (jj + 0.5) / w70 * W * s).astype(int)
        v = np.floor(y0 + (ii + 0.5) / h70 * H * s).astype(int)
        if u.min() < 0 or v.min() < 0 or u.max() >= SIZE or v.max() >= SIZE:
            return -1.0
        x = a240[:, v, u].ravel() - a240[:, v, u].mean()
        y = a70[:, ii, jj].ravel() - a70[:, ii, jj].mean()
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))

    coarse = np.meshgrid(np.arange(0, h70, 4), np.arange(0, w70, 4), indexing="ij")
    c = max((score(x, y, 1.0, coarse), x, y) for y in range(0, SIZE - 36, 2) for x in range(0, SIZE - 64, 2))
    full = np.meshgrid(np.arange(h70), np.arange(w70), indexing="ij")
    fine = max((score(x, y, s, full), x, y, s)
               for s in np.round(np.arange(0.96, 1.0401, 0.005), 3)
               for x in np.arange(c[1] - 1.5, c[1] + 1.51, 0.25)
               for y in np.arange(c[2] - 1.5, c[2] + 1.51, 0.25))
    r, x0, y0, s = fine
    dlon, dlat = s / KM_LON, s / KM_LAT                         # degrees per 240 km pixel
    lon_left, lat_top = RADAR_LON_LEFT - x0 * dlon, RADAR_LAT_TOP + y0 * dlat
    out = {"pairs": len(a70), "correlation": r, "x0_px": x0, "y0_px": y0, "km_per_px": s,
           "bounds": {"lat_top": lat_top, "lat_bottom": lat_top - SIZE * dlat,
                      "lon_left": lon_left, "lon_right": lon_left + SIZE * dlon},
           "centre": {"lat": lat_top - SIZE / 2 * dlat, "lon": lon_left + SIZE / 2 * dlon},
           "uncertainty": "position +/-0.25 px near Singapore; scale +/-1% -> +/-2.5 px at the image edges",
           "colour_legend": "identical 33-colour ramp to the 70 km product"}
    path = ROOT / "results" / "georef_240km.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
