#!/usr/bin/env python3
"""
build_era5_predictors.py -- Hourly large-scale weather predictors for the nowcaster
(tasks/plan_era5_conditioning.md).

At 28 km, ERA5 is nearly uniform across the 35 x 63 km radar domain, so instead of
maps the nowcaster gets a short vector describing "what kind of weather day it is",
averaged over the ERA5 window (N 3.0, W 102.0, S -0.5, E 105.5):

    tcwv          total column water vapour (moisture available for storms)
    t2m, msl      surface temperature and pressure
    q850, q500    humidity low and mid troposphere
    t850, t500    temperature low and mid troposphere (their difference ~ instability)
    u850, v850    low-level wind (sea-breeze / monsoon flow)
    u500, v500    mid-level wind (storm steering)

ERA5 is a reanalysis published ~6 days late, so these are an UPPER BOUND on what
large-scale weather can add; a live system would need real-time forecasts.

Input:  data/raw/era5/era5_corrdiff_2026.zarr (scripts/download_era5_corrdiff.bat)
Output: data/processed/era5_predictors.nc  (time, feature) float32, raw units
Usage:  python scripts/build_era5_predictors.py
"""

from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "era5" / "era5_corrdiff_2026.zarr"
OUT = ROOT / "data" / "processed" / "era5_predictors.nc"

SINGLE = ["tcwv", "t2m", "msl"]
LEVEL = [("q", 850), ("q", 500), ("t", 850), ("t", 500), ("u", 850), ("v", 850), ("u", 500), ("v", 500)]


def main() -> None:
    ds = xr.open_zarr(SRC)
    # area weights: cos(latitude), the standard for a lat-lon average
    w = np.cos(np.deg2rad(ds.latitude))
    cols, names = [], []
    for v in SINGLE:
        cols.append(ds[v].weighted(w).mean(("latitude", "longitude")).values)
        names.append(v)
    for v, p in LEVEL:
        cols.append(ds[v].sel(pressure_level=p).weighted(w).mean(("latitude", "longitude")).values)
        names.append(f"{v}{p}")
    X = np.stack(cols, axis=1).astype(np.float32)
    ok = np.isfinite(X).all(1)
    out = xr.Dataset({"x": (("time", "feature"), X[ok])},
                     coords={"time": ds.time.values[ok], "feature": names},
                     attrs={"source": "ERA5 (Copernicus/ECMWF), area-weighted mean over N3 W102 S-0.5 E105.5",
                            "note": "reanalysis, ~6-day lag: an upper bound on real-time value"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(OUT)
    t = out.time.values
    print(f"{OUT.name}: {len(t)} hours ({str(t[0])[:13]} .. {str(t[-1])[:13]}), features {names}")
    print("means:", {n: round(float(m), 4) for n, m in zip(names, X[ok].mean(0))})


if __name__ == "__main__":
    main()
