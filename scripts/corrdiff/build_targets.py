#!/usr/bin/env python3
"""
build_targets.py -- Hourly ~2 km rain targets for the CorrDiff spike (tasks/spike_corrdiff.md).

Target at ERA5 hour t = mean NEA radar rain rate over (t-1h, t], the same window
ERA5's hourly total_precipitation accumulates over, so the two are comparable in
mm/hr. The 0.29 km radar is block-averaged 7 x 7 -> 2.03 km and cropped to 16 x 28
cells (both divisible by 4, for two U-Net downsamplings). An hour is kept only if
at least MIN_FRAMES of its 12 five-minute frames exist and none is blank.

Output: data/processed/corrdiff/targets_2km.nc
    rain  (time, y, x) float32 mm/hr, hourly mean
    lat, lon (y), (x)  block-centre coordinates
    n_frames (time)    radar frames that went into each hour
Usage:  python scripts/corrdiff/build_targets.py
"""

from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "corrdiff" / "targets_2km.nc"
BLOCK = 7
ROWS = slice(0, 16 * BLOCK)          # 112 of 120 rows
COLS = slice(10, 10 + 28 * BLOCK)    # 196 of 217 columns, centred
MIN_FRAMES = 10


def block_mean(a: np.ndarray) -> np.ndarray:
    """(..., 16*B, 28*B) -> (..., 16, 28)."""
    *lead, h, w = a.shape
    return a.reshape(*lead, h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(-3, -1))


def main() -> None:
    da = xr.open_zarr(ROOT / "data" / "processed" / "radar.zarr", consolidated=True)["rain_rate"]
    times = da.time.values
    lat = da.lat.values[ROWS].reshape(16, BLOCK).mean(1)
    lon = da.lon.values[COLS].reshape(28, BLOCK).mean(1)

    # ERA5 hours covered by the archive: the first full hour after the start
    hours = np.arange(times[0].astype("datetime64[h]") + np.timedelta64(1, "h"),
                      times[-1].astype("datetime64[h]") + np.timedelta64(1, "h"),
                      np.timedelta64(1, "h")).astype("datetime64[ns]")
    # frame -> hour ending at or after it: a frame at 12:05..13:00 belongs to hour 13:00
    frame_hour = (times - np.timedelta64(1, "ns")).astype("datetime64[h]").astype("datetime64[ns]") \
        + np.timedelta64(1, "h")

    kept_t, kept_rain, kept_n = [], [], []
    block = 2000                                   # read the archive in chunks (RAM)
    sums, counts, bad = {}, {}, set()
    for a in range(0, len(times), block):
        frames = da.isel(time=slice(a, a + block), lat=ROWS, lon=COLS).values.astype(np.float32)
        for k, fh in enumerate(frame_hour[a:a + block]):
            f = frames[k]
            if np.isnan(f).any():
                bad.add(fh)
                continue
            sums[fh] = sums.get(fh, 0) + block_mean(f)
            counts[fh] = counts.get(fh, 0) + 1
    for h in hours:
        n = counts.get(h, 0)
        if n >= MIN_FRAMES and h not in bad:
            kept_t.append(h)
            kept_rain.append(sums[h] / n)
            kept_n.append(n)

    ds = xr.Dataset(
        {"rain": (("time", "y", "x"), np.stack(kept_rain).astype(np.float32), {"units": "mm/hr",
                  "long_name": "hourly mean radar rain rate over (t-1h, t]"}),
         "n_frames": ("time", np.array(kept_n, np.int8))},
        coords={"time": np.array(kept_t), "lat": ("y", lat), "lon": ("x", lon)},
        attrs={"source": "NEA 70 km rain-area radar (c) NEA/MSS, via data/processed/radar.zarr",
               "grid": f"{BLOCK}x{BLOCK} block mean of 0.29 km pixels = {0.29 * BLOCK:.2f} km"})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    wet = (ds.rain > 0.5).mean().item()
    print(f"{OUT.name}: {len(kept_t)} of {len(hours)} hours kept "
          f"({str(kept_t[0])[:13]} .. {str(kept_t[-1])[:13]}), grid 16x28 at {0.29 * BLOCK:.2f} km, "
          f"lat {lat.min():.3f}..{lat.max():.3f}, lon {lon.min():.3f}..{lon.max():.3f}, "
          f"wet (>0.5 mm/hr) {wet:.1%}, max {float(ds.rain.max()):.1f} mm/hr")


if __name__ == "__main__":
    main()
