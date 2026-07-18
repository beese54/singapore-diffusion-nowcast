"""
Repro for task 4.8: blank (all-NaN) radar frames leak into RadarDataset samples.

Part 1 — archive audit: per-frame NaN-fraction distribution across radar.zarr.
         Decides the frame-validity rule (all-NaN vs any-NaN vs threshold).
Part 2 — dataset audit: for each split, count samples whose *read* frames
         (context block + target frame) contain any NaN, via RadarDataset's
         real index list and window layout.

Pre-fix expectation: Part 2 counts > 0. Post-fix expectation: all zero.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.radar_dataset import RadarDataset, ZARR_PATH  # noqa: E402

import xarray as xr  # noqa: E402


def part1_frame_audit() -> None:
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    rain = ds["rain_rate"].values
    ds.close()
    T = rain.shape[0]
    nan_frac = np.isnan(rain).mean(axis=(1, 2))

    all_nan = int((nan_frac == 1.0).sum())
    any_nan = int((nan_frac > 0.0).sum())
    partial = nan_frac[(nan_frac > 0.0) & (nan_frac < 1.0)]
    print(f"frames total          : {T}")
    print(f"frames all-NaN        : {all_nan}")
    print(f"frames any-NaN        : {any_nan}")
    print(f"frames partial-NaN    : {len(partial)}")
    if len(partial):
        print(f"  partial NaN-fraction: min={partial.min():.4f} "
              f"median={np.median(partial):.4f} max={partial.max():.4f}")
        # Distribution shape: how many partials are 'nearly blank' vs 'nearly clean'
        for thr in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99):
            print(f"  partial frames with NaN-frac >= {thr}: {(partial >= thr).sum()}")


def part2_dataset_audit() -> None:
    for split in ("train", "val", "test"):
        dset = RadarDataset(split=split, heavy_rain_oversample=1)
        poisoned = 0
        for k in range(len(dset)):
            t = dset.indices[k]
            ctx = dset.rain[t - dset.context_frames: t]
            tgt = dset.rain[t + dset.target_offset]
            if np.isnan(ctx).any() or np.isnan(tgt).any():
                poisoned += 1
        print(f"{split:5s}: {poisoned}/{len(dset)} samples contain NaN")


if __name__ == "__main__":
    print("=== Part 1: per-frame NaN audit ===")
    part1_frame_audit()
    print("\n=== Part 2: NaN-poisoned samples per split ===")
    part2_dataset_audit()
