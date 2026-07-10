"""
radar_dataset.py — PyTorch Dataset wrapping the radar zarr archive.

Each sample:
- context : (context_frames, H, W) float32 — past radar frames, normalised
- target  : (1, H, W) float32           — future radar frame, normalised

Normalisation: rain rates are log1p-transformed then mapped to [-1, 1]
using dataset-level statistics. This handles the heavy-tailed distribution
of precipitation (most pixels are zero; few are very large values).
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent.parent.parent
ZARR_PATH = ROOT / "data" / "processed" / "radar.zarr"
STATS_PATH = ROOT / "data" / "processed" / "radar_stats.json"

CONTEXT_FRAMES = 6    # 6 × 5 min = 30 min of history
TARGET_OFFSET = 6     # predict 6 steps ahead = +30 min (change to 12/18 for 60/90 min)


def compute_stats(zarr_path: Path = ZARR_PATH, save_path: Path = STATS_PATH) -> dict:
    """Compute dataset-level log1p statistics and save to JSON."""
    ds = xr.open_zarr(zarr_path, consolidated=True)
    rain = ds["rain_rate"].values  # (T, H, W)
    ds.close()

    log_rain = np.log1p(rain)
    stats = {
        "log_mean": float(log_rain.mean()),
        "log_std": float(log_rain.std()),
        "log_max": float(log_rain.max()),
        "n_samples": int(rain.shape[0]),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_stats() -> dict:
    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            return json.load(f)
    return compute_stats()


class RadarDataset(Dataset):
    """
    Parameters
    ----------
    split         : 'train', 'val', or 'test'
    context_frames: number of past frames to use as conditioning
    target_offset : how many steps ahead to predict
    zarr_path     : path to radar.zarr (default: data/processed/radar.zarr)
    val_frac      : fraction of data for validation (last N days)
    test_frac     : fraction of data for test (last M days)
    heavy_rain_oversample : factor to oversample heavy rain events (≥10 mm/hr)
    """

    def __init__(
        self,
        split: str = "train",
        context_frames: int = CONTEXT_FRAMES,
        target_offset: int = TARGET_OFFSET,
        zarr_path: Path = ZARR_PATH,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        heavy_rain_oversample: int = 3,
    ):
        self.context_frames = context_frames
        self.target_offset = target_offset

        ds = xr.open_zarr(zarr_path, consolidated=True)
        self.rain = ds["rain_rate"].values.astype(np.float32)  # (T, H, W)
        times = ds["time"].values  # datetime64, sorted (enforced at ingest)
        ds.close()

        T = self.rain.shape[0]
        n_val = max(1, int(T * val_frac))
        n_test = max(1, int(T * test_frac))
        n_train = T - n_val - n_test

        # Minimum index for a valid sample: need context_frames before + target_offset after
        min_idx = context_frames
        max_idx = T - target_offset - 1

        if split == "train":
            valid_range = range(min_idx, n_train)
        elif split == "val":
            valid_range = range(n_train, n_train + n_val)
        else:
            valid_range = range(n_train + n_val, T - target_offset)

        # Build index list; optionally oversample heavy-rain events
        self.indices = [i for i in valid_range if i <= max_idx]

        # Gap-aware filtering: __getitem__ assumes positional adjacency means
        # 5-min steps, but the archive has holes (missed scrapes). A sample at
        # t uses frames [t - context_frames, t + target_offset]; keep it only
        # if every step in that span is exactly 5 minutes, otherwise the
        # context/lead-time semantics are silently wrong.
        step_ok = np.diff(times) == np.timedelta64(5, "m")
        ok_cumsum = np.concatenate(([0], np.cumsum(step_ok)))

        def _contiguous(t: int) -> bool:
            a, b = t - context_frames, t + target_offset
            return ok_cumsum[b] - ok_cumsum[a] == b - a

        n_before = len(self.indices)
        self.indices = [i for i in self.indices if _contiguous(i)]
        n_dropped = n_before - len(self.indices)
        if n_dropped:
            print(f"RadarDataset[{split}]: dropped {n_dropped}/{n_before} samples "
                  f"spanning archive gaps ({len(self.indices)} remain)")

        if heavy_rain_oversample > 1 and split == "train":
            heavy = [i for i in self.indices
                     if self.rain[i].max() >= 10.0]
            self.indices = self.indices + heavy * (heavy_rain_oversample - 1)

        # Normalisation stats
        stats = load_stats()
        self.log_mean = stats["log_mean"]
        self.log_std = max(stats["log_std"], 1e-6)

    def __len__(self) -> int:
        return len(self.indices)

    def _normalise(self, rain: np.ndarray) -> np.ndarray:
        """log1p → zero-mean unit-variance → clamp to [-3, 3]."""
        x = np.log1p(rain)
        x = (x - self.log_mean) / self.log_std
        return np.clip(x, -3.0, 3.0)

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        """Invert normalisation: normalised → mm/hr."""
        log_rain = x.float() * self.log_std + self.log_mean
        return torch.expm1(log_rain.clamp(min=0))

    def __getitem__(self, idx: int) -> dict:
        t = self.indices[idx]

        # Context: frames [t - context_frames, ..., t-1]
        ctx_frames = self.rain[t - self.context_frames: t]       # (C, H, W)
        # Target: frame at t + target_offset
        target_frame = self.rain[t + self.target_offset]         # (H, W)

        context = self._normalise(ctx_frames)
        target = self._normalise(target_frame[np.newaxis, ...])  # (1, H, W)

        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
        }
