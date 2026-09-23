"""
radar_dataset.py — PyTorch Dataset wrapping the radar zarr archive.

Each sample:
- context : (context_frames, H, W) float32 — past radar frames, normalised
- target  : (1, H, W) float32           — future radar frame, normalised

Normalisation: rain rates are log1p-transformed then mapped to [-1, 1]
using dataset-level statistics. This handles the heavy-tailed distribution
of precipitation (most pixels are zero; few are very large values).

Storage: the archive is held in RAM as uint8 codes into a float32 value table
rather than as a float32 array — the product is banded (33 NEA levels plus
zero), so this is lossless and 4x smaller, and one copy is shared by every
split. See _load_codes() and _attach_shared(); verified by
tasks/repro/codebook_dataset_repro.py.
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

# One encoded archive per process, shared by every split (see _attach_shared).
# Keyed on (resolved zarr path, frame count).
_CODE_CACHE: dict[tuple[str, int], tuple] = {}


def compute_stats(zarr_path: Path = ZARR_PATH, save_path: Path = STATS_PATH) -> dict:
    """Compute dataset-level log1p statistics and save to JSON.

    Streamed in blocks rather than materialising the whole archive: `.values`
    on the full array is 3.45 GB at 33k frames, and this may run on a machine
    that is already tight (see RadarDataset._load_codes).
    """
    ds = xr.open_zarr(zarr_path, consolidated=True)
    da = ds["rain_rate"]
    T = int(da.sizes["time"])

    # Mean and std from running sums, so no full copy is ever held. nan-aware:
    # the archive can contain blank (all-NaN) frames from failed scrapes.
    n = 0
    total = 0.0
    total_sq = 0.0
    log_max = -np.inf
    for a in range(0, T, 3000):
        blk = np.log1p(da.isel(time=slice(a, min(a + 3000, T))).values)
        finite = blk[~np.isnan(blk)].astype(np.float64)
        if finite.size:
            n += finite.size
            total += float(finite.sum())
            total_sq += float(np.square(finite).sum())
            log_max = max(log_max, float(finite.max()))
    ds.close()

    mean = total / max(n, 1)
    stats = {
        "log_mean": mean,
        "log_std": float(np.sqrt(max(total_sq / max(n, 1) - mean * mean, 0.0))),
        "log_max": float(log_max),
        "n_samples": T,
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
        times = ds["time"].values  # datetime64, sorted (enforced at ingest)
        # Codebook storage instead of a float32 array: see _load_codes(). Sets
        # self.codes (uint8), self.values (float32 table), self.frame_bad and
        # self.frame_max.
        #
        # Shared across splits: train.py builds a "train" and a "val" dataset
        # over the same archive, and each used to load its own full copy -- 2 x
        # 3.46 GB, which is essentially the whole ~7 GB that got two runs reaped.
        # The arrays are read-only after load, so one copy serves every split.
        # Keyed on (path, frame count) so a grown archive is never served stale.
        self._attach_shared(zarr_path, ds["rain_rate"])
        ds.close()

        T = self.codes.shape[0]
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

        # Blank-frame filtering: failed scrapes leave all-NaN frames at valid
        # timestamps, so they pass the contiguity check above but would feed
        # NaN into training. A frame with any NaN is unusable; keep a sample
        # only if every frame in its full span [t - context_frames,
        # t + target_offset] is NaN-free (symmetric with _contiguous).
        frame_bad = self.frame_bad  # computed during the encoding pass
        bad_cumsum = np.concatenate(([0], np.cumsum(frame_bad)))

        def _clean(t: int) -> bool:
            a, b = t - context_frames, t + target_offset
            return bad_cumsum[b + 1] - bad_cumsum[a] == 0

        n_before = len(self.indices)
        contiguous = [i for i in self.indices if _contiguous(i)]
        n_gap = n_before - len(contiguous)
        self.indices = [i for i in contiguous if _clean(i)]
        n_nan = len(contiguous) - len(self.indices)
        if n_gap or n_nan:
            print(f"RadarDataset[{split}]: dropped {n_gap}/{n_before} samples "
                  f"spanning archive gaps and {n_nan} containing blank (NaN) "
                  f"frames ({len(self.indices)} remain)")

        if heavy_rain_oversample > 1 and split == "train":
            heavy = [i for i in self.indices
                     if self.frame_max[i] >= 10.0]
            self.indices = self.indices + heavy * (heavy_rain_oversample - 1)

        # Normalisation stats
        stats = load_stats()
        self.log_mean = stats["log_mean"]
        self.log_std = max(stats["log_std"], 1e-6)

    # Reserved code for blank (all-NaN) frames from failed scrapes. Real codes
    # start at 0, so 255 can never collide while there are <=255 rain levels.
    NAN_CODE = 255
    # 1000, not 3000: the transient during encoding is what matters, not the
    # read size. np.searchsorted returns int64, so a 3000-frame block costs
    # ~625 MB of intermediate on top of the block itself -- measured a 3.17 GB
    # peak, which partly defeats the point of shrinking the archive. 1000
    # frames keeps the peak near the steady state. Zarr chunks are 288 frames.
    _BLOCK = 1000

    def _attach_shared(self, zarr_path, da) -> None:
        """Reuse an already-encoded archive if this process has one."""
        key = (str(Path(zarr_path).resolve()), int(da.sizes["time"]))
        cached = _CODE_CACHE.get(key)
        if cached is None:
            self._load_codes(da)
            _CODE_CACHE.clear()          # only ever one archive in play
            _CODE_CACHE[key] = (self.codes, self.values,
                                self.frame_bad, self.frame_max)
        else:
            self.codes, self.values, self.frame_bad, self.frame_max = cached

    def _load_codes(self, da) -> None:
        """Load the archive as uint8 codes into a float32 value table.

        The radar product is *banded*: every pixel is one of NEA's 33 discrete
        rain-rate levels, plus zero (verified 2026-09-23: exactly 34 distinct
        values archive-wide). Holding it as float32 therefore wastes 4 bytes to
        store 6 bits of real information -- 3.45 GB at 33k frames, which is the
        single largest term in this process's footprint and what got two
        training runs reaped for memory pressure.

        Storing codes instead is **lossless**, not approximate: the value table
        holds the original float32 bit patterns, so reconstruction is exact and
        normalised output is bit-identical to the previous implementation. That
        matters -- it means a run can resume across this change without its
        inputs shifting.

        Cost: one extra pass to discover the distinct values (~7 s for 33k
        frames). Falls back to float16 if the archive somehow exceeds 255
        levels, so an unexpected ingest degrades precision rather than crashing.
        """
        T = da.sizes["time"]

        # Pass 1 — discover the value set, per-frame NaN flags and per-frame max
        # (the latter two are needed by the filters below and would otherwise
        # each require their own full-array scan).
        uniq: set[float] = set()
        frame_bad = np.zeros(T, dtype=bool)
        frame_max = np.zeros(T, dtype=np.float32)
        for a in range(0, T, self._BLOCK):
            blk = da.isel(time=slice(a, min(a + self._BLOCK, T))).values
            b = a + blk.shape[0]
            nan = np.isnan(blk)
            frame_bad[a:b] = nan.any(axis=(1, 2))
            frame_max[a:b] = np.nan_to_num(blk, nan=0.0).max(axis=(1, 2))
            uniq.update(float(v) for v in np.unique(blk[~nan]))
        self.frame_bad = frame_bad
        self.frame_max = frame_max

        if len(uniq) > self.NAN_CODE:
            # Not expected for a banded product; keep training possible anyway.
            print(f"RadarDataset: {len(uniq)} distinct values exceeds the "
                  f"{self.NAN_CODE}-level codebook; falling back to float16 "
                  f"storage (lossy)")
            self.values = None
            self.codes = np.empty((T, da.sizes["lat"], da.sizes["lon"]), np.float16)
            for a in range(0, T, self._BLOCK):
                blk = da.isel(time=slice(a, min(a + self._BLOCK, T))).values
                self.codes[a:a + blk.shape[0]] = blk.astype(np.float16)
            return

        # Pass 2 — encode. searchsorted is exact here because every value in the
        # array is a member of the table by construction.
        self.values = np.array(sorted(uniq), dtype=np.float32)
        self.codes = np.empty((T, da.sizes["lat"], da.sizes["lon"]), np.uint8)
        for a in range(0, T, self._BLOCK):
            blk = da.isel(time=slice(a, min(a + self._BLOCK, T))).values
            b = a + blk.shape[0]
            nan = np.isnan(blk)
            enc = np.searchsorted(self.values,
                                  np.nan_to_num(blk, nan=0.0)).astype(np.uint8)
            enc[nan] = self.NAN_CODE
            self.codes[a:b] = enc
            del enc

            # Verify the round-trip on the first block rather than trusting it:
            # a silent encoding error would poison every sample.
            if a == 0:
                back = self._decode(self.codes[a:b])
                ok = np.allclose(back[~nan], blk[~nan], rtol=0, atol=0)
                if not ok:
                    raise ValueError("RadarDataset: codebook round-trip is not "
                                     "exact; refusing to train on re-encoded data")

        mb_before = T * da.sizes["lat"] * da.sizes["lon"] * 4 / 1e6
        mb_after = self.codes.nbytes / 1e6
        print(f"RadarDataset: {len(uniq)} rain levels -> uint8 codebook, "
              f"{mb_after:,.0f} MB in RAM (was {mb_before:,.0f} MB as float32)")

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        """Codes -> float32 mm/hr. NaN code maps back to NaN."""
        if self.values is None:
            return codes.astype(np.float32)          # float16 fallback path
        out = self.values[np.minimum(codes, len(self.values) - 1)]
        return np.where(codes == self.NAN_CODE, np.float32("nan"), out)

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
        ctx_frames = self._decode(self.codes[t - self.context_frames: t])  # (C,H,W)
        # Target: frame at t + target_offset
        target_frame = self._decode(self.codes[t + self.target_offset])    # (H, W)

        context = self._normalise(ctx_frames)
        target = self._normalise(target_frame[np.newaxis, ...])  # (1, H, W)

        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
        }
