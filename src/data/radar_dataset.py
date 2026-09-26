"""
radar_dataset.py — PyTorch Dataset wrapping the radar zarr archive.

Each sample:
- context : (context_frames, H, W) float32 — past radar frames, normalised
- target  : (1, H, W) float32           — future radar frame, normalised

Normalisation: rain rates are log1p-transformed then mapped linearly to
[-1, 1] against log_max, with NO clipping. This handles the heavy-tailed
distribution of precipitation (most pixels are zero; few are very large)
while keeping every one of NEA's 33 rain levels distinct -- see normalise_rain()
for why the earlier z-score-and-clamp was fatal.

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

# Training crop. The rain field is spatially sparse -- 3.3% of pixels wet on a
# full 120x217 frame -- so a full-frame batch is mostly dry pixels no matter how
# heavily wet FRAMES are oversampled. Cropping around a raining pixel raises the
# wet fraction and cuts compute per step (measured, rain-centred):
#
#   full 120x217   3.31% wet   26,040 px   1.0x
#   96 x 96        5.18% wet    9,216 px   2.8x
#   64 x 64        7.95% wet    4,096 px   6.4x   <- chosen
#   48 x 48       10.14% wet    2,304 px  11.3x
#
# NOT ENABLED BY DEFAULT. Cropping was introduced for the same falsified
# class-imbalance diagnosis as intensity_alpha, and it turned out to be actively
# harmful: rain-centred crops are ~9% wet while full frames are ~0.3% wet, so the
# model learned a prior where rain is common and was then asked to generate
# almost-dry scenes. A 300k-step run with crops converged (LR 5.6e-15, last four
# checkpoints identical) to median -0.697 and 16.4% dry against a 99.7% dry
# target -- a stable wrong answer, and train/inference distribution mismatch is
# the most likely reason. Pass crop_size=64 explicitly to experiment.
# Must stay divisible by 8 (three downsamplings in ConditionedUNet).
CROP_SIZE = 64
# Fraction of training crops centred on a raining pixel; the rest are uniform so
# the model still sees fully dry scenes and domain edges.
CROP_RAIN_FRAC = 0.8

# Split boundaries, PINNED by date (2026-09-26). They used to be "last 10% =
# test, previous 10% = val" of an archive that grows every morning, so every
# split moved daily: evaluations from different days scored different test
# periods, and a cache silently compared yesterday's forecasts with today's
# observations (lesson L028). Fixing the dates makes every result reproducible
# and lets models trained on different days share one test period. Frames after
# SPLIT_TEST_END are excluded from all three splits -- a fresh holdout.
SPLIT_VAL_START = np.datetime64("2026-09-02T06:50")
SPLIT_TEST_START = np.datetime64("2026-09-14T03:20")
SPLIT_TEST_END = np.datetime64("2026-09-25T23:55")      # inclusive

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


# The model-input transforms, as plain functions so that live inference
# (src/inference/nowcast.py) builds its input with the SAME code as training,
# without encoding the whole archive. RadarDataset delegates to these.

def normalise_rain(rain: np.ndarray, log_max: float) -> np.ndarray:
    """log1p -> [-1, 1] against log_max. No clipping.

    The previous version z-scored and clamped to [-3, 3], which silently
    destroyed the entire signal of interest. This field is 97.5% zeros, so
    log_std is only 0.215 and 3 sigma reaches just **0.96 mm/hr** -- 2 mm/hr
    sits at 5 sigma, 20 at 14 sigma, 100 at 21.4 sigma. The result was that
    **30 of the 35 rain levels mapped to exactly +3.0**, 65.9% of all wet
    pixels were pinned at that ceiling, and the 35-level ramp collapsed to 6
    distinct values. A 300k-step run trained on that target collapsed to
    predicting the constant dry value (max output 0.006 mm/hr) and scored
    FSS 0.000 against persistence's 0.070 -- strictly worse than assuming
    the last frame persists.

    Mapping log1p(rain) linearly onto [-1, 1] by log_max keeps every level
    distinct, uses the full DDPM input range, and cannot clip: rain=0 -> -1,
    0.5 -> -0.824, 2 -> -0.524, 20 -> +0.32, 100 -> +1.
    """
    x = np.log1p(rain) / log_max               # [0, 1]
    return (2.0 * x - 1.0).astype(np.float32)  # [-1, 1]


def denormalise_rain(x: torch.Tensor, log_max: float) -> torch.Tensor:
    """Invert normalise_rain: normalised -> mm/hr."""
    log_rain = (x.float() + 1.0) * 0.5 * log_max
    return torch.expm1(log_rain.clamp(min=0))


def time_features(t_last: np.datetime64, h: int, w: int) -> np.ndarray:
    """sin/cos of Singapore local hour at the last observed frame, (2, h, w).

    Singapore convection is strongly diurnal -- afternoon sea-breeze storms --
    and the model had no clock at all. The evaluation showed its remaining
    error is where storms FORM and DIE, not how they move (a shift by the
    true displacement still lost to persistence), so the time of day is
    cheap information about exactly that. Uses the last OBSERVED frame, which
    is known at forecast time. Times are UTC; SGT = UTC + 8.
    """
    ts = np.datetime64(t_last).astype("datetime64[m]").astype(np.int64)  # minutes
    hour = ((ts / 60.0) + 8.0) % 24.0
    ang = 2.0 * np.pi * hour / 24.0
    out = np.empty((2, h, w), dtype=np.float32)
    out[0].fill(np.sin(ang))
    out[1].fill(np.cos(ang))
    return out


def build_context(frames: np.ndarray, t_last: np.datetime64, log_max: float,
                  time_channels: bool) -> np.ndarray:
    """The model's input from raw mm/hr frames (CF, h, w), oldest first: the
    normalised frames, with time channels PREPENDED when the model uses them.
    Order matters: context[-1] must stay the last radar frame, because the
    residual base and the persistence baseline both read it from there."""
    x = normalise_rain(frames, log_max)
    if not time_channels:
        return x
    return np.concatenate([time_features(t_last, x.shape[1], x.shape[2]), x], axis=0)


class RadarDataset(Dataset):
    """
    Parameters
    ----------
    split         : 'train', 'val', or 'test'
    context_frames: number of past frames to use as conditioning
    target_offset : how many steps ahead to predict
    zarr_path     : path to radar.zarr (default: data/processed/radar.zarr)
    val_frac, test_frac : ignored -- splits are pinned by date (see
                    SPLIT_VAL_START / SPLIT_TEST_START / SPLIT_TEST_END)
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
        crop_size: int | None = None,
        time_channels: bool = False,
    ):
        self.context_frames = context_frames
        self.target_offset = target_offset
        # Hour-of-day as two extra input channels (see _time_features).
        self.time_channels = bool(time_channels)
        # Rain-centred cropping is a TRAINING-ONLY device. val/test stay full
        # frame so metrics remain comparable across runs and match inference.
        self.crop_size = crop_size if split == "train" else None
        self.split = split
        self._rng = np.random.default_rng(1234)

        ds = xr.open_zarr(zarr_path, consolidated=True)
        times = ds["time"].values  # datetime64, sorted (enforced at ingest)
        self.times = times         # needed for time-of-day features
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
        vs = int(np.searchsorted(times, SPLIT_VAL_START))
        ts = int(np.searchsorted(times, SPLIT_TEST_START))
        te = int(np.searchsorted(times, SPLIT_TEST_END, side="right"))  # exclusive
        self.split_bounds = {"train": (0, vs), "val": (vs, ts), "test": (ts, te)}
        lo, hi = self.split_bounds[split]

        # Anchor t reads context t-CF..t-1 and TARGET t+target_offset. Keep an
        # anchor only if its target lies inside its own split: the old fraction
        # logic let training anchors near the boundary take their target from
        # the validation period.
        min_idx = max(context_frames, lo)
        self.indices = [i for i in range(min_idx, hi) if i + target_offset < hi]

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

        # Normalisation stats. Only log_max is used by the transform below --
        # deliberately, because log_max is the top of NEA's fixed 33-level ramp
        # (log1p(100)) and therefore does not drift as the archive grows, whereas
        # log_mean/log_std do. They are kept for reference and diagnostics.
        stats = load_stats()
        self.log_mean = stats["log_mean"]
        self.log_std = max(stats["log_std"], 1e-6)
        self.log_max = max(stats["log_max"], 1e-6)

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

    @property
    def in_channels(self) -> int:
        """Channels of the context tensor the model receives."""
        return self.context_frames + (2 if self.time_channels else 0)

    def _time_features(self, i_last: int, h: int, w: int) -> np.ndarray:
        """See time_features(); i_last is the archive index of the last observed frame."""
        return time_features(self.times[i_last], h, w)

    def _context(self, t: int, y0: int = 0, x0: int = 0,
                 c: int | None = None) -> np.ndarray:
        """The model's input for anchor t: frames t-CF..t-1, normalised, with
        optional time channels PREPENDED. Order matters: context[-1] must stay
        the last radar frame, because the residual base and the persistence
        baseline both read it from there."""
        if c is None:
            codes = self.codes[t - self.context_frames: t]
        else:
            codes = self.codes[t - self.context_frames: t, y0:y0 + c, x0:x0 + c]
        return build_context(self._decode(codes), self.times[t - 1],
                             self.log_max, self.time_channels)

    def context_at(self, t: int) -> torch.Tensor:
        """Full-frame model input for anchor t (last observed frame t-1). For
        callers that pick their own times, e.g. flood-event scoring -- so they
        cannot assemble a context that differs from training."""
        return torch.from_numpy(self._context(t))

    def _crop_origin(self, t: int) -> tuple[int, int]:
        """Top-left of the training crop; (0, 0) when cropping is off.

        Centred on a randomly chosen raining pixel of the TARGET frame with
        probability CROP_RAIN_FRAC, else uniform. Centring on the target (not the
        context) is deliberate: the target is what the loss is computed against,
        so that is where wet pixels need to land.
        """
        c = self.crop_size
        if c is None:
            return 0, 0
        H, W = self.codes.shape[1], self.codes.shape[2]
        cy = cx = None
        if self._rng.random() < CROP_RAIN_FRAC:
            tgt = self.codes[t + self.target_offset]
            ys, xs = np.nonzero(tgt > 0)          # code 0 is exactly 0.0 mm/hr
            if len(ys):
                j = int(self._rng.integers(len(ys)))
                cy, cx = int(ys[j]), int(xs[j])
        if cy is None:
            cy = int(self._rng.integers(H))
            cx = int(self._rng.integers(W))
        y0 = int(np.clip(cy - c // 2, 0, max(H - c, 0)))
        x0 = int(np.clip(cx - c // 2, 0, max(W - c, 0)))
        return y0, x0

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        """Codes -> float32 mm/hr. NaN code maps back to NaN."""
        if self.values is None:
            return codes.astype(np.float32)          # float16 fallback path
        out = self.values[np.minimum(codes, len(self.values) - 1)]
        return np.where(codes == self.NAN_CODE, np.float32("nan"), out)

    def __len__(self) -> int:
        return len(self.indices)

    def _normalise(self, rain: np.ndarray) -> np.ndarray:
        """See normalise_rain()."""
        return normalise_rain(rain, self.log_max)

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        """Invert normalisation: normalised -> mm/hr."""
        return denormalise_rain(x, self.log_max)

    def __getitem__(self, idx: int) -> dict:
        t = self.indices[idx]

        # Context: frames [t - context_frames, ..., t-1]
        y0, x0 = self._crop_origin(t)
        c = self.crop_size
        context = self._context(t, y0, x0, c)
        if c is None:
            tgt_codes = self.codes[t + self.target_offset]
        else:
            tgt_codes = self.codes[t + self.target_offset, y0:y0 + c, x0:x0 + c]
        target_frame = self._decode(tgt_codes)    # (h, w)
        target = self._normalise(target_frame[np.newaxis, ...])  # (1, H, W)

        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
        }
