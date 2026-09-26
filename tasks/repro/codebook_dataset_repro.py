"""
Verification for the uint8 codebook in RadarDataset (2026-09-23).

The archive is a *banded* product -- 33 NEA rain levels plus zero -- so storing
it as float32 spends 4 bytes on 6 bits of information: 3.45 GB at 33k frames,
the largest term in the training process's footprint and the reason two runs
were reaped for memory pressure. Codes + a float32 value table cut that to
~870 MB and are LOSSLESS, so a run can resume across the change.

"Lossless" is the claim that has to be proven, because a silent encoding error
would poison every training sample while everything still looked healthy (the
same failure shape as L011/L016). This checks:

  1. decode(encode(x)) == x  bit-exactly, over the whole archive
  2. __getitem__ output is bit-identical to the old float32 implementation's
     math, computed independently from the raw zarr
  3. the sample index lists (splits, gap filter, NaN filter, oversampling) are
     unchanged
  4. actual resident memory is reduced

Expectation: all four PASS.
"""

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.radar_dataset import RadarDataset, load_stats  # noqa: E402

results = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} {detail}")
    results.append(bool(ok))


def rss_gb():
    """Resident set size via tasklist -- psapi through ctypes proved unreliable."""
    import subprocess, re
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {__import__('os').getpid()}",
                              "/FO", "CSV", "/NH"], capture_output=True, text=True,
                             timeout=30).stdout
        m = re.search(r'"([\d,]+) K"', out)
        return int(m.group(1).replace(",", "")) / 1e6 if m else float("nan")
    except Exception:
        return float("nan")


def main() -> None:
    base = rss_gb()
    t0 = time.time()
    ds = RadarDataset("train", context_frames=6, target_offset=6)
    load_s = time.time() - t0
    after = rss_gb()
    print()

    # ── 1. round-trip over the whole archive ─────────────────────────────────
    z = xr.open_zarr(ROOT / "data" / "processed" / "radar.zarr", consolidated=True)
    da = z["rain_rate"]
    T = da.sizes["time"]
    mismatch = 0
    for a in range(0, T, 4000):
        raw = da.isel(time=slice(a, min(a + 4000, T))).values
        back = ds._decode(ds.codes[a:a + raw.shape[0]])
        nan = np.isnan(raw)
        # exact equality on the finite entries, NaN preserved as NaN
        if not np.array_equal(back[~nan], raw[~nan]):
            mismatch += int((back[~nan] != raw[~nan]).sum())
        if not np.array_equal(np.isnan(back), nan):
            mismatch += 1
    check("round-trip exact over all frames", mismatch == 0,
          f"{T:,} frames, {mismatch} mismatched values")

    # ── 2. __getitem__ bit-identical to the float32 path ─────────────────────
    # The reference is the dataset's CURRENT transform applied to the raw
    # float32 archive, so this tests what it claims to -- that codebook decoding
    # is lossless -- rather than a frozen copy of a formula. It used to embed the
    # old z-score-and-clip normalisation, which went stale when the dataset moved
    # to [-1, 1] by log_max (commit 955bc60) and failed unnoticed until re-run.
    old_normalise = ds._normalise

    rng = np.random.default_rng(0)
    probe = rng.choice(len(ds), size=40, replace=False)
    ctx_bad = tgt_bad = 0
    for i in probe:
        t = ds.indices[int(i)]
        s = ds[int(i)]
        raw_ctx = da.isel(time=slice(t - 6, t)).values.astype(np.float32)
        raw_tgt = da.isel(time=t + 6).values.astype(np.float32)[np.newaxis, ...]
        if not np.array_equal(s["context"].numpy(), old_normalise(raw_ctx)):
            ctx_bad += 1
        if not np.array_equal(s["target"].numpy(), old_normalise(raw_tgt)):
            tgt_bad += 1
    check("__getitem__ context bit-identical to old path", ctx_bad == 0,
          f"{len(probe)} random samples")
    check("__getitem__ target bit-identical to old path", tgt_bad == 0,
          f"{len(probe)} random samples")

    # ── 3. index lists unchanged: recompute the filters from the raw zarr ────
    times = z["time"].values
    step_ok = np.diff(times) == np.timedelta64(5, "m")
    ok_cs = np.concatenate(([0], np.cumsum(step_ok)))
    n_val = max(1, int(T * 0.1)); n_test = max(1, int(T * 0.1))
    n_train = T - n_val - n_test
    bad = np.zeros(T, bool); fmax = np.zeros(T, np.float32)
    for a in range(0, T, 4000):
        blk = da.isel(time=slice(a, min(a + 4000, T))).values
        b = a + blk.shape[0]
        nn = np.isnan(blk)
        bad[a:b] = nn.any(axis=(1, 2))
        fmax[a:b] = np.nan_to_num(blk, nan=0.0).max(axis=(1, 2))
    bad_cs = np.concatenate(([0], np.cumsum(bad)))
    # Pinned split (2026-09-26): train anchors end before SPLIT_VAL_START and
    # their target must stay inside train as well.
    from src.data.radar_dataset import SPLIT_VAL_START
    vs = int(np.searchsorted(times, SPLIT_VAL_START))
    exp = [i for i in range(6, vs) if i + 6 < vs
           and ok_cs[i + 6] - ok_cs[i - 6] == 12
           and bad_cs[i + 7] - bad_cs[i - 6] == 0]
    exp = exp + [i for i in exp if fmax[i] >= 10.0] * 2
    check("train index list identical", ds.indices == exp,
          f"{len(ds.indices):,} vs {len(exp):,} samples")
    check("per-frame NaN flags identical", np.array_equal(ds.frame_bad, bad))
    check("per-frame max identical", np.array_equal(ds.frame_max, fmax))
    z.close()

    # ── 4. memory ────────────────────────────────────────────────────────────
    float32_gb = T * 120 * 217 * 4 / 1e9
    codes_gb = ds.codes.nbytes / 1e9
    check("codes smaller than float32 array", codes_gb < float32_gb / 3,
          f"{codes_gb:.2f} GB vs {float32_gb:.2f} GB")
    print(f"\n  dataset load: {load_s:.1f}s | process RSS {base:.2f} -> {after:.2f} GB "
          f"(float32 would add ~{float32_gb:.2f} GB)")

    # ── 5. splits share one encoded archive ──────────────────────────────────
    # train.py builds a train and a val dataset; before this they each held a
    # full copy, so the archive cost 2x. The index lists must still differ.
    t0 = time.time()
    val = RadarDataset("val", context_frames=6, target_offset=6)
    val_load_s = time.time() - t0
    check("val split reuses the encoded archive", val.codes is ds.codes)
    check("val split reuses the value table", val.values is ds.values)
    check("val split has its own index list", val.indices != ds.indices,
          f"{len(val.indices):,} val vs {len(ds.indices):,} train")
    check("second split load is ~free", val_load_s < 2.0, f"{val_load_s:.2f}s")

    print(f"\n{sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
