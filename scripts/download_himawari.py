#!/usr/bin/env python3
"""
download_himawari.py -- Stream Himawari-9 infrared strips and keep only the Singapore crop
(tasks/plan_satellite.md; src/data/satellite.py).

One file per band per UTC day: data/processed/sat/<band>/<YYYYMMDD>.npz with
    times  scan start times (datetime64[m]) actually present
    bt     (n, 111, 111) float16 brightness temperature, K
    missing  scan start times the archive does not have
A finished day is skipped, so an interrupted run resumes by re-running the same command
(the day in progress is redone, ~10 min). Raw strips never touch the disk except as a
temporary file that is deleted immediately.

Usage:  python scripts/download_himawari.py --start 2026-09-02 --end 2026-09-25 [--bands B13 B08]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.satellite import BANDS, SCAN, crop, day_path, fetch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="first UTC day, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="last UTC day, inclusive")
    ap.add_argument("--bands", nargs="+", default=list(BANDS))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    days = np.arange(np.datetime64(args.start, "D"), np.datetime64(args.end, "D") + 1)
    pool = ThreadPoolExecutor(args.workers)
    for day in days:
        for band in args.bands:
            out = day_path(band, str(day).replace("-", ""))
            if out.exists():
                continue
            t0 = time.time()
            slots = np.arange(np.datetime64(day, "m"), np.datetime64(day + 1, "m"), SCAN)
            raws = pool.map(lambda t: fetch(t, band), slots)            # downloads overlap the reads
            times, bts, missing = [], [], []
            for t, raw in zip(slots, raws):
                if raw is None:
                    missing.append(t)
                    continue
                times.append(t)
                bts.append(crop(raw, t, band).astype(np.float16))
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, times=np.array(times, "datetime64[m]"),
                                bt=np.stack(bts) if bts else np.empty((0, 111, 111), np.float16),
                                missing=np.array(missing, "datetime64[m]"))
            tmp.replace(out)
            print(f"{day} {band}: {len(times)} scans, {len(missing)} missing, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
