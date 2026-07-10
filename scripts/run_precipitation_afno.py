"""
run_precipitation_afno.py — Stage 2: PrecipitationAFNO hindcast over Singapore.

Uses earth2studio WB2ERA5 (global ERA5 from WeatherBench2 / Google Cloud) as input
to PrecipitationAFNO, then crops the 0.25-degree output to the Singapore bounding box.

Behaviour
---------
- Fetches global ERA5 from WB2ERA5 for each requested timestamp
- Runs PrecipitationAFNO inference (GPU if available, else CPU)
- Crops output to Singapore bbox and saves per-timestamp NetCDF under data/stage2/
- Skips timestamps already processed (resumable)
- Saves checkpoints/stage2_complete.flag when all timestamps are done

Usage
-----
    python scripts/run_precipitation_afno.py                      # default hindcast dates
    python scripts/run_precipitation_afno.py --dry-run            # print config, no fetch
    python scripts/run_precipitation_afno.py --dates 2022-04-22   # specific dates
    python scripts/run_precipitation_afno.py --status             # show progress
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

# Direct submodule imports bypass broken ecmwf/pyproj in the package __init__
from earth2studio.data.wb2 import WB2ERA5
from earth2studio.models.dx import PrecipitationAFNO

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "stage2"
CHECKPOINT_FLAG = ROOT / "checkpoints" / "stage2_complete.flag"

# Singapore bounding box (padded slightly for context)
SG_LAT_MIN, SG_LAT_MAX = 0.8, 1.6
SG_LON_MIN, SG_LON_MAX = 103.4, 104.2

# PrecipitationAFNO input variables (order matters — matches model checkpoint)
AFNO_VARS = [
    "u10m", "v10m", "t2m", "sp", "msl",
    "t850", "u1000", "v1000", "z1000",
    "u850", "v850", "z850",
    "u500", "v500", "z500", "t500",
    "z50", "r500", "r850", "tcwv",
]

# Default hindcast timestamps: known Singapore heavy-rain events within WB2ERA5 range
# WB2ERA5 covers 1959-01-01 to 2023-01-10 at 6-hourly intervals (UTC)
DEFAULT_DATES = [
    datetime(2022, 4, 22, 0),   # major flash flood event
    datetime(2022, 4, 22, 6),
    datetime(2022, 4, 22, 12),
    datetime(2022, 4, 22, 18),
    datetime(2021, 1,  7, 0),   # northeast monsoon heavy rain
    datetime(2021, 1,  7, 6),
    datetime(2021, 1,  7, 12),
    datetime(2021, 1,  7, 18),
    datetime(2020, 12, 27, 0),  # another monsoon event
    datetime(2020, 12, 27, 6),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument(
        "--dates", nargs="+", default=None,
        help="YYYY-MM-DD or YYYY-MM-DDTHH:MM — runs all four 6-hourly steps if only date given",
    )
    return p.parse_args()


def expand_dates(date_strs: list[str]) -> list[datetime]:
    out = []
    for s in date_strs:
        if "T" in s or len(s) > 10:
            out.append(datetime.fromisoformat(s))
        else:
            d = datetime.fromisoformat(s)
            out.extend([d + timedelta(hours=h) for h in (0, 6, 12, 18)])
    return out


def already_done(ts: datetime) -> bool:
    nc_path = OUT_DIR / f"precip_{ts.strftime('%Y%m%dT%H%M')}.nc"
    return nc_path.exists()


def show_status(timestamps: list[datetime]) -> None:
    done = [t for t in timestamps if already_done(t)]
    pending = [t for t in timestamps if not already_done(t)]
    print(f"Done:    {len(done)}/{len(timestamps)}")
    if pending:
        print("Pending:")
        for t in pending:
            print(f"  {t.isoformat()}")


def run_inference(timestamps: list[datetime], dry_run: bool) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if dry_run:
        print(f"[dry-run] Would process {len(timestamps)} timestamps:")
        for t in timestamps:
            status = "skip" if already_done(t) else "run"
            print(f"  [{status}] {t.isoformat()}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PrecipitationAFNO model...")
    package = PrecipitationAFNO.load_default_package()
    model = PrecipitationAFNO.load_model(package).to(device)
    model.eval()
    print("Model loaded.")

    # WB2ERA5 returns lat 90→-90 with 721 points (includes south pole).
    # PrecipitationAFNO needs 720 points (endpoint=False). We drop the last lat row.
    data_source = WB2ERA5(cache=True, verbose=True)

    for ts in timestamps:
        if already_done(ts):
            print(f"[skip] {ts.isoformat()} already processed")
            continue

        print(f"[fetch] {ts.isoformat()}")
        raw = data_source(ts, AFNO_VARS)  # shape: (1, 20, 721, 1440)

        # Trim WB2ERA5's extra south-pole lat row: 721 → 720
        raw_np = raw.values[:, :, :720, :]  # (1, 20, 720, 1440)

        x = torch.from_numpy(raw_np).float().to(device)

        # Build coord dict matching model.input_coords()
        import collections
        coords = collections.OrderedDict({
            "batch": np.array([np.datetime64(ts)]),
            "variable": np.array(AFNO_VARS),
            "lat": np.linspace(90, -90, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        })

        print(f"[infer] {ts.isoformat()}")
        with torch.inference_mode():
            out_tensor, out_coords = model(x, coords)

        # out_tensor: (1, 1, 720, 1440) — tp in metres
        tp_global = out_tensor.cpu().numpy()[0, 0]  # (720, 1440)

        lat_arr = out_coords["lat"]
        lon_arr = out_coords["lon"]

        # Crop to Singapore bbox
        lat_mask = (lat_arr >= SG_LAT_MIN) & (lat_arr <= SG_LAT_MAX)
        lon_mask = (lon_arr >= SG_LON_MIN) & (lon_arr <= SG_LON_MAX)
        tp_sg = tp_global[np.ix_(lat_mask, lon_mask)]

        ds = xr.Dataset(
            {"tp": (["lat", "lon"], tp_sg * 1000, {"units": "mm", "long_name": "Total precipitation (6-hourly)"})},
            coords={
                "lat": lat_arr[lat_mask],
                "lon": lon_arr[lon_mask],
                "time": ts,
            },
            attrs={
                "model": "PrecipitationAFNO",
                "source": "WB2ERA5",
                "valid_time": ts.isoformat(),
                "region": "Singapore",
                "resolution_deg": 0.25,
            },
        )

        nc_path = OUT_DIR / f"precip_{ts.strftime('%Y%m%dT%H%M')}.nc"
        ds.to_netcdf(nc_path)
        print(f"[saved] {nc_path.name}  max_tp={tp_sg.max()*1000:.2f} mm")

    # Write completion flag if all timestamps done
    remaining = [t for t in timestamps if not already_done(t)]
    if not remaining:
        CHECKPOINT_FLAG.parent.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FLAG.write_text(datetime.utcnow().isoformat())
        print(f"\nAll done. Flag written: {CHECKPOINT_FLAG}")


def main():
    args = parse_args()

    timestamps = expand_dates(args.dates) if args.dates else DEFAULT_DATES

    if args.status:
        show_status(timestamps)
        return

    pending = [t for t in timestamps if not already_done(t)]
    print(f"Timestamps: {len(timestamps)} total, {len(pending)} pending")

    if not pending and not args.dry_run:
        print("Nothing to do.")
        return

    run_inference(timestamps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
