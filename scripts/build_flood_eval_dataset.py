#!/usr/bin/env python3
"""
Cross-reference PUB flood labels against the radar archive to build the
Stage 5 evaluation dataset.

For each flood label (from collect_flood_labels.py) this finds the nearest
radar frame in time and records the observed rain intensity at that frame.
The result is the ground-truth event table the nowcaster is scored against:
"when PUB reported a flood/risk, what did the radar actually show?"

Both clocks are UTC: flood_labels.parquet stores tz-aware UTC; radar.zarr
stores tz-naive UTC (scrape_radar.py names each PNG by its UTC instant and
only shifts +8h to build NEA's SGT-labelled URL). The join is therefore
direct, with no timezone offset.

Usage:
    python scripts/build_flood_eval_dataset.py                 # build/refresh
    python scripts/build_flood_eval_dataset.py --tolerance-min 10
    python scripts/build_flood_eval_dataset.py --status        # report, no write
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data" / "processed" / "flood_labels.parquet"
ZARR_PATH = ROOT / "data" / "processed" / "radar.zarr"
GEO_CACHE_PATH = ROOT / "data" / "processed" / "geocode_cache.json"
OUT_PATH = ROOT / "data" / "processed" / "flood_eval_dataset.parquet"

# A label is considered "matched" if a radar frame exists within this window.
DEFAULT_TOLERANCE_MIN = 15

# Event types that represent an actual/imminent flood (vs. generic rain warnings).
FLOOD_EVENT_TYPES = {"FLASH_FLOOD", "FLOOD_RISK"}


def load_radar_times() -> pd.DatetimeIndex:
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    times = pd.DatetimeIndex(ds.time.values).sort_values()
    ds.close()
    return times


def radar_frame_metrics(frame_index: int, cell: tuple[int, int] | None,
                        radius: int) -> dict:
    """Grid-wide max/mean (mm/hr) at a radar frame, plus the value at one cell
    and the max within a +/-radius-cell neighbourhood of it.

    A single 0.3 km cell is too brittle for point matching (radar pixel
    registration + the flood site rarely sits exactly on the rain peak), so
    the neighbourhood max is the meteorologically meaningful spatial value.
    """
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    frame = ds.rain_rate.isel(time=frame_index).values
    ds.close()
    out = {"max": float(np.nanmax(frame)), "mean": float(np.nanmean(frame)),
           "cell": np.nan, "near": np.nan}
    if cell is not None:
        li, lo = cell
        if 0 <= li < frame.shape[0] and 0 <= lo < frame.shape[1]:
            out["cell"] = float(frame[li, lo])
            sub = frame[max(0, li - radius): li + radius + 1,
                        max(0, lo - radius): lo + radius + 1]
            out["near"] = float(np.nanmax(sub))
    return out


def load_geocode() -> dict:
    """Map raw location string -> geocode entry (lat/lon/idx), if cache exists."""
    if GEO_CACHE_PATH.exists():
        return json.loads(GEO_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def first_location(raw) -> str | None:
    locs = json.loads(raw) if isinstance(raw, str) else (raw or [])
    for loc in locs:
        if loc and loc.strip():
            return loc.strip()
    return None


def build(tolerance_min: int, cell_radius: int) -> pd.DataFrame:
    labels = pd.read_parquet(LABELS_PATH)
    # Normalise event time to tz-naive UTC to match the radar index.
    ev = pd.to_datetime(labels["event_datetime"], utc=True).dt.tz_convert(None)
    labels = labels.assign(_event_utc=ev).sort_values("_event_utc").reset_index(drop=True)

    radar_times = load_radar_times()
    radar_vals = radar_times.values  # numpy datetime64, sorted
    tol = pd.Timedelta(minutes=tolerance_min)
    geo = load_geocode()

    rows = []
    for _, lab in labels.iterrows():
        event = lab["_event_utc"]
        # Nearest radar frame by absolute time delta.
        pos = int(np.searchsorted(radar_vals, np.datetime64(event)))
        candidates = [p for p in (pos - 1, pos) if 0 <= p < len(radar_times)]
        best_idx, best_delta = None, None
        for p in candidates:
            delta = radar_times[p] - event
            if best_delta is None or abs(delta) < abs(best_delta):
                best_idx, best_delta = p, delta

        matched = best_idx is not None and abs(best_delta) <= tol

        # Spatial: resolve this label's location to a radar cell (if geocoded).
        loc_str = first_location(lab["locations"])
        g = geo.get(loc_str) if loc_str else None
        has_cell = bool(g and g.get("lat") is not None)
        cell = (g["lat_idx"], g["lon_idx"]) if has_cell else None

        if matched:
            m = radar_frame_metrics(best_idx, cell, cell_radius)
            matched_time = radar_times[best_idx]
            delta_min = best_delta / pd.Timedelta(minutes=1)
        else:
            m = {"max": np.nan, "mean": np.nan, "cell": np.nan, "near": np.nan}
            matched_time = pd.NaT
            delta_min = np.nan

        rows.append({
            "message_id": lab["message_id"],
            "event_datetime": event,
            "message_type": lab["message_type"],
            "is_flood_event": lab["message_type"] in FLOOD_EVENT_TYPES,
            "locations": lab["locations"],
            "matched": bool(matched),
            "matched_radar_time": matched_time,
            "delta_min": round(delta_min, 1) if matched else np.nan,
            "radar_max_rain_mmhr": round(m["max"], 2) if matched else np.nan,
            "radar_mean_rain_mmhr": round(m["mean"], 4) if matched else np.nan,
            "radar_frame_index": best_idx if matched else -1,
            # Spatial (from geocode_flood_labels.py); NaN/-1 when not geocoded.
            "location_str": loc_str,
            "geocoded": has_cell,
            "location_lat": g["lat"] if has_cell else np.nan,
            "location_lon": g["lon"] if has_cell else np.nan,
            "location_lat_idx": g["lat_idx"] if has_cell else -1,
            "location_lon_idx": g["lon_idx"] if has_cell else -1,
            "radar_rain_at_cell_mmhr": round(m["cell"], 2) if (matched and has_cell) else np.nan,
            "radar_rain_near_cell_mmhr": round(m["near"], 2) if (matched and has_cell) else np.nan,
            "cell_radius": cell_radius,
        })

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, tolerance_min: int) -> None:
    n = len(df)
    n_flood = int(df["is_flood_event"].sum())
    n_matched = int(df["matched"].sum())
    n_flood_matched = int((df["is_flood_event"] & df["matched"]).sum())
    n_geo = int(df.get("geocoded", pd.Series(dtype=bool)).sum()) if "geocoded" in df else 0
    print(f"Tolerance        : +/- {tolerance_min} min")
    print(f"Total labels     : {n}")
    print(f"  flood events   : {n_flood}  (FLASH_FLOOD / FLOOD_RISK)")
    print(f"Matched to radar : {n_matched}/{n}   (flood events: {n_flood_matched}/{n_flood})")
    print(f"Geocoded (spatial): {n_geo}   "
          f"(run geocode_flood_labels.py to add/refresh)")
    if n_matched:
        mm = df.loc[df["matched"], "radar_max_rain_mmhr"]
        print(f"Radar max rain at matched events: "
              f"min={mm.min():.1f}  median={mm.median():.1f}  max={mm.max():.1f} mm/hr")
    spatial = df[df.get("geocoded", False) & df["matched"]] if "geocoded" in df else df.iloc[0:0]
    if len(spatial):
        rad = int(spatial["cell_radius"].iloc[0])
        print(f"Rain at geocoded flood cells (cell value | neighbourhood max r={rad}):")
        for _, r in spatial.iterrows():
            print(f"  {r['event_datetime']}  {str(r['location_str'])[:42]:<42} "
                  f"cell({r['location_lat_idx']},{r['location_lon_idx']}) "
                  f"= {r['radar_rain_at_cell_mmhr']} | {r['radar_rain_near_cell_mmhr']} mm/hr")
    unmatched = df.loc[~df["matched"], "event_datetime"]
    if len(unmatched):
        print(f"Unmatched (radar gap) : {len(unmatched)}")
        for t in unmatched:
            print(f"  {t}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 5 flood evaluation dataset")
    parser.add_argument("--tolerance-min", type=int, default=DEFAULT_TOLERANCE_MIN,
                        help="Max minutes between a label and the nearest radar frame")
    parser.add_argument("--cell-radius", type=int, default=5,
                        help="Neighbourhood radius (cells) for spatial rain max (~0.3km/cell)")
    parser.add_argument("--status", action="store_true",
                        help="Report the existing dataset without rebuilding")
    args = parser.parse_args()

    if args.status:
        if not OUT_PATH.exists():
            print(f"No dataset yet at {OUT_PATH.relative_to(ROOT)}. Run without --status to build.")
            return
        df = pd.read_parquet(OUT_PATH)
        print_summary(df, args.tolerance_min)
        return

    if not LABELS_PATH.exists():
        print(f"ERROR: {LABELS_PATH.relative_to(ROOT)} not found. "
              f"Run scripts/collect_flood_labels.py first.")
        sys.exit(1)
    if not ZARR_PATH.exists():
        print(f"ERROR: {ZARR_PATH.relative_to(ROOT)} not found. "
              f"Run scripts/preprocess_radar.py first.")
        sys.exit(1)

    df = build(args.tolerance_min, args.cell_radius)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_PATH.relative_to(ROOT)}\n")
    print_summary(df, args.tolerance_min)


if __name__ == "__main__":
    main()
