"""
preprocess_radar.py — Convert archived NEA radar PNGs to a quantitative zarr dataset.

Behaviour
---------
- Reads PNG files from data/raw/radar/
- Converts colour-coded rainfall intensity to mm/hr using NEA's colour scale
- Crops to Singapore domain and stores as data/processed/radar.zarr
- Already-processed timestamps are detected by zarr chunk existence → skipped
- Safe to re-run (idempotent)

Usage
-----
    python scripts/preprocess_radar.py                  # process all new PNGs
    python scripts/preprocess_radar.py --status         # report coverage
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parent.parent
RADAR_DIR = ROOT / "data" / "raw" / "radar"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ROOT / "checkpoints"

console = Console()

ZARR_PATH = OUTPUT_DIR / "radar.zarr"

# ---------------------------------------------------------------------------
# NEA rain-area colour scale
# ---------------------------------------------------------------------------
# The NEA rain-area product ("dpsri ... dBR") renders rainfall as a *banded*
# overlay using exactly 33 discrete colours. Pixels are either fully opaque
# (alpha=255, rain) or fully transparent (alpha=0, no rain) — never blended.
#
# ORDERING (high confidence): the light→heavy sequence below was derived from
# the archive itself, by three independent methods that agree exactly:
#   1. Distance-transform depth — each band's mean depth inside a rain cell
#      increases monotonically (1.43 → 12.36 px) for the outer bands.
#   2. Spatial adjacency — a greedy walk over the band-adjacency graph
#      recovers this exact chain; adjacency counts fall away monotonically
#      toward the heavy end (287k → 473), the signature of a banded scale.
#   3. RGB ramp continuity — the sequence is a continuous path in colour space.
#
# ABSOLUTE VALUES (approximate): NEA publishes no per-colour mm/hr table, only
# band descriptions (light 0.5–2, moderate 2–10, heavy 10–30, intense >30).
# The product is dBR (decibels of rain rate, 10·log10 R), so equal-width dBR
# bands are log-uniform in mm/hr — values below are log-uniform from 0.5 to
# 100 mm/hr, i.e. 0.719 dBR per band. The resulting families land where NEA
# describes them: teal+early green = light, green/yellow = moderate,
# orange = heavy, red/magenta = intense.
#
# Suitable for training (monotonic and self-consistent). NOT calibrated for
# absolute flood thresholds — that needs regression against data.gov.sg rain
# gauges, which can be done later without another re-ingest.
NEA_COLOUR_RAMP: list[tuple[tuple[int, int, int], float]] = [
    ((  0, 255, 255),   0.50),   #  0   -3.01 dBR
    ((  0, 239, 239),   0.59),   #  1   -2.29 dBR
    ((  0, 209, 213),   0.70),   #  2   -1.57 dBR
    ((  0, 186, 191),   0.82),   #  3   -0.85 dBR
    ((  0, 151, 154),   0.97),   #  4   -0.13 dBR
    ((  0, 131, 125),   1.14),   #  5    0.59 dBR
    ((  0, 128,  69),   1.35),   #  6    1.30 dBR
    ((  0, 137,  56),   1.59),   #  7    2.02 dBR
    ((  0, 162,  53),   1.88),   #  8    2.74 dBR
    ((  0, 183,  41),   2.22),   #  9    3.46 dBR
    ((  0, 202,  17),   2.62),   # 10    4.18 dBR
    ((  0, 218,  13),   3.09),   # 11    4.90 dBR
    ((  0, 245,   7),   3.65),   # 12    5.62 dBR
    ((  0, 255,   0),   4.30),   # 13    6.34 dBR
    (( 67, 255,  65),   5.08),   # 14    7.06 dBR
    (( 72, 255,  70),   5.99),   # 15    7.78 dBR
    ((255, 255,  59),   7.07),   # 16    8.49 dBR
    ((255, 255,   0),   8.34),   # 17    9.21 dBR
    ((255, 240,   0),   9.85),   # 18    9.93 dBR
    ((255, 220,   0),  11.62),   # 19   10.65 dBR
    ((255, 198,   0),  13.71),   # 20   11.37 dBR
    ((255, 178,   0),  16.18),   # 21   12.09 dBR
    ((255, 165,   0),  19.10),   # 22   12.81 dBR
    ((255, 138,   0),  22.53),   # 23   13.53 dBR
    ((255, 114,   0),  26.59),   # 24   14.25 dBR
    ((255,  73,   0),  31.38),   # 25   14.97 dBR
    ((255,  31,   0),  37.03),   # 26   15.69 dBR
    ((229,   0,   0),  43.70),   # 27   16.40 dBR
    ((193,   0,   0),  51.57),   # 28   17.12 dBR
    ((182,   0, 106),  60.85),   # 29   17.84 dBR
    ((210,   0, 165),  71.81),   # 30   18.56 dBR
    ((212,   0, 170),  84.74),   # 31   19.28 dBR
    ((255,   0, 255), 100.00),   # 32   20.00 dBR
]

# Exact-match lookup keyed by packed RGB. Sorted so searchsorted() can be used;
# a colour that is not in the table is NEVER snapped to a neighbour — it is
# counted in UNKNOWN_COLOURS and mapped to 0.0 so it shows up in validation
# instead of silently becoming plausible-looking rain.
_PACKED = np.array([(r << 16) | (g << 8) | b for (r, g, b), _ in NEA_COLOUR_RAMP],
                   dtype=np.int64)
_ORDER = np.argsort(_PACKED)
_LUT_KEYS = _PACKED[_ORDER]
_LUT_VALS = np.array([v for _, v in NEA_COLOUR_RAMP], dtype=np.float32)[_ORDER]

# Accumulates {packed_rgb: count} for opaque pixels not in the ramp (diagnostic).
UNKNOWN_COLOURS: dict[int, int] = {}

# ---------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------
# NEA rain-area PNGs are 217×120 and cover a fixed box. Bounds are from the
# weather.gov.sg page source (map_latitude_top / _bottom, map_longitude_left /
# _right) and match the values used by cheeaun/checkweather-sg against these
# same images. Internal cross-check: 0.002602°/px in lat vs 0.002604°/px in
# lon — square pixels, ~0.29 km, which is what the NEA product documents.
# The whole domain is kept: no crop and no resampling. Rain advecting in from
# outside the coastline is exactly what a nowcaster needs.
RADAR_LAT_TOP = 1.4572
RADAR_LAT_BOTTOM = 1.1450
RADAR_LON_LEFT = 103.565
RADAR_LON_RIGHT = 104.130
RADAR_SHAPE = (120, 217)  # (rows/lat, cols/lon) — native PNG size

# Pixel *centres*, so a coordinate lookup lands on the cell that contains it.
_DLAT = (RADAR_LAT_TOP - RADAR_LAT_BOTTOM) / RADAR_SHAPE[0]
_DLON = (RADAR_LON_RIGHT - RADAR_LON_LEFT) / RADAR_SHAPE[1]
RADAR_LATS = RADAR_LAT_TOP - _DLAT * (np.arange(RADAR_SHAPE[0]) + 0.5)   # descending
RADAR_LONS = RADAR_LON_LEFT + _DLON * (np.arange(RADAR_SHAPE[1]) + 0.5)  # ascending

# The 240 km wide-range product (480 x 480 PNG, data/raw/radar_240km) has no
# published bounds; they are MEASURED by scripts/georef_240km.py (results/
# georef_240km.json): the 70 km domain's rain matches the 240 km image at
# correlation 0.95 with square 1 km pixels, i.e. 480 x 480 km centred on
# Singapore ("240 km" is the radius). Cross-checked against NEA's coastline
# basemap (same scale to 1 %, position to ~3 km). Uncertainty: +/-0.25 km near
# Singapore, +/-2.5 km at the image edges. Same 33-colour legend as the 70 km
# product, so rgba_to_rain_rate() applies unchanged.
RADAR240_LAT_TOP = 3.5056
RADAR240_LAT_BOTTOM = -0.8354
RADAR240_LON_LEFT = 101.8178
RADAR240_LON_RIGHT = 106.1297
RADAR240_SHAPE = (480, 480)


def rgba_to_rain_rate(img_array: np.ndarray) -> np.ndarray:
    """Convert an RGBA NEA radar frame to mm/hr via exact colour matching.

    Transparent pixels (alpha=0) are no-rain by definition and map to 0.0.
    Opaque pixels are looked up exactly in the 33-colour ramp; anything else
    is recorded in UNKNOWN_COLOURS and left at 0.0.
    """
    h, w = img_array.shape[:2]
    rgb = img_array[:, :, :3].astype(np.int64)
    packed = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]

    if img_array.shape[2] == 4:
        opaque = img_array[:, :, 3] > 0
    else:  # defensive: a non-alpha frame means every pixel is a colour claim
        opaque = np.ones((h, w), dtype=bool)

    rain = np.zeros((h, w), dtype=np.float32)
    if not opaque.any():
        return rain

    keys = packed[opaque]
    idx = np.searchsorted(_LUT_KEYS, keys)
    idx_clipped = np.clip(idx, 0, len(_LUT_KEYS) - 1)
    hit = _LUT_KEYS[idx_clipped] == keys

    vals = np.where(hit, _LUT_VALS[idx_clipped], np.float32(0.0))
    rain[opaque] = vals

    if not hit.all():
        for k, n in zip(*np.unique(keys[~hit], return_counts=True)):
            UNKNOWN_COLOURS[int(k)] = UNKNOWN_COLOURS.get(int(k), 0) + int(n)

    return rain


def parse_timestamp(filename: str) -> datetime | None:
    try:
        return datetime.strptime(filename[:13], "%Y%m%d_%H%M")
    except ValueError:
        return None


def load_existing_timestamps() -> set:
    """Load all already-processed timestamps from zarr in one shot."""
    if not ZARR_PATH.exists():
        return set()
    try:
        ds = xr.open_zarr(ZARR_PATH, consolidated=True)
        times = set(pd.DatetimeIndex(ds.time.values).to_pydatetime().tolist())
        ds.close()
        return times
    except Exception:
        return set()


def process_png(png_path: Path) -> np.ndarray | None:
    """Load a PNG and convert it to a native-resolution mm/hr field.

    No crop and no resampling: the frame is stored exactly as NEA renders it.
    Resampling a *banded* field is lossy in a way that manufactures rain —
    bilinear interpolation between a 4.30 mm/hr cell and a dry cell invents
    intermediate rates that exist nowhere in the source. Frames whose size is
    not the documented 217×120 are rejected rather than reshaped, so a change
    in the NEA product surfaces as an error instead of silent misalignment.
    """
    try:
        arr = np.array(Image.open(png_path))
    except Exception:
        return None

    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    if arr.shape[:2] != RADAR_SHAPE:
        return None

    return rgba_to_rain_rate(arr)


def write_batch_to_zarr(times: list[datetime], rain_stack: np.ndarray,
                        target: Path | None = None) -> None:
    """Write a batch of frames to zarr in a single atomic operation."""
    target = target or ZARR_PATH
    ds = xr.Dataset(
        {"rain_rate": xr.DataArray(
            rain_stack,
            dims=["time", "lat", "lon"],
            attrs={"units": "mm/hr", "long_name": "Rain rate from NEA X-band radar"},
        )},
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "lat": RADAR_LATS,
            "lon": RADAR_LONS,
        },
    )
    ds = ds.chunk({"time": 288, "lat": -1, "lon": -1})  # one day per chunk

    encoding = {"time": {"units": "minutes since 1970-01-01", "dtype": "float64"}}
    if target.exists():
        # encoding must not be re-specified for existing variables when appending
        # safe_chunks=False: single-threaded sequential write, no parallel corruption risk
        ds.to_zarr(target, mode="a", append_dim="time", consolidated=True, safe_chunks=False)
    else:
        ds.to_zarr(target, mode="w", consolidated=True, encoding=encoding)


def ensure_sorted_archive(zarr_path: Path | None = None) -> bool:
    """Guarantee the zarr time axis is strictly increasing (no duplicates).

    Appends can backfill older frames after newer ones (e.g. PNGs that sync
    in late), which silently breaks every consumer that indexes by position
    (RadarDataset context/target windows, frame lookups). If disorder or
    duplicate timestamps are found, the archive is rewritten sorted via a
    temp store + atomic rename. Returns True if a rewrite happened.
    """
    store = zarr_path or ZARR_PATH
    ds = xr.open_zarr(store, consolidated=True)
    times = pd.DatetimeIndex(ds.time.values)
    n_dups = int(times.duplicated().sum())
    if times.is_monotonic_increasing and n_dups == 0:
        ds.close()
        return False

    console.print(f"[yellow]Time axis disorder detected "
                  f"(monotonic={times.is_monotonic_increasing}, duplicates={n_dups}). "
                  f"Rewriting archive sorted...[/yellow]")
    ds = ds.load()
    ds_sorted = ds.sortby("time")
    if n_dups:
        _, first_occurrence = np.unique(ds_sorted.time.values, return_index=True)
        ds_sorted = ds_sorted.isel(time=first_occurrence)
    ds.close()

    tmp_path = store.with_name(store.name + ".tmp")
    bak_path = store.with_name(store.name + ".bak")
    for stale in (tmp_path, bak_path):
        if stale.exists():
            import shutil
            shutil.rmtree(stale)

    ds_sorted = ds_sorted.chunk({"time": 288, "lat": -1, "lon": -1})
    encoding = {"time": {"units": "minutes since 1970-01-01", "dtype": "float64"}}
    ds_sorted.to_zarr(tmp_path, mode="w", consolidated=True, encoding=encoding)

    store.rename(bak_path)
    tmp_path.rename(store)

    # Verify the rewrite before discarding the original
    check = xr.open_zarr(store, consolidated=True)
    check_times = pd.DatetimeIndex(check.time.values)
    ok = check_times.is_monotonic_increasing and not check_times.duplicated().any()
    n_frames = len(check_times)
    check.close()
    if not ok:
        raise RuntimeError(f"Sorted rewrite failed verification; original kept at {bak_path}")
    import shutil
    shutil.rmtree(bak_path)
    console.print(f"[green]Archive re-sorted: {n_frames} frames, strictly increasing.[/green]")
    return True


def heal_blank_frames() -> int:
    """Re-ingest all-NaN frames whose source PNG still exists on disk.

    Interrupted or racing appends can commit timestamps to the time axis
    without writing their rain_rate rows; zarr then silently serves
    fill_value=NaN for them, and the skip-by-existing-timestamp logic never
    retries. The PNGs are still on disk, so the data is recoverable: find
    all-NaN rows, re-run process_png, and write the rows in place (the time
    axis is untouched). Runs after every append, so both past corruption and
    any future interrupted append self-heal on the next run.
    Returns the number of frames healed.
    """
    if not ZARR_PATH.exists():
        return 0
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    times = pd.DatetimeIndex(ds.time.values)
    nan_mask = np.isnan(ds["rain_rate"].values).all(axis=(1, 2))
    ds.close()
    nan_pos = np.nonzero(nan_mask)[0]
    if len(nan_pos) == 0:
        return 0

    console.print(f"[yellow]{len(nan_pos)} all-NaN frames found; healing from PNGs...[/yellow]")
    healed_frames: dict[int, np.ndarray] = {}
    no_png = failed = 0
    for pos in nan_pos:
        png_path = RADAR_DIR / (times[pos].strftime("%Y%m%d_%H%M") + ".png")
        if not png_path.exists():
            no_png += 1
            continue
        frame = process_png(png_path)
        if frame is None:
            failed += 1
            continue
        healed_frames[int(pos)] = frame

    if healed_frames:
        # Group contiguous positions into runs → one region write per run
        root = zarr.open_group(str(ZARR_PATH), mode="r+")
        rr = root["rain_rate"]
        positions = sorted(healed_frames)
        run_start = prev = positions[0]
        runs = []
        for p in positions[1:]:
            if p == prev + 1:
                prev = p
            else:
                runs.append((run_start, prev))
                run_start = prev = p
        runs.append((run_start, prev))
        for a, b in runs:
            rr[a:b + 1] = np.stack([healed_frames[p] for p in range(a, b + 1)], axis=0)

    console.print(f"[green]Healed {len(healed_frames)} frames"
                  f"{f'; {no_png} have no PNG' if no_png else ''}"
                  f"{f'; {failed} failed to convert' if failed else ''}.[/green]")
    return len(healed_frames)


def rebuild_archive(target: Path, batch_size: int = 2016) -> int:
    """Build a fresh archive from every PNG on disk, into a NEW store.

    Used to re-ingest after a change to the colour mapping or georeferencing.
    Writes to `target` and never touches the live archive, so the existing
    store stays readable and the swap stays a rename (see docs/history/migration_plan.md).
    Returns the number of frames written.
    """
    if target.exists():
        console.print(f"[red]{target} already exists — remove it or pick another path.[/red]")
        return 0

    png_files = sorted(RADAR_DIR.glob("*.png"))
    if not png_files:
        console.print("[yellow]No PNG files found.[/yellow]")
        return 0

    UNKNOWN_COLOURS.clear()
    console.print(f"Rebuilding {len(png_files)} PNGs -> {target.name} "
                  f"(grid {RADAR_SHAPE[0]}x{RADAR_SHAPE[1]}, no crop, no resample)")

    written = errors = 0
    batch_times: list[datetime] = []
    batch_frames: list[np.ndarray] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting", total=len(png_files))

        for png_path in png_files:
            dt = parse_timestamp(png_path.stem)
            if dt is None:
                progress.advance(task)
                continue
            frame = process_png(png_path)
            if frame is None:
                errors += 1
            else:
                batch_times.append(dt)
                batch_frames.append(frame)
            if len(batch_times) >= batch_size:
                order = sorted(range(len(batch_times)), key=lambda i: batch_times[i])
                write_batch_to_zarr([batch_times[i] for i in order],
                                    np.stack([batch_frames[i] for i in order], axis=0),
                                    target=target)
                written += len(batch_times)
                batch_times, batch_frames = [], []
            progress.advance(task)

    if batch_times:
        order = sorted(range(len(batch_times)), key=lambda i: batch_times[i])
        write_batch_to_zarr([batch_times[i] for i in order],
                            np.stack([batch_frames[i] for i in order], axis=0),
                            target=target)
        written += len(batch_times)

    console.print(f"[green]Wrote {written} frames[/green]"
                  f"{f'; {errors} PNGs rejected' if errors else ''}.")
    if UNKNOWN_COLOURS:
        total_px = sum(UNKNOWN_COLOURS.values())
        console.print(f"[red]WARNING: {total_px} opaque pixels in "
                      f"{len(UNKNOWN_COLOURS)} colours are not in the NEA ramp:[/red]")
        for k, n in sorted(UNKNOWN_COLOURS.items(), key=lambda kv: -kv[1])[:10]:
            console.print(f"    ({k >> 16 & 255}, {k >> 8 & 255}, {k & 255})  {n} px")
    else:
        console.print("[green]All opaque pixels matched the 33-colour NEA ramp exactly.[/green]")
    return written


def print_status() -> None:
    if not ZARR_PATH.exists():
        console.print("[red]radar.zarr does not exist yet.[/red]")
    else:
        try:
            ds = xr.open_zarr(ZARR_PATH, consolidated=True)
            times = pd.DatetimeIndex(ds.time.values) if len(ds.time) > 0 else []
            console.print(f"[green]radar.zarr: {len(times)} timestamps[/green]")
            if len(times) > 0:
                console.print(f"  First: {times[0]}")
                console.print(f"  Last : {times[-1]}")
            ds.close()
        except Exception as e:
            console.print(f"[red]Error reading zarr: {e}[/red]")

    raw_pngs = list(RADAR_DIR.glob("*.png"))
    console.print(f"Raw PNGs in archive: {len(raw_pngs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess NEA radar PNGs into zarr dataset")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--repair", action="store_true",
                        help="Only check/repair time-axis ordering of the existing archive")
    parser.add_argument("--heal", action="store_true",
                        help="Only re-ingest all-NaN frames from their PNGs (no new ingest)")
    parser.add_argument("--rebuild", metavar="PATH",
                        help="Re-ingest every PNG into a NEW store at PATH (does not "
                             "touch the live archive); used after a colour/grid change")
    args = parser.parse_args()

    if args.rebuild:
        target = Path(args.rebuild)
        if not target.is_absolute():
            target = OUTPUT_DIR / target
        if rebuild_archive(target):
            ensure_sorted_archive(target)
        return

    if args.status:
        print_status()
        return

    if args.repair:
        if not ZARR_PATH.exists():
            console.print("[red]radar.zarr does not exist yet.[/red]")
            return
        if not ensure_sorted_archive():
            console.print("[green]Archive already sorted; nothing to repair.[/green]")
        return

    if args.heal:
        if heal_blank_frames() == 0:
            console.print("[green]No all-NaN frames; nothing to heal.[/green]")
        return

    png_files = sorted(RADAR_DIR.glob("*.png"))
    if not png_files:
        console.print("[yellow]No PNG files found. Run scrape_radar.py first.[/yellow]")
        return

    existing = load_existing_timestamps()
    console.print(f"Existing zarr timestamps: {len(existing)} | Raw PNGs: {len(png_files)}")

    saved = skipped = errors = 0
    batch_times: list[datetime] = []
    batch_frames: list[np.ndarray] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Reading PNGs", total=len(png_files))

        for png_path in png_files:
            dt = parse_timestamp(png_path.stem)
            if dt is None:
                progress.advance(task)
                continue

            if dt in existing:
                skipped += 1
                progress.advance(task)
                continue

            progress.update(task, description=f"[cyan]{dt.strftime('%Y-%m-%d %H:%M')}[/cyan]")

            try:
                rain_field = process_png(png_path)
                if rain_field is not None:
                    batch_times.append(dt)
                    batch_frames.append(rain_field)
                    saved += 1
                else:
                    errors += 1
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Re-run to resume.[/yellow]")
                break
            except Exception as e:
                errors += 1

            progress.advance(task)

    if batch_times:
        console.print(f"Writing {len(batch_times)} frames to zarr...")
        # Sort by time before writing
        order = sorted(range(len(batch_times)), key=lambda i: batch_times[i])
        batch_times = [batch_times[i] for i in order]
        rain_stack = np.stack([batch_frames[i] for i in order], axis=0)
        write_batch_to_zarr(batch_times, rain_stack)
        # Per-batch sorting is not enough: a batch may backfill frames older
        # than the archive tail, so enforce global order after every append.
        ensure_sorted_archive()
        console.print("[green]Write complete.[/green]")

    # Always heal, even on runs with no new PNGs: a previous interrupted
    # append may have left NaN rows behind.
    heal_blank_frames()

    console.print(f"[bold]Done:[/bold] {saved} processed, {skipped} skipped, {errors} errors")

    # Check for stage completion (≥90 days of data)
    if ZARR_PATH.exists():
        try:
            ds = xr.open_zarr(ZARR_PATH, consolidated=True)
            n_days = len(set(pd.DatetimeIndex(ds.time.values).normalize()))
            ds.close()
            if n_days >= 90:
                flag = CHECKPOINT_DIR / "stage3_complete.flag"
                flag.touch()
                console.print(f"[bold green]Stage 3 complete ({n_days} days) → {flag}[/bold green]")
            else:
                console.print(f"[yellow]Archive has {n_days}/90 days. Keep running scrape_radar.py daily.[/yellow]")
        except Exception:
            pass


if __name__ == "__main__":
    main()
