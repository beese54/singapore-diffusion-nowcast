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

# NEA rain area colour scale (approximate RGB → mm/hr mapping)
# Based on standard WMO colour conventions used by NEA
# Values from NEA radar legend: light blue=trace, green=light, yellow=moderate, red=heavy
COLOUR_SCALE = [
    # (R, G, B) → mm/hr (upper bound of bin)
    ((0, 0, 0), 0.0),         # black = no rain / background
    ((255, 255, 255), 0.0),   # white = no data
    ((173, 216, 230), 0.5),   # light blue = trace (<0.5)
    ((0, 255, 255), 1.0),     # cyan = very light (0.5–1)
    ((0, 200, 0), 2.0),       # green = light (1–2)
    ((0, 255, 0), 5.0),       # bright green = light-moderate (2–5)
    ((255, 255, 0), 10.0),    # yellow = moderate (5–10)
    ((255, 165, 0), 20.0),    # orange = moderate-heavy (10–20)
    ((255, 0, 0), 40.0),      # red = heavy (20–40)
    ((200, 0, 200), 80.0),    # magenta = very heavy (40–80)
    ((128, 0, 128), 100.0),   # purple = extreme (>80)
]

# Build a fast lookup table (nearest-colour in RGB space)
_COLOUR_ARRAY = np.array([c for c, _ in COLOUR_SCALE], dtype=np.float32)
_RATE_ARRAY = np.array([r for _, r in COLOUR_SCALE], dtype=np.float32)

# Singapore bounding box pixel coordinates in the 480×480 NEA composite image
# These are approximate and may need calibration against known coordinates
# NEA 70km range composite: centre ~1.35°N, 103.82°E; each pixel ~0.29km
# Full image covers roughly: 0.9°N–1.8°N, 103.3°E–104.3°E
SG_PIXEL_ROW_MIN = 80   # ~1.0°N
SG_PIXEL_ROW_MAX = 270  # ~1.6°N
SG_PIXEL_COL_MIN = 130  # ~103.5°E
SG_PIXEL_COL_MAX = 350  # ~104.1°E

# Derived lat/lon grid for the cropped SG domain
SG_LATS = np.linspace(1.6, 1.0, SG_PIXEL_ROW_MAX - SG_PIXEL_ROW_MIN)
SG_LONS = np.linspace(103.5, 104.1, SG_PIXEL_COL_MAX - SG_PIXEL_COL_MIN)
SG_GRID_SHAPE = (len(SG_LATS), len(SG_LONS))


def rgb_to_rain_rate(img_array: np.ndarray) -> np.ndarray:
    """Convert an RGB image array to mm/hr rain rate using nearest-colour lookup."""
    h, w, _ = img_array.shape
    flat_rgb = img_array[:, :, :3].reshape(-1, 3).astype(np.float32)

    # Vectorised nearest-colour: squared Euclidean distance in RGB space
    diffs = flat_rgb[:, np.newaxis, :] - _COLOUR_ARRAY[np.newaxis, :, :]
    dist2 = (diffs ** 2).sum(axis=2)
    nearest_idx = dist2.argmin(axis=1)
    rain_flat = _RATE_ARRAY[nearest_idx]
    return rain_flat.reshape(h, w)


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
    """Load a PNG, crop to Singapore domain, convert to mm/hr. Returns (H, W) array."""
    try:
        img = Image.open(png_path).convert("RGB")
        arr = np.array(img)
    except Exception as e:
        return None

    if arr.ndim != 3 or arr.shape[2] < 3:
        return None

    rain_rate = rgb_to_rain_rate(arr)

    # Crop to Singapore domain (guard against images smaller than expected)
    r0 = min(SG_PIXEL_ROW_MIN, arr.shape[0] - 1)
    r1 = min(SG_PIXEL_ROW_MAX, arr.shape[0])
    c0 = min(SG_PIXEL_COL_MIN, arr.shape[1] - 1)
    c1 = min(SG_PIXEL_COL_MAX, arr.shape[1])

    crop = rain_rate[r0:r1, c0:c1]

    # Resize to target SG grid shape if needed
    if crop.shape != SG_GRID_SHAPE:
        from PIL import Image as PILImage
        crop_img = PILImage.fromarray(crop).resize(
            (SG_GRID_SHAPE[1], SG_GRID_SHAPE[0]), PILImage.BILINEAR
        )
        crop = np.array(crop_img)

    return crop.astype(np.float32)


def write_batch_to_zarr(times: list[datetime], rain_stack: np.ndarray) -> None:
    """Write a batch of frames to zarr in a single atomic operation."""
    ds = xr.Dataset(
        {"rain_rate": xr.DataArray(
            rain_stack,
            dims=["time", "lat", "lon"],
            attrs={"units": "mm/hr", "long_name": "Rain rate from NEA X-band radar"},
        )},
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "lat": SG_LATS,
            "lon": SG_LONS,
        },
    )
    ds = ds.chunk({"time": 288, "lat": -1, "lon": -1})  # one day per chunk

    encoding = {"time": {"units": "minutes since 1970-01-01", "dtype": "float64"}}
    if ZARR_PATH.exists():
        # encoding must not be re-specified for existing variables when appending
        # safe_chunks=False: single-threaded sequential write, no parallel corruption risk
        ds.to_zarr(ZARR_PATH, mode="a", append_dim="time", consolidated=True, safe_chunks=False)
    else:
        ds.to_zarr(ZARR_PATH, mode="w", consolidated=True, encoding=encoding)


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
    args = parser.parse_args()

    if args.status:
        print_status()
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
        console.print("[green]Write complete.[/green]")

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
