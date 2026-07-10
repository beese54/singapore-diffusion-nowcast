"""
download_era5.py — Resumable ERA5 download for Singapore bounding box.

Behaviour
---------
- Downloads month-by-month into a single zarr store at data/raw/era5/singapore_YYYY.zarr
- Already-downloaded months are detected by checking zarr chunks → skipped automatically
- Safe to interrupt (Ctrl+C) and re-run at any time

Usage
-----
    python scripts/download_era5.py                     # download 2022-2023
    python scripts/download_era5.py --year 2023         # single year
    python scripts/download_era5.py --year 2023 --month 6  # single month
    python scripts/download_era5.py --status            # show what's downloaded
"""

import argparse
import calendar
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr
import zarr
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "era5"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

# Singapore bounding box with ~30 km buffer: N/W/S/E
SG_BBOX = [1.6, 103.5, 1.0, 104.1]   # [N, W, S, E] for CDS API

# Surface/single-level variables
SURFACE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_column_water_vapour",
    "total_precipitation",
    "surface_pressure",
]

# Pressure-level variables
PRESSURE_VARS = ["specific_humidity", "temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"]
PRESSURE_LEVELS = ["1000", "850", "500", "250"]

HOURS = [f"{h:02d}:00" for h in range(24)]

DEFAULT_YEARS = [2022, 2023]


def zarr_store_path(year: int) -> Path:
    return DATA_DIR / f"singapore_{year}.zarr"


def already_downloaded(year: int, month: int) -> bool:
    """Check whether a given year-month is already in the zarr store."""
    store_path = zarr_store_path(year)
    if not store_path.exists():
        return False
    try:
        ds = xr.open_zarr(store_path)
        # Support both 'time' and 'valid_time' coordinate names
        time_coord = "time" if "time" in ds.coords else "valid_time"
        times = ds[time_coord].values
        target = np.datetime64(f"{year}-{month:02d}", "M")
        present = np.any((times.astype("datetime64[M]")) == target)
        ds.close()
        return bool(present)
    except Exception:
        return False


def download_month(year: int, month: int) -> Path:
    """Download one calendar month and return path to the temporary NetCDF file."""
    days_in_month = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, days_in_month + 1)]

    tmp_surface = DATA_DIR / f"tmp_surface_{year}{month:02d}.nc"
    tmp_pressure = DATA_DIR / f"tmp_pressure_{year}{month:02d}.nc"

    c = cdsapi.Client(quiet=True)

    # Surface variables
    if not tmp_surface.exists():
        console.print(f"  [cyan]Downloading surface vars {year}-{month:02d}...[/cyan]")
        tmp_surface_raw = DATA_DIR / f"tmp_surface_{year}{month:02d}.raw"
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": SURFACE_VARS,
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": HOURS,
                "area": SG_BBOX,
                "format": "netcdf",
            },
            str(tmp_surface_raw),
        )
        # CDS v2 delivers surface data as a zip with one or two .nc files
        # (instant vars + accumulated vars, e.g. tp). Merge all into one file.
        if zipfile.is_zipfile(tmp_surface_raw):
            import gc as _gc2
            nc_names = []
            part_paths = []
            with zipfile.ZipFile(tmp_surface_raw, "r") as z:
                nc_names = [n for n in z.namelist() if n.endswith(".nc")]
                if not nc_names:
                    raise RuntimeError(f"No .nc file found inside zip for {year}-{month:02d}")
                for i, name in enumerate(nc_names):
                    dest = DATA_DIR / f"_surf_{year}{month:02d}_part{i}.nc"
                    dest.write_bytes(z.read(name))
                    part_paths.append(dest)
            if len(part_paths) == 1:
                part_paths[0].rename(tmp_surface)
            else:
                datasets = [xr.open_dataset(p, engine="netcdf4") for p in part_paths]
                merged = xr.merge(datasets, compat="override").load()
                for ds in datasets:
                    ds.close()
                _gc2.collect()
                merged.to_netcdf(str(tmp_surface))
                merged.close()
                for p in part_paths:
                    p.unlink(missing_ok=True)
            tmp_surface_raw.unlink(missing_ok=True)
        else:
            tmp_surface_raw.rename(tmp_surface)

    # Pressure-level variables — split into two half-months to stay under CDS field limits
    # (5 vars × 4 levels × 24 hours × 31 days = 14,880 fields exceeds the per-request cap)
    if not tmp_pressure.exists():
        mid = days_in_month // 2
        half_days = [days[:mid], days[mid:]]
        tmp_halves = [
            DATA_DIR / f"tmp_pressure_{year}{month:02d}_h{i}.nc"
            for i in range(2)
        ]

        for i, (half, tmp_half) in enumerate(zip(half_days, tmp_halves)):
            if not tmp_half.exists():
                console.print(f"  [cyan]Downloading pressure-level vars {year}-{month:02d} part {i+1}/2...[/cyan]")
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": PRESSURE_VARS,
                        "pressure_level": PRESSURE_LEVELS,
                        "year": str(year),
                        "month": f"{month:02d}",
                        "day": half,
                        "time": HOURS,
                        "area": SG_BBOX,
                        "format": "netcdf",
                    },
                    str(tmp_half),
                )

        # Merge the two halves along valid_time (the actual time dimension in CDS v2 output)
        import gc as _gc
        ds0 = xr.open_dataset(tmp_halves[0], engine="netcdf4")
        ds1 = xr.open_dataset(tmp_halves[1], engine="netcdf4")
        merged = xr.concat([ds0, ds1], dim="valid_time").load()
        ds0.close()
        ds1.close()
        _gc.collect()
        merged.to_netcdf(str(tmp_pressure))
        merged.close()
        for tmp_half in tmp_halves:
            tmp_half.unlink(missing_ok=True)

    return tmp_surface, tmp_pressure


def append_to_zarr(year: int, month: int, tmp_surface: Path, tmp_pressure: Path) -> None:
    """Merge the two NetCDF files and append to the zarr store."""
    import gc

    store_path = zarr_store_path(year)

    ds_surf = xr.open_dataset(tmp_surface, engine="netcdf4")
    ds_pres = xr.open_dataset(tmp_pressure, engine="netcdf4")

    # Drop CDS metadata variables that differ between datasets and are not needed
    for drop_var in ("expver", "number"):
        if drop_var in ds_surf:
            ds_surf = ds_surf.drop_vars(drop_var)
        if drop_var in ds_pres:
            ds_pres = ds_pres.drop_vars(drop_var)

    # CDS v2 uses valid_time as the time dimension; rename to 'time' for consistency
    for rename_from in ("valid_time",):
        if rename_from in ds_surf.dims and "time" not in ds_surf.dims:
            ds_surf = ds_surf.rename({rename_from: "time"})
        if rename_from in ds_pres.dims and "time" not in ds_pres.dims:
            ds_pres = ds_pres.rename({rename_from: "time"})

    ds = xr.merge([ds_surf, ds_pres], compat="override")

    # Load into memory so we can close file handles before writing zarr (Windows lock fix)
    ds = ds.load()
    ds_surf.close()
    ds_pres.close()
    gc.collect()

    # Chunk by time (1 day = 24h per chunk), lat/lon small so keep whole
    chunks = {"time": 24, "latitude": -1, "longitude": -1}
    if "pressure_level" in ds.dims:
        chunks["pressure_level"] = -1
    ds = ds.chunk(chunks)

    if store_path.exists():
        ds.to_zarr(store_path, mode="a", append_dim="time")
    else:
        ds.to_zarr(store_path, mode="w")

    ds.close()
    gc.collect()
    # Windows may hold a file lock briefly after close(); ignore — tmp files are safe to delete manually
    try:
        tmp_surface.unlink(missing_ok=True)
        tmp_pressure.unlink(missing_ok=True)
    except OSError:
        pass


def print_status(years: list[int]) -> None:
    table = Table(title="ERA5 Download Status")
    table.add_column("Year-Month", style="cyan")
    table.add_column("Status", style="green")

    for year in years:
        for month in range(1, 13):
            status = "[green]Downloaded[/green]" if already_downloaded(year, month) else "[red]Missing[/red]"
            table.add_row(f"{year}-{month:02d}", status)

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 data for Singapore bounding box")
    parser.add_argument("--year", type=int, help="Download only this year (default: 2022 and 2023)")
    parser.add_argument("--month", type=int, help="Download only this month (requires --year)")
    parser.add_argument("--status", action="store_true", help="Show download status and exit")
    args = parser.parse_args()

    years = [args.year] if args.year else DEFAULT_YEARS

    if args.status:
        print_status(years)
        return

    if args.month and not args.year:
        console.print("[red]--month requires --year[/red]")
        sys.exit(1)

    months_to_download = [(y, m) for y in years for m in range(1, 13)]
    if args.year and args.month:
        months_to_download = [(args.year, args.month)]

    skipped = 0
    downloaded = 0
    errors = []

    for year, month in months_to_download:
        label = f"{year}-{month:02d}"
        if already_downloaded(year, month):
            console.print(f"[dim]  {label}: already in zarr, skipping[/dim]")
            skipped += 1
            continue

        console.print(f"\n[bold]{label}[/bold]")
        try:
            tmp_surf, tmp_pres = download_month(year, month)
            console.print(f"  [yellow]Merging into zarr...[/yellow]")
            append_to_zarr(year, month, tmp_surf, tmp_pres)
            console.print(f"  [green]Done → {zarr_store_path(year)}[/green]")
            downloaded += 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Re-run to resume from this month.[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"  [red]ERROR: {e}[/red]")
            errors.append((label, str(e)))

    console.print(f"\n[bold]Summary:[/bold] {downloaded} downloaded, {skipped} skipped, {len(errors)} errors")
    if errors:
        for label, err in errors:
            console.print(f"  [red]{label}: {err}[/red]")
        sys.exit(1)

    # Write stage flag if all months for all years are present
    all_done = all(already_downloaded(y, m) for y in years for m in range(1, 13))
    if all_done:
        flag = CHECKPOINT_DIR / "stage1_complete.flag"
        flag.touch()
        console.print(f"[bold green]Stage 1 complete -> {flag}[/bold green]")


if __name__ == "__main__":
    main()
