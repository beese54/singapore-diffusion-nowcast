"""
validate_era5.py — Check ERA5 zarr store for missing timestamps and required variables.

Usage
-----
    python scripts/validate_era5.py                  # validate 2022-2023
    python scripts/validate_era5.py --year 2023      # single year
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "era5"
console = Console()

REQUIRED_SURFACE_VARS = {"u10", "v10", "t2m", "msl", "tcwv", "tp", "sp"}
REQUIRED_PRESSURE_VARS = {"q", "t", "z", "u", "v"}
REQUIRED_LEVELS = {250, 500, 850, 1000}


def validate_year(year: int) -> bool:
    store_path = DATA_DIR / f"singapore_{year}.zarr"
    if not store_path.exists():
        console.print(f"[red]MISSING: {store_path}[/red]")
        return False

    ds = xr.open_zarr(store_path, consolidated=True)
    ok = True

    # Check time coverage
    expected_start = pd.Timestamp(f"{year}-01-01")
    expected_end = pd.Timestamp(f"{year}-12-31 23:00")
    expected_times = pd.date_range(expected_start, expected_end, freq="h")
    actual_times = pd.DatetimeIndex(ds.time.values)
    missing_times = expected_times.difference(actual_times)

    if len(missing_times) > 0:
        console.print(f"[red]MISSING {len(missing_times)} timestamps in {year}[/red]")
        if len(missing_times) <= 20:
            for t in missing_times:
                console.print(f"  [red]{t}[/red]")
        ok = False
    else:
        console.print(f"[green]Time coverage {year}: complete ({len(actual_times)} hours)[/green]")

    # Check surface variables
    missing_surf = REQUIRED_SURFACE_VARS - set(ds.data_vars)
    if missing_surf:
        console.print(f"[red]Missing surface vars: {missing_surf}[/red]")
        ok = False
    else:
        console.print(f"[green]Surface variables: all present[/green]")

    # Check pressure-level variables and levels
    level_dim = next((d for d in ("pressure_level", "level") if d in ds.dims), None)
    if level_dim:
        actual_levels = set(ds[level_dim].values.tolist())
        missing_levels = REQUIRED_LEVELS - actual_levels
        if missing_levels:
            console.print(f"[red]Missing pressure levels: {missing_levels}[/red]")
            ok = False
        else:
            console.print(f"[green]Pressure levels: {sorted(actual_levels)}[/green]")

        missing_pres = REQUIRED_PRESSURE_VARS - set(ds.data_vars)
        if missing_pres:
            console.print(f"[red]Missing pressure vars: {missing_pres}[/red]")
            ok = False
        else:
            console.print(f"[green]Pressure variables: all present[/green]")
    else:
        console.print(f"[yellow]WARNING: no pressure-level dimension found — pressure data missing[/yellow]")
        ok = False

    ds.close()
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ERA5 zarr store")
    parser.add_argument("--year", type=int, help="Validate only this year")
    args = parser.parse_args()

    years = [args.year] if args.year else [2022, 2023]
    all_ok = True

    for year in years:
        console.rule(f"[bold]Validating {year}[/bold]")
        ok = validate_year(year)
        all_ok = all_ok and ok

    if all_ok:
        console.print("\n[bold green]All checks passed.[/bold green]")
        sys.exit(0)
    else:
        console.print("\n[bold red]Validation failed — re-run download_era5.py to fill gaps.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
