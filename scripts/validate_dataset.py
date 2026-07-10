"""
validate_dataset.py — Check radar zarr store for temporal gaps and data quality.

Usage
-----
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --window-days 7   # check last 7 days only
"""

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = ROOT / "data" / "processed" / "radar.zarr"
console = Console()

EXPECTED_INTERVAL_MIN = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-days", type=int, default=0, help="Check only the last N days (0 = all)")
    args = parser.parse_args()

    if not ZARR_PATH.exists():
        console.print("[red]radar.zarr not found. Run preprocess_radar.py first.[/red]")
        return

    ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    times = pd.DatetimeIndex(sorted(ds.time.values))
    ds.close()

    if args.window_days > 0:
        cutoff = times[-1] - pd.Timedelta(days=args.window_days)
        times = times[times >= cutoff]

    if len(times) == 0:
        console.print("[yellow]No timestamps in selected window.[/yellow]")
        return

    # Compute expected timestamps
    expected = pd.date_range(times[0], times[-1], freq=f"{EXPECTED_INTERVAL_MIN}min")
    missing = expected.difference(times)
    gap_pct = 100.0 * len(missing) / max(len(expected), 1)

    n_days = len(set(times.date))

    # Summary table
    table = Table(title="Radar Dataset Coverage")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("First timestamp", str(times[0])[:16])
    table.add_row("Last timestamp", str(times[-1])[:16])
    table.add_row("Total timestamps", str(len(times)))
    table.add_row("Days covered", str(n_days))
    table.add_row("Missing timestamps", str(len(missing)))
    table.add_row("Gap rate", f"{gap_pct:.1f}%")
    console.print(table)

    # Per-day coverage
    day_counts = times.to_series().groupby(times.date).count()
    sparse = day_counts[day_counts < 240]
    if not sparse.empty:
        console.print(f"\n[yellow]{len(sparse)} days with <240 images (possible gaps):[/yellow]")
        for date, count in sparse.head(15).items():
            console.print(f"  {date}: {count} images ({count * 5 // 60}h {count * 5 % 60}min)")

    if gap_pct < 5.0:
        console.print(f"\n[bold green]Quality check PASSED (gap rate {gap_pct:.1f}% < 5%)[/bold green]")
    else:
        console.print(f"\n[bold red]Quality check FAILED (gap rate {gap_pct:.1f}% >= 5%)[/bold red]")
        console.print("Run scrape_radar.py with --hours to backfill missing periods.")


if __name__ == "__main__":
    main()
