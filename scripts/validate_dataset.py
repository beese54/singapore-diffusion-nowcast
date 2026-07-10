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

# The 2026-05-22..28 scheduler ramp-up had large structural gaps; the DoD
# quality criterion (amended 2026-07-10) is scoped to steady state onwards.
STEADY_STATE_START = pd.Timestamp("2026-05-29")
MEDIAN_7D_GAP_THRESHOLD_PCT = 10.0


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

    # DoD criterion (amended 2026-07-10): median 7-day-window gap rate over
    # the steady-state period. Overall gap rate above is informational only —
    # the ramp-up week is frozen history and travel-day gaps are expected.
    daily = times.to_series().groupby(times.normalize()).count()
    # exclude partial first/last days and the ramp-up week
    steady = daily.iloc[1:-1]
    steady = steady[steady.index >= STEADY_STATE_START]
    verdict_shown = False
    if len(steady) >= 7:
        window_totals = steady.rolling(7).sum().dropna()
        gap_7d = (1.0 - window_totals / (288.0 * 7)) * 100.0
        median_gap = float(gap_7d.median())
        console.print(f"\nSteady-state (from {STEADY_STATE_START.date()}) 7-day windows: "
                      f"{len(gap_7d)} | median gap {median_gap:.1f}% | "
                      f"worst {float(gap_7d.max()):.1f}%")
        if median_gap < MEDIAN_7D_GAP_THRESHOLD_PCT:
            console.print(f"[bold green]Quality check PASSED "
                          f"(median 7-day gap {median_gap:.1f}% < {MEDIAN_7D_GAP_THRESHOLD_PCT:.0f}%)[/bold green]")
        else:
            console.print(f"[bold red]Quality check FAILED "
                          f"(median 7-day gap {median_gap:.1f}% >= {MEDIAN_7D_GAP_THRESHOLD_PCT:.0f}%)[/bold red]")
            console.print("Run scrape_radar.py with --hours to backfill missing periods.")
        verdict_shown = True
    if not verdict_shown:
        console.print("\n[yellow]Fewer than 7 steady-state days in range; no quality verdict.[/yellow]")


if __name__ == "__main__":
    main()
