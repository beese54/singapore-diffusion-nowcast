"""
scrape_radar.py — Incremental archiver for NEA Singapore radar imagery.

Behaviour
---------
- Downloads radar rain-area PNG images from NEA's public weather portal
- Stores each image as data/raw/radar/YYYYMMDD_HHMM.png
- Already-downloaded images are skipped
- Respectful rate limiting (1 request/s by default)
- Safe to run repeatedly; each session extends the archive

Usage
-----
    python scripts/scrape_radar.py                      # archive past 24h
    python scripts/scrape_radar.py --hours 720          # archive past 30 days (first run)
    python scripts/scrape_radar.py --status             # report archive coverage
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parent.parent
RADAR_DIR = ROOT / "data" / "raw" / "radar"
RADAR_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = ROOT / "checkpoints"

console = Console()

# Direct radar image URL — no API key required.
# Format: YYYYMMDDHHmm (12 digits, SGT) + 0000
# Images are served at 5-minute intervals from weather.gov.sg
RADAR_URL_TEMPLATE = "https://www.weather.gov.sg/files/rainarea/50km/v2/dpsri_70km_{ts}0000dBR.dpsri.png"
RADAR_REFERER = "https://www.weather.gov.sg/weather-rain-area-50km/"

REQUEST_INTERVAL_S = 1.2   # ~50 req/min, well under typical server limits
RADAR_INTERVAL_MIN = 5     # NEA updates every 5 minutes


def round_to_5min(dt: datetime) -> datetime:
    minutes = (dt.minute // 5) * 5
    return dt.replace(minute=minutes, second=0, microsecond=0)


def image_path(dt: datetime) -> Path:
    return RADAR_DIR / dt.strftime("%Y%m%d_%H%M.png")


def fetch_rain_area(dt: datetime, session: requests.Session, debug: bool = False) -> bytes | None:
    """Fetch the rain area PNG directly from weather.gov.sg (no API key required).

    Timestamps use Singapore time (UTC+8). Images update every 5 minutes.
    URL format: dpsri_70km_YYYYMMDDHHmm0000dBR.dpsri.png
    """
    # NEA serves images in SGT (UTC+8)
    sgt_dt = dt + timedelta(hours=8)
    ts = sgt_dt.strftime("%Y%m%d%H%M")
    url = RADAR_URL_TEMPLATE.format(ts=ts)

    try:
        resp = session.get(url, timeout=15)
        if debug:
            console.print(f"[dim]Radar URL: {url} -> {resp.status_code} {len(resp.content)}b[/dim]")
        if resp.status_code == 200 and len(resp.content) > 280:
            return resp.content
    except requests.RequestException:
        pass

    return None


def archive_range(start_dt: datetime, end_dt: datetime) -> tuple[int, int, int]:
    """Download all 5-minute intervals in [start_dt, end_dt]. Returns (saved, skipped, errors)."""
    saved = skipped = errors = 0

    # Build list of timestamps
    current = round_to_5min(start_dt)
    timestamps = []
    while current <= end_dt:
        timestamps.append(current)
        current += timedelta(minutes=RADAR_INTERVAL_MIN)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": RADAR_REFERER,
    })

    first_request = True
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Archiving radar", total=len(timestamps))

        for dt in timestamps:
            dest = image_path(dt)
            if dest.exists():
                skipped += 1
                progress.advance(task)
                continue

            progress.update(task, description=f"[cyan]{dt.strftime('%Y-%m-%d %H:%M')}[/cyan]")

            try:
                img_bytes = fetch_rain_area(dt, session, debug=first_request)
                first_request = False
                if img_bytes:
                    dest.write_bytes(img_bytes)
                    saved += 1
                else:
                    errors += 1
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Re-run to continue.[/yellow]")
                break
            except Exception as e:
                errors += 1

            time.sleep(REQUEST_INTERVAL_S)
            progress.advance(task)

    return saved, skipped, errors


def print_status() -> None:
    files = sorted(RADAR_DIR.glob("*.png"))
    if not files:
        console.print("[red]No radar images found in archive.[/red]")
        return

    dates = set()
    for f in files:
        try:
            dates.add(datetime.strptime(f.stem[:8], "%Y%m%d").date())
        except ValueError:
            pass

    console.print(f"[green]Archive: {len(files)} images across {len(dates)} days[/green]")
    if dates:
        console.print(f"  First: {min(dates)}")
        console.print(f"  Last:  {max(dates)}")

    # Check expected images per day (288 = 24h × 12 per hour)
    from collections import Counter
    counts = Counter()
    for f in files:
        try:
            counts[f.stem[:8]] += 1
        except Exception:
            pass

    sparse_days = [(d, c) for d, c in counts.items() if c < 240]
    if sparse_days:
        console.print(f"[yellow]{len(sparse_days)} days with <240 images (possible gaps):[/yellow]")
        for d, c in sorted(sparse_days)[:10]:
            console.print(f"  {d}: {c} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive NEA Singapore radar imagery")
    parser.add_argument("--hours", type=int, default=24, help="Hours of history to archive (default: 24)")
    parser.add_argument("--status", action="store_true", help="Show archive status and exit")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    end_dt = round_to_5min(datetime.now(timezone.utc).replace(tzinfo=None))
    start_dt = end_dt - timedelta(hours=args.hours)

    console.print(f"Archiving radar: [cyan]{start_dt}[/cyan] -> [cyan]{end_dt}[/cyan]")
    saved, skipped, errors = archive_range(start_dt, end_dt)
    console.print(f"[bold]Done:[/bold] {saved} saved, {skipped} skipped, {errors} errors")

    if saved + skipped > 0:
        console.print(f"[dim]Run scrape_radar.py again each session to extend the archive.[/dim]")


if __name__ == "__main__":
    main()
