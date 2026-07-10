"""
visualise_radar.py — Overlay NEA radar rain rate on a Singapore map.

Outputs
-------
  outputs/radar_overview.png   — grid of top rainfall frames + domain time-series
  outputs/radar_event.gif      — 3-hour animation around the heaviest rain event

Usage
-----
    python scripts/visualise_radar.py                  # full output
    python scripts/visualise_radar.py --no-anim        # skip GIF (faster)
    python scripts/visualise_radar.py --event 2026-06-01T14:00  # animate around a specific time
"""

import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio.v2 as imageio
import io

ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = ROOT / "data" / "processed" / "radar.zarr"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# --- Singapore domain (must match preprocess_radar.py) ---
SG_LON_MIN, SG_LON_MAX = 103.5, 104.1
SG_LAT_MIN, SG_LAT_MAX = 1.0, 1.6

# --- NEA-style rainfall colormap (transparent for 0, then blue→green→yellow→red→purple) ---
_RADAR_COLORS = [
    (0.00, (1.00, 1.00, 1.00, 0.0)),   # 0 mm/hr → fully transparent
    (0.01, (0.68, 0.85, 0.90, 0.6)),   # trace → light blue
    (0.05, (0.00, 1.00, 1.00, 0.7)),   # 0.5 mm/hr → cyan
    (0.12, (0.00, 0.78, 0.00, 0.8)),   # 2 mm/hr → green
    (0.25, (0.00, 1.00, 0.00, 0.85)),  # 5 mm/hr → bright green
    (0.40, (1.00, 1.00, 0.00, 0.9)),   # 10 mm/hr → yellow
    (0.60, (1.00, 0.65, 0.00, 0.95)),  # 20 mm/hr → orange
    (0.80, (1.00, 0.00, 0.00, 1.0)),   # 40 mm/hr → red
    (0.92, (0.78, 0.00, 0.78, 1.0)),   # 80 mm/hr → magenta
    (1.00, (0.50, 0.00, 0.50, 1.0)),   # 100+ mm/hr → purple
]

_RADAR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nea_radar",
    [(pos, rgba) for pos, rgba in _RADAR_COLORS],
)

VMAX = 80.0   # mm/hr — colour scale ceiling (purple = above this)


def load_zarr() -> xr.Dataset:
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    return ds


def make_map_axes(fig, rect, title: str = ""):
    """Return a cartopy GeoAxes with Singapore basemap."""
    ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
    ax.set_extent([SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX], crs=ccrs.PlateCarree())

    # Basemap features
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "land", "10m",
        facecolor="#e8e0d0", edgecolor="none", zorder=0,
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "ocean", "10m",
        facecolor="#c8dff5", edgecolor="none", zorder=0,
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "10m",
        facecolor="none", edgecolor="#555555", linewidth=0.6, zorder=2,
    ))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "10m",
        facecolor="none", edgecolor="#888888", linewidth=0.4, zorder=2,
    ))

    # Gridlines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.3, color="#aaaaaa", alpha=0.6, linestyle="--",
        x_inline=False, y_inline=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([103.6, 103.8, 104.0])
    gl.ylocator = mticker.FixedLocator([1.1, 1.3, 1.5])
    gl.xlabel_style = {"size": 5, "color": "#444"}
    gl.ylabel_style = {"size": 5, "color": "#444"}

    if title:
        ax.set_title(title, fontsize=7, pad=3)

    return ax


def plot_radar_frame(ax, lons, lats, rain_2d, alpha: float = 0.85):
    """Overlay rain rate on an existing cartopy axes."""
    masked = np.ma.masked_where(rain_2d < 0.3, rain_2d)
    mesh = ax.pcolormesh(
        lons, lats, masked,
        cmap=_RADAR_CMAP,
        vmin=0, vmax=VMAX,
        transform=ccrs.PlateCarree(),
        shading="auto",
        alpha=alpha,
        zorder=1,
    )
    return mesh


def find_top_frames(ds: xr.Dataset, n: int = 9):
    """Return indices of frames with highest spatial-max rain rate."""
    rain = ds["rain_rate"]
    # Compute max rain per frame in chunks to avoid loading 4000 frames at once
    chunk_size = 200
    maxvals = []
    n_t = len(rain.time)
    for i in range(0, n_t, chunk_size):
        chunk = rain.isel(time=slice(i, i + chunk_size)).values
        maxvals.append(chunk.max(axis=(1, 2)))
    maxvals = np.concatenate(maxvals)
    top_idx = np.argsort(maxvals)[::-1][:n]
    top_idx = sorted(top_idx)  # chronological order
    return top_idx, maxvals


def compute_timeseries(ds: xr.Dataset):
    """Return (times, domain_mean_rain) for the time series subplot."""
    rain = ds["rain_rate"]
    chunk_size = 200
    means = []
    n_t = len(rain.time)
    for i in range(0, n_t, chunk_size):
        chunk = rain.isel(time=slice(i, i + chunk_size)).values
        means.append(np.nanmean(chunk, axis=(1, 2)))
    means = np.concatenate(means)
    times = pd.DatetimeIndex(ds.time.values)
    return times, means


def overview_figure(ds: xr.Dataset, top_idx: list, maxvals: np.ndarray,
                    times_ts: pd.DatetimeIndex, means_ts: np.ndarray) -> Path:
    """9-panel rain event grid + time series → outputs/radar_overview.png"""
    lons = ds.lon.values
    lats = ds.lat.values
    times = pd.DatetimeIndex(ds.time.values)

    n_panels = min(9, len(top_idx))
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig = plt.figure(figsize=(14, 4 * nrows + 3), facecolor="#1a1a2e")
    fig.suptitle(
        "NEA Radar — Rain Rate over Singapore",
        fontsize=14, color="white", y=0.98, fontweight="bold",
    )

    # Layout: map grid in top portion, time series at bottom
    map_height = 0.80 * (nrows / (nrows + 0.8))
    bottom_strip = 0.13
    top_strip = 1.0 - bottom_strip - 0.05

    # ---- Map panels ----
    for panel_i, t_idx in enumerate(top_idx[:n_panels]):
        row = panel_i // ncols
        col = panel_i % ncols
        w = 0.30
        h = top_strip / nrows * 0.88
        left = 0.03 + col * 0.325
        bottom = top_strip - (row + 1) * (top_strip / nrows) + bottom_strip + 0.02

        ts_label = pd.Timestamp(times[t_idx]).strftime("%Y-%m-%d %H:%M UTC")
        max_rr = maxvals[t_idx]
        title = f"{ts_label}\nmax {max_rr:.1f} mm/hr"

        ax = make_map_axes(fig, [left, bottom, w, h], title=title)
        ax.set_facecolor("#1a1a2e")

        rain_2d = ds["rain_rate"].isel(time=t_idx).values
        mesh = plot_radar_frame(ax, lons, lats, rain_2d)

    # ---- Shared colorbar ----
    cbar_ax = fig.add_axes([0.94, bottom_strip + 0.04, 0.015, top_strip - 0.04])
    sm = plt.cm.ScalarMappable(cmap=_RADAR_CMAP, norm=mcolors.Normalize(0, VMAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Rain rate (mm/hr)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cbar.set_ticks([0, 1, 2, 5, 10, 20, 40, 80])
    cbar.ax.set_facecolor("#1a1a2e")

    # ---- Time series strip ----
    ax_ts = fig.add_axes([0.06, 0.02, 0.86, bottom_strip - 0.03], facecolor="#12122a")
    ax_ts.fill_between(times_ts, means_ts, alpha=0.7, color="#00ccff", linewidth=0)
    ax_ts.plot(times_ts, means_ts, color="#00ccff", linewidth=0.6)
    # Mark the frames shown in panels
    for t_idx in top_idx[:n_panels]:
        ax_ts.axvline(times_ts[t_idx], color="#ff6644", linewidth=0.8, alpha=0.7)
    ax_ts.set_ylabel("Domain mean\n(mm/hr)", color="white", fontsize=7)
    ax_ts.set_xlabel("Date (UTC)", color="white", fontsize=7)
    ax_ts.tick_params(colors="white", labelsize=6)
    ax_ts.spines[:].set_color("#555")
    ax_ts.xaxis.label.set_color("white")

    out_path = OUT_DIR / "radar_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def animation_gif(ds: xr.Dataset, centre_idx: int, half_window: int = 18) -> Path:
    """
    Generate a GIF of radar frames around a peak event.
    centre_idx: index of the peak frame
    half_window: frames either side (default ±18 = ±1.5 hours at 5-min cadence)
    """
    n_t = len(ds.time)
    start = max(0, centre_idx - half_window)
    end = min(n_t, centre_idx + half_window + 1)
    frame_indices = list(range(start, end))

    lons = ds.lon.values
    lats = ds.lat.values
    times = pd.DatetimeIndex(ds.time.values)
    rain_block = ds["rain_rate"].isel(time=slice(start, end)).values

    frames_png = []
    print(f"Rendering {len(frame_indices)} animation frames...")

    for i, t_idx in enumerate(frame_indices):
        fig = plt.figure(figsize=(6, 5), facecolor="#1a1a2e")
        ts_label = pd.Timestamp(times[t_idx]).strftime("%Y-%m-%d %H:%M UTC")
        max_rr = float(rain_block[i].max())

        ax = make_map_axes(fig, [0.05, 0.08, 0.82, 0.82],
                           title=f"NEA Radar — {ts_label}\nmax {max_rr:.1f} mm/hr")
        ax.set_facecolor("#1a1a2e")
        plot_radar_frame(ax, lons, lats, rain_block[i])

        # Colorbar
        cbar_ax = fig.add_axes([0.89, 0.1, 0.025, 0.75])
        sm = plt.cm.ScalarMappable(cmap=_RADAR_CMAP, norm=mcolors.Normalize(0, VMAX))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("mm/hr", color="white", fontsize=7)
        cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
        cbar.set_ticks([0, 1, 5, 10, 20, 40, 80])
        cbar.ax.set_facecolor("#1a1a2e")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        buf.seek(0)
        frames_png.append(imageio.imread(buf))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(frame_indices)} frames done")

    out_path = OUT_DIR / "radar_event.gif"
    imageio.mimsave(out_path, frames_png, fps=4, loop=0)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-anim", action="store_true", help="Skip GIF animation")
    parser.add_argument("--event", type=str, default=None,
                        help="Centre the animation on a specific ISO timestamp (e.g. 2026-06-01T14:00)")
    args = parser.parse_args()

    print("Loading radar.zarr...")
    ds = load_zarr()
    n_t = len(ds.time)
    times = pd.DatetimeIndex(ds.time.values)
    print(f"  {n_t} frames  |  {times[0]}  to  {times[-1]}")

    print("Computing frame statistics...")
    top_idx, maxvals = find_top_frames(ds, n=9)

    print("Computing domain time series...")
    times_ts, means_ts = compute_timeseries(ds)

    print("Generating overview figure...")
    overview_figure(ds, top_idx, maxvals, times_ts, means_ts)

    if not args.no_anim:
        if args.event:
            target = pd.Timestamp(args.event)
            diffs = np.abs((times - target).total_seconds())
            centre_idx = int(diffs.argmin())
        else:
            # Default: animate around the heaviest recorded frame
            centre_idx = int(np.argmax(maxvals))

        peak_time = pd.Timestamp(times[centre_idx]).strftime("%Y-%m-%d %H:%M UTC")
        print(f"Generating GIF around peak frame: {peak_time}")
        animation_gif(ds, centre_idx, half_window=18)

    ds.close()
    print("\nDone. Open outputs/ to view results.")


if __name__ == "__main__":
    main()
