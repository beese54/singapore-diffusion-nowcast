"""
animate_radar.py — Timelapse MP4 of all collected NEA radar frames.

Strategy: render the Singapore basemap ONCE, then use PIL alpha-compositing
to overlay radar data per frame (<5ms/frame vs ~200ms for matplotlib redraw).

Output
------
  outputs/radar_timelapse.mp4

Usage
-----
    python scripts/animate_radar.py              # all frames, 20fps
    python scripts/animate_radar.py --fps 10     # slower playback
    python scripts/animate_radar.py --skip 2     # every 2nd frame (faster render)
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ZARR_PATH = ROOT / "data" / "processed" / "radar.zarr"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SG_LON_MIN, SG_LON_MAX = 103.5, 104.1
SG_LAT_MIN, SG_LAT_MAX = 1.0, 1.6
VMAX = 80.0
HEADER_H = 48   # px — dark strip at top for timestamp text

_RADAR_COLORS = [
    (0.000, (1.00, 1.00, 1.00, 0.00)),
    (0.010, (0.68, 0.85, 0.90, 0.60)),
    (0.050, (0.00, 1.00, 1.00, 0.72)),
    (0.120, (0.00, 0.78, 0.00, 0.82)),
    (0.250, (0.00, 1.00, 0.00, 0.87)),
    (0.400, (1.00, 1.00, 0.00, 0.92)),
    (0.600, (1.00, 0.65, 0.00, 0.96)),
    (0.800, (1.00, 0.00, 0.00, 1.00)),
    (0.920, (0.78, 0.00, 0.78, 1.00)),
    (1.000, (0.50, 0.00, 0.50, 1.00)),
]
_RADAR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nea_radar", [(p, rgba) for p, rgba in _RADAR_COLORS]
)
_NORM = mcolors.Normalize(0, VMAX)


def _try_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_basemap() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Render Singapore basemap with colorbar.
    Returns:
        bg_rgb  — uint8 (H, W, 3) numpy array
        map_box — (left, top, right, bottom) pixel coords of the data extent
                  within bg_rgb (i.e. where to paste radar overlay)
    """
    fig = plt.figure(figsize=(8.0, 6.0), dpi=100, facecolor="#1a1a2e")

    # Map axes — leave right margin for colorbar
    ax = fig.add_axes([0.01, 0.04, 0.88, 0.94], projection=ccrs.PlateCarree())
    ax.set_extent([SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_facecolor("#1a1a2e")

    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "land", "10m", facecolor="#e8e0d0", edgecolor="none", zorder=0))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "ocean", "10m", facecolor="#c4d9f0", edgecolor="none", zorder=0))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "coastline", "10m",
        facecolor="none", edgecolor="#555", linewidth=0.7, zorder=2))
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_0_boundary_lines_land", "10m",
        facecolor="none", edgecolor="#888", linewidth=0.4, zorder=2))

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#777",
                      alpha=0.6, linestyle="--", x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([103.6, 103.8, 104.0])
    gl.ylocator = mticker.FixedLocator([1.1, 1.3, 1.5])
    gl.xlabel_style = {"size": 7, "color": "#ccc"}
    gl.ylabel_style = {"size": 7, "color": "#ccc"}

    # Colorbar
    cbar_ax = fig.add_axes([0.91, 0.06, 0.025, 0.88])
    sm = plt.cm.ScalarMappable(cmap=_RADAR_CMAP, norm=_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("mm / hr", color="white", fontsize=8, labelpad=4)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=6)
    cbar.set_ticks([0, 1, 2, 5, 10, 20, 40, 80])
    cbar.ax.set_facecolor("#1a1a2e")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Pixel extent of the map data area within the figure
    # transData: (lon, lat) → figure-level pixel coords (origin = bottom-left)
    fig_h = fig.canvas.get_width_height()[1]
    def data_to_img(lon, lat):
        """Convert data (lon,lat) → image pixel (x, y) with y=0 at top."""
        px, py = ax.transData.transform((lon, lat))
        return int(round(px)), int(round(fig_h - py))

    img_left, img_top    = data_to_img(SG_LON_MIN, SG_LAT_MAX)
    img_right, img_bottom = data_to_img(SG_LON_MAX, SG_LAT_MIN)

    # Clamp to figure bounds
    W, H = fig.canvas.get_width_height()
    img_left   = max(0, img_left)
    img_top    = max(0, img_top)
    img_right  = min(W, img_right)
    img_bottom = min(H, img_bottom)

    bg_rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(H, W, 4)[:, :, :3].copy()
    plt.close(fig)

    return bg_rgb, (img_left, img_top, img_right, img_bottom)


def rain_to_rgba(rain_2d: np.ndarray) -> np.ndarray:
    """Apply radar colormap; set transparent where rain < 0.3 mm/hr."""
    rgba = _RADAR_CMAP(_NORM(np.clip(rain_2d, 0, VMAX)))  # (H, W, 4), float 0–1
    rgba[rain_2d < 0.3] = 0.0
    return (rgba * 255).astype(np.uint8)


def build_frame(
    bg_pil: Image.Image,
    rain_2d: np.ndarray,
    map_box: tuple[int, int, int, int],
    timestamp_str: str,
    max_rr: float,
    font_lg: ImageFont.ImageFont,
    font_sm: ImageFont.ImageFont,
) -> np.ndarray:
    """Composite one radar frame onto the basemap and prepend header. Returns BGR uint8."""
    img_left, img_top, img_right, img_bottom = map_box
    map_w = img_right - img_left
    map_h = img_bottom - img_top

    # Radar → RGBA PIL image, resized to match the map pixel box
    radar_rgba = rain_to_rgba(rain_2d)           # (190, 220, 4)
    radar_pil  = Image.fromarray(radar_rgba, "RGBA")
    radar_pil  = radar_pil.resize((map_w, map_h), Image.BILINEAR)

    # Alpha-composite onto a copy of the basemap
    frame_pil = bg_pil.copy()
    frame_pil.alpha_composite(radar_pil, dest=(img_left, img_top))

    # Convert to RGB numpy
    frame_rgb = np.array(frame_pil.convert("RGB"))

    # Prepend header strip
    H, W = frame_rgb.shape[:2]
    full = np.empty((H + HEADER_H, W, 3), dtype=np.uint8)
    full[HEADER_H:] = frame_rgb

    # Draw header using PIL
    hdr = Image.new("RGB", (W, HEADER_H), color=(18, 18, 38))
    draw = ImageDraw.Draw(hdr)
    draw.text((10, 8),  timestamp_str,               fill=(180, 200, 255), font=font_lg)
    draw.text((10, 26), f"max {max_rr:.1f} mm/hr",   fill=(255, 140,  80), font=font_sm)
    draw.text((W - 230, 8),  "NEA X-band radar",     fill=(120, 120, 160), font=font_sm)
    draw.text((W - 230, 24), "SG Weather POC",       fill=( 80,  80, 120), font=font_sm)
    full[:HEADER_H] = np.array(hdr)

    # Return BGR for OpenCV
    return full[:, :, ::-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",  type=int, default=20, help="Playback frame rate (default 20)")
    parser.add_argument("--skip", type=int, default=1,  help="Use every Nth frame (default 1 = all)")
    args = parser.parse_args()

    print("Loading radar.zarr...")
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    times = pd.DatetimeIndex(ds.time.values)
    n_t   = len(times)
    print(f"  {n_t} frames  |  {times[0]}  to  {times[-1]}")

    frame_indices = list(range(0, n_t, args.skip))
    n_frames = len(frame_indices)
    print(f"  Animating {n_frames} frames at {args.fps} fps  "
          f"(duration ~{n_frames / args.fps:.0f}s)")

    print("Rendering basemap (once)...")
    bg_rgb, map_box = render_basemap()
    bg_pil = Image.fromarray(bg_rgb).convert("RGBA")
    BG_H, BG_W = bg_rgb.shape[:2]
    print(f"  Basemap: {BG_W}x{BG_H}px  |  map box: {map_box}")

    font_lg = _try_font(14)
    font_sm = _try_font(11)

    out_path = OUT_DIR / "radar_timelapse.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    video_w  = BG_W
    video_h  = BG_H + HEADER_H
    vout     = cv2.VideoWriter(str(out_path), fourcc, args.fps, (video_w, video_h))

    print(f"Writing {out_path} ({video_w}x{video_h}) ...")

    rain_var = ds["rain_rate"]
    CHUNK = 288  # matches zarr chunk size — reads one chunk per .values call

    done = 0
    for chunk_start in range(0, n_t, CHUNK):
        chunk_end    = min(n_t, chunk_start + CHUNK)
        chunk_rain   = rain_var.isel(time=slice(chunk_start, chunk_end)).values  # loads one chunk

        for local_i in range(chunk_end - chunk_start):
            global_i = chunk_start + local_i
            if global_i not in frame_indices:
                continue

            rain_2d   = chunk_rain[local_i]
            max_rr    = float(np.nanmax(rain_2d))
            ts        = pd.Timestamp(times[global_i])
            ts_str    = ts.strftime("%Y-%m-%d  %H:%M UTC")

            bgr = build_frame(bg_pil, rain_2d, map_box, ts_str, max_rr, font_lg, font_sm)
            vout.write(bgr)
            done += 1

        pct = done / n_frames * 100
        print(f"  {done}/{n_frames} frames  ({pct:.1f}%)", end="\r", flush=True)

    vout.release()
    ds.close()

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone.  {out_path}  ({size_mb:.1f} MB)")
    print(f"Duration: {n_frames / args.fps:.1f}s at {args.fps} fps")


if __name__ == "__main__":
    main()
