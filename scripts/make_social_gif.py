#!/usr/bin/env python3
"""
make_social_gif.py -- Animated summary of an out-of-sample flash flood for posts.

Left: observed radar at each target time. Right: the 30-min model's chance of
>=10 mm/hr for that time, from the forecast issued 35 min earlier. Bottom: rain
and warning probability at the flood site, building up frame by frame.
Every frame carries the NEA attribution (docs/DATA.md).

Inputs: data/processed/radar.zarr, data/processed/eval_cache/case_27sep_lead30.npz
Output: docs/img/linkedin/27sep_nowcast.gif
Usage:  python scripts/make_social_gif.py
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img" / "linkedin" / "27sep_nowcast.gif"
SGT = np.timedelta64(8, "h")
SITE, R = (63, 78), 3                       # Neo Pee Teck Lane / Pasir Panjang Rd
FLOOD = "11:57"
CREDIT = ("Radar imagery © NEA / Meteorological Service Singapore (weather.gov.sg) — personal, "
          "non-commercial, informational use only. Research prototype, not a warning service.")

LEVELS = [0.5, 1, 2, 5, 10, 20, 30, 50, 100]
CMAP = ListedColormap(["#c6e8ff", "#7cc3f5", "#2f8fd8", "#3fbf4f", "#f2e03c", "#f59a23", "#e8321e", "#9b1bb5"])
CMAP.set_under((0, 0, 0, 0))
NORM = BoundaryNorm(LEVELS, CMAP.N)
BG, INK, MUTED, ACCENT = "#0c1319", "#e5ebf0", "#98a6b3", "#5cc3dd"


def hm(t):
    return str((np.datetime64(t, "m") + SGT))[11:16]


def main():
    z = np.load(ROOT / "data" / "processed" / "eval_cache" / "case_27sep_lead30.npz")
    ens, off = z["ens"].astype(np.float32), int(z["lead_steps"])
    times = z["target_times"].astype("datetime64[ns]")
    da = xr.open_zarr(ROOT / "data" / "processed" / "radar.zarr", consolidated=True)["rain_rate"]
    lat, lon = da.lat.values, da.lon.values
    ext = [lon.min(), lon.max(), lat.min(), lat.max()]
    obs = np.stack([da.sel(time=t).values for t in times])
    i, j = SITE
    box = (slice(i - R, i + R + 1), slice(j - R, j + R + 1))
    site_obs = obs[:, box[0], box[1]].max(axis=(1, 2))
    site_p10 = (ens[:, :, box[0], box[1]].max(axis=(2, 3)) >= 10).mean(1)
    p10 = (ens >= 10).mean(1)
    slat, slon = lat[i], lon[j]

    import cartopy.feature as cfeature
    land = list(cfeature.NaturalEarthFeature("physical", "land", "10m").intersecting_geometries(
        [ext[0] - 0.05, ext[1] + 0.05, ext[2] - 0.05, ext[3] + 0.05]))

    frames, durations = [], []
    first = int(np.argmax(site_p10 >= 0.25))
    for k, t in enumerate(times):
        fig = plt.figure(figsize=(12, 6.75), dpi=100, facecolor=BG)
        fig.text(0.03, 0.935, "27 Sep 2026 · Pasir Panjang flash flood — a 30-minute AI nowcast vs what happened",
                 color=INK, fontsize=16, weight="bold")
        fig.text(0.03, 0.895, "Forecast made from radar only, 35 min before each time shown. Out of sample: "
                              "the model never saw this day.", color=MUTED, fontsize=10.5)
        axes = [fig.add_axes([0.03, 0.39, 0.46, 0.46]), fig.add_axes([0.51, 0.39, 0.46, 0.46])]
        panels = [(obs[k], CMAP, NORM, f"What the radar showed at {hm(t)}"),
                  (p10[k], "magma_r", None, f"Forecast made at {hm(t - np.timedelta64((off + 1) * 5, 'm'))}: "
                                             f"chance of heavy rain (≥10 mm/hr)")]
        for ax, (field, cmap, norm, title) in zip(axes, panels):
            ax.set_facecolor("#060c11")
            if norm is not None:
                ax.imshow(np.ma.masked_less(field, 0.5), extent=ext, origin="upper", cmap=cmap, norm=norm,
                          interpolation="nearest")
            else:
                ax.imshow(np.ma.masked_less(field, 0.01), extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=1,
                          interpolation="nearest")
            for g in land:
                for poly in getattr(g, "geoms", [g]):
                    x, y = poly.exterior.xy
                    ax.plot(x, y, color=(0.85, 0.9, 0.93, 0.5), lw=0.7)
            ax.plot(slon, slat, "o", ms=11, mfc="none", mec="white", mew=3.5)
            ax.plot(slon, slat, "o", ms=11, mfc="none", mec="#ff2d20", mew=2)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#243240")
            ax.set_title(title, color=INK, fontsize=11, loc="left")
        # time series
        ax = fig.add_axes([0.07, 0.1, 0.86, 0.22], facecolor=BG)
        tt = [hm(x) for x in times]
        xs = np.arange(len(times))
        ax.bar(xs[:k + 1], site_p10[:k + 1] * 100, color="#fb923c", alpha=0.85, width=0.55,
               label="forecast chance of ≥10 mm/hr at the site (%)")
        ax.plot(xs[:k + 1], site_obs[:k + 1], color=INK, lw=2.2, marker="o", ms=3,
                label="observed rain at the site (mm/hr)")
        fl = (int(FLOOD[:2]) * 60 + int(FLOOD[3:]) - (int(tt[0][:2]) * 60 + int(tt[0][3:]))) / 10
        ax.axvline(fl, color="#ff2d20", ls="--", lw=1.4)
        # headroom above 100 holds the legend and this label, clear of the data
        ax.text(fl + 0.1, 116, f"flash flood reported {FLOOD}", color="#ff6b5f", fontsize=9.5, va="center")
        ax.set_xlim(-0.6, len(times) - 0.4); ax.set_ylim(0, 128)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(xs[::2]); ax.set_xticklabels(tt[::2], color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=9)
        for s in ax.spines.values():
            s.set_color("#243240")
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=MUTED, ncol=2,
                  borderaxespad=0.2)
        if k >= first:
            # sits in the dry stretch before the storm, where no data is drawn
            ax.annotate(f"first warning issued {hm(times[first] - np.timedelta64((off + 1) * 5, 'm'))},\n"
                        f"dry on radar: 82 min before the flood report",
                        xy=(first, site_p10[first] * 100), xytext=(-0.45, 58), color=ACCENT, fontsize=9.5,
                        arrowprops=dict(arrowstyle="->", color=ACCENT))
        fig.text(0.03, 0.02, CREDIT, color=MUTED, fontsize=7.8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
        durations.append(2600 if k == first else (2200 if k == len(times) - 1 else 900))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"{OUT}: {len(frames)} frames, {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
