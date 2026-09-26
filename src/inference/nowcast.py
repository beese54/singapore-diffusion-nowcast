"""
nowcast.py -- Probabilistic radar nowcast from the trained diffusion models.

ONE MODEL PER LEAD. Each checkpoint is stamped with the target_offset it was
trained for, and that stamp -- never a command-line number -- decides what it
forecasts. (The previous version of this script ran the 30-min model three
times and labelled the copies 30, 60 and 90 min; it also normalised with a
z-score the model was never trained on.) The input is built by the same
functions the training dataset uses (src.data.radar_dataset.build_context),
from the zarr frames alone, so there is no second copy of the transform to
drift out of sync.

A nominal "30-min" model predicts frame t+6 from frames t-6..t-1, i.e. 35 min
after the last observed frame; "60" and "90" are 65 and 95 min.

Usage
-----
    python src/inference/nowcast.py --time "2026-09-22 16:15" \\
        --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt \\
                     checkpoints/nowcaster/lead60/ckpt_step_100000.pt \\
                     checkpoints/nowcaster/lead90/ckpt_step_100000.pt

--time is the LAST OBSERVED radar frame (the issue time), in SGT unless --utc
is given; default is the newest frame in the archive. The context frames must
be present, 5 min apart and free of blank (NaN) frames, or the run is refused.

Output: results/nowcast_<issue time SGT>/
    lead<NN>.zarr   rain_rate (member, lat, lon) mm/hr, with issue/valid times
    summary.png     ensemble mean and P(>=10 mm/hr) for every lead
The total wall time (model loading, data read, sampling, saving) is printed:
the Stage 5 budget is < 5 min for 8 members x 3 leads on the RTX 4060.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_model  # noqa: E402  (the single stamp-aware loader)
from src.data.radar_dataset import (ZARR_PATH, build_context,  # noqa: E402
                                    denormalise_rain, load_stats)

SGT = np.timedelta64(8, "h")
STEP = np.timedelta64(5, "m")
HEAVY = 10.0            # mm/hr, the heavy-rain threshold used throughout Stage 5


def read_context(da: xr.DataArray, issue: np.datetime64, n: int) -> np.ndarray:
    """The n frames ending at `issue` (inclusive), oldest first, (n, H, W) mm/hr.
    Refuses rather than forecasting from a gappy or partly blank history."""
    times = da.time.values
    i = int(np.searchsorted(times, issue))
    if i >= len(times) or times[i] != issue:
        sys.exit(f"No radar frame at {issue} UTC in the archive.")
    if i + 1 < n:
        sys.exit(f"Need {n} frames of history before {issue} UTC; the archive starts later.")
    want = issue - np.arange(n - 1, -1, -1) * STEP
    got = times[i + 1 - n: i + 1]
    if not np.array_equal(got, want):
        missing = np.setdiff1d(want, got).astype("datetime64[m]")
        sys.exit(f"Context frames missing (archive gap), UTC: {[str(m) for m in missing]}")
    frames = da.isel(time=slice(i + 1 - n, i + 1)).values.astype(np.float32)
    if np.isnan(frames).any():
        sys.exit("A context frame is blank (failed scrape); refusing to forecast from it.")
    return frames


def save_lead(out: Path, ens: np.ndarray, da: xr.DataArray, issue: np.datetime64,
              valid: np.datetime64, lead_min: int, ckpt: Path) -> None:
    ds = xr.Dataset(
        {"rain_rate": (("member", "lat", "lon"), ens.astype(np.float32), {"units": "mm/hr"})},
        coords={"member": np.arange(ens.shape[0]), "lat": da.lat.values, "lon": da.lon.values},
        attrs={"issue_time_utc": str(issue), "valid_time_utc": str(valid),
               "lead_nominal_min": lead_min, "minutes_after_last_frame": lead_min + 5,
               "checkpoint": str(ckpt)})
    ds.to_zarr(out / f"lead{lead_min}.zarr", mode="w", consolidated=True)


def save_summary(out: Path, results: list[dict], da: xr.DataArray, issue: np.datetime64) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    levels = [0.5, 1, 2, 5, 10, 20, 30, 50, 100]
    cmap = ListedColormap(["#c6e8ff", "#7cc3f5", "#2f8fd8", "#3fbf4f",
                           "#f2e03c", "#f59a23", "#e8321e", "#9b1bb5"])
    cmap.set_under("white")
    norm = BoundaryNorm(levels, cmap.N)
    lat, lon = da.lat.values, da.lon.values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    hm = lambda t: str((t + SGT).astype("datetime64[m]"))[11:16]  # noqa: E731

    fig, axes = plt.subplots(len(results), 2, figsize=(10, 2.8 * len(results)),
                             squeeze=False, layout="constrained")
    for row, r in zip(axes, results):
        im = row[0].imshow(r["ens"].mean(0), extent=extent, origin="upper", cmap=cmap, norm=norm)
        row[0].set_title(f"+{r['lead']} min ensemble mean, valid {hm(r['valid'])} SGT", fontsize=9)
        ip = row[1].imshow((r["ens"] >= HEAVY).mean(0), extent=extent, origin="upper",
                           cmap="magma_r", vmin=0, vmax=1)
        row[1].set_title(f"+{r['lead']} min P(>= {HEAVY:g} mm/hr)", fontsize=9)
        for a in row:
            a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=axes[:, 0], shrink=0.7, label="mm/hr", extend="both")
    fig.colorbar(ip, ax=axes[:, 1], shrink=0.7, label="probability")
    fig.suptitle(f"Nowcast issued {hm(issue)} SGT (last radar frame)")
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Probabilistic radar nowcast, one model per lead")
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="one checkpoint per lead; each forecasts its own stamped lead")
    ap.add_argument("--time", help="last observed frame, 'YYYY-MM-DD HH:MM' (SGT unless --utc); "
                                   "default: newest frame in the archive")
    ap.add_argument("--utc", action="store_true", help="--time is UTC")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--seed", type=int, help="seed each lead's sampling (reproducible output)")
    ap.add_argument("--radar-zarr", default=str(ZARR_PATH))
    ap.add_argument("--out", help="output folder (default results/nowcast_<issue SGT>)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for c in args.checkpoint:
        m = load_model(ROOT / c, device)
        models.append((Path(c), m))
    offs = [m.target_offset for _, m in models]
    if len(set(offs)) != len(offs):
        sys.exit(f"Two checkpoints forecast the same lead (target_offsets {offs}); "
                 f"pass one checkpoint per lead.")
    models.sort(key=lambda cm: cm[1].target_offset)

    da = xr.open_zarr(args.radar_zarr, consolidated=True)["rain_rate"]
    if args.time:
        issue = np.datetime64(args.time.replace(" ", "T"), "m").astype("datetime64[ns]")
        if not args.utc:
            issue = issue - SGT
    else:
        issue = da.time.values[-1]
    cf_max = max(m.data_cfg["context_frames"] for _, m in models)
    frames = read_context(da, issue, cf_max)
    log_max = max(load_stats()["log_max"], 1e-6)       # as RadarDataset
    t_load = time.perf_counter() - t0

    out = Path(args.out) if args.out else \
        ROOT / "results" / f"nowcast_{str((issue + SGT).astype('datetime64[m]')).replace('-', '').replace(':', '').replace('T', '_')}"
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for ckpt, m in models:
        ts = time.perf_counter()
        cf, tc = m.data_cfg["context_frames"], m.data_cfg["time_channels"]
        ctx = build_context(frames[-cf:], issue, log_max, tc)
        if args.seed is not None:
            torch.manual_seed(args.seed)
        with torch.no_grad():
            e = m.ensemble_sample(torch.from_numpy(ctx).unsqueeze(0).to(device),
                                  n_members=args.members, eta=1.0)
        ens = denormalise_rain(e.cpu().float()[0, :, 0], log_max).numpy()   # (M, H, W)
        lead = m.target_offset * 5
        valid = issue + (m.target_offset + 1) * STEP
        save_lead(out, ens, da, issue, valid, lead, ckpt)
        results.append(dict(lead=lead, valid=valid, ens=ens))
        print(f"  +{lead} min ({lead + 5} min after the last frame), valid "
              f"{str((valid + SGT).astype('datetime64[m]'))[11:]} SGT: {args.members} members, "
              f"max {ens.max():.1f} mm/hr, {time.perf_counter() - ts:.1f} s")

    save_summary(out, results, da, issue)
    total = time.perf_counter() - t0
    print(f"\nIssued {str((issue + SGT).astype('datetime64[m]'))} SGT -> {out}")
    print(f"Wall time {total:.1f} s (load {t_load:.1f} s) for {args.members} members x "
          f"{len(results)} lead(s)")


if __name__ == "__main__":
    main()
