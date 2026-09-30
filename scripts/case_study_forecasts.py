#!/usr/bin/env python3
"""
case_study_forecasts.py -- Ensemble forecasts for the 22 Sep 2026 flash floods.

Produces the inputs for notebooks/02_nowcast_evaluation.ipynb and
notebooks/03_flood_risk_overlay.ipynb, so the notebooks never need the GPU or
the full dataset and can run while a model trains.

For each target frame every 10 min across the storm (07:30-10:30 UTC =
15:30-18:30 SGT) it forecasts with the checkpoint's OWN lead (its stamped
target_offset), i.e. forecasts are issued `lead` minutes earlier. Output:

  data/processed/eval_cache/case_22sep_lead<nominal minutes>.npz
    ens          (N, members, H, W) float16 mm/hr
    target_times (N,) datetime64[m], UTC
    anchor_idx   (N,) archive index of each forecast's anchor
    lead_steps   target_offset in 5-min steps

Usage:
    python scripts/case_study_forecasts.py --checkpoint checkpoints/nowcaster/ckpt_step_300000.pt
    python scripts/case_study_forecasts.py --checkpoint checkpoints/nowcaster/lead60/ckpt_step_100000.pt
    # any other event (UTC window, name used in the output file):
    python scripts/case_study_forecasts.py --checkpoint ... --start 2026-09-27T02:30 --end 2026-09-27T04:30 --name 27sep
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import load_model  # noqa: E402
from src.data.radar_dataset import (ZARR_PATH, build_context,  # noqa: E402
                                    denormalise_rain, load_stats)

START = np.datetime64("2026-09-22T07:30")
END = np.datetime64("2026-09-22T10:30")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--start", default=str(START), help="first target, UTC (default: 22 Sep case)")
    ap.add_argument("--end", default=str(END), help="last target, UTC")
    ap.add_argument("--name", default="22sep", help="case name in the output file")
    args = ap.parse_args()
    start, end = np.datetime64(args.start, "m"), np.datetime64(args.end, "m")

    model = load_model(ROOT / args.checkpoint, "cuda" if torch.cuda.is_available() else "cpu")
    off = model.target_offset
    cf, tc = model.data_cfg["context_frames"], model.data_cfg["time_channels"]
    log_max = max(load_stats()["log_max"], 1e-6)       # as RadarDataset

    # Only this afternoon's frames are read (not the whole archive, ~1 GB in
    # RAM), through the same build_context() the training dataset uses.
    da = xr.open_zarr(ZARR_PATH, consolidated=True)["rain_rate"]
    times = da.time.values.astype("datetime64[m]")
    targets = np.arange(start, end + np.timedelta64(1, "m"), np.timedelta64(10, "m"))
    # Context = the cf frames ending (off + 1) steps before the target, looked up
    # by time, so a frame NEA never published elsewhere in the storm does not
    # shift the window. A target whose own frames have a gap is skipped.
    pos = {t: i for i, t in enumerate(times)}
    step = np.timedelta64(5, "m")
    keep, ctx_idx = [], []
    for k, t in enumerate(targets):
        need = [t - (off + cf - j) * step for j in range(cf)]
        if t in pos and all(n in pos for n in need):
            keep.append(k)
            ctx_idx.append([pos[n] for n in need])
        else:
            print(f"skip target {t}: its context or target frame is missing from the archive")
    if not keep:
        sys.exit("no target has a complete context")
    lo, hi = min(c[0] for c in ctx_idx), max(c[-1] for c in ctx_idx) + 1
    block = da.isel(time=slice(lo, hi)).values.astype(np.float32)
    if np.isnan(block).any():
        sys.exit("a context frame is blank (failed scrape)")

    ens = np.empty((len(keep), args.members, *block.shape[1:]), np.float16)
    dev = next(model.parameters()).device
    for n, (k, c) in enumerate(zip(keep, ctx_idx)):
        ctx = build_context(block[c[0] - lo: c[-1] + 1 - lo], times[c[-1]], log_max, tc,
                            model.data_cfg.get("motion_minutes", 0))
        torch.manual_seed(9000 + k)                 # k: position in the full target list
        with torch.no_grad():
            e = model.ensemble_sample(torch.from_numpy(ctx).unsqueeze(0).to(dev),
                                      n_members=args.members, eta=1.0)
        ens[n] = denormalise_rain(e.cpu().float()[0, :, 0], log_max).numpy()

    out = ROOT / "data" / "processed" / "eval_cache" / f"case_{args.name}_lead{off * 5}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ens=ens, target_times=targets[keep].astype("datetime64[m]"),
                        anchor_idx=np.array([c[-1] + 1 for c in ctx_idx]), lead_steps=off)
    print(f"{out.name}: {len(keep)} forecasts x {args.members} members, "
          f"lead {off * 5} min nominal ({(off + 1) * 5} min from last frame)")


if __name__ == "__main__":
    main()
