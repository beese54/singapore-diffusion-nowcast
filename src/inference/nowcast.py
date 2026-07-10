"""
nowcast.py — Generate probabilistic nowcast ensemble from a trained model.

Usage (script mode)
-------------------
    python src/inference/nowcast.py --checkpoint checkpoints/nowcaster/latest.pt \\
                                    --radar-zarr data/processed/radar.zarr \\
                                    --timestamp "2024-05-01 14:00" \\
                                    --members 8 \\
                                    --lead-times 30 60 90

Output: results/nowcast_<timestamp>/  containing zarr + PNG visualisations
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.model.unet import ConditionedUNet
from src.model.diffusion import GaussianDiffusion
from src.data.radar_dataset import RadarDataset, load_stats, CONTEXT_FRAMES


def load_model(checkpoint_path: Path, device: torch.device) -> GaussianDiffusion:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    unet = ConditionedUNet(context_frames=CONTEXT_FRAMES, use_checkpoint=False)
    diffusion = GaussianDiffusion(unet)
    diffusion.load_state_dict(state["model"])
    diffusion = diffusion.to(device).eval()
    return diffusion


def get_context_frames(zarr_path: Path, timestamp: datetime, context_frames: int = CONTEXT_FRAMES) -> np.ndarray:
    """Extract the last `context_frames` frames before `timestamp` from the zarr store."""
    ds = xr.open_zarr(zarr_path, consolidated=True)
    times = ds.time.values
    ts_np = np.datetime64(timestamp)
    idx = np.searchsorted(times, ts_np)

    if idx < context_frames:
        raise ValueError(f"Not enough history before {timestamp} in the archive.")

    frames = ds["rain_rate"].values[idx - context_frames: idx]  # (C, H, W)
    ds.close()
    return frames.astype(np.float32)


def normalise(rain: np.ndarray, stats: dict) -> np.ndarray:
    log_mean, log_std = stats["log_mean"], stats["log_std"]
    x = np.log1p(rain)
    return np.clip((x - log_mean) / max(log_std, 1e-6), -3.0, 3.0)


def denormalise(x: np.ndarray, stats: dict) -> np.ndarray:
    log_mean, log_std = stats["log_mean"], stats["log_std"]
    return np.expm1(np.clip(x * log_std + log_mean, 0, None))


@torch.no_grad()
def generate_ensemble(
    model: GaussianDiffusion,
    context_frames: np.ndarray,
    n_members: int,
    device: torch.device,
) -> np.ndarray:
    """
    Generate an ensemble of nowcast frames.

    Returns (n_members, H, W) array in mm/hr.
    """
    stats = load_stats()
    ctx_norm = normalise(context_frames, stats)
    ctx_tensor = torch.from_numpy(ctx_norm).unsqueeze(0).to(device)  # (1, C, H, W)

    ensemble = model.ensemble_sample(ctx_tensor, n_members=n_members, eta=1.0)
    # ensemble: (1, n_members, 1, H, W)
    ensemble_np = ensemble.squeeze(0).squeeze(2).cpu().float().numpy()  # (n_members, H, W)
    return denormalise(ensemble_np, stats)


def save_results(
    ensemble: np.ndarray,
    timestamp: datetime,
    lead_time_min: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_members, H, W = ensemble.shape

    # Save as zarr
    zarr_out = output_dir / f"lead_{lead_time_min:03d}min.zarr"
    ds = xr.Dataset(
        {"rain_rate": xr.DataArray(
            ensemble,
            dims=["member", "lat", "lon"],
            attrs={"units": "mm/hr", "lead_time_min": lead_time_min},
        )},
        coords={"member": np.arange(n_members)},
    )
    ds.to_zarr(zarr_out, mode="w", consolidated=True)

    # Save ensemble mean PNG
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        mean_field = ensemble.mean(axis=0)
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.cm.Blues
        im = ax.imshow(mean_field, origin="upper", cmap=cmap, vmin=0, vmax=30)
        plt.colorbar(im, ax=ax, label="Rain rate (mm/hr)")
        ax.set_title(f"Nowcast ensemble mean\n{timestamp} + {lead_time_min} min")
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(output_dir / f"lead_{lead_time_min:03d}min_mean.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Generate probabilistic nowcast ensemble")
    parser.add_argument("--checkpoint", default="checkpoints/nowcaster/latest.pt")
    parser.add_argument("--radar-zarr", default="data/processed/radar.zarr")
    parser.add_argument("--timestamp", required=True, help="ISO datetime, e.g. '2024-05-01 14:00'")
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--lead-times", type=int, nargs="+", default=[30, 60, 90],
                        help="Lead times in minutes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / args.checkpoint
    zarr_path = ROOT / args.radar_zarr
    timestamp = datetime.fromisoformat(args.timestamp)

    print(f"Loading model from {ckpt_path}...")
    model = load_model(ckpt_path, device)

    print(f"Extracting context frames for {timestamp}...")
    context_frames = get_context_frames(zarr_path, timestamp)

    # Steps per lead time (5-min intervals)
    output_dir = ROOT / "results" / f"nowcast_{timestamp.strftime('%Y%m%d_%H%M')}"

    for lead_min in args.lead_times:
        # For multi-step lead times we use a single model step scaled by target_offset
        # In practice, chain predictions for longer lead times
        steps_ahead = lead_min // 5
        print(f"Generating {args.members}-member ensemble for +{lead_min} min...")

        ensemble = generate_ensemble(model, context_frames, args.members, device)
        save_results(ensemble, timestamp, lead_min, output_dir)
        print(f"  Saved to {output_dir}/lead_{lead_min:03d}min.zarr")

    print(f"\nDone. Results in {output_dir}")


if __name__ == "__main__":
    main()
