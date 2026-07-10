"""
evaluate.py — Evaluate the trained nowcaster against a persistence baseline.

Metrics
-------
- FSS (Fractions Skill Score): spatial skill at given intensity threshold
- CRPS (Continuous Ranked Probability Score): ensemble calibration
- MAE / RMSE on ensemble mean

Outputs
-------
- results/evaluation_report.json
- results/fss_curves.png

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --threshold 20 --lead 30
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.unet import ConditionedUNet
from src.model.diffusion import GaussianDiffusion
from src.data.radar_dataset import RadarDataset, load_stats, CONTEXT_FRAMES, TARGET_OFFSET

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fss(pred: np.ndarray, obs: np.ndarray, threshold: float, scale: int = 1) -> float:
    """
    Fractions Skill Score for a single threshold.

    pred, obs : (H, W) arrays in mm/hr
    threshold : rain-rate threshold in mm/hr
    scale     : neighbourhood radius in pixels (1 = pixel-by-pixel)
    """
    from scipy.ndimage import uniform_filter

    pred_frac = uniform_filter((pred >= threshold).astype(float), size=scale * 2 + 1)
    obs_frac = uniform_filter((obs >= threshold).astype(float), size=scale * 2 + 1)

    fss_num = np.mean((pred_frac - obs_frac) ** 2)
    fss_ref = np.mean(pred_frac ** 2) + np.mean(obs_frac ** 2)

    if fss_ref < 1e-10:
        return 1.0  # both empty → perfect
    return 1.0 - fss_num / fss_ref


def crps_ensemble(ensemble: np.ndarray, obs: np.ndarray) -> float:
    """
    Mean CRPS for an ensemble forecast.

    ensemble : (M, H, W)
    obs      : (H, W)
    """
    M = ensemble.shape[0]
    # CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    mae_term = np.mean(np.abs(ensemble - obs[np.newaxis, ...]))
    spread = 0.0
    count = 0
    for i in range(M):
        for j in range(i + 1, M):
            spread += np.mean(np.abs(ensemble[i] - ensemble[j]))
            count += 1
    spread_term = spread / max(count, 1)
    return float(mae_term - 0.5 * spread_term)


def load_model(checkpoint_path: Path, device: torch.device) -> GaussianDiffusion:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    unet = ConditionedUNet(context_frames=CONTEXT_FRAMES, use_checkpoint=False)
    diffusion = GaussianDiffusion(unet)
    diffusion.load_state_dict(state["model"])
    return diffusion.to(device).eval()


def persistence_forecast(context_frames: np.ndarray) -> np.ndarray:
    """Naive baseline: predict the last observed frame unchanged."""
    return context_frames[-1]  # (H, W)


def score_flood_events(model, device, stats, args) -> dict:
    """Score model against geocoded real flood events from flood_eval_dataset.parquet."""
    import pandas as pd

    parquet_path = ROOT / "data" / "processed" / "flood_eval_dataset.parquet"
    if not parquet_path.exists():
        return {"error": "flood_eval_dataset.parquet not found — run build_flood_eval_dataset.py first"}

    df = pd.read_parquet(parquet_path)
    events = df[(df["geocoded"] == True) & (df["is_flood_event"] == True)].copy()

    if events.empty:
        return {"n_events": 0, "note": "No geocoded flood events available yet"}

    radar_zarr = ROOT / "data" / "processed" / "radar.zarr"
    ds = xr.open_zarr(str(radar_zarr), consolidated=True)
    radar_times = ds.time.values

    def denorm(x):
        return np.expm1(np.clip(x * stats["log_std"] + stats["log_mean"], 0, None))

    lead_steps = args.lead // 5  # convert lead time to 5-min steps
    context_needed = CONTEXT_FRAMES + lead_steps

    hits_model = 0
    hits_persist = 0
    n_scored = 0

    for _, row in events.iterrows():
        lat_idx = int(row["location_lat_idx"])
        lon_idx = int(row["location_lon_idx"])

        # Find radar frame matching the flood event time
        event_t = np.datetime64(row["matched_radar_time"])
        idx = int(np.searchsorted(radar_times, event_t))

        # We need context frames that end `lead_steps` before the event
        ctx_end = idx - lead_steps
        ctx_start = ctx_end - CONTEXT_FRAMES
        if ctx_start < 0 or ctx_end >= len(radar_times):
            continue

        ctx_frames = ds["rain_rate"].values[ctx_start:ctx_end].astype(np.float32)  # (C, H, W)
        obs_frame = ds["rain_rate"].values[idx].astype(np.float32)                  # (H, W)

        # Persistence: predict last context frame
        persist_pred_cell = float(ctx_frames[-1, lat_idx, lon_idx])
        hit_persist = persist_pred_cell >= args.threshold

        # Model ensemble mean at flood cell
        ctx_norm = np.stack([
            np.clip((np.log1p(f) - stats["log_mean"]) / max(stats["log_std"], 1e-6), -3.0, 3.0)
            for f in ctx_frames
        ])
        ctx_tensor = torch.from_numpy(ctx_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            ens = model.ensemble_sample(ctx_tensor, n_members=args.members, eta=1.0)
            ens_np = ens.cpu().float().numpy()[0, :, 0, :, :]  # (M, H, W)
        ens_mm = denorm(ens_np)
        model_pred_cell = float(ens_mm.mean(axis=0)[lat_idx, lon_idx])
        hit_model = model_pred_cell >= args.threshold

        hits_model += int(hit_model)
        hits_persist += int(hit_persist)
        n_scored += 1

    ds.close()

    note = (
        "Events are FLOOD_RISK warnings (rain approaches, flood lags rain). "
        "Low POD expected until FLASH_FLOOD events accumulate."
    )

    return {
        "n_events_geocoded": len(events),
        "n_events_scored": n_scored,
        "lead_time_min": args.lead,
        "threshold_mm_hr": args.threshold,
        "pod_model": round(hits_model / max(n_scored, 1), 3),
        "pod_persistence": round(hits_persist / max(n_scored, 1), 3),
        "note": note,
    }


def load_random_model(device: torch.device) -> GaussianDiffusion:
    """Instantiate model with random weights for smoke testing."""
    unet = ConditionedUNet(context_frames=CONTEXT_FRAMES, use_checkpoint=False)
    diffusion = GaussianDiffusion(unet)
    return diffusion.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/nowcaster/latest.pt")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Rain-rate threshold for FSS in mm/hr (default 2.0 ≈ 20dBZ)")
    parser.add_argument("--lead", type=int, default=30, help="Lead time in minutes")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: random-weight model, 10 samples, no stage5 flag")
    parser.add_argument("--flood-eval", action="store_true",
                        help="Also score against geocoded flood events in flood_eval_dataset.parquet")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        print("[SMOKE] Using random-weight model (no checkpoint loaded)")
        model = load_random_model(device)
        args.n_samples = min(args.n_samples, 10)
    else:
        ckpt_path = ROOT / args.checkpoint
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        model = load_model(ckpt_path, device)

    stats = load_stats()

    target_offset = args.lead // 5  # 5-min steps

    test_ds = RadarDataset("test", context_frames=CONTEXT_FRAMES, target_offset=target_offset)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    def denorm(x):
        return np.expm1(np.clip(x * stats["log_std"] + stats["log_mean"], 0, None))

    model_fss_scores = []
    persist_fss_scores = []
    crps_scores = []
    mae_scores = []
    n_eval = min(args.n_samples, len(test_ds))

    print(f"Evaluating on {n_eval} test samples, threshold={args.threshold} mm/hr, lead={args.lead}min")

    for i, batch in enumerate(test_loader):
        if i >= n_eval:
            break

        ctx = batch["context"].to(device)           # (1, C, H, W)
        tgt = batch["target"].squeeze(0).squeeze(0).numpy()  # (H, W) normalised

        # Generate ensemble
        with torch.no_grad():
            ens = model.ensemble_sample(ctx, n_members=args.members, eta=1.0)
            ens_np = ens.cpu().float().numpy()[0, :, 0, :, :]  # (M, H, W)

        obs_mm = denorm(tgt)
        ens_mm = denorm(ens_np)
        ens_mean = ens_mm.mean(axis=0)

        ctx_mm = denorm(batch["context"].squeeze(0).numpy())  # (C, H, W)
        persist_mm = persistence_forecast(ctx_mm)

        model_fss_scores.append(fss(ens_mean, obs_mm, args.threshold))
        persist_fss_scores.append(fss(persist_mm, obs_mm, args.threshold))
        crps_scores.append(crps_ensemble(ens_mm, obs_mm))
        mae_scores.append(float(np.mean(np.abs(ens_mean - obs_mm))))

        if i % 20 == 0:
            print(f"  {i}/{n_eval}  FSS(model)={np.mean(model_fss_scores):.3f}  "
                  f"FSS(persist)={np.mean(persist_fss_scores):.3f}")

    report = {
        "lead_time_min": args.lead,
        "threshold_mm_hr": args.threshold,
        "n_samples": n_eval,
        "model_fss_mean": float(np.mean(model_fss_scores)),
        "persistence_fss_mean": float(np.mean(persist_fss_scores)),
        "fss_skill": float(np.mean(model_fss_scores) - np.mean(persist_fss_scores)),
        "crps_mean": float(np.mean(crps_scores)),
        "mae_mean": float(np.mean(mae_scores)),
        "model_beats_persistence": float(np.mean(model_fss_scores)) > float(np.mean(persist_fss_scores)),
    }

    out_path = RESULTS_DIR / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Evaluation Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {out_path}")

    if args.flood_eval:
        flood_report = score_flood_events(model, device, stats, args)
        report["flood_eval"] = flood_report
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print("\n=== Flood Event Evaluation ===")
        for k, v in flood_report.items():
            print(f"  {k}: {v}")

    if args.smoke:
        print("\n[SMOKE] Pipeline OK — random weights, results not meaningful.")
        return

    if report["model_beats_persistence"]:
        print("\n[PASS] Model beats persistence baseline.")
        flag = ROOT / "checkpoints" / "stage5_complete.flag"
        flag.touch()
        print(f"Stage 5 complete → {flag}")
    else:
        print("\n[FAIL] Model does not beat persistence — continue training.")


if __name__ == "__main__":
    main()
