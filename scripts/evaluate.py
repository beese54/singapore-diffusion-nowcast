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
from src.data.radar_dataset import (RadarDataset, load_stats, CONTEXT_FRAMES, TARGET_OFFSET,
                                    SPLIT_TEST_END, SPLIT_TEST_START)

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


def fss_parts(prob: np.ndarray, obs_bin: np.ndarray, win: int) -> tuple[float, float]:
    """One sample's contribution to POOLED FSS: (numerator, denominator).

    FSS = 1 - sum(num) / sum(den) over the evaluation set (Roberts & Lean 2008).
    Averaging per-sample FSS instead -- what this script used to do -- heavily
    punishes near-dry samples where a few misplaced pixels score 0, and made the
    model look ~3x worse than persistence when the pooled scores are a tie.
    `prob` is a probability field in [0, 1]; a single forecast is passed as its
    0/1 exceedance field.
    """
    from scipy.ndimage import uniform_filter
    pf = uniform_filter(prob.astype(np.float64), size=win, mode="constant")
    of = uniform_filter(obs_bin.astype(np.float64), size=win, mode="constant")
    return float(np.sum((pf - of) ** 2)), float(np.sum(pf ** 2) + np.sum(of ** 2))


def crps_pixelwise(ens: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-element CRPS of an ensemble (M, ...) against obs (...). For a single
    forecast (M = 1) this reduces to the absolute error."""
    ens = ens.astype(np.float32)
    t1 = np.abs(ens - obs[None]).mean(0)
    M = ens.shape[0]
    t2 = np.zeros_like(obs, dtype=np.float32)
    for i in range(M):
        t2 += np.abs(ens[i][None] - ens).sum(0)
    return t1 - 0.5 * t2 / (M * M)


def bootstrap_ci(stat, n: int, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of stat(indices) by resampling evaluation samples."""
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, n)) for _ in range(reps)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def load_model(checkpoint_path: Path, device: torch.device,
               parameterization: str | None = None) -> GaussianDiffusion:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Input layout from the checkpoint; absent means the pre-option legacy
    # layout (6 frames, no time channels) by construction -- every checkpoint
    # written since goes through train.py's single _ckpt_state().
    cf = int(state.get("context_frames", CONTEXT_FRAMES))
    tc = bool(state.get("time_channels", False))
    unet = ConditionedUNet(context_frames=cf + (2 if tc else 0), use_checkpoint=False)
    # Never guess. This used to default an unstamped checkpoint to 'eps'; the
    # finished v-prediction model turned out to be unstamped (a save site in
    # train.py missed the field), so it was decoded as eps and scored MAE 29
    # mm/hr with no error raised. A silent default converted a missing field
    # into a wrong answer. Now: the stamp wins, --parameterization fills a gap,
    # and disagreement or absence stops the run.
    stamped = state.get("parameterization")
    if stamped and parameterization and stamped != parameterization:
        sys.exit(f"{checkpoint_path.name} is stamped '{stamped}' but "
                 f"--parameterization {parameterization} was given. Refusing.")
    param = stamped or parameterization
    if param is None:
        sys.exit(f"{checkpoint_path.name} has no parameterization stamp. Pass "
                 f"--parameterization v or eps explicitly -- decoding under the "
                 f"wrong one gives plausible-looking, wrong results.")
    # Unlike parameterization, a missing 'residual' key is not a gap to guess
    # across: the option did not exist before, and every checkpoint written since
    # comes from train.py's single _ckpt_state() constructor, which always sets
    # it. Absent therefore means full-frame by construction.
    residual = bool(state.get("residual", False))
    print(f"Loaded {checkpoint_path.name}: step {state.get('step', '?')}, "
          f"parameterization '{param}', residual={residual}, "
          f"context_frames={cf}, time_channels={tc}, "
          f"target_offset={int(state.get('target_offset', 6))}")
    diffusion = GaussianDiffusion(unet, parameterization=param, residual=residual)
    diffusion.load_state_dict(state["model"])
    # Callers build their dataset from this so the input matches training.
    diffusion.data_cfg = {"context_frames": cf, "time_channels": tc}
    # Each model forecasts ONE lead. Unstamped checkpoints predate the stamp and
    # were all trained with the default target_offset of 6 (nominal 30 min).
    diffusion.target_offset = int(state.get("target_offset", 6))
    return diffusion.to(device).eval()


def persistence_forecast(context_frames: np.ndarray) -> np.ndarray:
    """Naive baseline: predict the last observed frame unchanged."""
    return context_frames[-1]  # (H, W)


def score_flood_events(model, device, rds, args) -> dict:
    """Score model against geocoded real flood events from flood_eval_dataset.parquet."""
    import pandas as pd

    parquet_path = ROOT / "data" / "processed" / "flood_eval_dataset.parquet"
    if not parquet_path.exists():
        return {"error": "flood_eval_dataset.parquet not found — run build_flood_eval_dataset.py first"}

    df = pd.read_parquet(parquet_path)
    events = df[(df["geocoded"] == True) & (df["is_flood_event"] == True)].copy()
    # Test-period events only: earlier ones fall in the train/val periods,
    # whose radar frames the model was fitted to (lesson L031).
    et = pd.to_datetime(events["matched_radar_time"]).values.astype("datetime64[m]")
    events = events[(et >= SPLIT_TEST_START) & (et <= SPLIT_TEST_END)]

    if events.empty:
        return {"n_events": 0, "note": "No geocoded flood events available yet"}

    radar_zarr = ROOT / "data" / "processed" / "radar.zarr"
    ds = xr.open_zarr(str(radar_zarr), consolidated=True)
    radar_times = ds.time.values

    # Normalisation comes from the dataset -- never a local copy. This script
    # used to carry its own z-score-and-clip formulas, which silently went stale
    # when the dataset switched to [-1, 1] by log_max: decoded that way, 20 mm/hr
    # reads as ~0.1 mm/hr and FSS at 2 mm/hr degenerates to "both empty".
    def denorm(x):
        return rds.denormalise(torch.as_tensor(x)).numpy()

    lead_steps = getattr(model, "target_offset", args.lead // 5)
    CF = rds.context_frames      # the model's own history length

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
        ctx_start = ctx_end - CF
        if ctx_start < 0 or ctx_end >= len(radar_times):
            continue

        ctx_frames = ds["rain_rate"].values[ctx_start:ctx_end].astype(np.float32)  # (C, H, W)
        obs_frame = ds["rain_rate"].values[idx].astype(np.float32)                  # (H, W)

        # Persistence: predict last context frame
        persist_pred_cell = float(ctx_frames[-1, lat_idx, lon_idx])
        hit_persist = persist_pred_cell >= args.threshold

        # Model ensemble mean at flood cell
        # Built by the dataset, identical to training (incl. time channels).
        ctx_tensor = rds.context_at(ctx_end).unsqueeze(0).to(device)
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
    parser.add_argument("--lead", type=int, default=None,
                        help="Nominal lead in minutes. Defaults to the checkpoint's own; "
                             "a mismatch is refused (one model per lead time).")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: random-weight model, 10 samples, no stage5 flag")
    parser.add_argument("--parameterization", choices=["v", "eps"], default=None,
                        help="Only needed for checkpoints without a stamp")
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
        model = load_model(ckpt_path, device, args.parameterization)

    stats = load_stats()

    # The lead is a property of the model, not a free choice at evaluation
    # time: scoring a 60-min model against the 30-min target frame would run
    # without error and be meaningless.
    target_offset = getattr(model, "target_offset", None)
    if target_offset is None:                            # random smoke model
        target_offset = (args.lead or 30) // 5
    elif args.lead is not None and args.lead // 5 != target_offset:
        sys.exit(f"--lead {args.lead} does not match the checkpoint's lead "
                 f"({target_offset * 5} min). One model per lead time.")
    args.lead = target_offset * 5

    cfg = getattr(model, "data_cfg", {"context_frames": CONTEXT_FRAMES,
                                       "time_channels": False})
    test_ds = RadarDataset("test", context_frames=cfg["context_frames"],
                           target_offset=target_offset,
                           time_channels=cfg["time_channels"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    def denorm(x):
        # Single source of truth: the dataset's own inverse transform.
        return test_ds.denormalise(torch.as_tensor(x)).numpy()

    # Stage 5 criterion (definition_of_done.md, amended 2026-09-25):
    #   PRIMARY   CRPS skill vs persistence > 0, 95% CI lower bound > 0
    #   TRACKED   pooled ensemble-probability FSS at the threshold vs persistence
    #   TARGET    the same at >= 10 mm/hr (heavy rain drives flash floods)
    WINDOWS = {"6.1 km": 21, "11.9 km": 41, "23.5 km": 81}     # 0.29 km pixels
    THRESHOLDS = sorted({args.threshold, 10.0})
    parts = {(thr, w, who): [] for thr in THRESHOLDS for w in WINDOWS
             for who in ("model", "persistence")}
    crps_m, crps_p, legacy_member_fss, legacy_persist_fss = [], [], [], []

    n_eval = min(args.n_samples, len(test_ds))
    # Spread over the whole test period. Taking the first n in time order
    # covered only ~17 hours of a ~12-day test set.
    picks = np.linspace(0, len(test_ds) - 1, n_eval).astype(int)
    print(f"Evaluating {n_eval} test samples spread over the test period, "
          f"lead={args.lead} min, {args.members} members")

    for k, i in enumerate(picks):
        batch = test_ds[int(i)]
        ctx = batch["context"].unsqueeze(0).to(device)
        torch.manual_seed(1000 + k)
        with torch.no_grad():
            ens = model.ensemble_sample(ctx, n_members=args.members, eta=1.0)
        ens_mm = denorm(ens.cpu().float().numpy()[0, :, 0])            # (M, H, W)
        obs_mm = denorm(batch["target"][0].numpy())
        persist_mm = denorm(batch["context"][-1].numpy())                # last frame

        for thr in THRESHOLDS:
            ob = obs_mm >= thr
            prob = (ens_mm >= thr).mean(0)
            pers = (persist_mm >= thr).astype(np.float32)
            for name, w in WINDOWS.items():
                parts[(thr, name, "model")].append(fss_parts(prob, ob, w))
                parts[(thr, name, "persistence")].append(fss_parts(pers, ob, w))

        rel = (obs_mm >= 0.5) | (persist_mm >= 0.5) | (ens_mm.max(0) >= 0.5)
        crps_m.append(float(crps_pixelwise(ens_mm, obs_mm)[rel].sum()))
        crps_p.append(float(np.abs(persist_mm - obs_mm)[rel].sum()))
        # legacy diagnostic: the old per-sample, single-forecast FSS
        legacy_member_fss.append(fss(ens_mm[0], obs_mm, args.threshold))
        legacy_persist_fss.append(fss(persist_mm, obs_mm, args.threshold))

        if k % 25 == 0:
            print(f"  {k}/{n_eval}", flush=True)

    n = len(crps_m)
    cm, cp = np.array(crps_m), np.array(crps_p)

    def skill(ix):
        return 1.0 - cm[ix].sum() / max(cp[ix].sum(), 1e-12)

    crps_skill = skill(np.arange(n))
    crps_ci = bootstrap_ci(skill, n)

    def pooled(arr, ix):
        return 1.0 - arr[ix, 0].sum() / max(arr[ix, 1].sum(), 1e-12)

    fss_report = {}
    for thr in THRESHOLDS:
        fss_report[f"{thr} mm/hr"] = {}
        for name in WINDOWS:
            a = np.array(parts[(thr, name, "model")])
            b = np.array(parts[(thr, name, "persistence")])
            all_ix = np.arange(n)
            fss_report[f"{thr} mm/hr"][name] = {
                "model": pooled(a, all_ix), "persistence": pooled(b, all_ix),
                "diff": pooled(a, all_ix) - pooled(b, all_ix),
                "diff_95ci": bootstrap_ci(lambda ix, a=a, b=b: pooled(a, ix) - pooled(b, ix), n)}

    report = {
        "lead_time_min": args.lead,
        "lead_note": (f"nominal {args.lead} min = {args.lead + 5} min from the last "
                      f"observed frame (context ends t-1, target t+{target_offset})"),
        "n_samples": n, "members": args.members,
        # The test split is pinned by date (radar_dataset.SPLIT_*); record the
        # anchors actually scored so results stay traceable.
        "test_period": [str(test_ds.times[test_ds.indices[int(picks[0])]])[:16],
                        str(test_ds.times[test_ds.indices[int(picks[-1])]])[:16]],
        "criterion": "PRIMARY: CRPS skill vs persistence > 0 with 95% CI lower bound > 0",
        "crps_skill": crps_skill, "crps_skill_95ci": crps_ci,
        "primary_pass": bool(crps_ci[0] > 0),
        "fss_pooled_ensemble_probability": fss_report,
        "legacy_per_sample_single_forecast_fss": {
            "model": float(np.mean(legacy_member_fss)),
            "persistence": float(np.mean(legacy_persist_fss)),
            "note": "old headline; per-sample averaging and single members -- diagnostic only"},
    }
    report["checkpoint"] = "smoke" if args.smoke else str(args.checkpoint)
    out_path = RESULTS_DIR / ("evaluation_report_smoke.json" if args.smoke
                              else "evaluation_report.json")

    def save(rep):
        # One model per lead, so each run fills its own lead's section of ONE
        # report (the DoD asks for 30/60/90 in evaluation_report.json). A file
        # from before the pinned split (single lead at top level) is replaced:
        # its test period is not comparable.
        full = {}
        if out_path.exists():
            full = json.loads(out_path.read_text(encoding="utf-8"))
        if "by_lead" not in full:
            full = {"by_lead": {}}
        full["by_lead"][str(rep["lead_time_min"])] = rep
        full["by_lead"] = dict(sorted(full["by_lead"].items(), key=lambda kv: int(kv[0])))
        out_path.write_text(json.dumps(full, indent=2), encoding="utf-8")

    save(report)

    print("\n=== Evaluation Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {out_path}")

    if args.flood_eval:
        flood_report = score_flood_events(model, device, test_ds, args)
        report["flood_eval"] = flood_report
        save(report)
        print("\n=== Flood Event Evaluation ===")
        for k, v in flood_report.items():
            print(f"  {k}: {v}")

    if args.smoke:
        print("\n[SMOKE] Pipeline OK — random weights, results not meaningful.")
        return

    # Beating persistence is ONE Stage 5 criterion, not Stage 5. The DoD also
    # requires FSS at 30/60/90 min, a qualitative case study, the flood-risk
    # overlay notebook and <5 min end-to-end inference. Writing
    # stage5_complete.flag here would self-certify the stage on a partial test
    # (lesson L004), so this reports the criterion and leaves the flag alone.
    lo, hi = report["crps_skill_95ci"]
    verdict = "PASS" if report["primary_pass"] else "FAIL"
    print(f"\n[{verdict}] PRIMARY -- CRPS skill vs persistence "
          f"{report['crps_skill']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    for thr, rows in report["fss_pooled_ensemble_probability"].items():
        tag = "TARGET (heavy rain)" if thr.startswith("10.0") else "TRACKED"
        r = rows["11.9 km"]
        clo, chi = r["diff_95ci"]
        sig = ("model better" if clo > 0 else "persistence better" if chi < 0
               else "not distinguishable")
        print(f"[{tag}] pooled ensemble FSS {thr}, 11.9 km: model {r['model']:.3f} vs "
              f"persistence {r['persistence']:.3f}  CI of diff [{clo:+.3f}, {chi:+.3f}] -> {sig}")
    print("Stage 5 flag NOT written -- the remaining DoD criteria are separate.")


if __name__ == "__main__":
    main()
