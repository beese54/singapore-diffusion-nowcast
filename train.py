"""
train.py — Train the radar diffusion nowcaster.

Features
--------
- Resumes automatically from latest checkpoint (--resume or auto-detected)
- Saves checkpoint every 1000 steps AND on Ctrl+C (SIGINT)
- Mixed precision for RTX 4060 8GB: bf16 where supported (no GradScaler
  needed), falling back to fp16 + GradScaler on older GPUs
- Gradient checkpointing to stay within VRAM budget
- TensorBoard logging to runs/

Usage
-----
    python train.py                                  # fresh training
    python train.py --resume                         # auto-resume from latest checkpoint
    python train.py --resume checkpoints/nowcaster/ckpt_step_5000.pt
    python train.py --smoke training.max_steps=100 training.batch_size=2  # smoke test

Use --smoke for short test runs: it isolates checkpoints to checkpoints/nowcaster/smoke/
(so --resume auto never finds them) and does NOT write stage4_complete.flag.
"""

import math
import os
import signal
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model.unet import ConditionedUNet, count_parameters
from src.model.diffusion import GaussianDiffusion
from src.data.radar_dataset import RadarDataset, compute_stats

# ── Hyperparameters (override via CLI: python train.py training.lr=1e-4) ─────
import argparse

DEFAULTS = {
    "max_steps": 300_000,
    "batch_size": 4,
    "lr": 2e-4,
    "warmup_steps": 2_000,
    "checkpoint_interval": 1_000,
    "log_interval": 100,
    "val_interval": 5_000,
    "context_frames": 6,
    "target_offset": 6,       # 30 min ahead
    "base_ch": 64,
    "ch_mults": (1, 2, 4, 8),
    "diffusion_steps": 1000,
    "inference_steps": 50,
    # 0, not 2: RadarDataset holds the whole archive in RAM (3.45 GB at 33k
    # frames), so workers add spawn-pickling cost and zero I/O benefit — and
    # on Windows the pickle of that array fails outright (OSError 22).
    "num_workers": 0,
}

CKPT_DIR = ROOT / "checkpoints" / "nowcaster"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint. No value = auto-find latest.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-test run: isolate checkpoints to a throwaway subdir "
                             "and skip writing stage4_complete.flag.")
    for key, val in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=val, type=type(val)
                            if not isinstance(val, tuple) else str)
    # Support Hydra-style overrides: python train.py training.lr=1e-4
    args, overrides = parser.parse_known_args()
    for kv in overrides:
        if "=" in kv:
            k, v = kv.split("=", 1)
            # Strip the optional "training." prefix (a literal prefix, not a
            # character set — lstrip() would mangle keys like num_workers).
            if k.startswith("training."):
                k = k[len("training."):]
            k = k.replace("-", "_")
            if k in DEFAULTS:
                try:
                    setattr(args, k, type(DEFAULTS[k])(v))
                except (ValueError, TypeError):
                    pass
    return args


def find_latest_checkpoint(ckpt_dir: Path = CKPT_DIR) -> Path | None:
    """Most recently WRITTEN checkpoint, not the highest step number.

    After a resume from an earlier step (e.g. rolling back past a diverged
    stretch), the abandoned run's checkpoints have higher numbers still sitting
    on disk -- ranking by step would send --resume auto straight back into the
    run that was just abandoned.
    """
    ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def prune_checkpoints(ckpt_dir: Path) -> int:
    """Keep the recent few plus periodic milestones; delete the rest.

    Each checkpoint is ~309 MB and the interval is 1,000 steps, so a full 300k
    run writes ~93 GB into a OneDrive-synced folder. Retention keeps that at
    roughly 5 GB without losing the ability to go back to a milestone.

    latest.pt is never touched -- it is what --resume auto reads.
    """
    # 5k milestones, not 25k: a NaN already appeared once at val step 15,000, so
    # being able to roll back to any 5k boundary is worth the disk. Caps at ~60
    # files (~18 GB) for a 300k run instead of ~300 (~93 GB).
    keep_recent, milestone = 3, 5_000
    ckpts = {}
    for p in ckpt_dir.glob("ckpt_step_*.pt"):
        try:
            ckpts[int(p.stem.rsplit("_", 1)[1])] = p
        except (ValueError, IndexError):
            continue  # unexpected name: leave it alone rather than guess
    # "Recent" means most recently WRITTEN, not highest step number. After a
    # resume from an earlier step, checkpoints from the abandoned run have higher
    # numbers, so ranking by step would keep those and delete the fresh ones.
    by_mtime = sorted(ckpts, key=lambda s: ckpts[s].stat().st_mtime)
    keep = set(by_mtime[-keep_recent:]) | {s for s in ckpts if s % milestone == 0}
    removed = 0
    for s in ckpts:
        if s not in keep:
            try:
                ckpts[s].unlink()
                removed += 1
            except OSError:
                pass  # locked by OneDrive sync; it will be caught next time
    return removed


def save_checkpoint(state: dict, step: int, ckpt_dir: Path = CKPT_DIR,
                    is_interrupt: bool = False) -> Path:
    path = ckpt_dir / f"ckpt_step_{step}.pt"
    torch.save(state, path)
    # Always update latest.pt symlink / copy
    latest = ckpt_dir / "latest.pt"
    torch.save(state, latest)
    tag = " (on interrupt)" if is_interrupt else ""
    pruned = prune_checkpoints(ckpt_dir)
    extra = f" | pruned {pruned}" if pruned else ""
    print(f"[ckpt] Saved{tag} -> {path}{extra}")
    return path


def load_checkpoint(path: Path, model: nn.Module, optimizer, scaler) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    # Only restore the scaler when it is actually in use. Restoring a COLLAPSED
    # scale is how the 2026-09-22 resume inherited a dead run: the checkpoint
    # carried scale=3.8e-37 and every step went nan immediately. A fresh scaler
    # re-finds its scale within a few hundred steps, so there is nothing to lose
    # and a stuck run to avoid.
    if scaler.is_enabled():
        saved_scale = state.get("scaler", {}).get("scale")
        if saved_scale is not None and saved_scale < 1.0:
            print(f"[ckpt] Ignoring collapsed GradScaler scale ({saved_scale:.3g}); "
                  f"starting the scaler fresh")
        else:
            scaler.load_state_dict(state["scaler"])
    step = state["step"]
    print(f"[ckpt] Resumed from {path} (step {step})")
    return step


def cosine_lr(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    import math
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    args = parse_args()

    # Smoke runs write to an isolated subdir so --resume auto can never pick up
    # toy checkpoints, and they never mark the stage complete.
    ckpt_dir = CKPT_DIR / "smoke" if args.smoke else CKPT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        print(f"[smoke] Test run: checkpoints -> {ckpt_dir} (stage flag will NOT be written)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

    # ── Dataset ──────────────────────────────────────────────────────────────
    zarr_path = ROOT / "data" / "processed" / "radar.zarr"
    if not zarr_path.exists():
        print("ERROR: data/processed/radar.zarr not found.")
        print("Run scripts/scrape_radar.py and scripts/preprocess_radar.py first.")
        sys.exit(1)

    # Compute stats if missing
    stats_path = ROOT / "data" / "processed" / "radar_stats.json"
    if not stats_path.exists():
        print("Computing dataset statistics...")
        compute_stats(zarr_path, stats_path)

    train_ds = RadarDataset("train", context_frames=args.context_frames,
                            target_offset=args.target_offset)
    val_ds = RadarDataset("val", context_frames=args.context_frames,
                          target_offset=args.target_offset)

    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # Fail fast with the real reason: the cryptic alternative is a truncated
    # pickle from the spawned worker, which reads as data corruption.
    if args.num_workers > 0:
        gb = train_ds.rain.nbytes / 1e9
        raise SystemExit(
            f"--num-workers {args.num_workers} is unsupported: RadarDataset is "
            f"in-memory ({gb:.2f} GB) and Windows spawn cannot pickle it to "
            f"workers. Use --num-workers 0 (the default); it is not slower here, "
            f"because there is no I/O to overlap."
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    unet = ConditionedUNet(
        context_frames=args.context_frames,
        base_ch=args.base_ch,
        use_checkpoint=True,   # always use gradient checkpointing for 8GB GPU
    )
    diffusion = GaussianDiffusion(unet, timesteps=args.diffusion_steps,
                                  inference_steps=args.inference_steps)
    diffusion = diffusion.to(device)

    n_params = count_parameters(unet)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # ── Optimizer + AMP dtype ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr,
                                  weight_decay=1e-4, betas=(0.9, 0.999))

    # bf16 over fp16 wherever the GPU supports it (Ada/Ampere+). fp16 killed the
    # 2026-09-22 run: the forward overflowed on ~1 step in 10, GradScaler halved
    # the scale each time, and growth_interval needs 2,000 CONSECUTIVE clean
    # steps to double it back -- impossible at that failure rate. The scale
    # ratcheted one-way from 2**19 down to 3.8e-37 by step 17,000, below fp32's
    # denormal floor, at which point scaling the loss underflows and unscaling
    # amplifies noise into inf: a self-sustaining collapse. Weights stayed
    # finite the whole time (the scaler correctly skipped those steps), which is
    # exactly why it hid for 7,000 steps while learning had already stopped.
    # bf16 has fp32's exponent range, so there is nothing to overflow and no
    # scaler to collapse -- measured 0 non-finite events over 40 steps vs fp16's
    # 1-in-10. GradScaler is therefore disabled for bf16, which makes every
    # scaler call below a pass-through.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype is torch.float16
    scaler = GradScaler("cuda", enabled=use_scaler)
    print(f"AMP dtype: {str(amp_dtype).replace('torch.', '')} | "
          f"GradScaler: {'on' if use_scaler else 'off (not needed for bf16)'}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume is not None:
        ckpt_path = (find_latest_checkpoint(ckpt_dir) if args.resume == "auto"
                     else Path(args.resume))
        if ckpt_path and ckpt_path.exists():
            start_step = load_checkpoint(ckpt_path, diffusion, optimizer, scaler)
        else:
            print(f"[warn] Checkpoint not found: {ckpt_path}. Starting fresh.")

    # ── SIGINT handler: save checkpoint on Ctrl+C ─────────────────────────────
    interrupted = {"flag": False}

    def _sigint_handler(sig, frame):
        if not interrupted["flag"]:
            print("\n[interrupt] Saving checkpoint before exit...")
            interrupted["flag"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── TensorBoard ────────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(ROOT / "runs"))

    # ── Training loop ─────────────────────────────────────────────────────────
    step = start_step
    train_iter = iter(train_loader)

    print(f"Starting training from step {step} / {args.max_steps}")

    while step < args.max_steps:
        # Check for interrupt signal
        if interrupted["flag"]:
            save_checkpoint(
                {"model": diffusion.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "step": step},
                step, ckpt_dir, is_interrupt=True,
            )
            print("[interrupt] Checkpoint saved. Exiting safely.")
            break

        # Get next batch (restart iterator when exhausted)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        context = batch["context"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        # Update learning rate
        lr = cosine_lr(step, args.lr, args.warmup_steps, args.max_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward + backward with mixed precision
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=amp_dtype):
            loss = diffusion.p_losses(target, context)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        # Logging
        if step % args.log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
            print(f"  step {step:>7d} | loss {loss.item():.4f} | lr {lr:.2e}")

        # Validation
        if step % args.val_interval == 0:
            diffusion.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vctx = vbatch["context"].to(device)
                    vtgt = vbatch["target"].to(device)
                    with autocast("cuda", dtype=amp_dtype):
                        vloss = diffusion.p_losses(vtgt, vctx)
                    val_losses.append(vloss.item())
            # Aggregate only the finite batches. Validation runs under fp16
            # autocast with no GradScaler to skip overflows, so a single batch
            # overflowing (inputs are clipped to [-3,3], so this is an
            # intermediate activation, not bad data) used to turn the mean over
            # all ~800 batches into nan and destroy the metric outright -- which
            # is what happened at step 15,000. Report the count instead, so an
            # overflow is a visible number rather than a lost epoch.
            finite = [v for v in val_losses if math.isfinite(v)]
            n_bad = len(val_losses) - len(finite)
            val_loss = sum(finite) / len(finite) if finite else float("nan")
            writer.add_scalar("val/loss", val_loss, step)
            writer.add_scalar("val/nonfinite_batches", n_bad, step)
            warn = f"  [!] {n_bad}/{len(val_losses)} batches non-finite" if n_bad else ""
            print(f"  [val] step {step} | val_loss {val_loss:.4f}{warn}")
            diffusion.train()

        # Checkpoint
        if step % args.checkpoint_interval == 0:
            save_checkpoint(
                {"model": diffusion.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "step": step},
                step, ckpt_dir,
            )

    # Final checkpoint on clean exit
    if not interrupted["flag"] and step == args.max_steps:
        save_checkpoint(
            {"model": diffusion.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "step": step},
            step, ckpt_dir,
        )
        # Write stage flag only for genuine full runs, never for smoke tests
        if not args.smoke:
            flag = ROOT / "checkpoints" / "stage4_complete.flag"
            flag.touch()
            print(f"[done] Training complete. Stage 4 flag -> {flag}")
        else:
            print("[smoke] Test run complete. Stage flag intentionally not written.")

    writer.close()


if __name__ == "__main__":
    main()
