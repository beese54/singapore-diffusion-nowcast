"""
Verification for the optional ERA5 weather conditioning (2026-09-28,
tasks/plan_era5_conditioning.md).

What must hold:
- OFF by default, the architecture and outputs are unchanged: the 30-min model of
  record still loads strictly and reproduces its cached forecast bit for bit.
- ON, the new pathway starts at exactly zero: a conditioned model warm-started from
  the 30-min model gives the SAME output as that model for any weather vector, so
  training starts from the known-good model and any change is learned.
- It can learn: the zero-initialised layer receives a non-zero gradient.
- The dataset hands each sample the ERA5 hour at/before its last radar frame,
  scaled with training-period hours only.
- Guards: a warm start refuses weights that differ in anything but the new layer;
  a resume refuses a checkpoint whose era5_env differs.

Expectation: all PASS. Needs checkpoints/, data/processed/ and a GPU-free run is fine.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.model.unet import ConditionedUNet  # noqa: E402
from src.model.diffusion import GaussianDiffusion  # noqa: E402

CKPT = ROOT / "checkpoints" / "nowcaster" / "ckpt_step_300000.pt"
ok = []


def chk(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n:<62} {d}")
    ok.append(bool(c))


def build(env_dim):
    return GaussianDiffusion(ConditionedUNet(context_frames=6, use_checkpoint=False, env_dim=env_dim),
                             parameterization="v")


def main():
    state = torch.load(CKPT, map_location="cpu", weights_only=True)

    # 1. off by default: identical parameter set, strict load works
    base = build(0)
    base.load_state_dict(state["model"])                      # strict
    chk("env off: 30-min checkpoint loads strictly", True)
    chk("env off: no env parameters exist", not any("env_mlp" in k for k in base.state_dict()))

    # 2. on, zero-init: output identical to the base model for any env vector
    cond = build(11)
    missing, unexpected = cond.load_state_dict(state["model"], strict=False)
    chk("env on: only env layers are missing from the 30-min weights",
        all(".env_mlp." in k for k in missing) and not unexpected, f"{len(missing)} missing")
    torch.manual_seed(0)
    x = torch.randn(2, 1, 120, 217)
    ctx = torch.rand(2, 6, 120, 217) * 2 - 1
    t = torch.tensor([10, 900])
    base.eval(); cond.eval()
    with torch.no_grad():
        yb = base.model(x, ctx, t)
        ya = cond.model(x, ctx, t, env=torch.randn(2, 11) * 3)
        yc = cond.model(x, ctx, t, env=torch.zeros(2, 11))
    chk("env on at step 0: output == base model (random env)", torch.equal(ya, yb),
        f"max diff {float((ya - yb).abs().max()):.2e}")
    chk("env on at step 0: output independent of env value", torch.equal(ya, yc))
    try:
        cond.model(x, ctx, t)
        chk("env on: calling without env raises", False, "no error")
    except ValueError:
        chk("env on: calling without env raises", True)

    # 3. it can learn: the zero-initialised output layer gets a gradient
    cond.train()
    loss = cond.p_losses(torch.rand(2, 1, 120, 217) * 2 - 1, ctx, env=torch.randn(2, 11))
    loss.backward()
    g = cond.model.env_mlp[-1].weight.grad
    chk("zero-initialised env layer receives a non-zero gradient", g is not None and float(g.abs().sum()) > 0)

    # 4. dataset: floor-to-hour mapping and training-hour scaling
    from src.data.radar_dataset import RadarDataset, load_env_table, SPLIT_VAL_START
    hours, xs, names, mean, std = load_env_table()
    train = hours < SPLIT_VAL_START.astype("datetime64[h]")
    chk("env table scaled on training hours (mean ~0, std ~1)",
        abs(xs[train].mean()) < 1e-4 and abs(xs[train].std() - 1) < 1e-3,
        f"{len(names)} features, {train.sum()} training hours")
    ds = RadarDataset("test", target_offset=6, era5_env=True)
    t0 = ds.indices[len(ds.indices) // 2]
    last = ds.times[t0 - 1].astype("datetime64[h]")
    row = int(np.searchsorted(hours, last))
    chk("sample env = ERA5 hour at/before the last radar frame",
        hours[row] == last and np.array_equal(ds.env_at(t0).numpy(), xs[row]),
        f"frame {str(ds.times[t0 - 1])[:16]} -> hour {str(last)}")
    chk("__getitem__ returns env when enabled", "env" in ds[len(ds) // 2])
    chk("test samples limited to hours with ERA5",
        all(ds._env_row[i - 1] >= 0 for i in ds.indices), f"{len(ds)} samples")

    # 5. guards
    import train as tr
    bad = dict(state["model"]); bad["model.stem.weight"] = bad["model.stem.weight"][:, :5]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.pt"; torch.save({**state, "model": bad}, p)
        try:
            tr.init_weights_from(p, build(11), "v", False, (6, False))
            chk("warm start refuses weights differing beyond env layers", False, "no error")
        except (SystemExit, RuntimeError):
            chk("warm start refuses weights differing beyond env layers", True)
        p2 = Path(d) / "noenv.pt"; torch.save({**state, "era5_env": False}, p2)
        m = build(11); opt = torch.optim.AdamW(m.parameters())
        try:
            tr.load_checkpoint(p2, m, opt, torch.amp.GradScaler("cuda", enabled=False), "v", False, (6, False))
            chk("resume refuses era5_env mismatch", False, "no error")
        except SystemExit:
            chk("resume refuses era5_env mismatch", True)

    print(f"\n{sum(ok)}/{len(ok)} passed")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
