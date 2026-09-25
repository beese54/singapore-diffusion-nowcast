"""
Verification for residual forecasting (2026-09-25).

Why: the full-frame v-prediction model uses its context (true frames FSS 0.095
vs frames from another time 0.039) but only weakly -- persistence scores 0.264.
It had to redraw the whole field from noise. With residual=True it diffuses the
change r = (x0 - last_frame)/2, so a model that learns nothing equals
persistence and anything learned is improvement on top.

Must hold:
  1. the residual encoding round-trips exactly
  2. a "know-nothing" model (always predicts zero change) reproduces the last
     context frame EXACTLY through the real DDIM sampler -- the whole point
  3. on real data the residual target is centred near 0, not piled at -1
  4. residual=False behaves as before (no persistence offset leaks in)
Expectation: all PASS.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.model.diffusion import GaussianDiffusion  # noqa: E402
from src.data.radar_dataset import RadarDataset    # noqa: E402

ok = []
def chk(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n:<54} {d}"); ok.append(bool(c))


class ZeroChange(nn.Module):
    """Predicts v consistent with x0 = 0 at every step, i.e. 'no change'.
    For v-param, x0 = sqrt(ab)*x_t - sqrt(1-ab)*v, so x0 = 0 needs
    v = sqrt(ab)/sqrt(1-ab) * x_t."""
    def __init__(self): super().__init__(); self.d = None
    def forward(self, x, context, t):
        ab = self.d.alphas_cumprod[t][:, None, None, None]
        return ab.sqrt() / (1 - ab).clamp(min=1e-12).sqrt() * x


def main():
    ds = RadarDataset("val")
    b = ds[len(ds) // 3]
    ctx = b["context"].unsqueeze(0)            # (1, 6, H, W), real frames
    x0 = b["target"].unsqueeze(0)              # (1, 1, H, W)
    base = ctx[:, -1:]

    # 1. round trip
    r = (x0 - base) * 0.5
    chk("residual lies in [-1, 1]", float(r.abs().max()) <= 1.0 + 1e-6,
        f"max |r| {float(r.abs().max()):.3f}")
    back = (base + 2.0 * r).clamp(-1, 1)
    chk("residual round-trips exactly", float((back - x0).abs().max()) < 1e-6,
        f"err {float((back - x0).abs().max()):.1e}")

    # 2. know-nothing model == persistence through the REAL sampler
    m = ZeroChange()
    d = GaussianDiffusion(m, parameterization="v", residual=True)
    m.d = d
    torch.manual_seed(0)
    with torch.no_grad():
        out = d.ddim_sample(ctx, x0.shape, eta=0.0)
    err = float((out - base).abs().max())
    chk("zero-change model reproduces persistence exactly", err < 1e-4,
        f"max |out - last frame| {err:.1e}")
    with torch.no_grad():
        ens = d.ensemble_sample(ctx, n_members=2, eta=1.0)
    err_e = float((ens[:, :, :, :, :] - base.unsqueeze(1)).abs().max())
    chk("...and through ensemble_sample with eta=1", err_e < 1e-3,
        f"max err {err_e:.1e}")

    # 3. residual targets centred near 0 on real data
    rs = []
    for i in np.linspace(0, len(ds) - 1, 30).astype(int):
        s = ds[int(i)]
        rs.append(((s["target"] - s["context"][-1:]) * 0.5).numpy().ravel())
    rs = np.concatenate(rs)
    full = np.concatenate([ds[int(i)]["target"].numpy().ravel()
                           for i in np.linspace(0, len(ds) - 1, 30).astype(int)])
    chk("residual target is centred near 0", abs(rs.mean()) < 0.01,
        f"mean {rs.mean():+.4f} (full-frame target mean {full.mean():+.4f})")
    chk("most residual pixels are exactly 0 (dry stays dry)",
        (np.abs(rs) < 1e-6).mean() > 0.9, f"{(np.abs(rs) < 1e-6).mean()*100:.1f}% zero")

    # 4. residual=False: the same zero model must NOT produce persistence
    m2 = ZeroChange(); d2 = GaussianDiffusion(m2, parameterization="v", residual=False); m2.d = d2
    with torch.no_grad():
        out2 = d2.ddim_sample(ctx, x0.shape, eta=0.0)
    chk("residual=False adds no persistence offset", float(out2.abs().max()) < 1e-4,
        f"max |out| {float(out2.abs().max()):.1e} (field 0, not last frame)")

    print(f"\n{sum(ok)}/{len(ok)} passed")
    sys.exit(0 if all(ok) else 1)

if __name__ == "__main__":
    main()
