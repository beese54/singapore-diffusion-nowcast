"""
Verification for heavy-rain neighbourhood loss weighting (2026-09-25).

Why: the 300k model lets storm cores decay too fast (12% of heavy pixels kept vs
27% observed) and scatters heavy cores across members. The weighting targets
the neighbourhood of heavy rain in the target OR the last observed frame, so
missing a core and wrongly killing/keeping one both cost more, while dry and
light-rain areas elsewhere keep weight 1.

Must hold: beta = 0 is exactly the old loss; the mask covers both sources with
the intended spread; weights average 1 (loss scale, hence effective LR,
unchanged); on real data ~1.8% of pixels carry ~28% of the weight at beta 20.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.model.diffusion import GaussianDiffusion  # noqa: E402
from src.data.radar_dataset import RadarDataset    # noqa: E402

ok = []
def chk(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n:<54} {d}"); ok.append(bool(c))

class Zero(nn.Module):
    def forward(self, x, context, t): return torch.zeros_like(x)

def main():
    d = GaussianDiffusion(Zero(), parameterization="v")
    thr = 0.0391                                   # ~10 mm/hr normalised
    x0 = torch.full((1, 1, 32, 32), -1.0); ctx = torch.full((1, 6, 32, 32), -1.0)
    x0[0, 0, 10, 10] = 0.5                         # heavy in target
    ctx[0, -1, 20, 20] = 0.5                       # heavy in last frame only
    t = torch.tensor([500])

    torch.manual_seed(0); base = d.p_losses(x0, ctx, t)
    torch.manual_seed(0); b0 = d.p_losses(x0, ctx, t, heavy_weight=0.0, heavy_thr=thr)
    chk("heavy_weight=0 is exactly the old loss", torch.equal(base, b0))

    # reconstruct the weight map the loss uses
    m = ((x0 >= thr) | (ctx[:, -1:] >= thr)).float()
    m = F.max_pool2d(m, 7, stride=1, padding=3)
    chk("target core weighted", m[0, 0, 10, 10] == 1)
    chk("last-frame-only core weighted (decay side)", m[0, 0, 20, 20] == 1)
    chk("dilation reaches 3 px", m[0, 0, 13, 10] == 1 and m[0, 0, 10, 13] == 1)
    chk("dilation stops at 4 px", m[0, 0, 14, 10] == 0)
    chk("dry area far away unweighted", m[0, 0, 0, 0] == 0)
    w = 1 + 20 * m; w = w / w.mean()
    chk("weights average 1", abs(float(w.mean()) - 1) < 1e-6)

    torch.manual_seed(0); b20 = d.p_losses(x0, ctx, t, heavy_weight=20.0, heavy_thr=thr)
    chk("heavy_weight>0 changes the loss", not torch.equal(base, b20))

    ds = RadarDataset("train"); rng = np.random.default_rng(0)
    real_thr = float(ds._normalise(np.array([10.0], dtype=np.float32))[0])
    idx = rng.choice(len(ds), 300, replace=False)
    X = torch.stack([ds[int(i)]["target"] for i in idx])
    L = torch.stack([ds[int(i)]["context"][-1:] for i in idx])
    mm = F.max_pool2d(((X >= real_thr) | (L >= real_thr)).float(), 7, 1, 3)
    f = float(mm.mean()); share = 21 * f / (21 * f + (1 - f))
    chk("real data: heavy neighbourhood is a small area", f < 0.05, f"{f*100:.2f}% of pixels")
    chk("real data: it carries a meaningful weight share", 0.15 < share < 0.45,
        f"{share*100:.0f}% of loss weight at beta 20")

    print(f"\n{sum(ok)}/{len(ok)} passed")
    sys.exit(0 if all(ok) else 1)

if __name__ == "__main__":
    main()
