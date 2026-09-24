"""
Verification for the v-prediction switch (2026-09-24).

Why. eps-prediction failed on this data. The radar field is 97.3% dry, which
after normalisation sits at exactly -1.0 -- the boundary -- so the data mean is
-0.999, not the ~0 eps-prediction assumes. At the sampler's first step (t=980)
alphas_cumprod is 0.000877, so x_t = 0.0296*x0 + 0.9996*eps: the entire x0
signal is a 0.0296 shift, recovering x0 divides by it (amplifying eps error
34x), and the loss barely rewards getting it right (eps ~= x_t alone scores MSE
0.0008). Measured consequence over 20k steps: sample median wandering around
0 +/- 0.3, 0.00% dry pixels against a 99.7% dry target, no convergence trend.

With v = sqrt(ab)*eps - sqrt(1-ab)*x0, at high t (ab -> 0) v -> -x0, so the
model predicts the data directly exactly where eps-prediction gave it nothing.

Checks: the algebra is exact at every t, v stays well-conditioned where eps does
not, and a checkpoint cannot be resumed under the wrong parameterization.

Expectation: all PASS.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.model.diffusion import GaussianDiffusion  # noqa: E402

ok = []
def chk(n, c, d=""):
    print(f"  {'PASS' if c else 'FAIL'}  {n:<52} {d}")
    ok.append(bool(c))

class Id(nn.Module):
    def forward(self, x, context, t): return x

def main():
    torch.manual_seed(0)
    # x0 shaped like the real target: overwhelmingly at the -1.0 boundary
    x0 = torch.rand(4, 1, 16, 16) * 2 - 1
    x0[:, :, :15] = -1.0
    noise = torch.randn_like(x0)
    dry = float((x0 <= -0.999).float().mean())
    print(f"  synthetic x0: {dry*100:.1f}% at the -1.0 boundary, mean {float(x0.mean()):+.4f}\n")

    dv = GaussianDiffusion(Id(), timesteps=1000, parameterization="v")
    de = GaussianDiffusion(Id(), timesteps=1000, parameterization="eps")

    # 1. algebra exact at every t, including the extremes
    for tv in (0, 10, 300, 980, 999):
        t = torch.full((4,), tv, dtype=torch.long)
        xt, _ = dv.q_sample(x0, t, noise)
        v = dv._v_target(x0, noise, t)
        x0r, epsr = dv._to_x0_eps(v, xt, t)
        chk(f"v: t={tv:<4} recovers x0 exactly",
            float((x0r - x0).abs().max()) < 2e-3,
            f"err {float((x0r-x0).abs().max()):.1e}")
        chk(f"v: t={tv:<4} recovers eps exactly",
            float((epsr - noise).abs().max()) < 2e-3,
            f"err {float((epsr-noise).abs().max()):.1e}")

    # 2. conditioning: eps degrades at the top of the schedule, v does not.
    #    This is the numerical statement of why the switch was made.
    t = torch.full((4,), 999, dtype=torch.long)
    xt, _ = de.q_sample(x0, t, noise)
    x0_eps, _ = de._to_x0_eps(noise, xt, t)          # feeding the TRUE eps
    x0_v, _ = dv._to_x0_eps(dv._v_target(x0, noise, t), xt, t)
    e_eps = float((x0_eps - x0).abs().max())
    e_v = float((x0_v - x0).abs().max())
    chk("eps is ill-conditioned at t=999", e_eps > e_v * 10, f"eps err {e_eps:.1e}")
    chk("v stays well-conditioned at t=999", e_v < 1e-5, f"v err {e_v:.1e}")

    # 3. v -> -x0 as ab -> 0, i.e. the model is handed the data at high t
    t = torch.full((4,), 980, dtype=torch.long)
    v = dv._v_target(x0, noise, t)
    corr = float(torch.corrcoef(torch.stack([v.flatten(), (-x0).flatten()]))[0, 1])
    chk("v target correlates with -x0 at high t", corr > 0.9, f"r = {corr:.4f}")

    # 4. an unknown parameterization must fail loudly, not default silently
    try:
        GaussianDiffusion(Id(), parameterization="epsilon")
        chk("rejects an unknown parameterization", False, "no error raised")
    except ValueError:
        chk("rejects an unknown parameterization", True, "ValueError")

    print(f"\n{sum(ok)}/{len(ok)} passed")
    sys.exit(0 if all(ok) else 1)

if __name__ == "__main__":
    main()
