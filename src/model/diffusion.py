"""
diffusion.py — DDPM diffusion process for radar nowcasting.

Implements:
- Cosine noise schedule (default; linear available), 1000 training steps
- Forward process: q(x_t | x_0)
- Reverse process: p_theta(x_{t-1} | x_t, context), context = past radar frames
- Training loss: MSE on the v-prediction target (default; eps available but
  fails on this near-binary data -- see train.py DEFAULTS "parameterization")
- Inference: DDIM sampler, 50 steps, eta=1 for ensembles
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from 'Improved DDPM' — better for precipitation (sparse signal)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0, 0.999)


class GaussianDiffusion(nn.Module):
    """
    DDPM with DDIM inference.

    Parameters
    ----------
    model        : the UNet noise predictor
    timesteps    : number of diffusion steps during training (default 1000)
    schedule     : 'linear' or 'cosine'
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule: str = "cosine",
        inference_steps: int = 50,
        parameterization: str = "v",
        residual: bool = False,
    ):
        super().__init__()
        self.model = model
        # Residual forecasting: learn the CHANGE from the last context frame
        # instead of the whole future frame. See _residual_base().
        self.residual = bool(residual)
        self.T = timesteps
        self.inference_steps = inference_steps
        if parameterization not in ("v", "eps"):
            raise ValueError(f"parameterization must be 'v' or 'eps', got {parameterization!r}")
        self.parameterization = parameterization

        # Noise schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers so they move with .to(device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt())
        self.register_buffer("posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    # ── Forward (noising) ────────────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy x_t from x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_omc * noise, noise

    # ── Training loss ────────────────────────────────────────────────────────

    def p_losses(
        self,
        x0: torch.Tensor,
        context: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        intensity_alpha: float = 0.0,
        heavy_weight: float = 0.0,
        heavy_thr: float | None = None,
        heavy_dilate: int = 7,
    ) -> torch.Tensor:
        """
        Compute training loss L_simple = E[||noise - model(x_t, context, t)||²].

        Parameters
        ----------
        x0      : (B, 1, H, W) clean target frame (normalised to [-1, 1])
        context : (B, C, H, W) conditioning (past radar frames)
        t       : (B,) timestep indices; sampled uniformly if None
        intensity_alpha : per-pixel loss weighting by rain intensity. 0 gives
                  plain MSE. See below for why the default is not 0.

        Intensity weighting
        -------------------
        Unweighted, this loss is dominated by dry pixels: only ~1.9% of pixels
        carry rain, so they receive ~1.9% of the gradient signal and predicting a
        uniform field is close to optimal. Two runs failed exactly there -- one
        collapsed to constant dry, the next to constant wet, both with a
        healthy-looking loss curve.

        Weighting each pixel by 1 + alpha * (x0 + 1) / 2 makes a dry pixel weight
        1 and a 100 mm/hr pixel weight 1 + alpha. Measured share of total loss
        weight landing on wet pixels:

            alpha=0    1.9%      alpha=50   20.9%
            alpha=10   6.4%      alpha=100  33.7%   <- default
            alpha=20  10.5%      alpha=200  50.0%

        Weights are rescaled to mean 1 so the loss magnitude -- and therefore the
        effective learning rate -- stays comparable to the unweighted version.
        """
        B = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.T, (B,), device=x0.device)

        field = x0                                   # weights use the real field
        if self.residual:
            x0 = (x0 - self._residual_base(context)) * 0.5

        x_t, noise = self.q_sample(x0, t)
        pred = self.model(x_t, context, t)
        target = (self._v_target(x0, noise, t)
                  if self.parameterization == "v" else noise)

        if not intensity_alpha and not heavy_weight:
            return F.mse_loss(pred, target)

        w = torch.ones_like(field)
        if intensity_alpha:
            w = w + intensity_alpha * (field.detach() + 1.0) * 0.5
        if heavy_weight:
            # Heavy-rain neighbourhood: pixels at or above heavy_thr in the
            # TARGET or in the LAST observed frame, dilated by heavy_dilate px.
            # Diagnosis of the 300k model: storm cores decay too fast (12% of
            # heavy pixels kept vs 27% observed) and are scattered across
            # members. Weighting both sides penalises missing a core AND
            # wrongly killing or keeping an existing one, while light rain and
            # dry areas elsewhere keep weight 1 -- unlike intensity_alpha, which
            # up-weighted all rain and pushed forecasts to rain everywhere.
            last = self._residual_base(context)          # last radar frame
            m = ((field >= heavy_thr) | (last >= heavy_thr)).float()
            if heavy_dilate > 1:
                m = F.max_pool2d(m, heavy_dilate, stride=1, padding=heavy_dilate // 2)
            w = w + heavy_weight * m.detach()
        w = w / w.mean()                      # keep the loss scale unchanged
        return (w * (pred - target) ** 2).mean()

    # ── DDIM inference ───────────────────────────────────────────────────────

    # ── parameterization helpers ─────────────────────────────────────────────
    #
    # eps-prediction fails on this data. The radar field is 97.3% dry, which after
    # normalisation sits at exactly -1.0 -- the boundary of the range -- so the
    # data mean is -0.999 rather than the ~0 that eps-prediction assumes. At the
    # first sampling step (t=980) alphas_cumprod is 0.000877, so
    #
    #     x_t = 0.0296 * x0 + 0.9996 * eps
    #
    # The whole x0 signal is a 0.0296 shift, and recovering x0 divides by that,
    # amplifying any eps error 34x. Meanwhile the training loss barely rewards
    # capturing it: predicting eps ~= x_t alone already scores MSE 0.0008. So the
    # model never learned the prior at high t, sampling started off-manifold and
    # stayed there -- measured sample median bounced around 0 +/- 0.3 across 20k
    # steps with 0.00% dry pixels against a 99.7% dry target.
    #
    # v-prediction fixes the incentive. With
    #
    #     v = sqrt(ab) * eps - sqrt(1-ab) * x0
    #
    # at high t (ab -> 0) v -> -x0, so the model predicts the DATA directly where
    # eps-prediction gave it nothing to learn. Identities used below:
    #
    #     x0  = sqrt(ab) * x_t - sqrt(1-ab) * v
    #     eps = sqrt(1-ab) * x_t + sqrt(ab) * v

    # ── residual forecasting ─────────────────────────────────────────────────
    #
    # Evaluated on the finished full-frame model: it USES its context (true
    # frames FSS 0.095 vs frames from another time 0.039, better on 30/40
    # samples) but only weakly -- persistence scores 0.264 on the same samples.
    # Persistence is strong at 30 min because storms mostly keep their shape and
    # position, yet the model had to redraw the whole field from noise and was
    # never told "the last frame, shifted a bit" is the right starting point.
    #
    # With residual=True the model diffuses over r = (x0 - last_frame) / 2
    # instead of x0. Both fields are in [-1, 1], so the difference is in [-2, 2]
    # and /2 puts it back in [-1, 1] where the sampler's clamp is valid. A model
    # that learns nothing now outputs r = 0, i.e. EXACTLY persistence, and
    # anything it learns is improvement on top. As a side effect the target is
    # centred on 0 (dry stays dry -> 0) rather than piled at the -1 boundary that
    # broke eps-prediction.

    @staticmethod
    def _residual_base(context: torch.Tensor) -> torch.Tensor:
        """Last context frame, (B, 1, H, W): the persistence forecast."""
        return context[:, -1:]

    def _v_target(self, x0: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * noise - sqrt_omc * x0

    def _to_x0_eps(self, out: torch.Tensor, x_t: torch.Tensor,
                   t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Model output -> (x0, eps), whichever parameterization is in use."""
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omc = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        if self.parameterization == "v":
            x0 = sqrt_ac * x_t - sqrt_omc * out
            eps = sqrt_omc * x_t + sqrt_ac * out
        else:
            eps = out
            x0 = (x_t - sqrt_omc * eps) / sqrt_ac.clamp(min=1e-8)
        return x0, eps

    @torch.no_grad()
    def ddim_sample(
        self,
        context: torch.Tensor,
        shape: tuple[int, ...],
        eta: float = 0.0,
        callback=None,
    ) -> torch.Tensor:
        """
        Generate a sample using DDIM (fast, deterministic when eta=0).

        Parameters
        ----------
        context : (B, C, H, W) conditioning frames
        shape   : output shape, e.g. (B, 1, H, W)
        eta     : stochasticity (0 = deterministic DDIM, 1 = DDPM)
        callback: optional f(step, x_t, x0_pred), called after every step with
                  the new x_t and that step's clean-sample estimate. Observes
                  only -- it draws no random numbers, so samples are unchanged.
                  Used to visualise denoising (scripts/build_dashboard.py).
        """
        device = context.device
        B = shape[0]

        # Uniformly spaced subset of timesteps for inference
        step_size = self.T // self.inference_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((B,), timesteps[i + 1] if i + 1 < len(timesteps) else 0,
                                device=device, dtype=torch.long)

            alpha_t = self.alphas_cumprod[t][:, None, None, None]
            alpha_prev = self.alphas_cumprod[t_prev][:, None, None, None]
            if i + 1 == len(timesteps):
                # Final step lands on the clean sample (alpha = 1), as in DDIM.
                # Using alphas_cumprod[0] = 0.999959 here left sqrt(1-a) = 0.0064
                # of noise in every returned sample -- small, but it meant a model
                # predicting "no change" did not return persistence exactly.
                alpha_prev = torch.ones_like(alpha_prev)

            out = self.model(x, context, t)
            x0_pred, noise_pred = self._to_x0_eps(out, x, t)
            x0_pred = x0_pred.clamp(-1, 1)

            # DDIM update
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
            direction = (1 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * noise_pred
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0

            x = alpha_prev.sqrt() * x0_pred + direction + noise
            if callback is not None:
                callback(i, x, x0_pred)

        if self.residual:
            # back from change-space to a rain field: persistence + 2 * change
            x = (self._residual_base(context) + 2.0 * x).clamp(-1, 1)
        return x

    @torch.no_grad()
    def ensemble_sample(
        self,
        context: torch.Tensor,
        n_members: int = 8,
        eta: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate an ensemble of n_members probabilistic forecasts.

        Returns (B, n_members, 1, H, W).
        """
        B, _, H, W = context.shape
        members = [self.ddim_sample(context, (B, 1, H, W), eta=eta) for _ in range(n_members)]
        return torch.stack(members, dim=1)
