"""
diffusion.py — DDPM diffusion process for radar nowcasting.

Implements:
- Linear noise schedule (beta schedule)
- Forward process: q(x_t | x_0)
- Reverse process: p_theta(x_{t-1} | x_t, context)
- Training loss: simplified L_simple (predict noise)
- Inference: DDIM sampler for fast generation (50 steps vs 1000 training steps)
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
    ):
        super().__init__()
        self.model = model
        self.T = timesteps
        self.inference_steps = inference_steps

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

        x_t, noise = self.q_sample(x0, t)
        noise_pred = self.model(x_t, context, t)

        if not intensity_alpha:
            return F.mse_loss(noise_pred, noise)

        w = 1.0 + intensity_alpha * (x0.detach() + 1.0) * 0.5
        w = w / w.mean()                      # keep the loss scale unchanged
        return (w * (noise_pred - noise) ** 2).mean()

    # ── DDIM inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        context: torch.Tensor,
        shape: tuple[int, ...],
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate a sample using DDIM (fast, deterministic when eta=0).

        Parameters
        ----------
        context : (B, C, H, W) conditioning frames
        shape   : output shape, e.g. (B, 1, H, W)
        eta     : stochasticity (0 = deterministic DDIM, 1 = DDPM)
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

            noise_pred = self.model(x, context, t)

            # Predicted x_0
            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            # DDIM update
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
            direction = (1 - alpha_prev - sigma ** 2).clamp(min=0).sqrt() * noise_pred
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0

            x = alpha_prev.sqrt() * x0_pred + direction + noise

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
