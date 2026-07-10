"""
unet.py — Conditioned UNet backbone for radar diffusion nowcasting.

Architecture
------------
Standard UNet with:
- Time-step embedding (sinusoidal) injected at every residual block
- Context conditioning: past radar frames concatenated along channel dim at input
- Skip connections between encoder and decoder
- GroupNorm (not BatchNorm) for compatibility with small batch sizes (batch=4)
- Self-attention at the bottleneck

Designed to fit within 8GB VRAM using fp16 and gradient checkpointing.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer diffusion timestep t → (B, dim) sinusoidal embedding."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time-step conditioning via AdaGN."""

    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int, groups: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch * 2)   # scale + shift

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def _forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale_shift = self.t_proj(self.act(t_emb))[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.conv1(self.act(self.norm1(x)))
        h = h * (1 + scale) + shift
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, t_emb, use_reentrant=False)
        return self._forward(x, t_emb)


class SelfAttention(nn.Module):
    """Single-head self-attention for the bottleneck."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.bmm(q.permute(0, 2, 1), k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, t_emb_dim, use_checkpoint=use_checkpoint)
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        h = self.res(x, t_emb)
        return self.down(h), h   # return downsampled + skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_emb_dim: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.res = ResBlock(in_ch + skip_ch, out_ch, t_emb_dim, use_checkpoint=use_checkpoint)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        # Handle odd spatial dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


class ConditionedUNet(nn.Module):
    """
    Diffusion UNet for precipitation nowcasting.

    Inputs
    ------
    x       : (B, 1, H, W)  — noisy target frame
    context : (B, C_ctx, H, W) — past radar frames concatenated along channel dim
    t       : (B,) int — diffusion timestep indices

    Output
    ------
    (B, 1, H, W) — predicted noise (or score)
    """

    def __init__(
        self,
        context_frames: int = 6,
        base_ch: int = 64,
        ch_mults: tuple[int, ...] = (1, 2, 4, 8),
        t_emb_dim: int = 256,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.t_emb_dim = t_emb_dim

        # Time embedding MLP
        self.t_mlp = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 4, t_emb_dim),
        )

        # Input: noisy frame (1 channel) + context frames (context_frames channels)
        in_ch = 1 + context_frames

        channels = [base_ch * m for m in ch_mults]

        # Stem
        self.stem = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        prev_ch = channels[0]
        for ch in channels[1:]:
            self.down_blocks.append(DownBlock(prev_ch, ch, t_emb_dim, use_checkpoint))
            prev_ch = ch

        # Bottleneck
        self.mid_res1 = ResBlock(prev_ch, prev_ch, t_emb_dim, use_checkpoint=use_checkpoint)
        self.mid_attn = SelfAttention(prev_ch)
        self.mid_res2 = ResBlock(prev_ch, prev_ch, t_emb_dim, use_checkpoint=use_checkpoint)

        # Decoder
        # skips.pop() yields channels[i] tensors (DownBlock output before striding),
        # so skip_ch must equal channels[i], not channels[i-1].
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_blocks.append(
                UpBlock(channels[i], channels[i], channels[i - 1], t_emb_dim, use_checkpoint)
            )
        prev_ch = channels[0]

        # Output head
        self.out = nn.Sequential(
            nn.GroupNorm(8, prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.t_emb_dim)
        t_emb = self.t_mlp(t_emb)

        # Concatenate noisy frame with conditioning context
        h = torch.cat([x, context], dim=1)
        h = self.stem(h)

        # Encoder
        skips = [h]
        for block in self.down_blocks:
            h, skip = block(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # Decoder
        for block in self.up_blocks:
            skip = skips.pop()
            h = block(h, skip, t_emb)

        return self.out(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
