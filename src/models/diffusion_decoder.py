"""Diffusion decoder arm (plan Task 7.2).

Ho, Jain, Abbeel 2020 (DDPM, arXiv:2006.11239) + Nichol & Dhariwal 2021
(iDDPM, arXiv:2102.09672). Conditional mel-spectrogram diffusion with a
lightweight U-Net, cosine beta schedule, eps-parameterization, classifier-
free guidance, and a 50-step DDIM sampler.

Conditioning:
    z_s (B, d_s)      -> added to the sinusoidal timestep embedding via a
                         linear projection (FiLM-style global conditioning).
    z_c (B, d_c, T)   -> concatenated to the U-Net input along the channel
                         dim, broadcast along the frequency axis. Preserves
                         time structure needed for equivariance.

Classifier-free guidance: per-sample Bernoulli dropout of conditioning
during training (p = `cond_dropout_p`); at inference sample with guidance
scale `w`. For `w = 1` we recover plain conditional sampling.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schedules + embeddings


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule (Nichol & Dhariwal 2021)."""
    steps = n_timesteps + 1
    t = torch.linspace(0, n_timesteps, steps, dtype=torch.float64) / n_timesteps
    alphas_cumprod = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(1e-6, 0.999).to(torch.float32)


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard Transformer-style positional encoding for discrete steps t."""
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------------------------------------------------------
# Minimal conditional U-Net — small enough for tests, shape matches the plan


class _FiLM(nn.Module):
    """Per-channel affine modulation from a conditioning vector."""

    def __init__(self, cond_dim: int, n_channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * n_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ss = self.to_scale_shift(cond)                  # (B, 2C)
        scale, shift = ss.chunk(2, dim=-1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film = _FiLM(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _SelfAttention2d(nn.Module):
    """Single-head self-attention over spatial positions. Gateable at low
    resolution to keep compute bounded."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)       # (B, HW, C)
        k = k.reshape(b, c, h * w)                         # (B, C, HW)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)        # (B, HW, C)
        # Cast to float32 before softmax — under AMP/fp16 the q @ k product
        # can overflow fp16 at non-trivial sequence length, silently NaN-ing
        # the entire loss. Cast back to the input dtype after softmax.
        logits = (q @ k * self.scale).float()
        attn = torch.softmax(logits, dim=-1).to(q.dtype)   # (B, HW, HW)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class _UNet(nn.Module):
    """Minimal 2-D U-Net with FiLM + optional self-attention at selected
    resolutions. Down-samples across freq axis; time axis preserved so z_c
    can be concatenated at the input."""

    def __init__(
        self,
        *,
        in_channels: int,          # 1 (noisy mel) + d_c (z_c concat)
        input_h: int,              # spatial height at level 0 (e.g. n_mels)
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        attn_resolutions: Sequence[int] = (16,),
        cond_dim: int = 128,
    ) -> None:
        super().__init__()
        self.channel_mult = list(channel_mult)
        # attn_resolutions = set of *spatial resolutions* at which to apply
        # self-attention (e.g. 16 → apply when feature-map height == 16).
        self.attn_resolutions = set(attn_resolutions)
        chs = [base_channels * m for m in self.channel_mult]
        # Per-level spatial height (before the level's downsample). Each
        # downsample halves the height; last level has no downsample.
        res_at_level = [max(1, input_h >> level) for level in range(len(chs))]

        self.in_conv = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        # Encoder: two ResBlocks + optional attention + downsample per level.
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        for level, c in enumerate(chs):
            self.down_blocks.append(nn.ModuleList([
                _ResBlock(prev, c, cond_dim),
                _ResBlock(c, c, cond_dim),
            ]))
            use_attn = res_at_level[level] in self.attn_resolutions
            self.down_attn.append(_SelfAttention2d(c) if use_attn else nn.Identity())
            prev = c
            # Downsample everywhere except the last level.
            if level < len(chs) - 1:
                self.downs.append(nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1))
            else:
                self.downs.append(nn.Identity())

        # Mid.
        self.mid_block1 = _ResBlock(prev, prev, cond_dim)
        self.mid_attn = _SelfAttention2d(prev)
        self.mid_block2 = _ResBlock(prev, prev, cond_dim)

        # Decoder — mirror; each level takes skip from down.
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.ups = nn.ModuleList()
        for level in reversed(range(len(chs))):
            c = chs[level]
            self.up_blocks.append(nn.ModuleList([
                _ResBlock(prev + c, c, cond_dim),
                _ResBlock(c, c, cond_dim),
            ]))
            use_attn = res_at_level[level] in self.attn_resolutions
            self.up_attn.append(_SelfAttention2d(c) if use_attn else nn.Identity())
            prev = c
            if level > 0:
                self.ups.append(nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1))
            else:
                self.ups.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(8, chs[0]), chs[0])
        self.out_conv = nn.Conv2d(chs[0], 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for blocks, attn, down in zip(self.down_blocks, self.down_attn, self.downs):
            for b in blocks:
                h = b(h, cond)
            if not isinstance(attn, nn.Identity):
                h = attn(h)
            skips.append(h)
            h = down(h)
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)
        for blocks, attn, up in zip(self.up_blocks, self.up_attn, self.ups):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for b in blocks:
                h = b(h, cond)
            if not isinstance(attn, nn.Identity):
                h = attn(h)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Decoder


class DiffusionDecoder(nn.Module):
    """DDPM-style mel diffusion with eps-parameterization, CFG dropout,
    and a DDIM sampler."""

    def __init__(
        self,
        *,
        d_s: int,
        d_c: int,
        n_mels: int,
        n_time: int,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        attn_resolutions: Sequence[int] = (16,),
        n_timesteps: int = 1000,
        cond_emb_dim: int = 128,
        cond_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 <= cond_dropout_p <= 1.0:
            raise ValueError(f"cond_dropout_p must be in [0,1], got {cond_dropout_p}")
        self.d_s = d_s
        self.d_c = d_c
        self.n_mels = n_mels
        self.n_time = n_time
        self.n_timesteps = int(n_timesteps)
        self.cond_dropout_p = float(cond_dropout_p)

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )

        # z_s + timestep -> FiLM conditioning vector.
        self.t_emb_dim = cond_emb_dim
        self.s_proj = nn.Linear(d_s, cond_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )

        self.unet = _UNet(
            in_channels=1 + d_c,
            input_h=n_mels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            attn_resolutions=attn_resolutions,
            cond_dim=cond_emb_dim,
        )

        # Flag set by last training_loss call — True = sample kept cond;
        # False = sample dropped (CFG null path). Plain attribute (not a
        # registered buffer) because we overwrite it every step; buffer
        # semantics (state_dict, .to() move) would silently clobber on
        # every assignment via `self.X = ...`.
        self.last_cond_kept: torch.Tensor = torch.ones(0, dtype=torch.bool)

    # -- conditioning ------------------------------------------------------

    def _build_cond(
        self,
        t: torch.Tensor,
        z_s: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """FiLM vector from timestep + (optionally masked) z_s."""
        t_raw = sinusoidal_timestep_embedding(t, self.t_emb_dim)
        t_vec = self.t_proj(t_raw)
        s_vec = self.s_proj(z_s) * keep_mask[:, None].to(z_s.dtype)
        return t_vec + s_vec

    def _concat_zc(
        self, x: torch.Tensor, z_c: torch.Tensor, keep_mask: torch.Tensor
    ) -> torch.Tensor:
        """Broadcast z_c (B, d_c, T) across frequency, concat to channels.

        Masked samples have their z_c channels zeroed — matches the null
        classifier-free signal.
        """
        b, _, n_mels, n_time = x.shape
        if z_c.shape[-1] != n_time:
            z_c = F.interpolate(z_c, size=n_time, mode="linear", align_corners=True)
        zc_expanded = z_c[:, :, None, :].expand(b, z_c.shape[1], n_mels, n_time)
        zc_expanded = zc_expanded * keep_mask[:, None, None, None].to(zc_expanded.dtype)
        return torch.cat([x, zc_expanded], dim=1)

    # -- training loss -----------------------------------------------------

    def training_loss(
        self, *, x0: torch.Tensor, z_s: torch.Tensor, z_c: torch.Tensor
    ) -> torch.Tensor:
        """L_simple = E[|| eps - eps_theta(x_t, t, z_s, z_c) ||^2]."""
        b = x0.shape[0]
        t = torch.randint(0, self.n_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        a_bar = self.alphas_cumprod[t].view(b, 1, 1, 1)
        x_t = a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise

        # Classifier-free dropout: per-sample Bernoulli mask.
        if self.training and self.cond_dropout_p > 0.0:
            keep = torch.rand(b, device=x0.device) > self.cond_dropout_p
        else:
            keep = torch.ones(b, dtype=torch.bool, device=x0.device)
        self.last_cond_kept = keep.clone().detach()

        cond = self._build_cond(t, z_s, keep)
        x_in = self._concat_zc(x_t, z_c, keep)
        pred = self.unet(x_in, cond)
        return F.mse_loss(pred, noise)

    # -- sampling ----------------------------------------------------------

    @torch.no_grad()
    def sample_ddim(
        self,
        *,
        z_s: torch.Tensor,
        z_c: torch.Tensor,
        n_steps: int = 50,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
        shape: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """DDIM sampler with classifier-free guidance.

        Args:
            guidance_scale: `w` in `eps = eps_uncond + w * (eps_cond - eps_uncond)`.
                `w = 1` recovers plain conditional sampling.
            eta: 0 = deterministic DDIM; 1 = DDPM-style stochastic.
            shape: output shape `(B, 1, n_mels, n_time)`; inferred from
                z_s if omitted.
        """
        b = z_s.shape[0]
        device = z_s.device
        if shape is None:
            shape = (b, 1, self.n_mels, self.n_time)
        x = torch.randn(shape, device=device)

        # Uniformly spaced step schedule.
        step_indices = torch.linspace(
            self.n_timesteps - 1, 0, n_steps, device=device
        ).long()

        keep_full = torch.ones(b, dtype=torch.bool, device=device)
        keep_null = torch.zeros(b, dtype=torch.bool, device=device)

        for i, t_val in enumerate(step_indices):
            t = torch.full((b,), int(t_val), device=device, dtype=torch.long)
            a_bar = self.alphas_cumprod[t].view(b, 1, 1, 1)

            cond_c = self._build_cond(t, z_s, keep_full)
            x_in_c = self._concat_zc(x, z_c, keep_full)
            eps_c = self.unet(x_in_c, cond_c)

            if guidance_scale != 1.0:
                cond_u = self._build_cond(t, z_s, keep_null)
                x_in_u = self._concat_zc(x, z_c, keep_null)
                eps_u = self.unet(x_in_u, cond_u)
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps = eps_c

            x0_hat = (x - (1.0 - a_bar).sqrt() * eps) / a_bar.sqrt().clamp_min(1e-8)

            if i == n_steps - 1:
                x = x0_hat
                break
            t_next = step_indices[i + 1]
            a_bar_next = self.alphas_cumprod[t_next].view(1, 1, 1, 1)
            # DDIM update — sigma via eta (0 => deterministic).
            sigma = eta * torch.sqrt(
                (1.0 - a_bar_next) / (1.0 - a_bar).clamp_min(1e-8)
                * (1.0 - a_bar / a_bar_next.clamp_min(1e-8))
            )
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = (
                a_bar_next.sqrt() * x0_hat
                + (1.0 - a_bar_next - sigma ** 2).clamp_min(0.0).sqrt() * eps
                + sigma * noise
            )
        return x

    @property
    def config(self) -> dict[str, Any]:
        return {
            "d_s": self.d_s,
            "d_c": self.d_c,
            "n_mels": self.n_mels,
            "n_time": self.n_time,
            "n_timesteps": self.n_timesteps,
            "cond_dropout_p": self.cond_dropout_p,
        }


# ---------------------------------------------------------------------------
# Real-time factor (DITTO-2, Wang 2024)


def real_time_factor(wall_time_seconds: float, audio_seconds: float) -> float:
    """RTF = wall-time / audio-duration. < 1 means faster than real time."""
    if audio_seconds <= 0:
        raise ValueError(f"audio duration must be > 0, got {audio_seconds}")
    return float(wall_time_seconds) / float(audio_seconds)
