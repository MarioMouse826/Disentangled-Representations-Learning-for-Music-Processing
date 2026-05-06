"""β-VAE baseline.

Reference: Higgins et al. 2017, "β-VAE: Learning Basic Visual Concepts with a
Constrained Variational Framework". Gaussian posterior + standard-normal
prior; objective

    L(β) = E_q[log p(x|z)]  −  β · KL(q(z|x) ‖ p(z))

Implementation notes
--------------------
- Encoder trunk: reuse `src.models.encoder.MelEncoder` — same trunk is
  shared with every baseline and with SC-VAE, so all methods operate over
  identical representations (apples-to-apples comparison the plan demands).
- Bottleneck: mean-pool freq from encoder output → flatten → linear heads
  for μ / log σ².
- Decoder: symmetric upsampling stack (Upsample + Conv + GN + SiLU), not
  ConvTranspose2d, to avoid checkerboard artifacts on the mel canvas.
  Final `F.interpolate(size=(n_mels, n_time))` fixes any off-by-one sizing
  from integer-scale upsampling.
- Reconstruction loss: L1 on log-mel (more perceptual than MSE; NSynth §app).
- Reparameterization: stochastic in train mode, deterministic `z = mu` in
  inference mode so eval passes are reproducible.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import MelEncoder


class BetaVAE(nn.Module):
    """β-VAE over log-mel spectrograms.

    Args:
        d_z: latent dimensionality.
        n_mels: expected input freq axis size.
        n_time: expected input time axis size.
        channels: encoder/decoder channel schedule.
        strides: per-block time strides (freq stays 1 per plan default).
        decoder_base_freq: freq resolution at the decoder bottleneck. Must
            divide `n_mels` cleanly (128 / 4 = 32 blocks-per-freq — checked).
    """

    def __init__(
        self,
        d_z: int = 64,
        n_mels: int = 128,
        n_time: int = 401,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2, 2),
        decoder_base_freq: int = 4,
    ) -> None:
        super().__init__()
        if d_z <= 0:
            raise ValueError(f"d_z must be positive, got {d_z}")
        if n_mels <= 0 or n_time <= 0:
            raise ValueError(f"n_mels and n_time must be positive")

        self.d_z = d_z
        self.n_mels = n_mels
        self.n_time = n_time
        self._channels = tuple(channels)
        self._strides = tuple(strides)
        self._decoder_base_freq = decoder_base_freq

        self.encoder = MelEncoder(channels=channels, strides=strides)

        # Encoder output time resolution (freq is mean-pooled → 1 after pool).
        self._t_enc = self._compute_encoder_time(n_time, self._strides)
        flat = self.encoder.out_channels * self._t_enc

        self.to_mu = nn.Linear(flat, d_z)
        self.to_logvar = nn.Linear(flat, d_z)

        # Decoder: linear → (B, C_max, base_freq, t_enc) → N up-blocks.
        self._decoder_c0 = self.encoder.out_channels
        self.from_z = nn.Linear(d_z, self._decoder_c0 * decoder_base_freq * self._t_enc)

        # Upsample stack: reverse channel schedule.
        rev = list(reversed(self._channels))
        self.up_blocks = nn.ModuleList()
        c_prev = rev[0]
        for c in rev[1:] + [rev[-1] // 2 if rev[-1] > 1 else 1]:
            self.up_blocks.append(_UpBlock(c_prev, c))
            c_prev = c
        self.final_conv = nn.Conv2d(c_prev, 1, kernel_size=3, padding=1)

    # -- shape helpers ---------------------------------------------------

    @staticmethod
    def _compute_encoder_time(n_time: int, strides: Sequence[int]) -> int:
        # Conv2d with kernel=3, stride=s, padding=1: out = ceil(L / s).
        t = n_time
        for s in strides:
            t = (t + s - 1) // s
        return t

    # -- config dump -----------------------------------------------------

    @property
    def config(self) -> dict:
        return {
            "d_z": self.d_z,
            "n_mels": self.n_mels,
            "n_time": self.n_time,
            "channels": self._channels,
            "strides": self._strides,
            "decoder_base_freq": self._decoder_base_freq,
            "encoder": self.encoder.config,
        }

    # -- encode / reparameterize / decode -------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.n_mels:
            raise ValueError(
                f"expected input shape (B, 1, {self.n_mels}, T), got {tuple(x.shape)}"
            )
        h = self.encoder(x)                  # (B, C, T_enc)
        h_flat = h.flatten(start_dim=1)       # (B, C * T_enc)
        return self.to_mu(h_flat), self.to_logvar(h_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Deterministic (z = mu) during inference for reproducible eval
        # passes; stochastic when training.
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_z(z)
        h = h.view(z.shape[0], self._decoder_c0, self._decoder_base_freq, self._t_enc)
        for block in self.up_blocks:
            h = block(h)
        h = self.final_conv(h)
        # Canonical output size — guards against integer-rounding drift from
        # the cascaded scale-factor-2 upsamples.
        if h.shape[-2:] != (self.n_mels, self.n_time):
            h = F.interpolate(
                h, size=(self.n_mels, self.n_time), mode="bilinear", align_corners=False
            )
        return h

    # -- forward + loss --------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}

    def loss(
        self,
        x: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self(x)
        # L1 reconstruction on log-mel (NSynth app: more perceptual than MSE).
        recon = F.l1_loss(out["x_hat"], x, reduction="mean")
        # Analytic Gaussian KL(q || N(0, I)) summed over d_z, averaged over batch.
        kl = (
            -0.5 * (1.0 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
        ).sum(dim=1).mean()
        total = recon + beta * kl
        return total, {"recon": recon.detach(), "kl": kl.detach()}


class _UpBlock(nn.Module):
    """Upsample (×2 bilinear, both dims) → Conv → GN → SiLU → Conv → GN → SiLU.

    Bilinear upsample + Conv avoids the checkerboard artifacts that plague
    ConvTranspose2d-based VAE decoders on high-resolution targets.
    """

    def __init__(self, in_c: int, out_c: int, groups: int = 8) -> None:
        super().__init__()
        g = groups if out_c % groups == 0 else 1
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
