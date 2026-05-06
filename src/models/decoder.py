"""SC-VAE decoder — maps `(z_s, z_c) → mel spectrogram`.

Design (plan Task 3.4)
----------------------
Input latents:
    z_s : (B, d_s)           — time-pooled style latent (invariance constrained)
    z_c : (B, d_c, T_enc)    — time-resolved content latent (equivariance constrained)

Output:
    x_hat : (B, 1, n_mels, n_time)

Fusion strategy
---------------
1. Broadcast z_s along the time axis → (B, d_s, T_enc), concat with z_c on
   channel dim → (B, d_s + d_c, T_enc). Preserves content time structure —
   critical for `L_equi` gradient to reach the decoder.
2. 1×1 Conv1d projects the fused channel count up to `C_max · base_freq`
   and reshapes to (B, C_max, base_freq, T_enc). A small `base_freq` (= 4
   by default) plus 5 ×2-upsamples gives the 128-mel target.
3. 5 `_UpBlock`s (bilinear upsample + 2× Conv-GN-SiLU) — same blocks as
   the baselines for apples-to-apples comparison. Avoids checkerboard
   artifacts of ConvTranspose2d.
4. Final 3×3 Conv to 1 channel, then `F.interpolate(size=(n_mels, n_time))`
   to lock the output to the exact training target size (guards against
   integer-rounding drift from cascaded ×2 upsamples).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines.beta_vae import _UpBlock


class SCDecoder(nn.Module):
    """Decoder for the Symmetry-Constrained VAE.

    Args:
        d_s: style-latent dim.
        d_c: content-latent dim.
        n_mels: target mel-spectrogram freq axis size.
        n_time: target time axis size.
        t_enc: encoder-output time resolution (matches `MelEncoder` output T').
        channels: bottom-up channel schedule (same convention as encoder —
            reversed internally for the upsample stack).
        base_freq: freq resolution at the decoder bottleneck. 5 ×2-upsamples
            from base_freq=4 reach 128 exactly.
    """

    def __init__(
        self,
        d_s: int,
        d_c: int,
        n_mels: int = 128,
        n_time: int = 401,
        t_enc: int = 13,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        base_freq: int = 4,
    ) -> None:
        super().__init__()
        if d_s <= 0:
            raise ValueError(f"d_s must be positive, got {d_s}")
        if d_c <= 0:
            raise ValueError(f"d_c must be positive, got {d_c}")
        if n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {n_mels}")
        if n_time <= 0:
            raise ValueError(f"n_time must be positive, got {n_time}")
        if t_enc <= 0:
            raise ValueError(f"t_enc must be positive, got {t_enc}")
        if base_freq <= 0:
            raise ValueError(f"base_freq must be positive, got {base_freq}")

        channels = tuple(channels)
        if not channels:
            raise ValueError("channels must be non-empty")

        self._d_s = d_s
        self._d_c = d_c
        self._n_mels = n_mels
        self._n_time = n_time
        self._t_enc = t_enc
        self._channels = channels
        self._base_freq = base_freq

        c_max = channels[-1]
        self._c_max = c_max

        # Fusion: (d_s + d_c, T_enc) → (c_max * base_freq, T_enc) via 1×1 Conv1d.
        # Reshape to (c_max, base_freq, T_enc) happens in forward().
        self.fuse = nn.Conv1d(d_s + d_c, c_max * base_freq, kernel_size=1)

        # Upsample stack — reverse channel schedule, same block as baselines.
        rev = list(reversed(channels))
        self.up_blocks = nn.ModuleList()
        c_prev = rev[0]
        for c in rev[1:] + [rev[-1] // 2 if rev[-1] > 1 else 1]:
            self.up_blocks.append(_UpBlock(c_prev, c))
            c_prev = c
        self.final_conv = nn.Conv2d(c_prev, 1, kernel_size=3, padding=1)

    @property
    def config(self) -> dict:
        return {
            "d_s": self._d_s,
            "d_c": self._d_c,
            "n_mels": self._n_mels,
            "n_time": self._n_time,
            "t_enc": self._t_enc,
            "base_freq": self._base_freq,
            "channels": self._channels,
        }

    def forward(self, z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        if z_s.ndim != 2:
            raise ValueError(
                f"expected z_s of shape (B, d_s), got 2-D required but {tuple(z_s.shape)}"
            )
        if z_c.ndim != 3:
            raise ValueError(
                f"expected z_c of shape (B, d_c, T_enc), got 3-D required but {tuple(z_c.shape)}"
            )
        if z_s.shape[0] != z_c.shape[0]:
            raise ValueError(
                f"batch size mismatch: z_s {z_s.shape[0]} vs z_c {z_c.shape[0]}"
            )
        if z_s.shape[1] != self._d_s:
            raise ValueError(f"expected d_s={self._d_s}, got z_s channel {z_s.shape[1]}")
        if z_c.shape[1] != self._d_c:
            raise ValueError(f"expected d_c={self._d_c}, got z_c channel {z_c.shape[1]}")
        if z_c.shape[2] != self._t_enc:
            raise ValueError(f"expected t_enc={self._t_enc}, got z_c time {z_c.shape[2]}")

        B = z_s.shape[0]

        # Broadcast z_s along time to match z_c.
        z_s_tiled = z_s.unsqueeze(-1).expand(B, self._d_s, self._t_enc)  # (B, d_s, T_enc)
        fused = torch.cat([z_s_tiled, z_c], dim=1)                        # (B, d_s + d_c, T_enc)

        h = self.fuse(fused)                                              # (B, c_max * base_freq, T_enc)
        h = h.view(B, self._c_max, self._base_freq, self._t_enc)          # (B, c_max, base_freq, T_enc)

        for block in self.up_blocks:
            h = block(h)

        x_hat = self.final_conv(h)                                        # (B, 1, ~n_mels, ~n_time)
        if x_hat.shape[-2:] != (self._n_mels, self._n_time):
            x_hat = F.interpolate(
                x_hat,
                size=(self._n_mels, self._n_time),
                mode="bilinear",
                align_corners=False,
            )
        return x_hat
