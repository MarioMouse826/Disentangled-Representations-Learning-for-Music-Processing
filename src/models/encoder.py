"""Shared convolutional encoder trunk used by every VAE variant.

Reference: Donahue et al. 2019 (Adversarial Audio Synthesis) + NSynth baseline.

Shape contract
--------------
Input:  log-mel spectrogram of shape `(B, 1, n_mels, T)` (as produced by
        `src.data.features.LogMel`).
Output: `(B, C_out, T')` where `C_out = channels[-1]` and `T' = T / prod(strides)`
        (ceiling division due to Conv2d padding).

The trunk is shared between:
- β-VAE / FactorVAE / β-TCVAE / AR-HVAE baselines (Phase 2)
- SC-VAE core (Phase 3) — feeds the style head (invariance-constrained) and
  content head (equivariance-constrained) via separate head modules.

Default stack: 5 ConvBlocks at channels=(32, 64, 128, 256, 512), time-axis
strides (2, 2, 2, 2, 2), freq-axis stride held at 1 until the final
mean-over-freq pool. This follows the plan's Task 2.1 spec exactly.

Optional `stride_freq=True` mode strides both spatial dims — cuts per-layer
compute and memory ~4×. Reserved for the compute-budget ablation.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two stride-once conv-groupnorm-silu pairs.

    The first conv does the requested spatial downsampling; the second is a
    refinement conv with stride 1 so the block has ~2× the effective depth
    of a single-conv downsampler.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_time: int,
        stride_freq: int = 1,
        groups: int = 8,
    ) -> None:
        super().__init__()
        # `num_groups` must divide `num_channels`; drop to 1 group if
        # out_channels is too small for the default 8.
        g = groups if out_channels % groups == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(stride_freq, stride_time),
                padding=(1, 1),
            ),
            nn.GroupNorm(g, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.GroupNorm(g, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MelEncoder(nn.Module):
    """Shared conv trunk over log-mel spectrograms.

    Args:
        channels: per-block output channel sizes. Length N → N ConvBlocks.
        strides: per-block time-axis strides. Must match `channels` length.
        stride_freq: if True, freq-axis is also strided by `strides[i]` per
            block (for the compute-saving ablation); if False, freq stays
            full-resolution and is mean-collapsed in forward.
        in_channels: input channel dim; 1 for log-mel.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2, 2),
        stride_freq: bool = False,
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        channels = tuple(channels)
        strides = tuple(strides)

        if not channels:
            raise ValueError("channels must be non-empty")
        if len(channels) != len(strides):
            raise ValueError(
                f"channels and strides must be same length: "
                f"got {len(channels)} vs {len(strides)}"
            )
        if any(c <= 0 for c in channels):
            raise ValueError(f"channels must be positive, got {channels}")
        if any(s <= 0 for s in strides):
            raise ValueError(f"strides must be positive, got {strides}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")

        self._channels = channels
        self._strides = strides
        self._stride_freq = stride_freq
        self._in_channels = in_channels

        blocks: list[ConvBlock] = []
        c_prev = in_channels
        for c, s in zip(channels, strides):
            blocks.append(
                ConvBlock(
                    in_channels=c_prev,
                    out_channels=c,
                    stride_time=s,
                    stride_freq=s if stride_freq else 1,
                )
            )
            c_prev = c
        self.blocks = nn.ModuleList(blocks)
        self.out_channels: int = channels[-1]

    # -- config dump (for run_manifest.json) -----------------------------

    @property
    def config(self) -> dict:
        return {
            "channels": self._channels,
            "strides": self._strides,
            "stride_freq": self._stride_freq,
            "in_channels": self._in_channels,
            "out_channels": self.out_channels,
        }

    # -- forward ---------------------------------------------------------

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 4:
            raise ValueError(
                f"expected 4-D log-mel input (B, 1, n_mels, T), got shape {tuple(mel.shape)}"
            )
        if mel.shape[1] != self._in_channels:
            raise ValueError(
                f"expected {self._in_channels} input channel(s), got {mel.shape[1]}"
            )
        h = mel
        for block in self.blocks:
            h = block(h)
        # Collapse freq axis. In default mode freq is still 128 here; in
        # stride_freq=True mode freq is 128 / prod(strides). Either way,
        # mean over dim=2 produces (B, C_out, T').
        return h.mean(dim=2)
