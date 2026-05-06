"""DDSP decoder arm (plan Task 7.1).

Engel, J., Hantrakul, L., Gu, C., Roberts, A., 2020. "DDSP: Differentiable
Digital Signal Processing." ICLR. arXiv:2001.04643.

Harmonic-plus-noise synthesizer conditioned on the SC-VAE latent split:

    z_c : (B, d_c, T)   equivariant / time-resolved   -> f0(t), loudness(t)
    z_s : (B, d_s)       invariant / pooled           -> harmonic_dist (alpha),
                                                         noise_filter magnitudes

Timbre stays invariant to pitch shift because the harmonic distribution
`alpha` and the FIR magnitudes come only from `z_s`, which is pooled over
time and trained to be invariant under the pitch-shift group action.

Output: `(B, 1, n_samples)` mono waveform.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conditioning head


class DDSPHead(nn.Module):
    """Maps (z_s, z_c) -> DDSP control signals.

    z_c -> f0(t), loudness(t) via two 1-D convs. f0 is produced as a
    sigmoid in [0, 1] and scaled to [0, f0_max_hz]; loudness is a
    positive envelope via softplus.

    z_s -> harmonic distribution alpha (softmax over K partials) and
    FIR noise-filter magnitudes (softplus) via a 2-layer MLP.
    """

    def __init__(
        self,
        *,
        d_s: int,
        d_c: int,
        n_harmonics: int,
        n_noise_taps: int,
        f0_max_hz: float = 2000.0,
        hidden_c: int = 64,
        hidden_s: int = 128,
    ) -> None:
        super().__init__()
        if n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics}")
        if n_noise_taps < 1:
            raise ValueError(f"n_noise_taps must be >= 1, got {n_noise_taps}")
        self.n_harmonics = n_harmonics
        self.n_noise_taps = n_noise_taps
        self.f0_max_hz = float(f0_max_hz)

        # z_c: (B, d_c, T) -> two 1-D conv heads at the same frame rate.
        self.f0_conv = nn.Sequential(
            nn.Conv1d(d_c, hidden_c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_c, 1, kernel_size=1),
        )
        self.loud_conv = nn.Sequential(
            nn.Conv1d(d_c, hidden_c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_c, 1, kernel_size=1),
        )
        # z_s: (B, d_s) -> MLP -> (harmonic_dist, noise_filter).
        self.mlp_s = nn.Sequential(
            nn.Linear(d_s, hidden_s),
            nn.GELU(),
            nn.Linear(hidden_s, n_harmonics + n_noise_taps),
        )

    def forward(
        self, z_s: torch.Tensor, z_c: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if z_s.ndim != 2:
            raise ValueError(f"z_s must be (B, d_s), got {tuple(z_s.shape)}")
        if z_c.ndim != 3:
            raise ValueError(f"z_c must be (B, d_c, T), got {tuple(z_c.shape)}")

        f0 = torch.sigmoid(self.f0_conv(z_c).squeeze(1)) * self.f0_max_hz   # (B, T)
        loud = F.softplus(self.loud_conv(z_c).squeeze(1))                    # (B, T)

        h = self.mlp_s(z_s)                                                  # (B, K+N)
        alpha_logits = h[:, : self.n_harmonics]
        filter_raw = h[:, self.n_harmonics :]
        alpha = torch.softmax(alpha_logits, dim=-1)                          # (B, K)
        noise_filter = F.softplus(filter_raw)                                # (B, N)
        return {
            "f0_hz": f0,
            "loudness": loud,
            "harmonic_dist": alpha,
            "noise_filter": noise_filter,
        }


# ---------------------------------------------------------------------------
# Synthesizers — differentiable additive + filtered-noise


def _upsample_linear(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Linearly upsample `(B, T)` to `(B, n_samples)`."""
    return F.interpolate(
        x.unsqueeze(1), size=n_samples, mode="linear", align_corners=True
    ).squeeze(1)


def _harmonic_synth(
    f0_hz: torch.Tensor,        # (B, T)
    loudness: torch.Tensor,     # (B, T)
    alpha: torch.Tensor,        # (B, K)
    sample_rate: float,
    n_samples: int,
) -> torch.Tensor:
    """Additive synth — sum of K sinusoids, anti-aliased."""
    b, k = alpha.shape
    f0_s = _upsample_linear(f0_hz, n_samples)              # (B, S)
    loud_s = _upsample_linear(loudness, n_samples)          # (B, S)

    # Partial frequencies: k * f0(t), (B, K, S).
    idx = torch.arange(1, k + 1, device=alpha.device, dtype=alpha.dtype)
    partial_hz = idx[None, :, None] * f0_s[:, None, :]

    # Anti-alias mask — zero partials above Nyquist.
    nyquist = 0.5 * sample_rate
    mask = (partial_hz < nyquist).to(alpha.dtype)

    # Phase = cumulative angular frequency; divide by sample_rate to get
    # radians-per-sample, then cumsum.
    omega = 2.0 * math.pi * partial_hz / sample_rate
    phase = torch.cumsum(omega, dim=-1)

    # alpha carries the K-dim static weights; broadcast over time.
    weights = alpha[:, :, None] * mask                    # (B, K, S)
    wave = (torch.sin(phase) * weights).sum(dim=1)         # (B, S)
    return wave * loud_s


def _filtered_noise_synth_batched(
    noise_filter: torch.Tensor, n_samples: int
) -> torch.Tensor:
    """Vectorized filtered-noise: frequency-domain multiply then irfft."""
    b, n = noise_filter.shape
    # Reinterpret `noise_filter` as magnitudes over an rFFT of length
    # `n_fft = 2 * (n - 1)`. To match n_samples, we upsample the magnitude
    # response via linear interpolation to the target rfft length.
    n_fft = max(64, 1 << ((n_samples - 1).bit_length()))   # next power of 2
    target_bins = n_fft // 2 + 1
    mag = F.interpolate(
        noise_filter.unsqueeze(1), size=target_bins, mode="linear",
        align_corners=True,
    ).squeeze(1)                                            # (B, target_bins)

    noise = torch.randn(b, n_fft, device=noise_filter.device, dtype=noise_filter.dtype)
    spec = torch.fft.rfft(noise)                            # (B, target_bins)
    spec = spec * mag.to(spec.dtype)
    filtered = torch.fft.irfft(spec, n=n_fft)               # (B, n_fft)
    return filtered[..., :n_samples]


# ---------------------------------------------------------------------------
# Decoder


class DDSPDecoder(nn.Module):
    """Harmonic + filtered-noise DDSP decoder for the SC-VAE.

    Args:
        d_s, d_c:    style / content latent widths.
        n_harmonics: number of additive partials K.
        n_noise_taps: magnitude bins for the filtered-noise branch.
        sample_rate: audio sample rate in Hz.
        n_samples:   output waveform length (e.g. 64000 for 4 s at 16 kHz).
        n_frames:    number of control-rate frames for (f0, loudness).
        f0_max_hz:   upper bound on the sigmoid-mapped fundamental.
    """

    def __init__(
        self,
        *,
        d_s: int,
        d_c: int,
        n_harmonics: int = 100,
        n_noise_taps: int = 65,
        sample_rate: int = 16000,
        n_samples: int = 64000,
        n_frames: int = 250,
        f0_max_hz: float = 2000.0,
    ) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_c = d_c
        self.sample_rate = float(sample_rate)
        self.n_samples = int(n_samples)
        self.n_frames = int(n_frames)
        self.head = DDSPHead(
            d_s=d_s, d_c=d_c,
            n_harmonics=n_harmonics, n_noise_taps=n_noise_taps,
            f0_max_hz=f0_max_hz,
        )

    def _partial_mask(self, f0_hz: torch.Tensor, sample_rate: float) -> torch.Tensor:
        """Boolean mask `(B, T, K)` of partials below Nyquist. Used in tests."""
        k = self.head.n_harmonics
        idx = torch.arange(1, k + 1, device=f0_hz.device, dtype=f0_hz.dtype)
        partial = idx[None, None, :] * f0_hz[..., None]
        return partial < 0.5 * sample_rate

    def forward(self, z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        ctrl = self.head(z_s, z_c)
        harm = _harmonic_synth(
            f0_hz=ctrl["f0_hz"],
            loudness=ctrl["loudness"],
            alpha=ctrl["harmonic_dist"],
            sample_rate=self.sample_rate,
            n_samples=self.n_samples,
        )
        noise = _filtered_noise_synth_batched(
            noise_filter=ctrl["noise_filter"],
            n_samples=self.n_samples,
        )
        wave = harm + noise                                    # (B, S)
        return wave.unsqueeze(1)                                # (B, 1, S)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "d_s": self.d_s,
            "d_c": self.d_c,
            "n_harmonics": self.head.n_harmonics,
            "n_noise_taps": self.head.n_noise_taps,
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "n_frames": self.n_frames,
            "f0_max_hz": self.head.f0_max_hz,
        }
