"""Multi-window Signal-to-Reconstruction Ratio (SRR).

    SRR(x, x_hat) = 10 * log10( sum |S(x)|^2 / sum |S(x) - S(x_hat)|^2 )

where `S` is the magnitude STFT averaged over multiple window sizes. Default
windows {512, 1024, 2048} cover short-term transients and longer harmonics.

Returns +inf when recon is identical, -inf when reference is zero on a
non-zero recon. Batched input `(B, T)` returns `(B,)`.
"""
from __future__ import annotations

from typing import Iterable

import torch


_DEFAULT_WINDOWS: tuple[int, ...] = (512, 1024, 2048)


def _mag_stft(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Magnitude STFT with Hann window and `hop = n_fft // 4`.

    Shape: `(..., n_fft//2 + 1, frames)`.
    """
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.abs()


def compute_srr(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    *,
    window_sizes: Iterable[int] = _DEFAULT_WINDOWS,
) -> torch.Tensor:
    """Multi-window SRR in dB.

    Args:
        x: reference waveform `(T,)` or `(B, T)`.
        x_hat: reconstruction with same shape as `x`.
        window_sizes: STFT `n_fft` values to average over.

    Returns:
        Scalar tensor when input is `(T,)`; `(B,)` tensor when `(B, T)`.
    """
    if x.shape != x_hat.shape:
        raise ValueError(
            f"x and x_hat must share shape, got {tuple(x.shape)} vs "
            f"{tuple(x_hat.shape)}"
        )
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be (T,) or (B, T), got shape {tuple(x.shape)}")

    windows = tuple(window_sizes)
    if not windows:
        raise ValueError("window_sizes must be non-empty")

    signal_energy_sum = None
    error_energy_sum = None
    batched = x.ndim == 2
    # Sum reduce dims: everything except batch.
    reduce_dims = (-2, -1) if batched else None

    for n_fft in windows:
        s_x = _mag_stft(x, n_fft)
        s_hat = _mag_stft(x_hat, n_fft)
        sig = (s_x ** 2).sum(dim=reduce_dims)
        err = ((s_x - s_hat) ** 2).sum(dim=reduce_dims)
        signal_energy_sum = sig if signal_energy_sum is None else signal_energy_sum + sig
        error_energy_sum = err if error_energy_sum is None else error_energy_sum + err

    # Average across windows (constant factor — does not change the ratio in
    # dB because sum/sum = mean/mean, but keeps semantics explicit).
    n = len(windows)
    signal_energy = signal_energy_sum / n
    error_energy = error_energy_sum / n

    # Vectorized inf/nan handling:
    #   error == 0  → +inf  (perfect recon)
    #   signal == 0 → -inf  (zero reference, non-zero recon)
    #   both == 0   → NaN   (degenerate silence case)
    eps = torch.finfo(signal_energy.dtype).tiny
    ratio = signal_energy / error_energy.clamp(min=eps)
    srr = 10.0 * torch.log10(ratio.clamp(min=eps))

    perfect = error_energy == 0
    silent = (signal_energy == 0) & (error_energy > 0)
    srr = torch.where(perfect, torch.full_like(srr, float("inf")), srr)
    srr = torch.where(silent, torch.full_like(srr, float("-inf")), srr)
    return srr
