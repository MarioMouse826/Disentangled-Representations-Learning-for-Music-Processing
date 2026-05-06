"""Reconstruction losses — multi-scale spectral (waveform) + L1 log-mel.

References:
- Yamamoto et al. 2020, "Parallel WaveGAN" (arXiv:1910.11480) — multi-scale
  STFT loss for neural vocoding.
- Engel et al. 2020, "DDSP" (arXiv:2001.04643) §3.2 — canonical spec of
  the sum-over-scales log-magnitude L1 loss used here.

Plan Task 3.7 splits recon into two paths:

1. **L1 on log-mel** (fast, every training step): the main training signal.
   SC-VAE decodes to a log-mel canvas directly, so L1 there matches the
   target grid exactly — no invertible-mel inference needed.

2. **Multi-scale spectral on waveforms** (expensive, periodic): auxiliary
   loss computed every N epochs on HiFi-GAN-reconstructed waveforms.
   Catches phase / high-frequency artifacts that log-mel loss is blind to.

Both paths are differentiable end-to-end so gradients from either path
propagate through the encoder + decoder without detours.
"""
from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn.functional as F
import torchaudio


_DEFAULT_FFT_SIZES: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)


# Cache STFT windows per (pid, device, dtype, n_fft). Keying by pid makes
# the cache safe under `fork` multiprocessing on Linux: each DataLoader
# worker process sees its own cache entries and never aliases a CUDA
# tensor across processes (CUDA contexts are not fork-safe). `spawn`
# workers start with an empty cache automatically. Naive per-call
# allocation would force GPU sync and hurt DataLoader throughput on
# small clips.
_WINDOW_CACHE: dict[tuple[int, str, torch.dtype, int], torch.Tensor] = {}


def _get_window(n_fft: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (os.getpid(), str(device), dtype, n_fft)
    w = _WINDOW_CACHE.get(key)
    if w is None:
        w = torch.hann_window(n_fft, device=device, dtype=dtype)
        _WINDOW_CACHE[key] = w
    return w


def _stft_log_magnitude(
    waveform: torch.Tensor,
    n_fft: int,
    hop_length: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Centered STFT magnitude in log scale. Returns `(B, F, T)`."""
    window = _get_window(n_fft, waveform.device, waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        return_complex=True,
    )
    mag = spec.abs()
    return torch.log(mag.clamp(min=eps))


def multi_scale_spectral_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    fft_sizes: Sequence[int] = _DEFAULT_FFT_SIZES,
) -> torch.Tensor:
    """L1 on log-magnitude STFT, summed over FFT sizes.

    Args:
        x, x_hat: `(B, T)` or `(B, 1, T)` waveforms.
        fft_sizes: FFT window sizes. Hop length per scale is `n_fft // 4`
            (standard). Default matches Engel et al. 2020 §3.2.

    Returns:
        Scalar loss = Σ_scales L1( log |STFT_n(x)|, log |STFT_n(x̂)| ).
    """
    if x.shape != x_hat.shape:
        raise ValueError(
            f"shape mismatch: x {tuple(x.shape)} vs x_hat {tuple(x_hat.shape)}"
        )
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(
                f"expected single channel in (B, 1, T), got {x.shape[1]}"
            )
        x = x.squeeze(1)
        x_hat = x_hat.squeeze(1)
    elif x.ndim != 2:
        raise ValueError(
            f"expected 2-D or 3-D waveform, got shape {tuple(x.shape)}"
        )
    if not fft_sizes:
        raise ValueError("fft_sizes must be a non-empty sequence")
    if any(n <= 0 for n in fft_sizes):
        raise ValueError(f"fft_sizes entries must be positive, got {tuple(fft_sizes)}")

    total = x.new_zeros(())
    for n_fft in fft_sizes:
        hop = max(n_fft // 4, 1)
        # Clip n_fft to signal length so very short clips don't crash STFT.
        effective_n_fft = min(n_fft, x.shape[-1])
        if effective_n_fft < 2:
            continue
        log_x = _stft_log_magnitude(x, effective_n_fft, max(effective_n_fft // 4, 1))
        log_xhat = _stft_log_magnitude(x_hat, effective_n_fft, max(effective_n_fft // 4, 1))
        total = total + F.l1_loss(log_x, log_xhat, reduction="mean")
    return total


def l1_log_mel_loss(
    mel: torch.Tensor,
    mel_hat: torch.Tensor,
) -> torch.Tensor:
    """L1 loss on log-mel spectrograms.

    `mel` and `mel_hat` are expected to already be in the log-dB scale
    produced by `src.data.features.LogMel` (or equivalent). Mean reduction
    over every element — simplest, matches β-VAE baseline.
    """
    if mel.shape != mel_hat.shape:
        raise ValueError(
            f"shape mismatch: mel {tuple(mel.shape)} vs mel_hat {tuple(mel_hat.shape)}"
        )
    return F.l1_loss(mel, mel_hat, reduction="mean")
