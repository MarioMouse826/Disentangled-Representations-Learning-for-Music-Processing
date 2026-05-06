"""Mel-spectrogram to waveform vocoder.

Default: Griffin-Lim via torchaudio (no extra deps, ships with PyTorch).
Params match the LogMel frontend defaults in `src/data/features.py`:
    sample_rate=16000, n_fft=400, hop_length=160, n_mels=128.

Accepts decoder output `(B, 1, n_mels, T_frames)` and returns `(B, T_audio)`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class GriffinLimVocoder(nn.Module):
    """Griffin-Lim mel inversion.

    Args:
        sample_rate: audio sample rate in Hz.
        n_fft: FFT size — must match the LogMel frontend.
        hop_length: STFT hop — must match the LogMel frontend.
        n_mels: mel bins — must match the LogMel frontend.
        n_iter: Griffin-Lim iterations (more = better quality, slower).
        power: spectrogram power exponent passed to GriffinLim.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        n_iter: int = 32,
        power: float = 1.0,
    ) -> None:
        super().__init__()
        n_stft = n_fft // 2 + 1
        # InverseMelScale: (*, n_mels, T) → (*, n_stft, T)
        self.inverse_mel = T.InverseMelScale(
            n_stft=n_stft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0.0,
            f_max=float(sample_rate // 2),
        )
        # GriffinLim expects a power spectrogram (power=2.0 matches LogMel's
        # MelSpectrogram which uses power=2.0 internally).
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=n_iter,
            power=2.0,
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to waveform.

        Args:
            mel: `(B, 1, n_mels, T)` or `(B, n_mels, T)` — log-mel output
                from `SCDecoder`.

        Returns:
            `(B, T_audio)` waveform tensor.
        """
        if mel.dim() == 4:
            mel = mel.squeeze(1)  # (B, n_mels, T)
        # LogMel uses AmplitudeToDB(stype="power"): db = 10*log10(power).
        # Invert: power = 10^(db/10).
        power_mel = torch.pow(10.0, mel / 10.0)  # (B, n_mels, T) power scale
        spec = self.inverse_mel(power_mel)        # (B, n_stft, T) power linear
        wav = self.griffin_lim(spec)              # (B, T_audio)
        return wav
