"""Log-mel spectrogram frontend.

Deterministic, fully differentiable mapping `waveform → log-mel`. Uses
`torchaudio.transforms.MelSpectrogram` + `AmplitudeToDB`, clamped to
`[top_db, 0] = [-80, 0]` dB as prescribed by NSynth / DDSP preprocessing.

Shape contract (torchaudio convention, preserved here):

    (T,)                 →  (n_mels, T_frames)
    (1, T)               →  (1, n_mels, T_frames)
    (batch, T)           →  (batch, n_mels, T_frames)
    (batch, channels, T) →  (batch, channels, n_mels, T_frames)

With defaults (sr=16000, n_fft=400, hop=160, center=True),
`T_frames = 1 + T // hop`.
"""
from __future__ import annotations

import torch
import torchaudio


class LogMel(torch.nn.Module):
    """Power-mel spectrogram in dB.

    Args:
        sample_rate: audio sample rate in Hz.
        n_fft: FFT size (samples). Default 400 ≈ 25 ms @ 16 kHz.
        hop_length: STFT hop (samples). Default 160 ≈ 10 ms @ 16 kHz.
        n_mels: number of mel bins.
        fmin: lower mel-filterbank edge (Hz).
        fmax: upper mel-filterbank edge (Hz). Must satisfy `fmin < fmax`.
        top_db: dynamic range in dB (output clipped to `[-top_db, 0]`).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 8000.0,
        top_db: float = 80.0,
    ) -> None:
        super().__init__()

        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {n_fft}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        if n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {n_mels}")
        if fmin < 0:
            raise ValueError(f"fmin must be non-negative, got {fmin}")
        if fmax <= fmin:
            raise ValueError(f"fmax ({fmax}) must exceed fmin ({fmin})")
        if top_db <= 0:
            raise ValueError(f"top_db must be positive, got {top_db}")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.top_db = top_db

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )
        # power-spectrogram → dB with a symmetric top_db clamp. We pass
        # `top_db=None` into torchaudio's AmplitudeToDB so its internal
        # auto-normalization does not shift the dB floor relative to the
        # silence reference — we handle the clamp ourselves in forward().
        self.a2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None)

    @property
    def config(self) -> dict:
        """Configuration dict for run-manifest logging."""
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "top_db": self.top_db,
        }

    # Note on n_mels=128 vs n_fft=400: at 16 kHz / 25 ms window, n_freqs = 201.
    # Packing 128 mel bins into 201 FFT bins leaves a handful of empty
    # filterbanks at low frequencies (torchaudio emits a one-time warning).
    # These bins hit the dB floor at −80 and are still deterministic +
    # differentiable, so downstream training is unaffected. Upstream fix would
    # be n_fft=1024 or n_mels=80; plan pins the current values.

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(waveform):
            raise TypeError(f"waveform must be a Tensor, got {type(waveform).__name__}")

        power = self.melspec(waveform)                # (..., n_mels, T')
        # Floor the power at 1e-10 before dB conversion so `log10(0)` never fires.
        # 1e-10 in power = -100 dB, well below the -80 dB clamp used downstream.
        power = power.clamp(min=1e-10)
        db = self.a2db(power)
        return db.clamp(min=-self.top_db, max=0.0).to(torch.float32)
