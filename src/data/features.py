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


class ConstantQ(torch.nn.Module):
    """Constant-Q transform for music analysis — bass-optimized representation.

    Uses nnAudio backend for GPU acceleration and differentiability.
    Logarithmic frequency resolution (1 semitone per bin) provides equal perceptual
    weight across the entire frequency spectrum, with superior low-frequency detail
    compared to mel-scale for bass instrument analysis.

    Args:
        sample_rate: audio sample rate in Hz (default 16000).
        n_bins_per_octave: CQT frequency bins per octave (default 12 = 1 semitone).
        n_octaves: number of octaves to span (default 7 = C1 to B7, ~32.7 Hz to 3951 Hz).
        fmin: minimum frequency in Hz (default 32.7 = C1, CREPE reference point).
        hop_length: STFT hop size in samples (default 160, matching LogMel @ 16 kHz).
        top_db: dynamic range in dB (output clipped to [-top_db, 0]).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_bins_per_octave: int = 12,
        n_octaves: int = 7,
        fmin: float = 32.7,
        hop_length: int = 160,
        top_db: float = 80.0,
    ) -> None:
        super().__init__()

        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if n_bins_per_octave <= 0:
            raise ValueError(f"n_bins_per_octave must be positive, got {n_bins_per_octave}")
        if n_octaves <= 0:
            raise ValueError(f"n_octaves must be positive, got {n_octaves}")
        if fmin <= 0:
            raise ValueError(f"fmin must be positive, got {fmin}")
        if hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {hop_length}")
        if top_db <= 0:
            raise ValueError(f"top_db must be positive, got {top_db}")

        try:
            from nnAudio.features import CQT as CQT_nnAudio
        except ImportError as e:
            raise ImportError(
                "nnAudio not installed. Install via: pip install nnAudio"
            ) from e

        self.sample_rate = sample_rate
        self.n_bins_per_octave = n_bins_per_octave
        self.n_octaves = n_octaves
        self.fmin = fmin
        self.hop_length = hop_length
        self.top_db = top_db
        self.n_bins_total = n_bins_per_octave * n_octaves

        # nnAudio CQT: handles both CPU and GPU tensors transparently.
        # Returns complex tensor (batch, n_bins, n_frames, 2) or native complex.
        self.cqt_transform = CQT_nnAudio(
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            bins_per_octave=n_bins_per_octave,
            n_bins=self.n_bins_total,
            window="hann",
            center=True,
            pad_mode="reflect",
            norm=None,  # No normalization; we handle dB conversion
        )

    @property
    def config(self) -> dict:
        """Configuration dict for run-manifest logging."""
        return {
            "sample_rate": self.sample_rate,
            "n_bins_per_octave": self.n_bins_per_octave,
            "n_octaves": self.n_octaves,
            "fmin": self.fmin,
            "hop_length": self.hop_length,
            "n_bins_total": self.n_bins_total,
            "top_db": self.top_db,
        }

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute constant-Q transform in dB.

        Args:
            waveform: audio waveform with shape (..., T).

        Returns:
            Log-magnitude CQT in dB with shape (..., n_bins_total, T_frames),
            clipped to [-top_db, 0].

        Raises:
            TypeError: if waveform is not a Tensor.
        """
        if not torch.is_tensor(waveform):
            raise TypeError(f"waveform must be a Tensor, got {type(waveform).__name__}")

        # nnAudio.CQT expects shape (batch, ..., time) and returns complex magnitude.
        # Handle optional mono channel: (1, T) → (T,) or (batch, 1, T) → (batch, T).
        orig_shape = waveform.shape
        if waveform.dim() > 2 and waveform.shape[-2] == 1:
            waveform = waveform.squeeze(-2)

        # Compute CQT: returns (batch, n_bins, n_frames) in magnitude
        # or (batch, n_bins, n_frames, 2) if complex representation.
        cqt_out = self.cqt_transform(waveform)

        # Extract magnitude if nnAudio returns complex tuple/stacked representation.
        # Most recent nnAudio versions return complex natively; fall back to abs().
        if cqt_out.dim() == 4 and cqt_out.shape[-1] == 2:
            # Real + imaginary stacked on last dim: (batch, n_bins, n_frames, 2)
            cqt_mag = torch.sqrt(
                cqt_out[..., 0] ** 2 + cqt_out[..., 1] ** 2 + 1e-10
            )
        else:
            # Assume native complex or magnitude output
            cqt_mag = torch.abs(cqt_out) + 1e-10

        # Convert magnitude to dB
        cqt_db = 20.0 * torch.log10(cqt_mag)
        cqt_db = cqt_db.clamp(min=-self.top_db, max=0.0)

        return cqt_db.to(torch.float32)
