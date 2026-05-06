"""MoisesDB bass wrapper that converts raw waveforms to log-mel spectrograms.

Adapts `MoisesDBBass` (which returns raw audio tuples) to the
`{"x": Tensor[1, n_mels, T], "labels": dict}` format expected by `LitModule`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.features import LogMel
from src.data.moisesdb import MoisesDBBass


class MoisesDBBassMel(Dataset):
    """MoisesDBBass + LogMel frontend.

    Returns items in the same schema as `NSynthBassCached`:
        {"x": Tensor[1, n_mels, T_frames], "labels": {"track_id": str_tensor}}

    Args:
        root: MoisesDB dataset root directory.
        split: "train", "valid", "test", or "all".
        crop_seconds: audio crop length in seconds.
        sample_rate: target sample rate in Hz.
        deterministic: use fixed-hop crops (True for val/test).
        n_mels: number of mel frequency bins.
        n_fft: FFT size matching the features pipeline.
        hop_length: STFT hop matching the features pipeline.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_seconds: float = 4.0,
        sample_rate: int = 16000,
        deterministic: bool = False,
        n_mels: int = 128,
        n_fft: int = 400,
        hop_length: int = 160,
    ) -> None:
        self._bass = MoisesDBBass(
            root=root,
            split=split,
            crop_seconds=crop_seconds,
            sample_rate=sample_rate,
            deterministic=deterministic,
        )
        self._mel = LogMel(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    def __len__(self) -> int:
        return len(self._bass)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        wav, label = self._bass[idx]
        # wav: (1, T) float32
        x = self._mel(wav)  # (1, n_mels, T_frames)
        # MoisesDBLabel has str fields (track_id, stem) — only keep int fields
        # that are safe to tensorify and meaningful for downstream grouping.
        return {
            "x": x,
            "labels": {
                "track_idx": torch.tensor(label["track_idx"], dtype=torch.long),
                "segment_idx": torch.tensor(label["segment_idx"], dtype=torch.long),
            },
        }
