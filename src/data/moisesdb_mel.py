"""MoisesDB bass wrapper that converts raw waveforms to log-mel or CQT spectrograms.

Adapts `MoisesDBBass` (which returns raw audio tuples) to the
`{"x": Tensor[1, n_bins, T], "labels": dict}` format expected by `LitModule`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from src.data.features import ConstantQ, LogMel
from src.data.moisesdb import MoisesDBBass


class MoisesDBBassMel(Dataset):
    """MoisesDBBass + feature frontend (LogMel or CQT).

    Returns items in the same schema as `NSynthBassCached`:
        {"x": Tensor[1, n_bins, T_frames], "labels": {"track_idx": ..., ...}}

    Args:
        root: MoisesDB dataset root directory.
        split: "train", "valid", "test", or "all".
        crop_seconds: audio crop length in seconds.
        sample_rate: target sample rate in Hz.
        deterministic: use fixed-hop crops (True for val/test).
        feature_type: "mel" or "cqt" (default "cqt").
        n_mels: number of mel frequency bins (only used if feature_type="mel").
        n_fft: FFT size (only used if feature_type="mel").
        hop_length: STFT hop (used for both mel and CQT; default 160).
        n_bins_per_octave: CQT bins per octave (only used if feature_type="cqt").
        n_octaves: CQT octaves (only used if feature_type="cqt").
        fmin: CQT minimum frequency (only used if feature_type="cqt").
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_seconds: float = 4.0,
        sample_rate: int = 16000,
        deterministic: bool = False,
        feature_type: Literal["mel", "cqt"] = "cqt",
        n_mels: int = 128,
        n_fft: int = 400,
        hop_length: int = 160,
        n_bins_per_octave: int = 12,
        n_octaves: int = 7,
        fmin: float = 32.7,
    ) -> None:
        if feature_type not in {"mel", "cqt"}:
            raise ValueError(f"feature_type must be 'mel' or 'cqt', got {feature_type!r}")

        self._bass = MoisesDBBass(
            root=root,
            split=split,
            crop_seconds=crop_seconds,
            sample_rate=sample_rate,
            deterministic=deterministic,
        )

        if feature_type == "cqt":
            self._feature = ConstantQ(
                sample_rate=sample_rate,
                n_bins_per_octave=n_bins_per_octave,
                n_octaves=n_octaves,
                fmin=fmin,
                hop_length=hop_length,
            )
        else:  # feature_type == "mel"
            self._feature = LogMel(
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
        x = self._feature(wav)  # (1, n_bins, T_frames) — n_bins is n_mels or n_cqt_bins
        # MoisesDBLabel has str fields (track_id, stem) — only keep int fields
        # that are safe to tensorify and meaningful for downstream grouping.
        return {
            "x": x,
            "labels": {
                "track_idx": torch.tensor(label["track_idx"], dtype=torch.long),
                "segment_idx": torch.tensor(label["segment_idx"], dtype=torch.long),
            },
        }
