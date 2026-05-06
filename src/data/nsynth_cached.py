"""HDF5-backed NSynth-bass mel cache reader.

Drop-in replacement for `NSynthBass` when mels have been pre-computed via
`scripts/prepare_nsynth_cache.py`. Reads indexed slices at training time
and avoids the CPU-bound STFT per-item.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class NSynthBassCached(Dataset):
    """Index-backed reader over an HDF5 produced by prepare_nsynth_cache.

    Returns items matching the `PairedPitchShiftCollate` contract:
        {"x": Tensor[1, n_mels, T], "labels": dict[str, Tensor]}

    Args:
        h5_path: HDF5 file from `scripts/prepare_nsynth_cache.py`.
        as_float32: promote mels to float32 on read (default True).
        drop_velocity: omit velocity label (default False).
    """

    _LABEL_KEYS: tuple[str, ...] = (
        "pitch", "velocity", "instrument_source", "instrument_id",
    )

    def __init__(
        self,
        h5_path: str | Path,
        *,
        as_float32: bool = True,
        drop_velocity: bool = False,
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"NSynthBassCached: missing cache at {self.h5_path}. "
                "Run scripts/prepare_nsynth_cache.py first."
            )
        self.as_float32 = bool(as_float32)
        self.drop_velocity = bool(drop_velocity)
        # Lazy open per worker — HDF5 handles are not fork-safe.
        self._h5: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            self._n: int = int(f["mel"].shape[0])
            self.sample_rate: int = int(f.attrs["sample_rate"])
            self.duration: float = float(f.attrs["duration"])
            self.n_mels: int = int(f.attrs["n_mels"])
            self._t_frames: int = int(f["mel"].shape[-1])

    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        f = self._ensure_open()
        # (n_mels, T) -> (1, n_mels, T) to match (channels=1, mel, time).
        mel = f["mel"][idx]
        if self.as_float32:
            mel = mel.astype(np.float32, copy=False)
        x = torch.from_numpy(mel).unsqueeze(0).contiguous()

        labels: dict[str, torch.Tensor] = {}
        for k in self._LABEL_KEYS:
            if k == "velocity" and self.drop_velocity:
                continue
            labels[k] = torch.tensor(int(f[k][idx]), dtype=torch.long)

        return {"x": x, "labels": labels}

    def __getstate__(self) -> dict[str, Any]:
        # Strip HDF5 handle on pickling (DataLoader workers get fresh handles).
        state = self.__dict__.copy()
        state["_h5"] = None
        return state
