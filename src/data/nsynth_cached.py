"""HDF5-backed NSynth-bass feature cache reader.

Drop-in replacement for `NSynthBass` when features (mel or CQT) have been pre-computed via
`scripts/prepare_nsynth_cache.py`. Reads indexed slices at training time
and avoids the CPU-bound feature extraction per-item.
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

    Supports both mel-spectrogram and CQT features. Detects feature type from HDF5 attrs.
    Returns items matching the `PairedPitchShiftCollate` contract:
        {"x": Tensor[1, n_bins, T], "labels": dict[str, Tensor]}

    Args:
        h5_path: HDF5 file from `scripts/prepare_nsynth_cache.py`.
        as_float32: promote features to float32 on read (default True).
        drop_velocity: omit velocity label (default False).
        feature_type: "auto" (detect from HDF5), "mel", or "cqt". Default "auto".
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
        feature_type: str = "auto",
    ) -> None:
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(
                f"NSynthBassCached: missing cache at {self.h5_path}. "
                "Run scripts/prepare_nsynth_cache.py first."
            )
        self.as_float32 = bool(as_float32)
        self.drop_velocity = bool(drop_velocity)
        self.feature_type = feature_type
        
        # Lazy open per worker — HDF5 handles are not fork-safe.
        self._h5: h5py.File | None = None
        
        with h5py.File(self.h5_path, "r") as f:
            # Auto-detect feature type from available HDF5 keys
            if feature_type == "auto":
                if "cqt" in f:
                    self._feature_key = "cqt"
                    self.feature_type = "cqt"
                elif "mel" in f:
                    self._feature_key = "mel"
                    self.feature_type = "mel"
                else:
                    raise ValueError(
                        f"HDF5 has neither 'cqt' nor 'mel' key. Available: {list(f.keys())}"
                    )
            elif feature_type == "cqt":
                if "cqt" not in f:
                    raise ValueError(f"HDF5 has no 'cqt' key (expected for feature_type='cqt')")
                self._feature_key = "cqt"
            elif feature_type == "mel":
                if "mel" not in f:
                    raise ValueError(f"HDF5 has no 'mel' key (expected for feature_type='mel')")
                self._feature_key = "mel"
            else:
                raise ValueError(f"feature_type must be 'auto', 'mel', or 'cqt', got {feature_type!r}")
            
            self._n: int = int(f[self._feature_key].shape[0])
            self.sample_rate: int = int(f.attrs["sample_rate"])
            self.duration: float = float(f.attrs["duration"])
            self.n_bins: int = int(f[self._feature_key].shape[1])  # n_mels or n_cqt_bins
            self._t_frames: int = int(f[self._feature_key].shape[-1])

    def _ensure_open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    @property
    def n_mels(self) -> int:
        """Backward-compatibility property. Returns n_bins (n_mels for mel, n_cqt_bins for CQT)."""
        return self.n_bins

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        f = self._ensure_open()
        # (n_bins, T) -> (1, n_bins, T) to match (channels=1, feature, time).
        feature = f[self._feature_key][idx]
        if self.as_float32:
            feature = feature.astype(np.float32, copy=False)
        x = torch.from_numpy(feature).unsqueeze(0).contiguous()

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
