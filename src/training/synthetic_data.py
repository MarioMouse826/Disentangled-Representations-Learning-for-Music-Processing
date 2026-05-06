"""Synthetic paired dataset for Hydra CLI smoke tests — no disk I/O."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticPairedDataset(Dataset):
    """Deterministic synthetic `(x, x_g, g_cents)` triplets at target log-mel shape.

    Used by `configs/data/synthetic.yaml` to bootstrap the Hydra pipeline
    without requiring NSynth / MoisesDB downloads. Real training swaps
    this out for `NSynthBass` + `PairedPitchShiftCollate`.
    """

    def __init__(
        self,
        n: int = 8,
        n_mels: int = 128,
        n_time: int = 401,
        seed: int = 0,
    ) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if n_mels <= 0 or n_time <= 0:
            raise ValueError(f"n_mels, n_time must be positive")
        gen = torch.Generator().manual_seed(seed)
        self._items = [
            {
                "x": torch.randn(1, n_mels, n_time, generator=gen),
                "x_g": torch.randn(1, n_mels, n_time, generator=gen),
                "g_cents": torch.tensor(100 * (i % 5 - 2), dtype=torch.long),
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        return self._items[idx]
