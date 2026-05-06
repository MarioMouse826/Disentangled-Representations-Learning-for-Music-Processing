"""Zero-shot OOD bass-line dataset backed by pre-computed shards.

Each shard on disk is a `torch.save`-serialized dict with the keys written by
`src.data.zero_shot_prepare.prepare_zero_shot_shards`:

    {
        "waveform":           Tensor[1, segment_samples] float32,
        "track_name":         str,
        "segment_idx":        int,
        "sample_rate":        int,
        "crepe_pitch_curve":  Tensor[frames] float32,
        "crepe_confidence":   Tensor[frames] float32,
        "beat_grid":          Tensor[num_beats] float32,
    }

This split keeps expensive CREPE + beat extraction out of the training loop.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import torch
from torch.utils.data import Dataset


class ZeroShotLabel(TypedDict):
    track_name: str
    segment_idx: int
    crepe_pitch_curve: torch.Tensor
    crepe_confidence: torch.Tensor
    beat_grid: torch.Tensor


class ZeroShotBass(Dataset):
    """Loads pre-computed zero-shot OOD bass shards.

    Args:
        shard_dir: directory containing `*.pt` shards written by
            `prepare_zero_shot_shards`.
    """

    def __init__(self, shard_dir: str | Path) -> None:
        self.shard_dir: Path = Path(shard_dir)
        if not self.shard_dir.is_dir():
            raise FileNotFoundError(f"shard_dir not found: {self.shard_dir}")
        self._shards: list[Path] = sorted(self.shard_dir.glob("*.pt"))
        if not self._shards:
            raise FileNotFoundError(f"no shards (*.pt) found under {self.shard_dir}")

    def __len__(self) -> int:
        return len(self._shards)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ZeroShotLabel]:
        # weights_only=True restricts deserialization to tensors + primitives;
        # shards are produced only by our preparer but pin the safe path so
        # future torch releases cannot silently broaden acceptance.
        data = torch.load(self._shards[idx], weights_only=True)
        waveform: torch.Tensor = data["waveform"].to(torch.float32)
        label: ZeroShotLabel = {
            "track_name": str(data["track_name"]),
            "segment_idx": int(data["segment_idx"]),
            "crepe_pitch_curve": data["crepe_pitch_curve"].to(torch.float32),
            "crepe_confidence": data["crepe_confidence"].to(torch.float32),
            "beat_grid": data["beat_grid"].to(torch.float32),
        }
        return waveform, label
