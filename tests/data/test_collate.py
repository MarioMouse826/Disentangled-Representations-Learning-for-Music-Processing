from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augment import PitchShiftGroup
from src.data.collate import PairedPitchShiftCollate, build_paired_loader


def _harmonic_bass(duration_s: float, freq: float, sr: int = 16000) -> torch.Tensor:
    t = np.arange(int(sr * duration_s)) / sr
    wave = np.zeros_like(t, dtype=np.float32)
    for k in range(1, 6):
        wave += (0.4 / k) * np.sin(2.0 * np.pi * k * freq * t).astype(np.float32)
    wave /= np.max(np.abs(wave)) + 1e-9
    return torch.from_numpy((0.6 * wave).astype(np.float32)).unsqueeze(0)


class _TinyBass(Dataset):
    """Deterministic in-memory fixture: 4 harmonic-bass clips + integer labels."""

    def __init__(self, n: int = 4, duration_s: float = 1.0) -> None:
        self._items = [
            (_harmonic_bass(duration_s, freq=110.0 + 20 * i), {"pitch": i + 40, "inst": i})
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        wav, meta = self._items[idx]
        return wav, {"pitch": torch.tensor(meta["pitch"]), "inst": torch.tensor(meta["inst"])}


def test_batch_schema_and_shapes() -> None:
    ds = _TinyBass(n=4, duration_s=0.5)
    aug = PitchShiftGroup(sample_rate=16000)
    collate = PairedPitchShiftCollate(aug=aug, seed=42)
    batch = collate([ds[i] for i in range(4)])
    assert set(batch.keys()) == {"x", "x_g", "g_cents", "labels"}
    assert batch["x"].shape == (4, 1, 8000)
    assert batch["x_g"].shape == (4, 1, 8000)
    assert batch["g_cents"].shape == (4,)
    assert batch["g_cents"].dtype == torch.long
    assert isinstance(batch["labels"], dict)
    assert batch["labels"]["pitch"].shape == (4,)
    assert batch["labels"]["pitch"].dtype == torch.long


def test_g_cents_range_and_quantum() -> None:
    aug = PitchShiftGroup(sample_rate=16000)
    collate = PairedPitchShiftCollate(
        aug=aug, seed=0, range_cents=(-500, 500), quantum=100
    )
    ds = _TinyBass(n=8, duration_s=0.5)
    batch = collate([ds[i] for i in range(8)])
    g = batch["g_cents"]
    assert torch.all(g >= -500)
    assert torch.all(g <= 500)
    assert torch.all(g % 100 == 0)


def test_per_item_vs_batch_uniform_g() -> None:
    aug = PitchShiftGroup(sample_rate=16000)
    ds = _TinyBass(n=4, duration_s=0.5)
    per_item = PairedPitchShiftCollate(aug=aug, seed=7, per_item_g=True)
    batch_uni = PairedPitchShiftCollate(aug=aug, seed=7, per_item_g=False)

    b1 = per_item([ds[i] for i in range(4)])
    b2 = batch_uni([ds[i] for i in range(4)])
    # Per-item mode: g can differ within a batch (probabilistic check).
    assert b1["g_cents"].unique().numel() >= 2 or b1["g_cents"].numel() == 1
    # Batch-uniform: all g equal.
    assert b2["g_cents"].unique().numel() == 1


def test_seeded_determinism() -> None:
    aug = PitchShiftGroup(sample_rate=16000)
    ds = _TinyBass(n=4, duration_s=0.5)
    c1 = PairedPitchShiftCollate(aug=aug, seed=123)
    c2 = PairedPitchShiftCollate(aug=aug, seed=123)
    items = [ds[i] for i in range(4)]
    b1 = c1(items)
    b2 = c2(items)
    assert torch.equal(b1["g_cents"], b2["g_cents"])
    assert torch.allclose(b1["x_g"], b2["x_g"])


def test_x_g_matches_direct_apply() -> None:
    # Sanity: x_g[i] must equal aug.apply(x[i], g_cents[i]).
    aug = PitchShiftGroup(sample_rate=16000)
    collate = PairedPitchShiftCollate(aug=aug, seed=0)
    ds = _TinyBass(n=3, duration_s=0.5)
    items = [ds[i] for i in range(3)]
    batch = collate(items)
    for i in range(3):
        expected = aug.apply(batch["x"][i], int(batch["g_cents"][i]))
        assert torch.allclose(batch["x_g"][i], expected, atol=1e-5)


def test_dataloader_num_workers_zero() -> None:
    # In-process DataLoader — the simplest integration path.
    aug = PitchShiftGroup(sample_rate=16000)
    ds = _TinyBass(n=4, duration_s=0.5)
    loader = build_paired_loader(
        ds, aug=aug, batch_size=2, num_workers=0, seed=1, shuffle=False
    )
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["x"].shape == (2, 1, 8000)


def test_dataloader_num_workers_gt_zero() -> None:
    # Multi-worker DataLoader — validates that the collate + aug are both
    # picklable and that worker-local RNGs yield sensible output.
    aug = PitchShiftGroup(sample_rate=16000)
    ds = _TinyBass(n=4, duration_s=0.5)
    loader = build_paired_loader(
        ds, aug=aug, batch_size=2, num_workers=2, seed=1, shuffle=False
    )
    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert b["x_g"].shape == b["x"].shape
        assert torch.isfinite(b["x_g"]).all()


def test_zero_range_produces_identity_pair() -> None:
    # Sampling g ∈ {0} via range_cents=(0,0) should produce x_g == x
    # (zero-shift short-circuits to bitwise-exact in PitchShiftGroup.apply).
    aug = PitchShiftGroup(sample_rate=16000)
    collate = PairedPitchShiftCollate(
        aug=aug, seed=0, range_cents=(0, 0), quantum=1
    )
    ds = _TinyBass(n=3, duration_s=0.5)
    batch = collate([ds[i] for i in range(3)])
    assert torch.all(batch["g_cents"] == 0)
    assert torch.equal(batch["x"], batch["x_g"])
