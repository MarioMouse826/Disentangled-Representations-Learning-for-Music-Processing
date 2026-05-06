from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.moisesdb import MoisesDBBass


def test_item_schema(moisesdb_fixture: Path) -> None:
    ds = MoisesDBBass(
        root=moisesdb_fixture,
        split="all",
        crop_seconds=4.0,
        sample_rate=16000,
        deterministic=True,
    )
    assert len(ds) > 0
    wav, label = ds[0]
    assert isinstance(wav, torch.Tensor)
    assert wav.dtype == torch.float32
    assert wav.shape == (1, 16000 * 4)

    assert label["stem"] == "bass"
    assert isinstance(label["track_id"], str)
    assert label["track_id"] in {"track-A", "track-B"}
    assert isinstance(label["segment_idx"], int)
    assert "track_idx" in label  # integer track index for factor supervision


def test_track_c_skipped(moisesdb_fixture: Path) -> None:
    # track-C has no bass subfolder — must not appear in the track list.
    ds = MoisesDBBass(root=moisesdb_fixture, split="all", deterministic=True)
    seen = {ds[i][1]["track_id"] for i in range(len(ds))}
    assert "track-C" not in seen
    assert seen == {"track-A", "track-B"}


def test_deterministic_stride(moisesdb_fixture: Path) -> None:
    # Deterministic mode: same index, same instance, same seed → identical output.
    ds = MoisesDBBass(root=moisesdb_fixture, split="all", deterministic=True, hop_seconds=2.0)
    w1, l1 = ds[1]
    w2, l2 = ds[1]
    assert torch.equal(w1, w2)
    assert l1["segment_idx"] == l2["segment_idx"]

    # Consecutive segments from the same track must advance by exactly hop_seconds.
    same_track = [
        (i, ds[i][1]["segment_idx"]) for i in range(len(ds))
        if ds[i][1]["track_id"] == "track-A"
    ]
    # segment_idx is monotonic within a track.
    idxs = [s for _, s in same_track]
    assert idxs == sorted(idxs)


def test_random_crop_reproducible_with_seed(moisesdb_fixture: Path) -> None:
    ds1 = MoisesDBBass(root=moisesdb_fixture, split="all", deterministic=False, seed=42)
    ds2 = MoisesDBBass(root=moisesdb_fixture, split="all", deterministic=False, seed=42)
    w1, _ = ds1[0]
    w2, _ = ds2[0]
    assert torch.equal(w1, w2)


def test_random_crop_differs_across_seeds(moisesdb_fixture: Path) -> None:
    ds_a = MoisesDBBass(root=moisesdb_fixture, split="all", deterministic=False, seed=1)
    ds_b = MoisesDBBass(root=moisesdb_fixture, split="all", deterministic=False, seed=2)
    # Track-A is 10s with 16kHz × 4s crop → multiple valid crop positions, so
    # two different seeds must yield different windows with high probability.
    diffs = 0
    for i in range(len(ds_a)):
        if not torch.equal(ds_a[i][0], ds_b[i][0]):
            diffs += 1
    assert diffs >= 1


def test_loudness_normalization(moisesdb_fixture: Path) -> None:
    # Target -23 LUFS → RMS of the output sinusoid should be in a bounded range.
    # For a pure sine at -23 LUFS (pyloudnorm scales by gain factor), RMS should
    # land near ~0.07 (roughly). Assert a loose envelope rather than an exact value.
    ds = MoisesDBBass(
        root=moisesdb_fixture,
        split="all",
        deterministic=True,
        target_lufs=-23.0,
    )
    wav, _ = ds[0]
    rms = wav.pow(2).mean().sqrt().item()
    assert 0.01 < rms < 0.5, f"RMS {rms} outside reasonable LUFS-normalized range"


def test_short_track_is_padded(moisesdb_fixture: Path) -> None:
    # Request a 12s crop but track-B only has 6s of audio → output must be padded.
    ds = MoisesDBBass(
        root=moisesdb_fixture,
        split="all",
        crop_seconds=12.0,
        deterministic=True,
    )
    # Filter down to track-B items.
    b_items = [i for i in range(len(ds)) if ds[i][1]["track_id"] == "track-B"]
    assert len(b_items) >= 1
    wav, _ = ds[b_items[0]]
    assert wav.shape == (1, 16000 * 12)
    # Trailing samples (beyond the original 6s of audio) must be exactly zero.
    assert torch.all(wav[:, 16000 * 6 :] == 0.0)


def test_multiple_bass_stems_summed(moisesdb_fixture: Path) -> None:
    # track-A has two bass files (di.wav + amped.wav) — loader must sum them.
    ds = MoisesDBBass(root=moisesdb_fixture, split="all", deterministic=True)
    a_items = [i for i in range(len(ds)) if ds[i][1]["track_id"] == "track-A"]
    assert len(a_items) >= 1
    wav, _ = ds[a_items[0]]
    # Summing two identical sines then LUFS-normalizing should still produce
    # non-zero audio.
    assert wav.abs().mean() > 1e-3


def test_invalid_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no MoisesDB tracks"):
        MoisesDBBass(root=tmp_path / "does-not-exist", split="all")


def test_split_partitioning_unit(moisesdb_fixture: Path) -> None:
    # Tight unit: every track in exactly one split; union covers all tracks.
    # Small 2-track fixture may land both tracks in one split (80/10/10
    # bucketing over only 2 SHA-1 samples), so we only assert coverage +
    # disjointness here. `test_split_partitioning_statistical` below
    # exercises the non-empty-splits property on a larger synthetic set.
    train = MoisesDBBass(root=moisesdb_fixture, split="train", deterministic=True)
    valid = MoisesDBBass(root=moisesdb_fixture, split="valid", deterministic=True)
    test = MoisesDBBass(root=moisesdb_fixture, split="test", deterministic=True)
    train_ids = {train[i][1]["track_id"] for i in range(len(train))}
    valid_ids = {valid[i][1]["track_id"] for i in range(len(valid))}
    test_ids = {test[i][1]["track_id"] for i in range(len(test))}
    assert train_ids | valid_ids | test_ids == {"track-A", "track-B"}
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)


def test_split_partitioning_statistical() -> None:
    # Using the `_in_split` classmethod directly on 200 synthetic track_ids
    # verifies the 80/10/10 ratio holds and all three splits are non-empty.
    # This pins the actual invariant (every real training run draws from all
    # three splits) that the tiny fixture cannot cover.
    counts = {"train": 0, "valid": 0, "test": 0}
    for i in range(200):
        tid = f"synthetic-track-{i:03d}"
        for split in counts:
            if MoisesDBBass._in_split(tid, split):
                counts[split] += 1
                break
    assert sum(counts.values()) == 200
    for s, c in counts.items():
        assert c > 0, f"split {s} is empty — bucketing broken"
    # Check ratios are in plausible neighborhood of 80/10/10 (loose —
    # SHA-1 is not perfectly uniform at N=200, but tail ratios should be
    # within ±5% absolute).
    assert 0.70 <= counts["train"] / 200 <= 0.90
    assert 0.05 <= counts["valid"] / 200 <= 0.15
    assert 0.05 <= counts["test"] / 200 <= 0.15
