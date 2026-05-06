from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.nsynth import NSynthBass


def test_nsynth_bass_item_schema(nsynth_fixture: Path) -> None:
    ds = NSynthBass(root=nsynth_fixture, split="train", sample_rate=16000, duration=4.0)

    # Fixture has 3 train entries; 2 are bass family — loader must filter to those.
    assert len(ds) == 2

    wav, label = ds[0]
    assert isinstance(wav, torch.Tensor)
    assert wav.dtype == torch.float32
    assert wav.shape == (1, 16000 * 4)

    assert label["pitch"].dtype == torch.long
    assert 21 <= int(label["pitch"]) <= 108
    assert "velocity" in label
    assert "instrument_source" in label
    assert "instrument_id" in label
    assert int(label["instrument_source"]) in {0, 1, 2}


def test_nsynth_deterministic_ordering(nsynth_fixture: Path) -> None:
    # Re-instantiating must yield the same first item — required for reproducible splits.
    ds1 = NSynthBass(root=nsynth_fixture, split="train")
    ds2 = NSynthBass(root=nsynth_fixture, split="train")
    wav1, lbl1 = ds1[0]
    wav2, lbl2 = ds2[0]
    assert torch.equal(wav1, wav2)
    assert int(lbl1["pitch"]) == int(lbl2["pitch"])


def test_nsynth_split_switching(nsynth_fixture: Path) -> None:
    train = NSynthBass(root=nsynth_fixture, split="train")
    valid = NSynthBass(root=nsynth_fixture, split="valid")
    assert len(train) == 2
    assert len(valid) == 1
    _, vlabel = valid[0]
    assert int(vlabel["pitch"]) == 30


def test_nsynth_variable_duration(nsynth_fixture: Path) -> None:
    # Truncation path: request 1s from a 4s clip.
    ds = NSynthBass(root=nsynth_fixture, split="train", sample_rate=16000, duration=1.0)
    wav, _ = ds[0]
    assert wav.shape == (1, 16000)

    # Padding path: request 5s from a 4s clip.
    ds_pad = NSynthBass(root=nsynth_fixture, split="train", sample_rate=16000, duration=5.0)
    wav_pad, _ = ds_pad[0]
    assert wav_pad.shape == (1, 16000 * 5)
    # Last second must be exactly zero-padded.
    assert torch.all(wav_pad[:, 16000 * 4 :] == 0.0)


def test_nsynth_transform_hook(nsynth_fixture: Path) -> None:
    # Transform must be applied post-load / post-length-fix.
    ds = NSynthBass(
        root=nsynth_fixture,
        split="train",
        transform=lambda w: w * 0.0,
    )
    wav, _ = ds[0]
    assert torch.all(wav == 0.0)


def test_nsynth_invalid_split_raises(nsynth_fixture: Path) -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        NSynthBass(root=nsynth_fixture, split="nope")


def test_nsynth_missing_manifest_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest"):
        NSynthBass(root=tmp_path, split="train")
