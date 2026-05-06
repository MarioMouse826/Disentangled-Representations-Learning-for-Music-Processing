from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.zero_shot import ZeroShotBass


def test_item_schema(zero_shot_shard_dir: Path) -> None:
    ds = ZeroShotBass(shard_dir=zero_shot_shard_dir)
    assert len(ds) > 0
    wav, label = ds[0]

    assert isinstance(wav, torch.Tensor)
    assert wav.dtype == torch.float32
    assert wav.shape == (1, 16000 * 4)

    assert "track_name" in label
    assert isinstance(label["track_name"], str)

    pitch = label["crepe_pitch_curve"]
    assert isinstance(pitch, torch.Tensor)
    # 4 seconds @ 100 Hz frame rate (10 ms step) = 400 frames (+/- 1 frame slack).
    assert abs(pitch.shape[0] - 400) <= 1
    assert pitch.dtype == torch.float32

    beats = label["beat_grid"]
    assert isinstance(beats, torch.Tensor)
    # Beat grid is a 1-D tensor of beat timestamps (seconds, relative to the
    # segment start); may be empty for sparse fixtures.
    assert beats.ndim == 1
    assert beats.dtype == torch.float32


def test_segment_count(zero_shot_shard_dir: Path) -> None:
    ds = ZeroShotBass(shard_dir=zero_shot_shard_dir)
    # money.wav (10s) → 4 segments (start=0,2,4,6), next start 8 would end at 12 > 10 → drop.
    # under_pressure.wav (6s) → 2 segments (start=0,2).
    # stand_by_me.wav (4s)     → 1 segment (start=0).
    # Total: 4 + 2 + 1 = 7.
    assert len(ds) == 7


def test_track_names_correct(zero_shot_shard_dir: Path) -> None:
    ds = ZeroShotBass(shard_dir=zero_shot_shard_dir)
    names = {ds[i][1]["track_name"] for i in range(len(ds))}
    assert names == {"money", "under_pressure", "stand_by_me"}


def test_segment_order_is_deterministic(zero_shot_shard_dir: Path) -> None:
    ds1 = ZeroShotBass(shard_dir=zero_shot_shard_dir)
    ds2 = ZeroShotBass(shard_dir=zero_shot_shard_dir)
    for i in range(len(ds1)):
        w1, l1 = ds1[i]
        w2, l2 = ds2[i]
        assert torch.equal(w1, w2)
        assert l1["track_name"] == l2["track_name"]
        assert int(l1["segment_idx"]) == int(l2["segment_idx"])


def test_pitch_tracks_fundamental(zero_shot_shard_dir: Path) -> None:
    # money.wav is a 110 Hz (A2) sinusoid — CREPE should predict ~110 Hz within
    # 10 cents on voiced frames. We take the median over high-confidence frames.
    ds = ZeroShotBass(shard_dir=zero_shot_shard_dir)
    for i in range(len(ds)):
        _, label = ds[i]
        if label["track_name"] == "money":
            pitch = label["crepe_pitch_curve"]
            confidence = label["crepe_confidence"]
            voiced = pitch[confidence > 0.5]
            if voiced.numel() > 10:
                median_hz = float(voiced.median())
                assert 100.0 < median_hz < 120.0, f"CREPE median {median_hz} Hz away from 110 Hz"
                return
    pytest.fail("no confident 'money' segment found")


def test_shard_dir_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ZeroShotBass(shard_dir=tmp_path / "nope")


def test_empty_shard_dir_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no shards"):
        ZeroShotBass(shard_dir=empty)
