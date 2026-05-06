"""Tests for HDF5-backed NSynth-bass mel cache reader.

Write a tiny synthetic HDF5 in the expected schema; verify the dataset
returns the contract items downstream code depends on (`{"x", "labels"}`).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.nsynth_cached import NSynthBassCached


def _write_fake_cache(path: Path, n: int = 4, n_mels: int = 16, t: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = 16000
        f.attrs["duration"] = 4.0
        f.attrs["n_mels"] = n_mels
        f.attrs["n_fft"] = 400
        f.attrs["hop_length"] = 160
        f.attrs["split"] = "test"
        f.create_dataset("mel",
                         data=np.random.default_rng(0).standard_normal((n, n_mels, t)).astype(np.float16),
                         chunks=(1, n_mels, t))
        f.create_dataset("pitch", data=np.arange(n, dtype=np.int16))
        f.create_dataset("velocity", data=np.full(n, 100, dtype=np.int16))
        f.create_dataset("instrument_source", data=np.zeros(n, dtype=np.int16))
        f.create_dataset("instrument_id", data=np.arange(n, dtype=np.int16) * 2)


def test_cached_reader_len_and_item_shapes(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"
    _write_fake_cache(path, n=8, n_mels=16, t=32)

    ds = NSynthBassCached(path)
    assert len(ds) == 8
    item = ds[3]
    assert set(item.keys()) == {"x", "labels"}
    assert item["x"].shape == (1, 16, 32)
    assert item["x"].dtype == torch.float32
    labels = item["labels"]
    assert labels["pitch"].item() == 3
    assert labels["velocity"].item() == 100
    assert labels["instrument_id"].item() == 6
    assert labels["pitch"].dtype == torch.long


def test_cached_reader_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_nsynth_cache"):
        NSynthBassCached(tmp_path / "does_not_exist.h5")


def test_cached_reader_drop_velocity_option(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"
    _write_fake_cache(path, n=2)
    ds = NSynthBassCached(path, drop_velocity=True)
    item = ds[0]
    assert "velocity" not in item["labels"]
    assert "pitch" in item["labels"]


def test_cached_reader_as_float16_preserves_dtype(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"
    _write_fake_cache(path, n=2, n_mels=8, t=16)
    ds = NSynthBassCached(path, as_float32=False)
    x = ds[0]["x"]
    assert x.dtype == torch.float16


def test_getstate_strips_h5_handle_for_fork_safety(tmp_path: Path) -> None:
    # HDF5 handles are not fork-safe — DataLoader workers receive a pickled
    # copy of the dataset, so __getstate__ must return a handle-less dict.
    path = tmp_path / "cache.h5"
    _write_fake_cache(path, n=4)
    ds = NSynthBassCached(path)
    _ = ds[0]  # open the handle
    assert ds._h5 is not None
    state = ds.__getstate__()
    assert state["_h5"] is None
    # Round-trip via __setstate__ / __dict__.update simulates pickle.
    ds2 = NSynthBassCached.__new__(NSynthBassCached)
    ds2.__dict__.update(state)
    assert ds2._h5 is None
    # Accessing after restore should reopen on demand.
    item = ds2[1]
    assert item["x"].shape == (1, 16, 32)


def test_cached_reader_round_trips_mel_values(tmp_path: Path) -> None:
    path = tmp_path / "cache.h5"
    n, n_mels, t = 3, 8, 10
    with h5py.File(path, "w") as f:
        f.attrs["sample_rate"] = 16000
        f.attrs["duration"] = 4.0
        f.attrs["n_mels"] = n_mels
        f.attrs["n_fft"] = 400
        f.attrs["hop_length"] = 160
        known = np.arange(n * n_mels * t, dtype=np.float16).reshape(n, n_mels, t)
        f.create_dataset("mel", data=known)
        f.create_dataset("pitch", data=np.zeros(n, dtype=np.int16))
        f.create_dataset("velocity", data=np.zeros(n, dtype=np.int16))
        f.create_dataset("instrument_source", data=np.zeros(n, dtype=np.int16))
        f.create_dataset("instrument_id", data=np.zeros(n, dtype=np.int16))
    ds = NSynthBassCached(path)
    x = ds[2]["x"].squeeze(0).numpy()
    np.testing.assert_array_equal(x.astype(np.float16), known[2])
