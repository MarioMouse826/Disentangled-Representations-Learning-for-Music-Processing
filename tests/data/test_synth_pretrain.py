"""Tests for the synthetic pre-training dataset (plan Task 7.3)."""
from __future__ import annotations

import torch

from src.data.synth_pretrain import SynthBassPretrainDataset


def test_dataset_len_matches_arg() -> None:
    ds = SynthBassPretrainDataset(n=64, sample_rate=8000, crop_seconds=0.25, seed=0)
    assert len(ds) == 64


def test_item_schema_and_shapes() -> None:
    sr = 8000
    crop = 0.25
    ds = SynthBassPretrainDataset(n=4, sample_rate=sr, crop_seconds=crop, seed=0)
    item = ds[0]
    assert set(item.keys()) >= {"x", "labels"}
    assert item["x"].shape == (1, int(sr * crop))
    assert item["x"].dtype == torch.float32
    assert torch.isfinite(item["x"]).all()
    labels = item["labels"]
    assert "pitch_hz" in labels and "timbre_id" in labels
    assert isinstance(labels["pitch_hz"], float)
    assert isinstance(labels["timbre_id"], int)


def test_item_is_seed_deterministic() -> None:
    ds_a = SynthBassPretrainDataset(n=4, sample_rate=8000, crop_seconds=0.25, seed=42)
    ds_b = SynthBassPretrainDataset(n=4, sample_rate=8000, crop_seconds=0.25, seed=42)
    assert torch.equal(ds_a[2]["x"], ds_b[2]["x"])
    assert ds_a[2]["labels"] == ds_b[2]["labels"]


def test_different_seeds_produce_different_items() -> None:
    ds_a = SynthBassPretrainDataset(n=4, sample_rate=8000, crop_seconds=0.25, seed=0)
    ds_b = SynthBassPretrainDataset(n=4, sample_rate=8000, crop_seconds=0.25, seed=1)
    assert not torch.allclose(ds_a[0]["x"], ds_b[0]["x"])


def test_pitch_range_respected() -> None:
    ds = SynthBassPretrainDataset(
        n=64,
        sample_rate=16000,
        crop_seconds=0.5,
        seed=0,
        pitch_hz_range=(40.0, 100.0),
    )
    for i in range(len(ds)):
        pitch = ds[i]["labels"]["pitch_hz"]
        assert 40.0 <= pitch <= 100.0


def test_timbre_id_within_preset_count() -> None:
    n_presets = 8
    ds = SynthBassPretrainDataset(
        n=50, sample_rate=8000, crop_seconds=0.25, seed=0, n_timbres=n_presets
    )
    for i in range(len(ds)):
        tid = ds[i]["labels"]["timbre_id"]
        assert 0 <= tid < n_presets


def test_synth_pitch_matches_requested_via_fft_peak() -> None:
    """Synthesized audio should peak energy near the requested f0."""
    sr = 16000
    crop = 0.5
    ds = SynthBassPretrainDataset(
        n=1,
        sample_rate=sr,
        crop_seconds=crop,
        seed=0,
        pitch_hz_range=(100.0, 100.0),  # fixed f0
    )
    x = ds[0]["x"][0]  # (T,)
    # Single-sided FFT magnitude.
    spec = torch.fft.rfft(x).abs()
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sr)
    peak_idx = int(torch.argmax(spec))
    peak_hz = float(freqs[peak_idx])
    # Bass f0=100 Hz: peak should be within ~30 Hz due to harmonic content.
    assert abs(peak_hz - 100.0) < 30.0 or abs(peak_hz - 200.0) < 30.0


def test_same_pitch_different_timbre_differs() -> None:
    """Same pitch across two timbre ids produces distinct spectra."""
    common = dict(n=1, sample_rate=8000, crop_seconds=0.25, pitch_hz_range=(60.0, 60.0))
    ds_a = SynthBassPretrainDataset(seed=0, n_timbres=16, **common)
    # Force different timbre by seeding so the ID differs — if seeds happen
    # to match, seed 0 vs seed 1 will yield different preset picks.
    ds_b = SynthBassPretrainDataset(seed=1, n_timbres=16, **common)
    ta = ds_a[0]["labels"]["timbre_id"]
    tb = ds_b[0]["labels"]["timbre_id"]
    if ta == tb:
        ds_b = SynthBassPretrainDataset(seed=2, n_timbres=16, **common)
    assert not torch.allclose(ds_a[0]["x"], ds_b[0]["x"], atol=1e-4)
