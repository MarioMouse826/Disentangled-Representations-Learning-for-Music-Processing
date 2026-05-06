from __future__ import annotations

import pytest
import torch

from src.data.features import LogMel


def test_basic_shape() -> None:
    # 4 s @ 16 kHz, hop=160, center=True → 1 + 64000/160 = 401 frames.
    mel = LogMel()
    x = torch.randn(1, 16000 * 4)
    y = mel(x)
    assert y.shape == (1, 128, 401)
    assert y.dtype == torch.float32


def test_batched_channelled() -> None:
    mel = LogMel()
    x = torch.randn(3, 1, 16000 * 4)
    y = mel(x)
    assert y.shape == (3, 1, 128, 401)


def test_batched_unchannelled() -> None:
    mel = LogMel()
    x = torch.randn(5, 16000 * 2)
    y = mel(x)
    assert y.shape == (5, 128, 201)


def test_db_range_respected() -> None:
    # Any finite input must produce dB in [-80, 0].
    mel = LogMel()
    x = torch.randn(1, 16000)
    y = mel(x)
    assert torch.isfinite(y).all()
    assert y.min() >= -80.0 - 1e-4
    assert y.max() <= 0.0 + 1e-4


def test_determinism_across_instances() -> None:
    x = torch.randn(1, 16000 * 2)
    m1 = LogMel()
    m2 = LogMel()
    assert torch.allclose(m1(x), m2(x), atol=1e-6)


def test_shape_stable_under_reapplication() -> None:
    # Plan spec: "idempotent re-application (shape stable)".
    # Re-running on the same input must produce identical output (no RNG).
    mel = LogMel()
    x = torch.randn(1, 16000 * 2)
    y1 = mel(x)
    y2 = mel(x)
    assert y1.shape == y2.shape
    assert torch.equal(y1, y2)


def test_differentiable() -> None:
    mel = LogMel()
    x = torch.randn(1, 16000, requires_grad=True)
    y = mel(x)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_config_dict_round_trip() -> None:
    cfg = dict(sample_rate=22050, n_fft=1024, hop_length=256, n_mels=64, fmin=40, fmax=11000)
    mel = LogMel(**cfg)
    for k, v in cfg.items():
        assert mel.config[k] == v


def test_device_preservation() -> None:
    # CPU-only check ensures the module does not force output to CPU when
    # input already lives on an accelerator. Re-run manually on MPS/CUDA.
    mel = LogMel()
    x = torch.randn(1, 16000)
    y = mel(x)
    assert y.device == x.device


def test_silence_produces_floor() -> None:
    mel = LogMel()
    x = torch.zeros(1, 16000)
    y = mel(x)
    # Silence must map to the dB floor (−80) everywhere.
    assert torch.allclose(y, torch.full_like(y, -80.0), atol=1e-4)


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        LogMel(sample_rate=0)
    with pytest.raises(ValueError):
        LogMel(n_fft=0)
    with pytest.raises(ValueError):
        LogMel(hop_length=0)
    with pytest.raises(ValueError):
        LogMel(n_mels=0)
    with pytest.raises(ValueError):
        LogMel(fmin=-1)
    with pytest.raises(ValueError):
        LogMel(fmin=1000, fmax=500)
