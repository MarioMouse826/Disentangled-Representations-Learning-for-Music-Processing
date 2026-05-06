from __future__ import annotations

import pytest
import torch

from src.models.encoder import MelEncoder


def test_default_shape_contract() -> None:
    # Input is 4-D log-mel (B, 1, n_mels, T). Default stack: 5 ConvBlocks
    # with strides (2,2,2,2,2) on the time axis; freq dim is left untouched
    # and then mean-pooled in forward. For T=401 and strides (2,2,2,2,2):
    #   401 -> 201 -> 101 -> 51 -> 26 -> 13
    # So T' = 13. (The plan's test-snippet claimed T'=25; that's a plan-text
    # arithmetic slip — the stride-5 time-only stack cannot yield 25 from
    # 401. We follow the plan's implementation spec, which is authoritative.)
    enc = MelEncoder()
    x = torch.randn(2, 1, 128, 401)
    y = enc(x)
    assert y.shape == (2, 512, 13)
    assert y.dtype == torch.float32


def test_batched() -> None:
    enc = MelEncoder()
    x = torch.randn(7, 1, 128, 401)
    y = enc(x)
    assert y.shape == (7, 512, 13)


def test_stride_freq_variant() -> None:
    # stride_freq=True strides both spatial dims, halving freq per block.
    # 128 freq / 2^5 = 4, then mean collapses → output (B, 512, T').
    enc = MelEncoder(stride_freq=True)
    x = torch.randn(2, 1, 128, 401)
    y = enc(x)
    assert y.shape == (2, 512, 13)


def test_configurable_channels_and_strides() -> None:
    enc = MelEncoder(channels=(16, 32, 64, 128), strides=(2, 2, 2, 2))
    x = torch.randn(1, 1, 128, 401)
    y = enc(x)
    # 401 / 2^4 = 25.0625 → 26 (conv output with padding=1)
    assert y.shape == (1, 128, 26)


def test_config_roundtrip() -> None:
    enc = MelEncoder(channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2, 2))
    cfg = enc.config
    assert cfg["channels"] == (32, 64, 128, 256, 512)
    assert cfg["strides"] == (2, 2, 2, 2, 2)
    assert cfg["out_channels"] == 512
    assert cfg["stride_freq"] is False


def test_differentiable() -> None:
    enc = MelEncoder()
    x = torch.randn(1, 1, 128, 401, requires_grad=True)
    y = enc(x)
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_deterministic_in_eval_mode() -> None:
    enc = MelEncoder()
    enc.train(False)  # disable any future dropout / BN drift
    x = torch.randn(1, 1, 128, 401)
    y1 = enc(x)
    y2 = enc(x)
    assert torch.equal(y1, y2)


def test_device_preservation() -> None:
    from src.utils.device import best_device

    dev = best_device()
    enc = MelEncoder().to(dev)
    x = torch.randn(1, 1, 128, 401, device=dev)
    y = enc(x)
    assert y.device == x.device
    assert y.shape == (1, 512, 13)


def test_out_channels_attr_matches_output() -> None:
    enc = MelEncoder(channels=(8, 16, 32), strides=(2, 2, 2))
    x = torch.randn(1, 1, 128, 401)
    y = enc(x)
    assert y.shape[1] == enc.out_channels == 32


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="channels"):
        MelEncoder(channels=(), strides=())
    with pytest.raises(ValueError, match="same length"):
        MelEncoder(channels=(32, 64), strides=(2,))
    with pytest.raises(ValueError, match="positive"):
        MelEncoder(channels=(0, 64, 128, 256, 512))
    with pytest.raises(ValueError, match="positive"):
        MelEncoder(channels=(32, 64, 128, 256, 512), strides=(1, 2, 2, 2, 0))


def test_rejects_3d_input() -> None:
    enc = MelEncoder()
    with pytest.raises(ValueError, match="4-D"):
        enc(torch.randn(1, 128, 401))


def test_rejects_wrong_channel_input() -> None:
    enc = MelEncoder()
    with pytest.raises(ValueError, match="channel"):
        enc(torch.randn(1, 3, 128, 401))  # expected 1-channel log-mel


def test_param_count_reasonable() -> None:
    # Plan stack at (32, 64, 128, 256, 512) channels expected well under
    # 10 M params. Pin a generous ceiling to catch future accidental bloat.
    enc = MelEncoder()
    n = sum(p.numel() for p in enc.parameters())
    assert n < 10_000_000, f"encoder has {n} params — unexpected bloat"
    assert n > 1_000_000, f"encoder has {n} params — suspiciously small"
