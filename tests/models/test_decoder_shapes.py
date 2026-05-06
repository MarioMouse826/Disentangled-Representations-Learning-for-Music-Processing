from __future__ import annotations

import pytest
import torch

from src.models.decoder import SCDecoder


def test_basic_shape() -> None:
    dec = SCDecoder(d_s=32, d_c=16, n_mels=128, n_time=401, t_enc=13)
    z_s = torch.randn(4, 32)
    z_c = torch.randn(4, 16, 13)
    out = dec(z_s, z_c)
    assert out.shape == (4, 1, 128, 401)
    assert out.dtype == torch.float32


def test_different_batch_sizes() -> None:
    dec = SCDecoder(d_s=16, d_c=8, n_mels=128, n_time=401, t_enc=13)
    for B in (1, 3, 9):
        z_s = torch.randn(B, 16)
        z_c = torch.randn(B, 8, 13)
        out = dec(z_s, z_c)
        assert out.shape == (B, 1, 128, 401)


def test_differentiable() -> None:
    dec = SCDecoder(d_s=32, d_c=16, n_mels=128, n_time=401, t_enc=13)
    z_s = torch.randn(2, 32, requires_grad=True)
    z_c = torch.randn(2, 16, 13, requires_grad=True)
    out = dec(z_s, z_c)
    out.mean().backward()
    assert z_s.grad is not None and torch.isfinite(z_s.grad).all()
    assert z_c.grad is not None and torch.isfinite(z_c.grad).all()
    for n, p in dec.named_parameters():
        assert p.grad is not None, f"no grad on {n}"
        assert torch.isfinite(p.grad).all()


def test_deterministic_given_input() -> None:
    dec = SCDecoder(d_s=16, d_c=8, n_mels=128, n_time=401, t_enc=13)
    z_s = torch.randn(2, 16)
    z_c = torch.randn(2, 8, 13)
    o1 = dec(z_s, z_c)
    o2 = dec(z_s, z_c)
    assert torch.equal(o1, o2)


def test_z_s_actually_influences_output() -> None:
    # Sanity: changing z_s must change the reconstruction. If z_s were
    # silently dropped somewhere in the fusion path, this would pass with
    # identical outputs.
    dec = SCDecoder(d_s=32, d_c=16, n_mels=128, n_time=401, t_enc=13)
    z_c = torch.randn(1, 16, 13)
    out_a = dec(torch.randn(1, 32), z_c)
    out_b = dec(torch.randn(1, 32), z_c)
    # Very small chance two random draws produce the same output; 64000+
    # element MSE is a strong discriminator.
    assert not torch.allclose(out_a, out_b, atol=1e-3)


def test_z_c_actually_influences_output() -> None:
    # Sanity: changing z_c must change the reconstruction.
    dec = SCDecoder(d_s=32, d_c=16, n_mels=128, n_time=401, t_enc=13)
    z_s = torch.randn(1, 32)
    out_a = dec(z_s, torch.randn(1, 16, 13))
    out_b = dec(z_s, torch.randn(1, 16, 13))
    assert not torch.allclose(out_a, out_b, atol=1e-3)


def test_z_c_time_structure_propagates() -> None:
    # Shifting z_c in time must shift the reconstructed mel in time (the
    # decoder preserves temporal structure, the precondition for content
    # equivariance). Exact pixel-shift is too strong due to bilinear
    # interpolation in F.interpolate; instead we verify that z_c time-
    # perturbations produce correlated reconstruction changes.
    dec = SCDecoder(d_s=16, d_c=8, n_mels=128, n_time=401, t_enc=13)
    z_s = torch.randn(1, 16)
    z_c = torch.randn(1, 8, 13)
    out_base = dec(z_s, z_c)
    # Perturb only the first half of time steps in z_c.
    z_c_mod = z_c.clone()
    z_c_mod[..., :6] += 1.0
    out_mod = dec(z_s, z_c_mod)
    diff = (out_base - out_mod).abs().sum(dim=(0, 1, 2))  # (T,)
    # Largest diffs should concentrate in the early time half.
    left_mass = float(diff[: 401 // 2].sum())
    right_mass = float(diff[401 // 2 :].sum())
    assert left_mass > right_mass, (
        f"z_c time-perturbation did not propagate to decoder output "
        f"(left {left_mass:.2f} vs right {right_mass:.2f})"
    )


def test_config_roundtrip() -> None:
    dec = SCDecoder(
        d_s=48, d_c=24, n_mels=128, n_time=401, t_enc=13,
        channels=(32, 64, 128, 256, 512), base_freq=4,
    )
    cfg = dec.config
    assert cfg["d_s"] == 48
    assert cfg["d_c"] == 24
    assert cfg["n_mels"] == 128
    assert cfg["n_time"] == 401
    assert cfg["t_enc"] == 13
    assert cfg["base_freq"] == 4
    assert cfg["channels"] == (32, 64, 128, 256, 512)


def test_device_preservation() -> None:
    from src.utils.device import best_device

    dev = best_device()
    dec = SCDecoder(d_s=16, d_c=8, n_mels=128, n_time=401, t_enc=13).to(dev)
    z_s = torch.randn(2, 16, device=dev)
    z_c = torch.randn(2, 8, 13, device=dev)
    out = dec(z_s, z_c)
    assert out.device.type == dev.type


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="d_s"):
        SCDecoder(d_s=0, d_c=8, n_mels=128, n_time=401, t_enc=13)
    with pytest.raises(ValueError, match="d_c"):
        SCDecoder(d_s=16, d_c=-1, n_mels=128, n_time=401, t_enc=13)
    with pytest.raises(ValueError, match="n_mels"):
        SCDecoder(d_s=16, d_c=8, n_mels=0, n_time=401, t_enc=13)
    with pytest.raises(ValueError, match="t_enc"):
        SCDecoder(d_s=16, d_c=8, n_mels=128, n_time=401, t_enc=0)
    with pytest.raises(ValueError, match="base_freq"):
        SCDecoder(d_s=16, d_c=8, n_mels=128, n_time=401, t_enc=13, base_freq=0)


def test_rejects_wrong_z_shapes() -> None:
    dec = SCDecoder(d_s=16, d_c=8, n_mels=128, n_time=401, t_enc=13)
    # Wrong z_s dim.
    with pytest.raises(ValueError, match="d_s"):
        dec(torch.randn(2, 32), torch.randn(2, 8, 13))
    # Wrong z_c dim.
    with pytest.raises(ValueError, match="d_c"):
        dec(torch.randn(2, 16), torch.randn(2, 16, 13))
    # Wrong t_enc.
    with pytest.raises(ValueError, match="t_enc"):
        dec(torch.randn(2, 16), torch.randn(2, 8, 20))
    # Wrong z_s rank.
    with pytest.raises(ValueError, match="2-D"):
        dec(torch.randn(2, 16, 3), torch.randn(2, 8, 13))
    # Wrong z_c rank.
    with pytest.raises(ValueError, match="3-D"):
        dec(torch.randn(2, 16), torch.randn(2, 8))
    # Batch size mismatch.
    with pytest.raises(ValueError, match="batch"):
        dec(torch.randn(2, 16), torch.randn(3, 8, 13))
