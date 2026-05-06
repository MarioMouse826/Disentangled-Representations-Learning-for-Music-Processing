from __future__ import annotations

import pytest
import torch

from src.models.content_head import ContentHead


def test_schema_and_shapes() -> None:
    head = ContentHead(in_channels=512, d_c=16)
    h = torch.randn(3, 512, 13)
    out = head(h)
    assert set(out.keys()) == {"mu_c", "logvar_c"}
    assert out["mu_c"].shape == (3, 16, 13)
    assert out["logvar_c"].shape == (3, 16, 13)


def test_preserves_time_axis() -> None:
    # Critical invariant: the content head is time-resolved (unlike the
    # style head which pools over time). Output T' must match input T'.
    head = ContentHead(in_channels=64, d_c=8)
    for t in (5, 13, 25, 101):
        h = torch.randn(2, 64, t)
        out = head(h)
        assert out["mu_c"].shape == (2, 8, t)
        assert out["logvar_c"].shape == (2, 8, t)


def test_deterministic_given_input() -> None:
    head = ContentHead(in_channels=64, d_c=16)
    h = torch.randn(2, 64, 10)
    out1 = head(h)
    out2 = head(h)
    for k in out1:
        assert torch.equal(out1[k], out2[k])


def test_differentiable_all_params() -> None:
    head = ContentHead(in_channels=64, d_c=16)
    h = torch.randn(2, 64, 10, requires_grad=True)
    out = head(h)
    (out["mu_c"].sum() + out["logvar_c"].sum()).backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    for n, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {n}"
        assert torch.isfinite(p.grad).all()


def test_logvar_clamped() -> None:
    head = ContentHead(in_channels=8, d_c=4, logvar_clamp=(-10.0, 10.0))
    h = torch.randn(1, 8, 3) * 1e6
    out = head(h)
    assert torch.all(out["logvar_c"] >= -10.0 - 1e-4)
    assert torch.all(out["logvar_c"] <= 10.0 + 1e-4)


def test_translational_equivariance_on_time_shift() -> None:
    # 1×1 convs over (B, C, T) are trivially time-equivariant: shifting the
    # input in time must shift the output by the same amount. This is the
    # *architectural* prerequisite for pitch-shift equivariance; the rotation
    # rho(g) (Task 3.3) + L_equi loss (Task 3.5) build on this.
    head = ContentHead(in_channels=16, d_c=8)
    h = torch.randn(1, 16, 20)
    # Shift input by 3 time steps via roll.
    h_shift = torch.roll(h, shifts=3, dims=-1)
    mu1 = head(h)["mu_c"]
    mu2 = head(h_shift)["mu_c"]
    expected = torch.roll(mu1, shifts=3, dims=-1)
    assert torch.allclose(mu2, expected, atol=1e-5)


def test_even_d_c_for_rotation_compatibility() -> None:
    # Task 3.3 requires d_c % 2 == 0 for the block-diagonal SO(2) rotation
    # representation. The ContentHead itself tolerates odd d_c (nothing in
    # 1×1-conv math requires even), but we add a soft guard via config flag.
    with pytest.raises(ValueError, match="d_c must be even"):
        ContentHead(in_channels=64, d_c=5, require_even_d_c=True)


def test_config_roundtrip() -> None:
    head = ContentHead(in_channels=128, d_c=24, logvar_clamp=(-8.0, 8.0))
    cfg = head.config
    assert cfg["in_channels"] == 128
    assert cfg["d_c"] == 24
    assert cfg["logvar_clamp"] == (-8.0, 8.0)


def test_device_preservation() -> None:
    from src.utils.device import best_device

    dev = best_device()
    head = ContentHead(in_channels=64, d_c=16).to(dev)
    h = torch.randn(2, 64, 10, device=dev)
    out = head(h)
    for v in out.values():
        assert v.device.type == dev.type


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="in_channels"):
        ContentHead(in_channels=0, d_c=16)
    with pytest.raises(ValueError, match="d_c"):
        ContentHead(in_channels=64, d_c=0)
    with pytest.raises(ValueError, match="logvar_clamp"):
        ContentHead(in_channels=64, d_c=16, logvar_clamp=(5.0, -5.0))


def test_rejects_wrong_input_dims() -> None:
    head = ContentHead(in_channels=64, d_c=16)
    with pytest.raises(ValueError, match="3-D"):
        head(torch.randn(2, 64))
    with pytest.raises(ValueError, match="in_channels"):
        head(torch.randn(2, 32, 10))
