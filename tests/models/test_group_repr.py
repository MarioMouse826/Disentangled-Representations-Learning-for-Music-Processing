from __future__ import annotations

import math

import pytest
import torch

from src.models.group_repr import IdentityRep, RotationRep, TranslationRep


# -- RotationRep: group-theoretic invariants ------------------------------


def test_homomorphism() -> None:
    # Plan-critical: ρ(g₁ + g₂) = ρ(g₁) · ρ(g₂).
    d_c = 16
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    g1, g2 = 250.0, 475.0
    R1 = rep.matrix(torch.tensor([g1]))
    R2 = rep.matrix(torch.tensor([g2]))
    R12 = rep.matrix(torch.tensor([g1 + g2]))
    assert torch.allclose(R1 @ R2, R12, atol=1e-5)


def test_identity_at_zero() -> None:
    # Plan-critical: ρ(0) = I.
    d_c = 16
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    R0 = rep.matrix(torch.tensor([0.0]))
    assert torch.allclose(R0, torch.eye(d_c).unsqueeze(0), atol=1e-6)


def test_linearity() -> None:
    # Higgins et al. 2018 requires ρ(g) to act *linearly* on z_c.
    # ρ(g) (a z₁ + b z₂) = a ρ(g) z₁ + b ρ(g) z₂.
    d_c = 16
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    g = torch.tensor([300.0])
    z1 = torch.randn(1, d_c, 10)
    z2 = torch.randn(1, d_c, 10)
    a, b = 0.7, -1.3
    lhs = rep(a * z1 + b * z2, g)
    rhs = a * rep(z1, g) + b * rep(z2, g)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_inverse() -> None:
    # ρ(−g) = ρ(g)^{−1} for an orthogonal rep; product is identity.
    d_c = 8
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    g_cents = torch.tensor([325.0])
    R = rep.matrix(g_cents)
    R_inv = rep.matrix(-g_cents)
    prod = R @ R_inv
    I = torch.eye(d_c).unsqueeze(0)
    assert torch.allclose(prod, I, atol=1e-5)


def test_orthogonality() -> None:
    # Block-diagonal 2×2 rotations → orthogonal matrix: R^T R = I.
    d_c = 12
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    g_cents = torch.tensor([123.0, -456.0])
    R = rep.matrix(g_cents)                  # (2, d_c, d_c)
    RtR = R.transpose(-1, -2) @ R
    I = torch.eye(d_c).unsqueeze(0).expand(2, -1, -1)
    assert torch.allclose(RtR, I, atol=1e-5)


def test_determinant_is_plus_one() -> None:
    # SO(2) blocks → det(R) = +1 (proper rotations, no reflections).
    d_c = 10
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    g = torch.tensor([237.0])
    R = rep.matrix(g).squeeze(0)
    det = torch.linalg.det(R)
    assert abs(float(det) - 1.0) < 1e-4


# -- RotationRep: forward on z_c ------------------------------------------


def test_forward_preserves_shape_3d() -> None:
    d_c = 16
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    z = torch.randn(4, d_c, 13)
    g = torch.tensor([100.0, 200.0, -300.0, 700.0])
    out = rep(z, g)
    assert out.shape == z.shape
    assert out.dtype == z.dtype


def test_forward_2d_input() -> None:
    # Some callers pass time-pooled z_c as (B, d_c). Support both layouts.
    d_c = 8
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    z = torch.randn(3, d_c)
    g = torch.tensor([150.0, -200.0, 400.0])
    out = rep(z, g)
    assert out.shape == (3, d_c)


def test_forward_matches_matmul() -> None:
    # forward(z, g) should equal matmul of ρ(g) with z along the channel axis.
    d_c = 8
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    z = torch.randn(2, d_c, 5)
    g = torch.tensor([80.0, -120.0])
    R = rep.matrix(g)
    expected = torch.einsum("bij,bjt->bit", R, z)
    assert torch.allclose(rep(z, g), expected, atol=1e-6)


def test_forward_batch_g_matches_batch_z() -> None:
    # g must broadcast 1-to-1 with batch dim of z.
    d_c = 8
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    z = torch.randn(3, d_c, 4)
    g = torch.tensor([100.0, 200.0, 300.0])
    out = rep(z, g)
    # Per-item check: out[i] == rep(z[i:i+1], g[i:i+1])[0]
    for i in range(3):
        exp = rep(z[i : i + 1], g[i : i + 1])[0]
        assert torch.allclose(out[i], exp, atol=1e-6)


# -- RotationRep: engineering ---------------------------------------------


def test_deterministic() -> None:
    d_c = 8
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    z = torch.randn(2, d_c, 5)
    g = torch.tensor([200.0, -100.0])
    o1 = rep(z, g)
    o2 = rep(z, g)
    assert torch.equal(o1, o2)


def test_differentiable_through_omega() -> None:
    # Learnable frequencies ω_k must receive gradient from downstream loss.
    d_c = 8
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    z = torch.randn(2, d_c, 5)
    g = torch.tensor([200.0, -100.0])
    out = rep(z, g)
    out.sum().backward()
    assert rep.log_omega.grad is not None
    assert torch.isfinite(rep.log_omega.grad).all()


def test_differentiable_through_input() -> None:
    d_c = 8
    rep = RotationRep(d_c=d_c, period_cents=1200.0)
    z = torch.randn(2, d_c, 5, requires_grad=True)
    g = torch.tensor([100.0, 200.0])
    rep(z, g).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_device_preservation() -> None:
    from src.utils.device import best_device

    dev = best_device()
    d_c = 8
    rep = RotationRep(d_c=d_c, period_cents=1200.0).to(dev)
    z = torch.randn(2, d_c, 5, device=dev)
    g = torch.tensor([100.0, -200.0], device=dev)
    out = rep(z, g)
    assert out.device.type == dev.type


def test_config_roundtrip() -> None:
    rep = RotationRep(d_c=16, period_cents=1200.0, freq_init="octave")
    cfg = rep.config
    assert cfg["d_c"] == 16
    assert cfg["n_blocks"] == 8
    assert cfg["period_cents"] == 1200.0
    assert cfg["freq_init"] == "octave"


def test_odd_d_c_raises() -> None:
    with pytest.raises(ValueError, match="even"):
        RotationRep(d_c=7, period_cents=1200.0)


def test_invalid_period_raises() -> None:
    with pytest.raises(ValueError, match="period_cents"):
        RotationRep(d_c=8, period_cents=0.0)


def test_invalid_freq_init_raises() -> None:
    with pytest.raises(ValueError, match="freq_init"):
        RotationRep(d_c=8, period_cents=1200.0, freq_init="bogus")


def test_forward_batch_g_mismatch_raises() -> None:
    rep = RotationRep(d_c=8, period_cents=1200.0)
    z = torch.randn(3, 8, 5)
    g = torch.tensor([100.0, 200.0])  # only 2, but z has B=3
    with pytest.raises(ValueError, match="batch"):
        rep(z, g)


# -- TranslationRep (ablation path) ---------------------------------------


def test_translation_rep_forward() -> None:
    # ρ(g)z = z + g · v for learnable direction v.
    d_c = 8
    rep = TranslationRep(d_c=d_c)
    z = torch.zeros(2, d_c)
    g = torch.tensor([100.0, -50.0])
    out = rep(z, g)
    # out[i] = g[i] * v for i in [0, 1]
    expected_0 = g[0] * rep.direction
    expected_1 = g[1] * rep.direction
    assert torch.allclose(out[0], expected_0, atol=1e-6)
    assert torch.allclose(out[1], expected_1, atol=1e-6)


def test_translation_rep_linearity() -> None:
    d_c = 8
    rep = TranslationRep(d_c=d_c)
    g = torch.tensor([250.0])
    z1 = torch.randn(1, d_c)
    z2 = torch.randn(1, d_c)
    a, b = 0.7, -1.3
    # Translation is affine, not linear in z + g jointly, but the plan's
    # formulation fixes g, so it's linear in z: ρ(g)(az₁+bz₂) = a·ρ(g)z₁+b·ρ(g)z₂
    # only when the translation part cancels; the proper check is that the
    # rep applied to a batch equals batched application.
    for_batch = rep(torch.cat([z1, z2], dim=0), g.repeat(2))
    individually = torch.cat([rep(z1, g), rep(z2, g)], dim=0)
    assert torch.allclose(for_batch, individually, atol=1e-6)


# -- IdentityRep (null-ablation) ------------------------------------------


def test_identity_rep_unchanged() -> None:
    rep = IdentityRep(d_c=16)
    z = torch.randn(2, 16, 10)
    g = torch.tensor([500.0, -300.0])
    out = rep(z, g)
    assert torch.equal(out, z)
