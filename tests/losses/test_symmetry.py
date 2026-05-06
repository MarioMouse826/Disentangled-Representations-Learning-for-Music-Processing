from __future__ import annotations

import pytest
import torch

from src.losses.symmetry import (
    equivariance_loss,
    invariance_loss,
    swap_consistency_loss,
)


# -- L_inv ----------------------------------------------------------------


def test_inv_zero_when_equal() -> None:
    # Plan-critical: L_inv = 0 when mu_s(x) == mu_s(T_g x).
    mu = torch.randn(4, 32)
    loss = invariance_loss(mu, mu.clone())
    assert float(loss) == 0.0


def test_inv_positive_on_random() -> None:
    a = torch.randn(8, 32)
    b = torch.randn(8, 32)
    loss = invariance_loss(a, b)
    assert float(loss) > 0.0
    assert torch.isfinite(loss)


def test_inv_matches_squared_l2_mean() -> None:
    # Pin the exact formula: ||Δ||²_2 summed over d_s, averaged over batch.
    torch.manual_seed(0)
    a = torch.randn(3, 16)
    b = torch.randn(3, 16)
    manual = ((a - b) ** 2).sum(dim=-1).mean()
    assert torch.allclose(invariance_loss(a, b), manual, atol=1e-6)


def test_inv_differentiable() -> None:
    a = torch.randn(4, 16, requires_grad=True)
    b = torch.randn(4, 16, requires_grad=True)
    loss = invariance_loss(a, b)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def test_inv_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        invariance_loss(torch.randn(4, 16), torch.randn(4, 32))


# -- L_equi ---------------------------------------------------------------


def _synthetic_rotation(d_c: int, theta: float) -> torch.Tensor:
    """Block-diagonal 2×2 rotation by `theta` radians; pair dims (0,1), (2,3), ..."""
    assert d_c % 2 == 0
    R = torch.zeros(d_c, d_c)
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    for k in range(d_c // 2):
        R[2 * k, 2 * k] = c
        R[2 * k, 2 * k + 1] = -s
        R[2 * k + 1, 2 * k] = s
        R[2 * k + 1, 2 * k + 1] = c
    return R


def test_equi_zero_on_synthetic_rotation() -> None:
    # Plan-critical: L_equi = 0 when mu_c(T_g x) = rho(g) · mu_c(x).
    torch.manual_seed(0)
    B, d_c, T = 4, 8, 13
    R = _synthetic_rotation(d_c, theta=0.3)          # (d_c, d_c)
    mu_c_x = torch.randn(B, d_c, T)
    # mu_c(T_g x) := R · mu_c(x) along the channel axis.
    mu_c_gx = torch.einsum("ij,bjt->bit", R, mu_c_x)
    rho_g_mu_c_x = torch.einsum("ij,bjt->bit", R, mu_c_x)
    loss = equivariance_loss(mu_c_gx, rho_g_mu_c_x)
    assert float(loss) < 1e-10


def test_equi_positive_on_random() -> None:
    mu_c_gx = torch.randn(2, 8, 10)
    rho_g_mu_c_x = torch.randn(2, 8, 10)
    loss = equivariance_loss(mu_c_gx, rho_g_mu_c_x)
    assert float(loss) > 0.0
    assert torch.isfinite(loss)


def test_equi_matches_formula() -> None:
    # Sum over (d_c, T), mean over batch only — matches L_inv scaling
    # so λ_inv = λ_equi weights symmetrically. Updated Phase-3 review.
    a = torch.randn(2, 6, 5)
    b = torch.randn(2, 6, 5)
    manual = ((a - b) ** 2).sum(dim=(1, 2)).mean()
    assert torch.allclose(equivariance_loss(a, b), manual, atol=1e-6)


def test_equi_rejects_2d_input() -> None:
    # Code-reviewer Phase-3 catch: 2-D (B, d_c) pooled content would have
    # sum(dim=1) collapse the wrong axis silently. Surface as ValueError.
    with pytest.raises(ValueError, match="3-D"):
        equivariance_loss(torch.randn(2, 8), torch.randn(2, 8))


def test_equi_differentiable() -> None:
    a = torch.randn(2, 8, 10, requires_grad=True)
    b = torch.randn(2, 8, 10, requires_grad=True)
    loss = equivariance_loss(a, b)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def test_equi_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        equivariance_loss(torch.randn(2, 8, 10), torch.randn(2, 8, 7))


# -- L_swap ---------------------------------------------------------------


def test_swap_zero_when_equal() -> None:
    mel = torch.randn(2, 1, 128, 401)
    loss = swap_consistency_loss(mel, mel.clone())
    assert float(loss) == 0.0


def test_swap_positive_on_random() -> None:
    a = torch.randn(2, 1, 128, 401)
    b = torch.randn(2, 1, 128, 401)
    loss = swap_consistency_loss(a, b)
    assert float(loss) > 0.0
    assert torch.isfinite(loss)


def test_swap_is_l1() -> None:
    a = torch.randn(1, 1, 16, 20)
    b = torch.randn(1, 1, 16, 20)
    manual = (a - b).abs().mean()
    assert torch.allclose(swap_consistency_loss(a, b), manual, atol=1e-6)


def test_swap_differentiable() -> None:
    a = torch.randn(1, 1, 16, 20, requires_grad=True)
    b = torch.randn(1, 1, 16, 20, requires_grad=True)
    loss = swap_consistency_loss(a, b)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_swap_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        swap_consistency_loss(torch.randn(1, 1, 128, 401), torch.randn(1, 1, 128, 400))


# -- device preservation --------------------------------------------------


def test_losses_device_preservation() -> None:
    from src.utils.device import best_device

    dev = best_device()
    a = torch.randn(4, 16, device=dev)
    b = torch.randn(4, 16, device=dev)
    assert invariance_loss(a, b).device.type == dev.type

    c = torch.randn(2, 8, 10, device=dev)
    d = torch.randn(2, 8, 10, device=dev)
    assert equivariance_loss(c, d).device.type == dev.type

    m1 = torch.randn(1, 1, 16, 16, device=dev)
    m2 = torch.randn(1, 1, 16, 16, device=dev)
    assert swap_consistency_loss(m1, m2).device.type == dev.type
