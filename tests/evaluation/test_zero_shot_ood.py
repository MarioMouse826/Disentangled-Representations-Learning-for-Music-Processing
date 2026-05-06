"""Tests for zero-shot OOD assessment (plan Task 6.5).

Two measurements per clip:
    (a) identity stability = per-dim variance of z_s across sliding windows.
    (b) equivariance ratio ER at shifts g ∈ {+/-100, ..., +/-1200} cents.
"""
from __future__ import annotations

import math

import torch

from src.evaluation.zero_shot_ood import (
    compute_identity_stability,
    compute_zero_shot_ood,
    identity_variance_scalar,
)


# -- identity stability ---------------------------------------------------


class _ConstantStyleModel(torch.nn.Module):
    """Oracle: z_s is constant across any window → variance = 0."""

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        return {
            "mu_s": torch.ones(b, 4),
            "mu_c": torch.zeros(b, 4),
        }


class _VaryingStyleModel(torch.nn.Module):
    """z_s depends on input mean → variance > 0 across windows."""

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        means = x.reshape(b, -1).mean(dim=-1, keepdim=True)
        mu_s = torch.cat([means, means * 2.0, means * 3.0, means * 4.0], dim=1)
        return {"mu_s": mu_s, "mu_c": torch.zeros(b, 4)}


def test_identity_stability_oracle_gives_zero_variance() -> None:
    torch.manual_seed(0)
    windows = torch.randn(6, 1, 8, 16)
    out = compute_identity_stability(
        model=_ConstantStyleModel(), windows=windows
    )
    assert out["per_dim_variance"].shape == (4,)
    assert torch.allclose(out["per_dim_variance"], torch.zeros(4), atol=1e-6)
    assert out["scalar_variance"] == 0.0


def test_identity_stability_varying_model_gives_positive_variance() -> None:
    torch.manual_seed(1)
    windows = torch.randn(8, 1, 8, 16) + torch.arange(8).float().reshape(8, 1, 1, 1)
    out = compute_identity_stability(
        model=_VaryingStyleModel(), windows=windows
    )
    assert (out["per_dim_variance"] > 0).all()
    assert out["scalar_variance"] > 0


def test_identity_variance_scalar_is_mean_of_per_dim() -> None:
    per_dim = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert math.isclose(identity_variance_scalar(per_dim), 0.25, rel_tol=1e-9)


# -- end-to-end zero-shot OOD --------------------------------------------


class _RotRep:
    def __init__(self, d_c: int = 4, period: float = 1200.0) -> None:
        self.d_c = d_c
        self.period = period

    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor:
        b = g_cents.shape[0]
        theta = (2.0 * math.pi / self.period) * g_cents
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.eye(self.d_c, dtype=g_cents.dtype).unsqueeze(0).expand(b, -1, -1).clone()
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        return R


class _IdentityAug:
    """Rotate first two channels of the spectrogram by `g_cents`."""

    def __init__(self, period: float = 1200.0) -> None:
        self.period = period

    def __call__(self, x: torch.Tensor, g_cents: torch.Tensor) -> torch.Tensor:
        theta = (2.0 * math.pi / self.period) * g_cents
        c = torch.cos(theta)[:, None, None]  # (B, 1, 1)
        s = torch.sin(theta)[:, None, None]
        out = x.clone()
        # Rotate first two mel-rows.
        out[:, :, 0, :] = c * x[:, :, 0, :] - s * x[:, :, 1, :]
        out[:, :, 1, :] = s * x[:, :, 0, :] + c * x[:, :, 1, :]
        return out


class _PerfectEquivariantModel(torch.nn.Module):
    """mu_s constant, mu_c = first 4 mel-row means — when aug rotates first 2
    channels, mu_c transforms exactly as rho(g) predicts."""

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        mu_s = torch.ones(b, 2)
        mu_c = x[:, 0].mean(dim=-1)[:, :4]  # (B, 4)
        return {"mu_s": mu_s, "mu_c": mu_c}


def test_compute_zero_shot_ood_returns_per_clip_metrics() -> None:
    torch.manual_seed(2)
    clips = [torch.randn(5, 1, 8, 16) for _ in range(3)]
    g_list = torch.tensor([-600.0, 600.0])
    out = compute_zero_shot_ood(
        model=_PerfectEquivariantModel(),
        clip_windows=clips,
        g_cents_list=g_list,
        group_rep=_RotRep(),
        augment=_IdentityAug(),
    )
    assert out["per_clip"].shape == (3, 2)  # columns: [variance, ER]
    assert out["mean_variance"] >= 0.0
    assert out["mean_equivariance_ratio"] >= 0.0


def test_compute_zero_shot_ood_is_low_on_perfect_oracle() -> None:
    torch.manual_seed(3)
    clips = [torch.randn(4, 1, 8, 16) * 0.01 + 1.0 for _ in range(3)]  # near-constant
    g_list = torch.tensor([-600.0, 600.0])
    out = compute_zero_shot_ood(
        model=_PerfectEquivariantModel(),
        clip_windows=clips,
        g_cents_list=g_list,
        group_rep=_RotRep(),
        augment=_IdentityAug(),
    )
    # mu_s is a constant vector → identity variance is exactly 0.
    assert out["mean_variance"] == 0.0
