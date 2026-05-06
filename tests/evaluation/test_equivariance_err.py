"""Tests for equivariance / invariance diagnostics.

    IR = ||mu_s(x) - mu_s(T_g x)||^2 / ||mu_s(x)||^2        (lower -> more invariant)
    ER = ||mu_c(T_g x) - rho(g) mu_c(x)||^2 / ||mu_c(T_g x)||^2   (lower -> more equivariant)

Averaged over the assessment set and g in {+/-100, ..., +/-1200} cents.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.equivariance_err import compute, compute_from_arrays


# -- perfectly-equivariant oracle ----------------------------------------


class _PerfectModel(torch.nn.Module):
    """Oracle: mu_s is always zero vector (invariant), mu_c is identity
    transform of input (so rho(g) * mu_c(x) equals mu_c(T_g x) when T_g is a
    rotation acting on the first d_c channels)."""

    def __init__(self, d_s: int, d_c: int) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_c = d_c

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # mu_s: constant zeros (invariant by construction).
        b = x.shape[0]
        mu_s = torch.zeros(b, self.d_s, device=x.device, dtype=x.dtype)
        # mu_c: pass through first d_c channels of the (B, d_c) input.
        mu_c = x[:, : self.d_c]
        return {"mu_s": mu_s, "mu_c": mu_c}


class _IdentityAugmenter:
    """Action on `x`: apply rotation (2-D rot block) by angle g cents to the
    first 2 channels. Matches the rotation representation exactly, so the
    oracle model above is perfectly equivariant."""

    def __init__(self, period_cents: float = 1200.0) -> None:
        self.period = period_cents

    def __call__(self, x: torch.Tensor, g_cents: torch.Tensor) -> torch.Tensor:
        theta = (2.0 * math.pi / self.period) * g_cents
        c = torch.cos(theta)
        s = torch.sin(theta)
        out = x.clone()
        out[:, 0] = c * x[:, 0] - s * x[:, 1]
        out[:, 1] = s * x[:, 0] + c * x[:, 1]
        return out


class _RotationRep:
    """rho(g) as a (d_c, d_c) rotation on the first 2 channels, identity
    elsewhere. Mirrors src.models.group_repr.RotationRep but simplified for
    the test oracle."""

    def __init__(self, d_c: int, period_cents: float = 1200.0) -> None:
        self.d_c = d_c
        self.period = period_cents

    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor:
        b = g_cents.shape[0]
        theta = (2.0 * math.pi / self.period) * g_cents
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.eye(self.d_c, dtype=g_cents.dtype, device=g_cents.device)
        R = R.unsqueeze(0).expand(b, -1, -1).clone()
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        return R


def test_perfect_oracle_scores_zero_on_both_metrics() -> None:
    torch.manual_seed(0)
    d_c = 4
    d_s = 3
    model = _PerfectModel(d_s=d_s, d_c=d_c)
    aug = _IdentityAugmenter()
    rep = _RotationRep(d_c=d_c)

    n = 64
    x = torch.randn(n, d_c)
    g_list = torch.tensor([-1200.0, -600.0, -100.0, 100.0, 600.0, 1200.0])

    out = compute_from_arrays(
        model=model, x=x, g_cents_list=g_list, group_rep=rep, augment=aug
    )
    # mu_s is zero → invariance ratio uses epsilon stabilization and returns 0.
    assert out["invariance_ratio"] < 1e-6
    # mu_c exactly matches rho(g) mu_c(x) → equivariance ratio ~ 0.
    assert out["equivariance_ratio"] < 1e-6
    assert out["per_g"].shape == (g_list.shape[0], 2)


def test_non_equivariant_model_scores_high_er() -> None:
    """A model whose mu_c(T_g x) is UNRELATED to rho(g) mu_c(x) should
    produce a high equivariance ratio."""

    class _Bad(torch.nn.Module):
        def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            # mu_c simply returns x (no rotation consistency with rho).
            # Then rho(g) mu_c(x) != mu_c(T_g x) because T_g x changes x.
            return {
                "mu_s": torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype),
                "mu_c": x[:, :4],
            }

    torch.manual_seed(1)
    model = _Bad()
    aug = _IdentityAugmenter()
    # Use translation in the aug but rotation in the rep — deliberately
    # inconsistent, so ER should be high.

    class _TranslationAug:
        def __call__(self, x: torch.Tensor, g_cents: torch.Tensor) -> torch.Tensor:
            return x + g_cents[:, None] * 0.01

    rep = _RotationRep(d_c=4)
    x = torch.randn(32, 4)
    g_list = torch.tensor([-600.0, -100.0, 100.0, 600.0])
    out = compute_from_arrays(
        model=model, x=x, g_cents_list=g_list, group_rep=rep, augment=_TranslationAug()
    )
    assert out["equivariance_ratio"] > 0.05


def test_invariance_ratio_captures_style_drift() -> None:
    """If mu_s varies strongly with g, invariance ratio should rise."""

    class _StyleDrift(torch.nn.Module):
        def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            # mu_s depends on the first channel of x → changes with T_g.
            b = x.shape[0]
            mu_s = torch.stack([x[:, 0], torch.zeros(b, device=x.device)], dim=1)
            return {"mu_s": mu_s, "mu_c": x[:, :4]}

    torch.manual_seed(2)
    model = _StyleDrift()
    rep = _RotationRep(d_c=4)
    aug = _IdentityAugmenter()
    x = torch.randn(64, 4) + 1.0  # non-zero baseline so denominator isn't tiny
    g_list = torch.tensor([-600.0, -100.0, 100.0, 600.0])
    out = compute_from_arrays(
        model=model, x=x, g_cents_list=g_list, group_rep=rep, augment=aug
    )
    assert out["invariance_ratio"] > 0.05


def test_rejects_empty_g_list() -> None:
    model = _PerfectModel(d_s=2, d_c=4)
    rep = _RotationRep(d_c=4)
    with pytest.raises(ValueError, match="g_cents_list"):
        compute_from_arrays(
            model=model,
            x=torch.randn(4, 4),
            g_cents_list=torch.empty(0),
            group_rep=rep,
            augment=_IdentityAugmenter(),
        )


def test_rejects_non_1d_g_list() -> None:
    model = _PerfectModel(d_s=2, d_c=4)
    rep = _RotationRep(d_c=4)
    with pytest.raises(ValueError, match="1-D"):
        compute_from_arrays(
            model=model,
            x=torch.randn(4, 4),
            g_cents_list=torch.zeros(2, 3),
            group_rep=rep,
            augment=_IdentityAugmenter(),
        )


def test_dataloader_wrapper_runs_end_to_end() -> None:
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self, n: int) -> None:
            self.x = torch.randn(n, 4)

        def __len__(self) -> int:
            return self.x.shape[0]

        def __getitem__(self, i: int) -> dict:
            return {"x": self.x[i]}

    def _col(batch: list[dict]) -> dict:
        return {"x": torch.stack([b["x"] for b in batch], dim=0)}

    model = _PerfectModel(d_s=2, d_c=4)
    rep = _RotationRep(d_c=4)
    aug = _IdentityAugmenter()
    loader = DataLoader(_DS(n=32), batch_size=8, collate_fn=_col)
    out = compute(
        model=model,
        dataloader=loader,
        g_cents_list=torch.tensor([-600.0, 600.0]),
        group_rep=rep,
        augment=aug,
    )
    assert out["invariance_ratio"] < 1e-6
    assert out["equivariance_ratio"] < 1e-6
