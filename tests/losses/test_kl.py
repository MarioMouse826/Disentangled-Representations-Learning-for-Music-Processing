from __future__ import annotations

import math

import pytest
import torch

from src.losses.kl import cyclical_beta, kl_subspace, kl_total


# -- kl_subspace ----------------------------------------------------------


def test_kl_zero_when_posterior_equals_prior() -> None:
    # Plan-critical (a): analytic Gaussian KL for N(0, I) vs N(0, I) = 0.
    mu = torch.zeros(8, 32)
    logvar = torch.zeros(8, 32)
    kl = kl_subspace(mu, logvar)
    assert abs(float(kl)) < 1e-6


def test_kl_positive_on_shifted_posterior() -> None:
    torch.manual_seed(0)
    mu = torch.randn(8, 16)
    logvar = torch.zeros(8, 16)
    kl = kl_subspace(mu, logvar)
    assert float(kl) > 0.0


def test_kl_matches_analytic_formula() -> None:
    # Direct per-element check against -0.5 * (1 + logvar - μ² - exp(logvar))
    # summed over dim, averaged over batch.
    torch.manual_seed(1)
    mu = torch.randn(4, 8) * 0.5
    logvar = torch.randn(4, 8) * 0.3
    manual_per_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    manual = manual_per_elem.sum(dim=-1).mean()
    assert torch.allclose(kl_subspace(mu, logvar), manual, atol=1e-6)


def test_free_bits_clamps_at_threshold() -> None:
    # Plan-critical (b): free-bits τ = 0.5 must clamp KL at ≥ 0.5 nats/dim.
    # Use (mu, logvar) == (0, 0) so raw KL is exactly 0 per dim → free-bits
    # floor is the only contribution.
    mu = torch.zeros(4, 10)
    logvar = torch.zeros(4, 10)
    kl = kl_subspace(mu, logvar, free_bits=0.5)
    assert abs(float(kl) - 5.0) < 1e-6  # 0.5 nats/dim × 10 dims


def test_free_bits_does_not_raise_when_above_threshold() -> None:
    # With per-example clamp semantics, the clamp is a no-op only when
    # EVERY per-example per-dim KL exceeds free_bits. Use a deterministic
    # offset (|mu|=3.0 → kl_per_dim=4.5 per example) so every entry is
    # safely above τ=0.1.
    mu = torch.full((16, 8), 3.0)
    logvar = torch.zeros(16, 8)
    kl_low_fb = kl_subspace(mu, logvar, free_bits=0.0)
    kl_high_fb = kl_subspace(mu, logvar, free_bits=0.1)
    assert torch.allclose(kl_low_fb, kl_high_fb, atol=1e-6)


def test_kl_handles_time_resolved_latent() -> None:
    # Content latent is (B, d_c, T'). The function must sum over time before
    # per-example + per-dim free-bits clamp.
    mu = torch.zeros(3, 8, 13)
    logvar = torch.zeros(3, 8, 13)
    kl = kl_subspace(mu, logvar, free_bits=0.1)
    assert abs(float(kl) - 0.8) < 1e-6  # 0.1 per dim × 8 dims


def test_free_bits_applied_per_example_not_batch_mean() -> None:
    # Code-reviewer Phase-3 catch: Kingma 2016 free-bits clamps per-example
    # THEN averages, not average THEN clamps. Construct a mixed batch where
    # the two orderings give different answers:
    #   example 0: zero KL  (mu=0, logvar=0 → kl_per_dim=0.0)
    #   example 1: large KL (mu=10, logvar=0 → kl_per_dim=50.0)
    # With τ=0.1:
    #   per-example order: clamp(0.0, 0.1)+clamp(50.0, 0.1) = 0.1 + 50.0 → mean = 25.05
    #   batch-mean order:  mean(0.0, 50.0)=25.0; clamp(25.0, 0.1) = 25.0
    # Difference is 0.05 × D per dim of mixed behavior.
    mu = torch.zeros(2, 4)
    mu[1] = 10.0
    logvar = torch.zeros(2, 4)
    kl = kl_subspace(mu, logvar, free_bits=0.1)
    # Per-dim raw: [0.0, 50.0]; clamp per-example → [0.1, 50.0]; mean → 25.05;
    # sum over 4 dims → 100.2.
    expected = 4 * (0.1 + 50.0) / 2
    assert abs(float(kl) - expected) < 1e-3, (
        f"kl={float(kl):.4f} does not match Kingma per-example order "
        f"expected={expected:.4f}"
    )


def test_kl_differentiable() -> None:
    mu = torch.randn(4, 16, requires_grad=True)
    logvar = torch.randn(4, 16, requires_grad=True)
    kl = kl_subspace(mu, logvar, free_bits=0.1)
    kl.backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    assert logvar.grad is not None and torch.isfinite(logvar.grad).all()


def test_kl_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        kl_subspace(torch.randn(4, 16), torch.randn(4, 32))


def test_kl_rejects_bad_rank() -> None:
    with pytest.raises(ValueError, match=r"2-D.*3-D"):
        kl_subspace(torch.randn(4), torch.randn(4))


def test_kl_negative_free_bits_raises() -> None:
    with pytest.raises(ValueError, match="free_bits"):
        kl_subspace(torch.zeros(2, 4), torch.zeros(2, 4), free_bits=-0.5)


# -- kl_total (per-subspace API) ------------------------------------------


def test_kl_total_returns_independent_scalars() -> None:
    # Plan-critical (d): kl_total must return independent (kl_s, kl_c) so
    # Phase-4 can weight them with independent β_s, β_c.
    mu_s = torch.randn(4, 16)
    logvar_s = torch.zeros(4, 16)
    mu_c = torch.randn(4, 8, 13)
    logvar_c = torch.zeros(4, 8, 13)
    kl_s, kl_c = kl_total(mu_s, logvar_s, mu_c, logvar_c, tau_s=0.1, tau_c=0.1)
    assert kl_s.ndim == 0
    assert kl_c.ndim == 0
    # Free bits floor: τ=0.1 × d=16 for style, τ=0.1 × 8 for content.
    assert float(kl_s) >= 1.6 - 1e-5
    assert float(kl_c) >= 0.8 - 1e-5


def test_kl_total_independent_free_bits() -> None:
    # Different τ per subspace must produce different floors.
    mu_s = torch.zeros(2, 10)
    logvar_s = torch.zeros(2, 10)
    mu_c = torch.zeros(2, 10, 5)
    logvar_c = torch.zeros(2, 10, 5)
    kl_s, kl_c = kl_total(mu_s, logvar_s, mu_c, logvar_c, tau_s=0.2, tau_c=0.5)
    assert abs(float(kl_s) - 2.0) < 1e-6  # 0.2 × 10
    assert abs(float(kl_c) - 5.0) < 1e-6  # 0.5 × 10


# -- cyclical_beta --------------------------------------------------------


def test_cyclical_beta_zero_at_cycle_start() -> None:
    # Plan-critical (c): β schedule returns 0.0 at cycle start.
    assert cyclical_beta(step=0, total_steps=1000, n_cycles=4) == 0.0


def test_cyclical_beta_max_at_cycle_end() -> None:
    # At the last step of a cycle (pos just below 1.0) — in the hold phase
    # → returns beta_max.
    cycle_len = 250  # 1000/4
    # One step before cycle rollover.
    b = cyclical_beta(step=cycle_len - 1, total_steps=1000, n_cycles=4)
    assert abs(b - 1.0) < 1e-5


def test_cyclical_beta_ramp_midpoint() -> None:
    # At ratio=0.5 midpoint of ramp: β = 0.5 * beta_max.
    cycle_len = 200  # 1000/5
    # pos = 0.25 / 0.5 = 0.5 of ramp → β = 0.5.
    b = cyclical_beta(step=int(cycle_len * 0.25), total_steps=1000, n_cycles=5, ratio=0.5)
    assert abs(b - 0.5) < 1e-5


def test_cyclical_beta_hold_phase() -> None:
    # After `ratio` of cycle, β holds at beta_max until rollover.
    # cycle_len=200, ratio=0.5 → ramp phase is pos ∈ [0, 0.5), hold is [0.5, 1).
    # step=120 → pos=0.6 > 0.5 → hold phase.
    b_past_mid = cyclical_beta(step=120, total_steps=200, n_cycles=1, ratio=0.5)
    b_late = cyclical_beta(step=199, total_steps=200, n_cycles=1, ratio=0.5)
    assert abs(b_past_mid - 1.0) < 1e-5
    assert abs(b_late - 1.0) < 1e-5


def test_cyclical_beta_min_max_custom() -> None:
    # Custom [β_min, β_max] range.
    assert abs(cyclical_beta(step=0, total_steps=100, n_cycles=1, beta_min=2.0, beta_max=8.0) - 2.0) < 1e-5
    # At ratio midpoint, half-way: β = 2 + 0.5*(8-2) = 5.
    assert abs(cyclical_beta(step=25, total_steps=100, n_cycles=1, ratio=0.5, beta_min=2.0, beta_max=8.0) - 5.0) < 1e-5


def test_cyclical_beta_periodicity() -> None:
    # Full cycle periodicity: step and step+cycle_len give same β.
    for step in (0, 13, 47, 99):
        b1 = cyclical_beta(step=step, total_steps=800, n_cycles=4)
        b2 = cyclical_beta(step=step + 200, total_steps=800, n_cycles=4)
        assert abs(b1 - b2) < 1e-5


def test_cyclical_beta_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="total_steps"):
        cyclical_beta(step=0, total_steps=0, n_cycles=1)
    with pytest.raises(ValueError, match="n_cycles"):
        cyclical_beta(step=0, total_steps=100, n_cycles=0)
    with pytest.raises(ValueError, match="ratio"):
        cyclical_beta(step=0, total_steps=100, n_cycles=1, ratio=1.5)
    with pytest.raises(ValueError, match="beta_min"):
        cyclical_beta(step=0, total_steps=100, n_cycles=1, beta_min=5.0, beta_max=2.0)
