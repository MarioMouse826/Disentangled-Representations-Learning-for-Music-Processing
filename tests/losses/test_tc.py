from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.losses.tc import (
    TCDiscriminator,
    batch_tc_decomposition,
    permute_latents,
    tc_discriminator_loss,
    tc_estimate_from_logits,
)


def test_permute_latents_preserves_marginals() -> None:
    torch.manual_seed(0)
    z = torch.randn(1024, 8)
    z_perm = permute_latents(z)
    # Each dim's mean/std must match (marginals preserved — only joint breaks).
    assert torch.allclose(z.mean(dim=0), z_perm.mean(dim=0), atol=1e-5)
    assert torch.allclose(z.std(dim=0), z_perm.std(dim=0), atol=1e-5)
    # Joint structure is destroyed — in expectation, sum(z * z_perm) != sum(z * z).
    # For random Gaussian, diag of covariance stays same but off-diag drops to ~0.
    cov_orig = (z - z.mean(0)).T @ (z - z.mean(0)) / (z.shape[0] - 1)
    cov_perm = (z_perm - z_perm.mean(0)).T @ (z_perm - z_perm.mean(0)) / (z_perm.shape[0] - 1)
    # Diagonal (variance) preserved within float precision.
    assert torch.allclose(cov_orig.diag(), cov_perm.diag(), atol=1e-5)


def test_permute_latents_shape_preserved() -> None:
    z = torch.randn(64, 16)
    z_perm = permute_latents(z)
    assert z_perm.shape == z.shape
    assert z_perm.dtype == z.dtype


def test_tc_goes_to_zero_on_factored_prior() -> None:
    # Kim & Mnih 2018: on samples drawn from a factored prior (independent
    # Gaussians), the discriminator cannot distinguish joint from factored,
    # so the TC estimate should converge near 0.
    torch.manual_seed(0)
    d_z = 8
    # Use a small hidden size to keep the test fast — 256 units × 3 layers
    # is sufficient to saturate on an 8-dim factored-vs-factored task.
    disc = TCDiscriminator(d_z=d_z, hidden=256, n_layers=3)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-4)

    for _ in range(1000):
        # Both "joint" and "factored" are drawn from the same factored prior.
        z_joint = torch.randn(256, d_z)
        z_factored = permute_latents(torch.randn(256, d_z))
        opt.zero_grad()
        l = tc_discriminator_loss(disc(z_joint), disc(z_factored))
        l.backward()
        opt.step()

    disc.train(False)
    z_eval = torch.randn(4096, d_z)
    tc = float(tc_estimate_from_logits(disc(z_eval)))
    assert abs(tc) < 0.15, f"TC on factored samples should be ~0, got {tc:.3f}"


def test_tc_is_positive_on_correlated_gaussians() -> None:
    # On jointly-correlated samples (ρ=0.9), discriminator should learn to
    # separate joint from factored → positive TC estimate > 0.3.
    torch.manual_seed(1)
    d_z = 4
    rho = 0.9
    L = torch.tensor(
        [
            [1.0, rho, rho, rho],
            [rho, 1.0, rho, rho],
            [rho, rho, 1.0, rho],
            [rho, rho, rho, 1.0],
        ]
    )
    L_chol = torch.linalg.cholesky(L)

    disc = TCDiscriminator(d_z=d_z, hidden=256, n_layers=3)
    opt = torch.optim.Adam(disc.parameters(), lr=1e-4)

    for _ in range(1500):
        eps = torch.randn(256, d_z)
        z_joint = eps @ L_chol.T
        z_factored = permute_latents(z_joint.detach().clone())
        opt.zero_grad()
        l = tc_discriminator_loss(disc(z_joint), disc(z_factored))
        l.backward()
        opt.step()

    disc.train(False)
    eps = torch.randn(4096, d_z)
    z_corr = eps @ L_chol.T
    tc = float(tc_estimate_from_logits(disc(z_corr)))
    assert tc > 0.3, f"TC on ρ=0.9 Gaussians should exceed 0.3, got {tc:.3f}"


def test_discriminator_output_shape() -> None:
    disc = TCDiscriminator(d_z=16, hidden=128, n_layers=3)
    z = torch.randn(7, 16)
    logits = disc(z)
    assert logits.shape == (7, 2)


def test_discriminator_loss_is_finite_and_bounded() -> None:
    torch.manual_seed(0)
    disc = TCDiscriminator(d_z=8, hidden=128, n_layers=3)
    z_j = torch.randn(32, 8)
    z_f = permute_latents(torch.randn(32, 8))
    loss = tc_discriminator_loss(disc(z_j), disc(z_f))
    assert torch.isfinite(loss)
    # Cross-entropy on 2-class, uniform init → ≈ log(2) ≈ 0.693.
    assert 0.0 <= float(loss) <= 3.0


def test_invalid_permute_input_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        permute_latents(torch.randn(4))
    with pytest.raises(ValueError, match="2-D"):
        permute_latents(torch.randn(4, 8, 2))


# -- batch_tc_decomposition (β-TCVAE, Chen et al. 2018) --------------------


def test_batch_tc_near_zero_on_factored_aggregate() -> None:
    # Per-item posterior means sampled from N(0, I) independently across
    # dims → aggregate posterior is factored → TC estimate ~ 0 up to the
    # minibatch-weighted sampling bias (±0.1 nats).
    torch.manual_seed(0)
    B, D = 512, 4
    mu = torch.randn(B, D)
    logvar = torch.zeros(B, D)
    eps = torch.randn(B, D)
    z = mu + eps  # z ~ N(mu, I) per item

    _, tc, _ = batch_tc_decomposition(z, mu, logvar)
    assert abs(float(tc)) < 0.2, f"TC on factored aggregate should be ~0, got {float(tc):.3f}"


def test_batch_tc_positive_on_correlated_aggregate() -> None:
    # Per-item posterior means along the diagonal (mu_i[j] = c_i for all j
    # with c_i ~ N(0, 1)) → aggregate posterior collapses onto the y=x=...
    # diagonal, which is maximally non-factored.
    torch.manual_seed(1)
    B, D = 512, 4
    c = torch.randn(B, 1)
    mu = c.expand(B, D).clone()
    logvar = torch.full((B, D), -4.0)  # near-zero noise for strong correlation
    eps = torch.randn(B, D) * (0.5 * logvar).exp()
    z = mu + eps

    _, tc, _ = batch_tc_decomposition(z, mu, logvar)
    assert float(tc) > 0.3, f"TC on correlated aggregate should exceed 0.3, got {float(tc):.3f}"


def test_batch_tc_sum_matches_kl() -> None:
    # Plan-load-bearing invariant: MI + TC + dimwise_KL == KL(q(z|x) || N(0,I))
    # evaluated at the same posterior samples.
    torch.manual_seed(2)
    B, D = 256, 6
    mu = torch.randn(B, D) * 0.5
    logvar = torch.randn(B, D) * 0.3
    eps = torch.randn(B, D)
    z = mu + eps * (0.5 * logvar).exp()

    mi, tc, dimwise = batch_tc_decomposition(z, mu, logvar)

    # Monte-Carlo KL(q || N(0,I)) at this z: E_q[log q(z|x) - log p(z)].
    import math as _m
    log_q_given_x = (
        -0.5 * (_m.log(2.0 * _m.pi) + logvar + (z - mu).pow(2) * (-logvar).exp())
    ).sum(dim=1).mean()
    log_p_z = (-0.5 * (_m.log(2.0 * _m.pi) + z.pow(2))).sum(dim=1).mean()
    mc_kl = log_q_given_x - log_p_z

    decomposed_sum = mi + tc + dimwise
    # Algebraic identity: mi + tc + dimwise_kl == log_q_z_given_x - log_p_z
    # up to float32 accumulation noise. The MWS estimator bias cancels
    # across the three terms (it's a fixed `log(B)` shift applied and
    # subtracted). Tighten to 1e-4 nats to catch sign flips and
    # missing-sum bugs that would otherwise hide at the previous 0.5-nat
    # threshold.
    assert abs(float(decomposed_sum) - float(mc_kl)) < 1e-4, (
        f"decomposition sum {float(decomposed_sum):.5f} != MC KL {float(mc_kl):.5f}"
    )


def test_batch_tc_shape_validation() -> None:
    with pytest.raises(ValueError, match="B, D"):
        batch_tc_decomposition(torch.randn(8), torch.randn(8, 4), torch.randn(8, 4))
    with pytest.raises(ValueError, match="B, D"):
        batch_tc_decomposition(
            torch.randn(8, 4), torch.randn(8, 3), torch.randn(8, 4)
        )


def test_batch_tc_rejects_batch_size_one() -> None:
    # MWS estimator degenerates silently at B=1 (logsumexp over one element
    # + log(1)=0 → every decomposition term is 0). Guard must raise.
    with pytest.raises(ValueError, match=r"B >= 2"):
        batch_tc_decomposition(
            torch.randn(1, 4), torch.randn(1, 4), torch.zeros(1, 4)
        )


def test_batch_tc_device_preservation() -> None:
    from src.utils.device import best_device

    dev = best_device()
    B, D = 64, 4
    mu = torch.randn(B, D, device=dev)
    logvar = torch.zeros(B, D, device=dev)
    z = mu + torch.randn(B, D, device=dev)
    mi, tc, dimwise = batch_tc_decomposition(z, mu, logvar)
    for t in (mi, tc, dimwise):
        assert t.device.type == dev.type
