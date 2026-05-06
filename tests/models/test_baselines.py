from __future__ import annotations

import pytest
import torch

from src.models.baselines.ar_hvae import AR_HVAE
from src.models.baselines.beta_tcvae import BetaTCVAE
from src.models.baselines.beta_vae import BetaVAE
from src.models.baselines.factor_vae import FactorVAE


@pytest.fixture
def mel() -> torch.Tensor:
    return torch.randn(2, 1, 128, 401)


def test_beta_vae_forward(mel: torch.Tensor) -> None:
    model = BetaVAE(d_z=64, n_mels=128, n_time=401)
    out = model(mel)
    assert set(out.keys()) == {"x_hat", "mu", "logvar", "z"}
    assert out["x_hat"].shape == mel.shape
    assert out["mu"].shape == (mel.shape[0], 64)
    assert out["logvar"].shape == (mel.shape[0], 64)
    assert out["z"].shape == (mel.shape[0], 64)
    for v in out.values():
        assert torch.isfinite(v).all()


def test_reparameterize_stochastic_in_train(mel: torch.Tensor) -> None:
    model = BetaVAE(d_z=32, n_mels=128, n_time=401)
    model.train(True)
    z1 = model(mel)["z"]
    z2 = model(mel)["z"]
    # Same input, two stochastic forward passes — z differs.
    assert not torch.equal(z1, z2)


def test_reparameterize_deterministic_in_inference(mel: torch.Tensor) -> None:
    # Default policy: inference mode returns z = mu (no noise). Pins behavior
    # for reproducible assessment passes over the test set.
    model = BetaVAE(d_z=32, n_mels=128, n_time=401)
    model.train(False)
    out1 = model(mel)
    out2 = model(mel)
    assert torch.equal(out1["z"], out2["z"])
    assert torch.allclose(out1["z"], out1["mu"])


def test_loss_returns_scalar_and_dict(mel: torch.Tensor) -> None:
    model = BetaVAE(d_z=32, n_mels=128, n_time=401)
    total, parts = model.loss(mel, beta=1.0)
    assert total.ndim == 0  # scalar
    assert {"recon", "kl"}.issubset(parts.keys())
    assert all(v.ndim == 0 for v in parts.values())


def test_kl_is_non_negative(mel: torch.Tensor) -> None:
    model = BetaVAE(d_z=32, n_mels=128, n_time=401)
    _, parts = model.loss(mel, beta=1.0)
    assert float(parts["kl"]) >= 0.0


def test_beta_scales_total_monotonically(mel: torch.Tensor) -> None:
    # Inference mode → deterministic z = mu → same mu/logvar across calls →
    # KL is fixed, so the total loss grows monotonically with beta when KL > 0.
    torch.manual_seed(0)
    model = BetaVAE(d_z=32, n_mels=128, n_time=401)
    model.train(False)
    t1, _ = model.loss(mel, beta=1.0)
    t4, _ = model.loss(mel, beta=4.0)
    t16, _ = model.loss(mel, beta=16.0)
    assert float(t1) <= float(t4) <= float(t16)


def test_gradient_flows_through_every_parameter(mel: torch.Tensor) -> None:
    model = BetaVAE(d_z=32, n_mels=128, n_time=401)
    total, _ = model.loss(mel, beta=1.0)
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient on: {missing}"
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {n}"


def test_device_preservation(mel: torch.Tensor) -> None:
    from src.utils.device import best_device

    dev = best_device()
    model = BetaVAE(d_z=32, n_mels=128, n_time=401).to(dev)
    mel = mel.to(dev)
    out = model(mel)
    for v in out.values():
        # Compare device type only — MPS stamps index=0 on outputs,
        # `best_device()` returns the index-less form. Either is valid.
        assert v.device.type == dev.type


def test_recon_zero_on_perfect_reconstruction() -> None:
    # Replace the decoder with an identity-zero map and verify recon loss
    # is exactly zero on a zero input. Sanity-check for loss plumbing, not
    # model quality.
    model = BetaVAE(d_z=16, n_mels=128, n_time=401)
    model.train(False)
    x = torch.zeros(1, 1, 128, 401)
    original_decode = model.decode

    def zero_decode(z: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    model.decode = zero_decode  # type: ignore[assignment]
    try:
        _, parts = model.loss(x, beta=1.0)
        assert float(parts["recon"]) == 0.0
    finally:
        model.decode = original_decode  # type: ignore[assignment]


def test_config_roundtrip() -> None:
    model = BetaVAE(d_z=48, n_mels=128, n_time=401)
    cfg = model.config
    assert cfg["d_z"] == 48
    assert cfg["n_mels"] == 128
    assert cfg["n_time"] == 401
    assert "encoder" in cfg and cfg["encoder"]["out_channels"] == 512


def test_invalid_d_z_raises() -> None:
    with pytest.raises(ValueError, match="d_z"):
        BetaVAE(d_z=0, n_mels=128, n_time=401)


def test_rejects_wrong_mel_shape() -> None:
    model = BetaVAE(d_z=32, n_mels=128, n_time=401)
    with pytest.raises(ValueError, match="expected"):
        model(torch.randn(1, 1, 64, 401))  # wrong n_mels


# -- FactorVAE ------------------------------------------------------------


def test_factor_vae_forward(mel: torch.Tensor) -> None:
    model = FactorVAE(d_z=32, n_mels=128, n_time=401, disc_hidden=256, disc_layers=3)
    out = model(mel)
    assert set(out.keys()) == {"x_hat", "mu", "logvar", "z"}
    assert out["x_hat"].shape == mel.shape
    assert out["z"].shape == (mel.shape[0], 32)


def test_factor_vae_loss_vae_returns_tc_term(mel: torch.Tensor) -> None:
    model = FactorVAE(d_z=32, n_mels=128, n_time=401, disc_hidden=256, disc_layers=3)
    total, parts, z = model.loss_vae(mel, beta=1.0, gamma=1.0)
    assert total.ndim == 0
    assert {"recon", "kl", "tc"}.issubset(parts.keys())
    assert z.shape == (mel.shape[0], 32)


def test_factor_vae_disc_loss_separates_joint_from_factored() -> None:
    # Train the discriminator briefly on synthetic correlated vs permuted
    # latents — loss should drop below log(2) ≈ 0.69 (chance baseline).
    torch.manual_seed(0)
    model = FactorVAE(d_z=8, n_mels=128, n_time=401, disc_hidden=128, disc_layers=3)
    opt = torch.optim.Adam(model.disc.parameters(), lr=1e-4)
    rho = 0.9
    from src.losses.tc import permute_latents

    for _ in range(400):
        eps = torch.randn(128, 8)
        # Correlated joint: add shared noise component.
        shared = torch.randn(128, 1)
        z_j = rho * shared + (1 - rho**2) ** 0.5 * eps
        z_f = permute_latents(z_j.detach().clone())
        opt.zero_grad()
        loss = model.loss_disc(z_j.detach(), z_f)
        loss.backward()
        opt.step()

    # After training: loss on fresh correlated data should be below chance.
    model.disc.train(False)
    eps = torch.randn(512, 8)
    shared = torch.randn(512, 1)
    z_j = rho * shared + (1 - rho**2) ** 0.5 * eps
    z_f = permute_latents(z_j.clone())
    loss_final = float(model.loss_disc(z_j, z_f))
    assert loss_final < 0.69, f"disc failed to learn: loss={loss_final:.3f}"


def test_factor_vae_vae_grad_does_not_update_disc(mel: torch.Tensor) -> None:
    # Kim & Mnih 2018: VAE step uses D frozen — gamma·TC term must provide
    # gradient to VAE params via z, but not to disc params.
    model = FactorVAE(d_z=32, n_mels=128, n_time=401, disc_hidden=128, disc_layers=3)
    # Capture disc params before.
    disc_before = [p.detach().clone() for p in model.disc.parameters()]

    total, _, _ = model.loss_vae(mel, beta=1.0, gamma=1.0)
    total.backward()

    # Disc params must have zero gradient (we blocked them inside loss_vae).
    for p in model.disc.parameters():
        assert p.grad is None or torch.all(p.grad == 0)


def test_factor_vae_config_includes_disc() -> None:
    model = FactorVAE(d_z=32, n_mels=128, n_time=401, disc_hidden=256, disc_layers=4)
    cfg = model.config
    assert cfg["disc_hidden"] == 256
    assert cfg["disc_layers"] == 4


def test_factor_vae_device_preservation(mel: torch.Tensor) -> None:
    from src.utils.device import best_device

    dev = best_device()
    model = FactorVAE(d_z=32, n_mels=128, n_time=401, disc_hidden=256, disc_layers=3).to(dev)
    mel = mel.to(dev)
    out = model(mel)
    for v in out.values():
        assert v.device.type == dev.type


# -- AR-HVAE --------------------------------------------------------------


def test_ar_hvae_forward_schema(mel: torch.Tensor) -> None:
    model = AR_HVAE(levels=(32, 16, 8), n_mels=128, n_time=401)
    out = model(mel)
    # x_hat + 3 per-level (mu, logvar, z, prior_mu, prior_logvar) bundles.
    assert "x_hat" in out
    assert out["x_hat"].shape == mel.shape
    for k in ("mu", "logvar", "z", "prior_mu", "prior_logvar"):
        assert len(out[k]) == 3, f"expected 3 levels for {k!r}, got {len(out[k])}"
    # Bottom (z_1) has dim 32, mid (z_2) has dim 16, top (z_3) has dim 8.
    assert out["z"][0].shape == (mel.shape[0], 32)
    assert out["z"][1].shape == (mel.shape[0], 16)
    assert out["z"][2].shape == (mel.shape[0], 8)


def test_ar_hvae_kl_positive(mel: torch.Tensor) -> None:
    # Plan-critical: every per-level KL term must be ≥ 0 (analytic Gaussian
    # KL cannot be negative). If any term went negative it would indicate a
    # sign flip in the KL formula.
    model = AR_HVAE(levels=(32, 16, 8), n_mels=128, n_time=401)
    _, parts = model.loss(mel)
    for i in range(3):
        kl_i = float(parts[f"kl_{i}"])
        # float32 KL accumulation can wobble as low as ~-1e-5 on non-CPU
        # backends (CUDA/MPS reorder reductions). A sign-flipped KL formula
        # would produce -O(d_z) magnitudes, far larger than this floor.
        assert kl_i >= -1e-4, f"level {i} KL is negative: {kl_i}"
    # Total KL should equal sum of per-level terms.
    kl_sum = sum(float(parts[f"kl_{i}"]) for i in range(3))
    total_kl = float(parts["kl"])
    assert abs(total_kl - kl_sum) < 1e-5


def test_ar_hvae_elbo_decomposition(mel: torch.Tensor) -> None:
    # ELBO = recon + sum_L KL_L. Pin this contract so future refactors
    # don't silently drop a level from the objective.
    model = AR_HVAE(levels=(32, 16, 8), n_mels=128, n_time=401)
    total, parts = model.loss(mel)
    expected = float(parts["recon"]) + sum(float(parts[f"kl_{i}"]) for i in range(3))
    assert abs(float(total) - expected) < 1e-5


def test_ar_hvae_grad_flows(mel: torch.Tensor) -> None:
    model = AR_HVAE(levels=(32, 16, 8), n_mels=128, n_time=401)
    total, _ = model.loss(mel)
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient on: {missing}"


def test_ar_hvae_deterministic_inference(mel: torch.Tensor) -> None:
    model = AR_HVAE(levels=(32, 16, 8), n_mels=128, n_time=401)
    model.train(False)
    o1 = model(mel)
    o2 = model(mel)
    assert torch.equal(o1["x_hat"], o2["x_hat"])
    for i in range(3):
        assert torch.equal(o1["z"][i], o2["z"][i])


def test_ar_hvae_device_preservation(mel: torch.Tensor) -> None:
    from src.utils.device import best_device

    dev = best_device()
    model = AR_HVAE(levels=(32, 16, 8), n_mels=128, n_time=401).to(dev)
    mel = mel.to(dev)
    out = model(mel)
    assert out["x_hat"].device.type == dev.type
    for i in range(3):
        assert out["z"][i].device.type == dev.type


def test_ar_hvae_invalid_levels_raise() -> None:
    with pytest.raises(ValueError, match="levels"):
        AR_HVAE(levels=(), n_mels=128, n_time=401)
    with pytest.raises(ValueError, match="positive"):
        AR_HVAE(levels=(32, 0, 8), n_mels=128, n_time=401)


# -- β-TCVAE --------------------------------------------------------------


def test_beta_tcvae_forward(mel: torch.Tensor) -> None:
    model = BetaTCVAE(d_z=32, n_mels=128, n_time=401)
    out = model(mel)
    assert out["x_hat"].shape == mel.shape
    assert out["z"].shape == (mel.shape[0], 32)


def test_beta_tcvae_loss_parts(mel: torch.Tensor) -> None:
    model = BetaTCVAE(d_z=32, n_mels=128, n_time=401)
    total, parts = model.loss(mel, beta=1.0)
    for k in ("recon", "mi", "tc", "dimwise_kl", "kl"):
        assert k in parts
        assert parts[k].ndim == 0
    # kl reported should equal mi + tc + dimwise_kl (decomposition).
    recomposed = float(parts["mi"]) + float(parts["tc"]) + float(parts["dimwise_kl"])
    assert abs(float(parts["kl"]) - recomposed) < 1e-5


def test_beta_tcvae_beta_scales_tc_only() -> None:
    # At α=γ=1 (defaults), β must multiply ONLY the TC term. Verify via
    # total-loss algebra: Δtotal = Δβ · tc must hold exactly (modulo
    # float32 noise). A refactor that accidentally applied β to mi or
    # dimwise_kl would break this identity.
    torch.manual_seed(0)
    model = BetaTCVAE(d_z=16, n_mels=128, n_time=401)
    model.train(False)  # deterministic z = mu
    x = torch.randn(2, 1, 128, 401)
    total1, p1 = model.loss(x, beta=1.0)
    total8, p8 = model.loss(x, beta=8.0)

    # MI, tc, dimwise_kl scalars depend only on (mu, logvar, z). In
    # inference mode z = mu so these are deterministic across calls → all
    # three components are bitwise identical when re-computed.
    assert torch.allclose(p1["mi"], p8["mi"], atol=1e-5)
    assert torch.allclose(p1["dimwise_kl"], p8["dimwise_kl"], atol=1e-5)
    assert torch.allclose(p1["tc"], p8["tc"], atol=1e-5)

    # Total-loss algebra: delta must equal (8 - 1) * tc, confirming that
    # beta multiplies only tc in the computed total.
    delta = float(total8 - total1)
    expected = 7.0 * float(p1["tc"])
    assert abs(delta - expected) < 1e-4, (
        f"beta delta {delta:.5f} != 7·tc = {expected:.5f} — beta is "
        f"scaling something other than tc"
    )


def test_beta_tcvae_grad_flows(mel: torch.Tensor) -> None:
    model = BetaTCVAE(d_z=32, n_mels=128, n_time=401)
    total, _ = model.loss(mel, beta=4.0)
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient on: {missing}"


def test_beta_tcvae_invalid_alpha_gamma_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BetaTCVAE(d_z=16, n_mels=128, n_time=401, alpha=-0.5)
    with pytest.raises(ValueError, match="non-negative"):
        BetaTCVAE(d_z=16, n_mels=128, n_time=401, gamma=-1.0)


def test_beta_tcvae_device_preservation(mel: torch.Tensor) -> None:
    from src.utils.device import best_device

    dev = best_device()
    model = BetaTCVAE(d_z=32, n_mels=128, n_time=401).to(dev)
    mel = mel.to(dev)
    out = model(mel)
    for v in out.values():
        assert v.device.type == dev.type
