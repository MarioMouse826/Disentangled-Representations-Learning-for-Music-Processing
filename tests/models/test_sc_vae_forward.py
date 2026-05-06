from __future__ import annotations

import pytest
import torch

from src.models.sc_vae import SCVAE


@pytest.fixture
def mel() -> torch.Tensor:
    return torch.randn(2, 1, 128, 401)


@pytest.fixture
def mel_g(mel: torch.Tensor) -> torch.Tensor:
    # Simulated T_g(x) — different content from mel.
    return mel + 0.5 * torch.randn_like(mel)


@pytest.fixture
def g_cents() -> torch.Tensor:
    return torch.tensor([200.0, -400.0])


def test_forward_schema(mel: torch.Tensor) -> None:
    model = SCVAE(d_s=32, d_c=16, n_mels=128, n_time=401)
    out = model(mel)
    expected_keys = {
        "x_hat", "mu_s", "logvar_s", "z_s",
        "mu_c", "logvar_c", "z_c", "attn_weights",
    }
    assert set(out.keys()) == expected_keys


def test_forward_shapes(mel: torch.Tensor) -> None:
    model = SCVAE(d_s=32, d_c=16, n_mels=128, n_time=401)
    out = model(mel)
    B = mel.shape[0]
    t_enc = model.t_enc
    assert out["x_hat"].shape == mel.shape
    assert out["mu_s"].shape == (B, 32)
    assert out["logvar_s"].shape == (B, 32)
    assert out["z_s"].shape == (B, 32)
    assert out["mu_c"].shape == (B, 16, t_enc)
    assert out["logvar_c"].shape == (B, 16, t_enc)
    assert out["z_c"].shape == (B, 16, t_enc)
    assert out["attn_weights"].shape == (B, t_enc)


def test_reparam_deterministic_in_inference(mel: torch.Tensor) -> None:
    model = SCVAE(d_s=32, d_c=16, n_mels=128, n_time=401)
    model.train(False)
    o1 = model(mel)
    o2 = model(mel)
    assert torch.equal(o1["z_s"], o2["z_s"])
    assert torch.equal(o1["z_c"], o2["z_c"])
    assert torch.allclose(o1["z_s"], o1["mu_s"])
    assert torch.allclose(o1["z_c"], o1["mu_c"])


def test_reparam_stochastic_in_train(mel: torch.Tensor) -> None:
    model = SCVAE(d_s=32, d_c=16, n_mels=128, n_time=401)
    model.train(True)
    z_s_1 = model(mel)["z_s"]
    z_s_2 = model(mel)["z_s"]
    assert not torch.equal(z_s_1, z_s_2)


def test_loss_no_symmetry_args(mel: torch.Tensor) -> None:
    # Without x_g/g_cents, loss has only recon + KL terms.
    model = SCVAE(d_s=32, d_c=16, n_mels=128, n_time=401)
    total, parts = model.loss(mel)
    assert total.ndim == 0
    assert {"recon", "kl_s", "kl_c"}.issubset(parts.keys())
    # Symmetry terms should NOT be present when no paired input provided.
    assert "L_inv" not in parts
    assert "L_equi" not in parts
    assert "L_swap" not in parts


def test_loss_with_symmetry(
    mel: torch.Tensor, mel_g: torch.Tensor, g_cents: torch.Tensor
) -> None:
    model = SCVAE(d_s=32, d_c=16, n_mels=128, n_time=401)
    total, parts = model.loss(
        mel, x_g=mel_g, g_cents=g_cents,
        beta_s=1.0, beta_c=1.0,
        tau_s=0.1, tau_c=0.1,
        lambda_inv=1.0, lambda_equi=1.0, lambda_swap=0.5,
    )
    assert total.ndim == 0
    for k in ("recon", "kl_s", "kl_c", "L_inv", "L_equi", "L_swap"):
        assert k in parts
        assert parts[k].ndim == 0


def test_loss_identity_rep_inv_zero_when_xg_equals_x(mel: torch.Tensor) -> None:
    # With rep=identity + x_g == x + g=0, symmetry losses should vanish
    # (inference mode makes z deterministic, so L_inv = 0 exactly).
    model = SCVAE(
        d_s=16, d_c=8, n_mels=128, n_time=401,
        rep_type="identity",
    )
    model.train(False)
    B = mel.shape[0]
    zero_g = torch.zeros(B)
    _, parts = model.loss(
        mel, x_g=mel, g_cents=zero_g,
        lambda_inv=1.0, lambda_equi=1.0, lambda_swap=1.0,
    )
    assert abs(float(parts["L_inv"])) < 1e-5
    assert abs(float(parts["L_equi"])) < 1e-5
    # L_swap: Dec(z_s, rho(0)·z_c) == Dec(z_s, z_c) == x_hat.
    # On random mel input, x_hat won't exactly equal x, so L_swap > 0 but
    # equal to the recon L1 (both compare Dec(z) to x).
    assert abs(float(parts["L_swap"]) - float(parts["recon"])) < 1e-5


def test_gradient_flows(
    mel: torch.Tensor, mel_g: torch.Tensor, g_cents: torch.Tensor
) -> None:
    model = SCVAE(d_s=32, d_c=16, n_mels=128, n_time=401)
    total, _ = model.loss(
        mel, x_g=mel_g, g_cents=g_cents,
        lambda_inv=1.0, lambda_equi=1.0, lambda_swap=1.0,
    )
    total.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad on: {missing}"
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {n}"

    # Plan-critical: L_swap + L_equi must push gradient into rep.log_omega
    # (the only learnable parameter of RotationRep). Magnitude check,
    # not just non-None — a None-check alone passes even when autograd
    # populates a zero-valued grad tensor, hiding detach bugs.
    rep = model.rep
    assert hasattr(rep, "log_omega"), "expected RotationRep with log_omega"
    assert rep.log_omega.grad is not None
    assert float(rep.log_omega.grad.abs().sum()) > 0.0, (
        "rep.log_omega received zero gradient — L_swap/L_equi chain broken"
    )


def test_device_preservation(mel: torch.Tensor) -> None:
    from src.utils.device import best_device

    dev = best_device()
    model = SCVAE(d_s=16, d_c=8, n_mels=128, n_time=401).to(dev)
    mel = mel.to(dev)
    out = model(mel)
    for k, v in out.items():
        assert v.device.type == dev.type, f"{k} on wrong device"


def test_rep_swapping() -> None:
    mel = torch.randn(1, 1, 128, 401)
    for rep_type in ("rotation", "translation", "identity"):
        model = SCVAE(d_s=8, d_c=8, n_mels=128, n_time=401, rep_type=rep_type)
        out = model(mel)
        assert out["x_hat"].shape == mel.shape
        assert model.config["rep_type"] == rep_type


def test_config_roundtrip() -> None:
    model = SCVAE(
        d_s=32, d_c=16, n_mels=128, n_time=401,
        rep_type="rotation", period_cents=1200.0, freq_init="octave",
    )
    cfg = model.config
    assert cfg["d_s"] == 32
    assert cfg["d_c"] == 16
    assert cfg["n_mels"] == 128
    assert cfg["n_time"] == 401
    assert cfg["rep_type"] == "rotation"
    assert "encoder" in cfg
    assert "style_head" in cfg
    assert "content_head" in cfg
    assert "decoder" in cfg


def test_invalid_rep_raises() -> None:
    with pytest.raises(ValueError, match="rep_type"):
        SCVAE(d_s=8, d_c=8, n_mels=128, n_time=401, rep_type="bogus")


def test_rotation_rep_requires_even_d_c() -> None:
    # rep_type="rotation" enforces d_c even because ContentHead is built
    # with require_even_d_c=True when paired with RotationRep.
    with pytest.raises(ValueError, match="even"):
        SCVAE(d_s=8, d_c=5, n_mels=128, n_time=401, rep_type="rotation")


def test_rejects_wrong_input_shape() -> None:
    model = SCVAE(d_s=8, d_c=8, n_mels=128, n_time=401)
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 64, 401))  # wrong n_mels


def test_loss_requires_both_xg_and_gcents(
    mel: torch.Tensor, mel_g: torch.Tensor, g_cents: torch.Tensor
) -> None:
    # Passing only one of (x_g, g_cents) must raise — sanity check the
    # paired-input contract.
    model = SCVAE(d_s=8, d_c=8, n_mels=128, n_time=401)
    with pytest.raises(ValueError, match="x_g.*g_cents"):
        model.loss(mel, x_g=mel_g)  # missing g_cents
    with pytest.raises(ValueError, match="x_g.*g_cents"):
        model.loss(mel, g_cents=g_cents)  # missing x_g
