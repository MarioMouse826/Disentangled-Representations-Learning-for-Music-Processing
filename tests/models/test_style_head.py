from __future__ import annotations

import pytest
import torch

from src.models.style_head import StyleHead


def test_schema_and_shapes() -> None:
    head = StyleHead(in_channels=512, d_s=32)
    h = torch.randn(3, 512, 13)
    out = head(h)
    assert set(out.keys()) == {"mu_s", "logvar_s", "attn_weights"}
    assert out["mu_s"].shape == (3, 32)
    assert out["logvar_s"].shape == (3, 32)
    assert out["attn_weights"].shape == (3, 13)


def test_attention_weights_sum_to_one() -> None:
    head = StyleHead(in_channels=512, d_s=32)
    h = torch.randn(5, 512, 13)
    out = head(h)
    # Softmax over time axis → sum per-batch-item = 1.
    sums = out["attn_weights"].sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Non-negative attention.
    assert torch.all(out["attn_weights"] >= 0)


def test_deterministic_given_input() -> None:
    head = StyleHead(in_channels=64, d_s=16)
    h = torch.randn(2, 64, 10)
    out1 = head(h)
    out2 = head(h)
    for k in out1:
        assert torch.equal(out1[k], out2[k])


def test_differentiable_all_params() -> None:
    head = StyleHead(in_channels=64, d_s=16)
    h = torch.randn(2, 64, 10, requires_grad=True)
    out = head(h)
    (out["mu_s"].sum() + out["logvar_s"].sum()).backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    for n, p in head.named_parameters():
        assert p.grad is not None, f"no grad on {n}"
        assert torch.isfinite(p.grad).all()


def test_uniform_attention_equals_mean_pool() -> None:
    # If the attention score network outputs a constant, softmax gives
    # uniform weights → pooled vector equals the arithmetic mean over time.
    # Zero out the query + pre-projection so all scores are identical.
    head = StyleHead(in_channels=32, d_s=8)
    with torch.no_grad():
        head.score_proj.weight.zero_()
        head.score_proj.bias.zero_()
        head.query.zero_()
    h = torch.randn(2, 32, 7)
    out = head(h)
    # Uniform attention over 7 time steps.
    expected_w = torch.full((2, 7), 1.0 / 7)
    assert torch.allclose(out["attn_weights"], expected_w, atol=1e-5)


def test_invariance_to_time_permutation_when_uniform() -> None:
    # With uniform attention, the pooled vector is the mean over time and
    # thus permutation-invariant on the time axis. This is the *architectural*
    # inductive bias; the full L_inv loss (Task 3.5) enforces this globally.
    head = StyleHead(in_channels=32, d_s=8)
    with torch.no_grad():
        head.score_proj.weight.zero_()
        head.score_proj.bias.zero_()
        head.query.zero_()
    h = torch.randn(1, 32, 6)
    perm = torch.randperm(6)
    h_perm = h[:, :, perm]
    out_a = head(h)
    out_b = head(h_perm)
    assert torch.allclose(out_a["mu_s"], out_b["mu_s"], atol=1e-5)


def test_logvar_clamped() -> None:
    # Extreme inputs must not produce unbounded logvar — stability clamp
    # keeps downstream KL / reparam numerics sane.
    head = StyleHead(in_channels=8, d_s=4, logvar_clamp=(-10.0, 10.0))
    h = torch.randn(1, 8, 3) * 1e6
    out = head(h)
    assert torch.all(out["logvar_s"] >= -10.0 - 1e-4)
    assert torch.all(out["logvar_s"] <= 10.0 + 1e-4)


def test_config_roundtrip() -> None:
    head = StyleHead(in_channels=128, d_s=24, hidden_score=64, logvar_clamp=(-8.0, 8.0))
    cfg = head.config
    assert cfg["in_channels"] == 128
    assert cfg["d_s"] == 24
    assert cfg["hidden_score"] == 64
    assert cfg["logvar_clamp"] == (-8.0, 8.0)


def test_device_preservation() -> None:
    from src.utils.device import best_device

    dev = best_device()
    head = StyleHead(in_channels=64, d_s=16).to(dev)
    h = torch.randn(2, 64, 10, device=dev)
    out = head(h)
    for v in out.values():
        assert v.device.type == dev.type


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="in_channels"):
        StyleHead(in_channels=0, d_s=16)
    with pytest.raises(ValueError, match="d_s"):
        StyleHead(in_channels=64, d_s=-1)
    with pytest.raises(ValueError, match="hidden_score"):
        StyleHead(in_channels=64, d_s=16, hidden_score=0)
    with pytest.raises(ValueError, match="logvar_clamp"):
        StyleHead(in_channels=64, d_s=16, logvar_clamp=(5.0, -5.0))


def test_rejects_wrong_input_dims() -> None:
    head = StyleHead(in_channels=64, d_s=16)
    with pytest.raises(ValueError, match="3-D"):
        head(torch.randn(2, 64))
    with pytest.raises(ValueError, match="in_channels"):
        head(torch.randn(2, 32, 10))
