"""Tests for DDSP decoder arm (plan Task 7.1).

Reference: Engel et al. 2020 (arXiv:2001.04643). Harmonic-plus-noise
synthesizer conditioned on:
    z_c (time-resolved) -> f0(t), loudness(t)  via 1D conv heads
    z_s (time-pooled)   -> harmonic_dist, noise_filter  via 2-layer MLP
"""
from __future__ import annotations

import torch

from src.models.ddsp_decoder import DDSPDecoder, DDSPHead


def test_ddsp_head_output_shapes() -> None:
    d_s, d_c, T = 4, 4, 64
    head = DDSPHead(
        d_s=d_s, d_c=d_c, n_harmonics=16, n_noise_taps=65, f0_max_hz=2000.0
    )
    z_s = torch.randn(2, d_s)
    z_c = torch.randn(2, d_c, T)
    out = head(z_s, z_c)
    assert out["f0_hz"].shape == (2, T)
    assert out["loudness"].shape == (2, T)
    assert out["harmonic_dist"].shape == (2, 16)
    assert out["noise_filter"].shape == (2, 65)
    # harmonic_dist sums to 1 across K (softmax).
    s = out["harmonic_dist"].sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_ddsp_head_f0_bounded_by_max_hz() -> None:
    head = DDSPHead(d_s=2, d_c=2, n_harmonics=8, n_noise_taps=17, f0_max_hz=1500.0)
    z_s = torch.randn(1, 2)
    z_c = torch.randn(1, 2, 16) * 100.0  # saturate the sigmoid
    out = head(z_s, z_c)
    assert (out["f0_hz"] >= 0).all()
    assert (out["f0_hz"] <= 1500.0).all()


def test_ddsp_decoder_output_shape() -> None:
    # Tiny config — 4 kHz / 0.25 s = 1024 samples.
    dec = DDSPDecoder(
        d_s=4, d_c=4, n_harmonics=16, n_noise_taps=65,
        sample_rate=4000, n_samples=1024, n_frames=64,
    )
    z_s = torch.randn(2, 4)
    z_c = torch.randn(2, 4, 64)
    wave = dec(z_s, z_c)
    assert wave.shape == (2, 1, 1024)
    assert torch.isfinite(wave).all()


def test_ddsp_decoder_plan_scale_shape() -> None:
    # Full scale per plan: (B, 1, 64000) at 16 kHz (4 s).
    dec = DDSPDecoder(
        d_s=8, d_c=8, n_harmonics=100, n_noise_taps=65,
        sample_rate=16000, n_samples=64000, n_frames=250,
    )
    z_s = torch.randn(1, 8)
    z_c = torch.randn(1, 8, 250)
    wave = dec(z_s, z_c)
    assert wave.shape == (1, 1, 64000)


def test_ddsp_decoder_gradients_flow_through_f0_and_alpha() -> None:
    dec = DDSPDecoder(
        d_s=4, d_c=4, n_harmonics=16, n_noise_taps=33,
        sample_rate=4000, n_samples=1024, n_frames=64,
    )
    z_s = torch.randn(1, 4, requires_grad=True)
    z_c = torch.randn(1, 4, 64, requires_grad=True)
    wave = dec(z_s, z_c)
    loss = wave.pow(2).mean()
    loss.backward()
    assert z_s.grad is not None and torch.isfinite(z_s.grad).all()
    assert z_c.grad is not None and torch.isfinite(z_c.grad).all()
    # Some parameter on the f0 head should accumulate grad.
    f0_head_grads = [p.grad for p in dec.head.f0_conv.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in f0_head_grads)
    # Some parameter on the MLP producing alpha should too.
    mlp_grads = [p.grad for p in dec.head.mlp_s.parameters() if p.grad is not None]
    assert any(g.abs().sum() > 0 for g in mlp_grads)


def test_ddsp_decoder_style_changes_output() -> None:
    # Swap z_s -> waveform must change (style conditions timbre).
    torch.manual_seed(0)
    dec = DDSPDecoder(
        d_s=4, d_c=4, n_harmonics=16, n_noise_taps=33,
        sample_rate=4000, n_samples=1024, n_frames=64,
    )
    dec.train(False)
    z_c = torch.randn(1, 4, 64)
    z_s_a = torch.randn(1, 4)
    z_s_b = torch.randn(1, 4) * 3.0
    wa = dec(z_s_a, z_c)
    wb = dec(z_s_b, z_c)
    assert not torch.allclose(wa, wb)


def test_ddsp_decoder_antialias_zeros_partials_above_nyquist() -> None:
    # When fundamental is already at nyquist, all higher partials must vanish.
    dec = DDSPDecoder(
        d_s=2, d_c=2, n_harmonics=10, n_noise_taps=17,
        sample_rate=4000, n_samples=512, n_frames=32,
    )
    # Force f0 near Nyquist by pushing z_c large positive; noise filter set
    # to zero effect by zeroing the MLP output path would be invasive —
    # instead, check internally that aliasing mask is applied.
    mask = dec._partial_mask(
        f0_hz=torch.full((1, 32), 1999.0), sample_rate=4000.0
    )
    # At f0=1999, nyquist=2000: only the first partial (k=1) survives.
    assert mask[..., 0].all()
    assert not mask[..., 1:].any()
