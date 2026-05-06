"""Tests for multi-window Signal-to-Reconstruction Ratio (SRR)."""
from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.srr import compute_srr


def test_srr_identical_is_inf() -> None:
    torch.manual_seed(0)
    x = torch.randn(16000)
    out = compute_srr(x, x)
    assert torch.isinf(out) and out > 0, f"expected +inf, got {out}"


def test_srr_zero_signal_on_nonzero_recon_is_neg_inf() -> None:
    # Reference is all zero → ratio 0 / e → log10(0) = -inf.
    x = torch.zeros(16000)
    x_hat = torch.randn(16000)
    out = compute_srr(x, x_hat)
    assert torch.isinf(out) and out < 0


def test_srr_noise_is_low() -> None:
    torch.manual_seed(1)
    x = torch.randn(16000)
    noise = torch.randn(16000)
    out = compute_srr(x, noise)
    # Pure noise recon should score near 0 dB — signal energy similar to
    # error energy when both are iid normal.
    assert out.item() < 5.0


def test_srr_small_perturbation_is_high() -> None:
    torch.manual_seed(2)
    x = torch.randn(16000)
    # x_hat = x + small noise → SRR should be high (>=20 dB).
    x_hat = x + 0.01 * torch.randn_like(x)
    out = compute_srr(x, x_hat)
    assert out.item() > 20.0


def test_srr_batched_input_returns_per_item() -> None:
    torch.manual_seed(3)
    B = 4
    x = torch.randn(B, 16000)
    x_hat = x + 0.01 * torch.randn_like(x)
    out = compute_srr(x, x_hat)
    assert out.shape == (B,)
    assert (out > 20.0).all()


def test_srr_custom_window_sizes() -> None:
    torch.manual_seed(4)
    x = torch.randn(8192)
    x_hat = x + 0.01 * torch.randn_like(x)
    out = compute_srr(x, x_hat, window_sizes=(256, 512))
    assert out.item() > 20.0


def test_srr_rejects_shape_mismatch() -> None:
    x = torch.randn(16000)
    x_hat = torch.randn(8000)
    with pytest.raises(ValueError, match="shape"):
        compute_srr(x, x_hat)


def test_srr_is_average_not_sum_across_windows() -> None:
    # Consistency check: single-window SRR and multi-window SRR with identical
    # window replicated three times should agree.
    torch.manual_seed(5)
    x = torch.randn(16000)
    x_hat = x + 0.05 * torch.randn_like(x)
    single = compute_srr(x, x_hat, window_sizes=(1024,))
    triple = compute_srr(x, x_hat, window_sizes=(1024, 1024, 1024))
    assert math.isclose(single.item(), triple.item(), rel_tol=1e-5)
