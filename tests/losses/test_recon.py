from __future__ import annotations

import pytest
import torch

from src.losses.recon import l1_log_mel_loss, multi_scale_spectral_loss


# -- multi-scale spectral loss (waveform domain) --------------------------


def test_spectral_zero_on_identical_inputs() -> None:
    # Plan-critical: identical waveforms produce zero loss.
    x = torch.randn(2, 1, 16000)
    loss = multi_scale_spectral_loss(x, x.clone())
    assert abs(float(loss)) < 1e-5


def test_spectral_positive_on_shuffled() -> None:
    # Plan-critical: mismatched waveforms produce positive loss.
    torch.manual_seed(0)
    x = torch.randn(2, 1, 16000)
    x_shuf = x[:, :, torch.randperm(16000)]
    loss = multi_scale_spectral_loss(x, x_shuf)
    assert float(loss) > 0.0
    assert torch.isfinite(loss)


def test_spectral_accepts_2d_input() -> None:
    # Support both (B, T) and (B, 1, T) layouts.
    x = torch.randn(3, 8000)
    loss = multi_scale_spectral_loss(x, x.clone())
    assert abs(float(loss)) < 1e-5


def test_spectral_sums_across_scales() -> None:
    # Total loss equals sum of per-scale losses.
    torch.manual_seed(1)
    x = torch.randn(1, 1, 8000)
    y = torch.randn(1, 1, 8000)
    fft_sizes = (1024, 256)
    total = multi_scale_spectral_loss(x, y, fft_sizes=fft_sizes)

    per_scale_sum = 0.0
    for n in fft_sizes:
        l = multi_scale_spectral_loss(x, y, fft_sizes=(n,))
        per_scale_sum += float(l)
    assert abs(float(total) - per_scale_sum) < 1e-5


def test_spectral_differentiable() -> None:
    x = torch.randn(1, 1, 4000, requires_grad=True)
    y = torch.randn(1, 1, 4000, requires_grad=True)
    loss = multi_scale_spectral_loss(x, y, fft_sizes=(512, 128))
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert x.grad.abs().sum() > 0


def test_spectral_device_preservation() -> None:
    from src.utils.device import best_device

    dev = best_device()
    x = torch.randn(1, 1, 4000, device=dev)
    y = torch.randn(1, 1, 4000, device=dev)
    loss = multi_scale_spectral_loss(x, y, fft_sizes=(512, 128))
    assert loss.device.type == dev.type


def test_spectral_default_scales_match_ddsp() -> None:
    # DDSP (Engel 2020 §3.2) prescribes {2048, 1024, 512, 256, 128, 64}.
    # Pin default so accidental shortening of the tuple doesn't silently
    # change experiment results.
    from src.losses.recon import _DEFAULT_FFT_SIZES
    assert _DEFAULT_FFT_SIZES == (2048, 1024, 512, 256, 128, 64)


def test_spectral_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        multi_scale_spectral_loss(torch.randn(1, 1, 4000), torch.randn(1, 1, 3000))


def test_spectral_bad_fft_size_raises() -> None:
    with pytest.raises(ValueError, match="fft_sizes"):
        multi_scale_spectral_loss(
            torch.randn(1, 4000), torch.randn(1, 4000), fft_sizes=()
        )
    with pytest.raises(ValueError, match="fft_sizes"):
        multi_scale_spectral_loss(
            torch.randn(1, 4000), torch.randn(1, 4000), fft_sizes=(0,)
        )


def test_spectral_bad_rank_raises() -> None:
    with pytest.raises(ValueError, match="2-D or 3-D"):
        multi_scale_spectral_loss(torch.randn(100), torch.randn(100))
    with pytest.raises(ValueError, match="2-D or 3-D"):
        multi_scale_spectral_loss(
            torch.randn(1, 1, 1, 100), torch.randn(1, 1, 1, 100)
        )


# -- L1 log-mel loss (mel domain) ----------------------------------------


def test_mel_zero_on_identical() -> None:
    m = torch.randn(2, 1, 128, 401)
    loss = l1_log_mel_loss(m, m.clone())
    assert float(loss) == 0.0


def test_mel_positive_on_shuffled() -> None:
    a = torch.randn(2, 1, 128, 401)
    b = torch.randn(2, 1, 128, 401)
    loss = l1_log_mel_loss(a, b)
    assert float(loss) > 0.0
    assert torch.isfinite(loss)


def test_mel_is_l1() -> None:
    a = torch.randn(1, 1, 16, 20)
    b = torch.randn(1, 1, 16, 20)
    manual = (a - b).abs().mean()
    assert torch.allclose(l1_log_mel_loss(a, b), manual, atol=1e-6)


def test_mel_differentiable() -> None:
    a = torch.randn(1, 1, 8, 10, requires_grad=True)
    b = torch.randn(1, 1, 8, 10, requires_grad=True)
    loss = l1_log_mel_loss(a, b)
    loss.backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


def test_mel_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        l1_log_mel_loss(torch.randn(1, 1, 128, 401), torch.randn(1, 1, 128, 400))
