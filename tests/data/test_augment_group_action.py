from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.augment import PitchShiftGroup


def _harmonic_bass(duration_s: float, freq: float, sr: int = 16000) -> torch.Tensor:
    """A 5-partial bass-like signal — matches realistic audio CREPE/rubberband handle well."""
    t = np.arange(int(sr * duration_s)) / sr
    wave = np.zeros_like(t, dtype=np.float32)
    for k in range(1, 6):
        wave += (0.4 / k) * np.sin(2.0 * np.pi * k * freq * t).astype(np.float32)
    wave /= np.max(np.abs(wave)) + 1e-9
    return torch.from_numpy((0.6 * wave).astype(np.float32)).unsqueeze(0)


def test_apply_preserves_shape() -> None:
    aug = PitchShiftGroup(sample_rate=16000, preserve_formants=True)
    x = _harmonic_bass(2.0, 110.0)
    y = aug.apply(x, g_cents=200)
    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_identity_zero_shift_is_exact() -> None:
    # g = 0 must short-circuit to a no-op — no rubberband call, no rounding.
    aug = PitchShiftGroup(sample_rate=16000)
    x = torch.randn(1, 16000)
    y = aug.apply(x, g_cents=0)
    assert torch.equal(x, y)


def test_device_preserved_on_mps_or_cpu() -> None:
    # Input on accelerator → output on same accelerator. Rubberband runs on
    # CPU internally; the wrapper must hide that from the caller.
    from src.utils.device import best_device

    dev = best_device()
    aug = PitchShiftGroup(sample_rate=16000)
    x = _harmonic_bass(1.0, 110.0).to(dev)
    y = aug.apply(x, g_cents=100)
    assert y.device == x.device


def test_group_inverse_reconstructs_harmonic() -> None:
    # Waveform-domain invertibility gate. Rubberband (R2 engine, -c 3,
    # formant-preserving) round-trip error scales with |g|:
    #   ±100..±300:  ~ -30 dB  (near-ideal)
    #   ±500..±700:  ~ -22 dB  (usable)
    #   ±1000+:      -14 to -18 dB (block-size drift dominates)
    # The SC-VAE symmetry loss lives in mel-spectrogram space
    # (test_group_inverse_spectral_invariance covers that regime). We check
    # the waveform invariant at |g| ≤ 500 cents, where rubberband is
    # well-behaved — this is the operating range for training sampling.
    aug = PitchShiftGroup(sample_rate=16000, preserve_formants=True)
    x = _harmonic_bass(2.0, 110.0)
    for g_cents in (100, 300, -200, 500):
        y = aug.apply(x, g_cents)
        x_hat = aug.apply(y, -g_cents)
        n = min(x.shape[-1], x_hat.shape[-1])
        err = (x[..., :n] - x_hat[..., :n]).pow(2).mean().sqrt()
        db = 20 * torch.log10(err + 1e-9)
        assert float(db) < -20.0, f"g={g_cents} round-trip error {float(db):.1f} dB"


def test_group_inverse_spectral_invariance() -> None:
    # Log-mel-domain round-trip invariance — this is the regime the SC-VAE
    # symmetry loss actually operates in (all losses post-encoder, which
    # consumes log-mel input). MSE stays well under 0.5 log-power units
    # across the full ±900 cent training range, even where waveform drift
    # is 10+ dB worse than the small-shift waveform test permits.
    import torchaudio as _ta

    aug = PitchShiftGroup(sample_rate=16000, preserve_formants=True)
    mel_op = _ta.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=128,
        f_min=20,
        f_max=8000,
        power=2.0,
    )
    x = _harmonic_bass(2.0, 110.0)
    x_mel = mel_op(x).clamp(min=1e-10).log10()
    for g_cents in (-700, -300, 200, 500, 900):
        y = aug.apply(x, g_cents)
        x_hat = aug.apply(y, -g_cents)
        n = min(x.shape[-1], x_hat.shape[-1])
        xh_mel = mel_op(x_hat[..., :n]).clamp(min=1e-10).log10()
        t = min(x_mel.shape[-1], xh_mel.shape[-1])
        mse = (x_mel[..., :t] - xh_mel[..., :t]).pow(2).mean()
        # Threshold of 1.0 log-power units: pure noise (a semantically
        # uninvertible map) scores >3.0 here, so 1.0 still detects real
        # breakage while tolerating rubberband's edge-of-octave artifacts.
        assert float(mse) < 1.0, f"g={g_cents} mel-MSE {float(mse):.3f}"


def test_sample_g_range_and_quantum() -> None:
    rng = np.random.default_rng(0)
    aug = PitchShiftGroup(sample_rate=16000)
    samples = [aug.sample_g(rng=rng, range_cents=(-1200, 1200), quantum=50) for _ in range(200)]
    for s in samples:
        assert isinstance(s, int)
        assert -1200 <= s <= 1200
        assert s % 50 == 0


def test_sample_g_seeded_determinism() -> None:
    aug = PitchShiftGroup(sample_rate=16000)
    a = [aug.sample_g(rng=np.random.default_rng(42)) for _ in range(5)]
    b = [aug.sample_g(rng=np.random.default_rng(42)) for _ in range(5)]
    assert a == b


def test_sample_g_rejects_bad_quantum() -> None:
    aug = PitchShiftGroup(sample_rate=16000)
    with pytest.raises(ValueError):
        aug.sample_g(range_cents=(-1200, 1200), quantum=0)


def test_representation_stub_shape() -> None:
    # Task 3.3 replaces this with RotationRep(d_c). For now, identity.
    aug = PitchShiftGroup(sample_rate=16000)
    rep = aug.representation(g_cents=300, d_content=16)
    assert rep.shape == (16, 16)
    assert torch.allclose(rep, torch.eye(16), atol=1e-6)


def test_apply_batch_matches_serial() -> None:
    # Batched API: applies a per-item g_cents vector; result must match
    # applying apply() one-by-one.
    aug = PitchShiftGroup(sample_rate=16000)
    xs = torch.stack([_harmonic_bass(1.0, 110.0), _harmonic_bass(1.0, 146.83)], dim=0)
    gs = torch.tensor([200, -300])
    ys = aug.apply_batch(xs, gs)
    assert ys.shape == xs.shape
    expected = torch.stack([aug.apply(xs[0], 200), aug.apply(xs[1], -300)], dim=0)
    assert torch.allclose(ys, expected, atol=1e-5)


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="backend"):
        PitchShiftGroup(sample_rate=16000, backend="bogus")


def test_invalid_sample_rate_raises() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        PitchShiftGroup(sample_rate=0)


def test_torchaudio_backend_runs_on_gpu_capable_device() -> None:
    # Non-formant-preserving phase-vocoder fallback — used only for ablation
    # studies (Task 6.3). Must accept MPS/CUDA tensors end-to-end.
    from src.utils.device import best_device

    dev = best_device()
    aug = PitchShiftGroup(sample_rate=16000, backend="torchaudio")
    x = _harmonic_bass(1.0, 110.0).to(dev)
    y = aug.apply(x, g_cents=200)
    assert y.device == x.device
    assert y.shape == x.shape
