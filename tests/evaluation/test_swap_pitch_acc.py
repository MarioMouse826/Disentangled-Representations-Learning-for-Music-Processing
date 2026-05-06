"""Tests for timbre-swap pitch accuracy (task-based perceptual eval).

Protocol: swap timbre z_s(x_b) onto content z_c(x_a); render via vocoder;
extract f0 from the rendered waveform with CREPE; compare to f0(x_a).
PitchAcc@50cents = fraction of frames within 50 cents of the target.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.evaluation.swap_pitch_acc import (
    bootstrap_ci,
    cents_error,
    frame_pitch_accuracy,
)


# -- cents_error ----------------------------------------------------------


def test_cents_error_zero_for_equal_freqs() -> None:
    f = torch.full((8,), 440.0)
    err = cents_error(f, f)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-4)


def test_cents_error_100_per_semitone() -> None:
    ref = torch.full((4,), 440.0)
    pred = torch.full((4,), 440.0 * 2 ** (1.0 / 12.0))  # +1 semitone
    err = cents_error(pred, ref)
    assert torch.allclose(err, torch.full_like(err, 100.0), atol=0.5)


def test_cents_error_sign_preserved_via_abs() -> None:
    # Function returns absolute cents error — sign-agnostic per the plan.
    ref = torch.full((4,), 440.0)
    up = torch.full((4,), 440.0 * 2 ** (1.0 / 12.0))
    down = torch.full((4,), 440.0 * 2 ** (-1.0 / 12.0))
    assert torch.allclose(cents_error(up, ref), cents_error(down, ref), atol=0.5)


def test_cents_error_ignores_unvoiced_zero_frames() -> None:
    # Frames with f0=0 (unvoiced) must not propagate NaN/Inf.
    ref = torch.tensor([440.0, 0.0, 440.0, 0.0])
    pred = torch.tensor([440.0, 440.0, 0.0, 0.0])
    err = cents_error(pred, ref)
    assert torch.isfinite(err).all()


# -- frame_pitch_accuracy -------------------------------------------------


def test_frame_accuracy_perfect_is_one() -> None:
    ref = torch.full((32,), 440.0)
    pred = ref.clone()
    voiced = torch.ones_like(ref, dtype=torch.bool)
    acc = frame_pitch_accuracy(pred, ref, voiced=voiced, tolerance_cents=50.0)
    assert acc == 1.0


def test_frame_accuracy_half_out_of_tolerance_is_half() -> None:
    ref = torch.full((32,), 440.0)
    pred = ref.clone()
    # Push the second half 200 cents off → out of 50-cent tolerance.
    pred[16:] = 440.0 * 2 ** (2.0 / 12.0)
    voiced = torch.ones_like(ref, dtype=torch.bool)
    acc = frame_pitch_accuracy(pred, ref, voiced=voiced, tolerance_cents=50.0)
    assert abs(acc - 0.5) < 1e-6


def test_frame_accuracy_requires_at_least_one_voiced_frame() -> None:
    ref = torch.full((8,), 440.0)
    pred = ref.clone()
    voiced = torch.zeros_like(ref, dtype=torch.bool)
    with pytest.raises(ValueError, match="voiced"):
        frame_pitch_accuracy(pred, ref, voiced=voiced, tolerance_cents=50.0)


# -- bootstrap_ci ---------------------------------------------------------


def test_bootstrap_ci_mean_is_empirical() -> None:
    rng = np.random.default_rng(0)
    vals = rng.uniform(0.7, 0.9, size=200).astype(np.float64)
    out = bootstrap_ci(vals, n_boot=500, alpha=0.05, random_state=0)
    assert abs(out["mean"] - vals.mean()) < 1e-9
    assert out["ci_low"] < out["mean"] < out["ci_high"]


def test_bootstrap_ci_width_shrinks_with_n() -> None:
    rng = np.random.default_rng(1)
    small = rng.normal(0.8, 0.05, size=20)
    large = rng.normal(0.8, 0.05, size=2000)
    w_small = bootstrap_ci(small, n_boot=200, alpha=0.05, random_state=0)
    w_large = bootstrap_ci(large, n_boot=200, alpha=0.05, random_state=0)
    assert (w_large["ci_high"] - w_large["ci_low"]) < (w_small["ci_high"] - w_small["ci_low"])


def test_bootstrap_ci_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci(np.array([], dtype=np.float64), n_boot=10)


# -- end-to-end with a mock CREPE -----------------------------------------


def test_compute_swap_pitch_accuracy_runs_with_mock_crepe() -> None:
    """End-to-end smoke: fake model + fake vocoder + fake pitch extractor.

    The fake extractor returns the ground-truth f0 for `x_a` so the
    reported accuracy should be ~1.0 regardless of the rendered waveform.
    """
    from src.evaluation.swap_pitch_acc import compute_swap_pitch_accuracy

    class _Model(torch.nn.Module):
        def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            b = x.shape[0]
            return {
                "mu_s": torch.zeros(b, 2),
                "mu_c": torch.zeros(b, 2),
            }

        def decode(self, z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
            return torch.zeros(z_s.shape[0], 1, 8, 16)

    def vocoder(mel: torch.Tensor) -> torch.Tensor:
        # (B, T_wave)
        return torch.zeros(mel.shape[0], 1024)

    def extractor(wave: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, F) f0 and (B, F) voiced mask.
        b = wave.shape[0]
        return torch.full((b, 4), 440.0), torch.ones(b, 4, dtype=torch.bool)

    pairs = [
        {
            "x_a": torch.randn(1, 1, 8, 16),
            "x_b": torch.randn(1, 1, 8, 16),
            "f0_a": torch.full((4,), 440.0),
            "voiced_a": torch.ones(4, dtype=torch.bool),
        }
        for _ in range(6)
    ]

    out = compute_swap_pitch_accuracy(
        model=_Model(),
        vocoder=vocoder,
        pairs=pairs,
        pitch_extractor=extractor,
        sample_rate=16000,
        tolerance_cents=50.0,
    )
    assert math.isclose(out["mean_accuracy"], 1.0, abs_tol=1e-6)
    assert out["ci95_low"] <= out["mean_accuracy"] <= out["ci95_high"]
    assert out["per_pair"].shape == (6,)
