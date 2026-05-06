"""Tests for FactorVAE score — Kim & Mnih 2018.

Procedure:
    1. Pick factor k, sample L minibatches with constant v_k.
    2. Per minibatch, compute Var_i(z) / Var_global(z) → find argmin dim.
    3. Majority-vote classifier (argmin dim -> factor) → accuracy on
       held-out votes = FactorVAE score.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.factor_vae_score import compute_factor_vae_from_arrays


def test_factor_vae_perfect_disentanglement_scores_high() -> None:
    rng = np.random.default_rng(0)
    n = 4096
    v0 = rng.integers(0, 5, size=n)
    v1 = rng.integers(0, 5, size=n)
    # Signal dims encode factors exactly; other dims are noise.
    z = np.column_stack([
        v0.astype(np.float32),
        v1.astype(np.float32),
        rng.standard_normal(n),
        rng.standard_normal(n),
    ])
    out = compute_factor_vae_from_arrays(
        z=z,
        factors={"v0": v0, "v1": v1},
        n_votes=200,
        batch_size=64,
        random_state=0,
    )
    assert out["factor_vae_score"] > 0.8
    assert out["classifier_table"].shape == (4, 2)


def test_factor_vae_random_latents_score_low() -> None:
    rng = np.random.default_rng(1)
    n = 4096
    v0 = rng.integers(0, 5, size=n)
    v1 = rng.integers(0, 5, size=n)
    z = rng.standard_normal((n, 4)).astype(np.float32)
    out = compute_factor_vae_from_arrays(
        z=z,
        factors={"v0": v0, "v1": v1},
        n_votes=200,
        batch_size=64,
        random_state=0,
    )
    # Random latents carry no information about factors → near chance (0.5).
    assert out["factor_vae_score"] < 0.7


def test_factor_vae_rejects_continuous_factor() -> None:
    rng = np.random.default_rng(2)
    n = 256
    pitch = rng.uniform(0, 1, size=n).astype(np.float32)
    z = rng.standard_normal((n, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="discrete"):
        compute_factor_vae_from_arrays(
            z=z,
            factors={"pitch": pitch},
            n_votes=10,
            batch_size=8,
            random_state=0,
        )


def test_factor_vae_skips_factors_with_single_value() -> None:
    rng = np.random.default_rng(3)
    n = 512
    v = rng.integers(0, 4, size=n)
    const = np.zeros(n, dtype=np.int64)
    z = np.column_stack([v.astype(np.float32), rng.standard_normal(n)])
    with pytest.warns(RuntimeWarning, match="constant"):
        out = compute_factor_vae_from_arrays(
            z=z,
            factors={"v": v, "const": const},
            n_votes=64,
            batch_size=32,
            random_state=0,
        )
    assert "v" in out["kept_factors"]
    assert "const" not in out["kept_factors"]
