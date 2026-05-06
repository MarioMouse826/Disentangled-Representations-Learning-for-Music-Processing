"""Tests for Modularity — Ridgeway & Mozer 2018."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.modularity import compute_modularity_from_arrays


def test_modularity_perfect_case_scores_high() -> None:
    rng = np.random.default_rng(0)
    n = 2048
    v0 = rng.integers(0, 8, size=n)
    v1 = rng.integers(0, 8, size=n)
    z = np.column_stack([
        v0.astype(np.float32),
        v1.astype(np.float32),
        rng.standard_normal(n),
        rng.standard_normal(n),
    ])
    out = compute_modularity_from_arrays(
        z=z, factors={"v0": v0, "v1": v1}, n_bins=20
    )
    assert out["modularity"] > 0.8
    assert out["mod_per_dim"].shape == (4,)


def test_modularity_entangled_case_scores_lower() -> None:
    rng = np.random.default_rng(1)
    n = 2048
    v0 = rng.integers(0, 5, size=n)
    v1 = rng.integers(0, 5, size=n)
    # Every latent dim mixes both factors equally → modularity per dim low.
    mix = v0.astype(np.float32) + v1.astype(np.float32)
    z = np.column_stack([mix + 0.01 * rng.standard_normal(n) for _ in range(4)])
    entangled = compute_modularity_from_arrays(
        z=z, factors={"v0": v0, "v1": v1}, n_bins=20
    )
    perfect = compute_modularity_from_arrays(
        z=np.column_stack([
            v0.astype(np.float32),
            v1.astype(np.float32),
            rng.standard_normal(n),
        ]),
        factors={"v0": v0, "v1": v1},
        n_bins=20,
    )
    assert entangled["modularity"] < perfect["modularity"]


def test_modularity_rejects_empty_factors() -> None:
    with pytest.raises(ValueError, match="factors"):
        compute_modularity_from_arrays(
            z=np.zeros((8, 2), dtype=np.float32), factors={}, n_bins=10
        )


def test_modularity_single_factor_collapses_to_one() -> None:
    # With K=1 the formula's denominator vanishes; implementation should
    # return 1.0 for each dim (nothing to compete against).
    rng = np.random.default_rng(2)
    n = 256
    v = rng.integers(0, 4, size=n)
    z = rng.standard_normal((n, 3)).astype(np.float32)
    out = compute_modularity_from_arrays(
        z=z, factors={"v": v}, n_bins=10
    )
    assert np.allclose(out["mod_per_dim"], 1.0)
