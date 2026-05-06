"""Tests for paired bootstrap CI95 + paired t-test + Cliff's delta (plan Task 6.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.report import (
    cliffs_delta,
    compare_models,
    paired_bootstrap_diff,
    paired_ttest_pvalue,
)


# -- paired_bootstrap_diff -----------------------------------------------


def test_bootstrap_zero_diff_centered_at_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, size=50)
    y = x.copy()
    out = paired_bootstrap_diff(x, y, n_boot=1000, alpha=0.05, random_state=0)
    assert abs(out["mean_diff"]) < 1e-9
    assert out["ci_low"] <= 0 <= out["ci_high"]


def test_bootstrap_positive_diff_ci_excludes_zero() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.01, size=100)
    x = base + 0.5         # candidate clearly higher
    y = base
    out = paired_bootstrap_diff(x, y, n_boot=2000, alpha=0.05, random_state=0)
    assert out["mean_diff"] > 0.4
    assert out["ci_low"] > 0.0  # clearly significant


def test_bootstrap_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        paired_bootstrap_diff(
            np.zeros(5), np.zeros(6), n_boot=10, random_state=0
        )


def test_bootstrap_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_diff(
            np.array([]), np.array([]), n_boot=10, random_state=0
        )


# -- paired_ttest_pvalue -------------------------------------------------


def test_ttest_identical_samples_pvalue_is_one() -> None:
    x = np.array([0.3, 0.4, 0.5, 0.35, 0.45])
    p = paired_ttest_pvalue(x, x.copy())
    assert p >= 0.99 or np.isnan(p)


def test_ttest_clearly_different_is_tiny() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0.5, 0.01, size=30)
    b = a - 0.2
    p = paired_ttest_pvalue(a, b)
    assert p < 0.001


def test_ttest_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        paired_ttest_pvalue(np.zeros(4), np.zeros(5))


# -- cliffs_delta --------------------------------------------------------


def test_cliffs_delta_identical_samples_is_zero() -> None:
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert cliffs_delta(x, x.copy()) == 0.0


def test_cliffs_delta_strictly_greater_is_one() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-1.0, -2.0, -3.0])
    assert cliffs_delta(a, b) == 1.0


def test_cliffs_delta_strictly_less_is_minus_one() -> None:
    a = np.array([-1.0, -2.0, -3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, b) == -1.0


def test_cliffs_delta_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        cliffs_delta(np.array([]), np.array([1.0]))


# -- compare_models ------------------------------------------------------


def _run(model: str, seed: int, metric: float) -> dict:
    return {"model": model, "seed": seed, "metrics": {"mig": metric}}


def test_compare_models_returns_full_stats_dict() -> None:
    baseline = [_run("beta_vae", s, 0.3 + 0.01 * s) for s in range(5)]
    candidate = [_run("sc_vae", s, 0.5 + 0.01 * s) for s in range(5)]
    runs = baseline + candidate
    out = compare_models(
        runs=runs,
        metric="mig",
        baseline="beta_vae",
        candidate="sc_vae",
        n_boot=500,
        random_state=0,
    )
    assert set(out.keys()) >= {"delta_mean", "ci_low", "ci_high", "p_value", "cliffs_delta"}
    assert out["delta_mean"] > 0.15
    assert out["ci_low"] > 0.0
    assert out["p_value"] < 0.01
    assert out["cliffs_delta"] == 1.0  # candidate strictly above baseline


def test_compare_models_requires_matching_seed_count() -> None:
    baseline = [_run("a", s, 0.3) for s in range(3)]
    candidate = [_run("b", s, 0.5) for s in range(5)]
    with pytest.raises(ValueError, match="seed"):
        compare_models(
            runs=baseline + candidate,
            metric="mig",
            baseline="a",
            candidate="b",
            n_boot=10,
            random_state=0,
        )


def test_compare_models_rejects_missing_metric() -> None:
    baseline = [{"model": "a", "seed": s, "metrics": {}} for s in range(3)]
    candidate = [_run("b", s, 0.5) for s in range(3)]
    with pytest.raises(ValueError, match="metric"):
        compare_models(
            runs=baseline + candidate,
            metric="mig",
            baseline="a",
            candidate="b",
            n_boot=10,
            random_state=0,
        )
