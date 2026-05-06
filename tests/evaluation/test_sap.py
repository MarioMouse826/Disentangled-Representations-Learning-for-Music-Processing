"""Tests for SAP (Separated Attribute Predictability).

Kumar, A., Sattigeri, P., Balakrishnan, A., 2018. "Variational Inference of
Disentangled Latent Concepts from Unlabeled Observations." ICLR.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.sap import compute_sap_from_arrays


_FAST: dict = {"max_iter": 2000}


def test_sap_perfect_disentanglement_scores_high() -> None:
    rng = np.random.default_rng(0)
    n = 2048
    # 4 classes: LinearSVC 1-D OvR caps signal-dim accuracy at ~1.0 and
    # noise-dim accuracy at ~1/K = 0.25, giving a realistic SAP gap of ~0.75.
    # At 6+ classes, 1-D OvR decision boundaries fail on mid-range classes
    # (signal accuracy collapses to ~0.67), masking the disentanglement signal.
    v0 = rng.integers(0, 4, size=n)
    v1 = rng.integers(0, 4, size=n)
    z = np.column_stack([
        v0.astype(np.float32) + 0.01 * rng.standard_normal(n),
        v1.astype(np.float32) + 0.01 * rng.standard_normal(n),
        rng.standard_normal(n),
        rng.standard_normal(n),
    ])
    out = compute_sap_from_arrays(
        z=z,
        factors={"v0": v0, "v1": v1},
        factor_types={"v0": "discrete", "v1": "discrete"},
        svc_kwargs=_FAST,
        random_state=0,
    )
    assert out["sap"] > 0.5
    assert out["score_matrix"].shape == (4, 2)
    # Top-scoring dim per factor is the signal dim.
    assert int(np.argmax(out["score_matrix"][:, 0])) == 0
    assert int(np.argmax(out["score_matrix"][:, 1])) == 1


def test_sap_continuous_factor_uses_r_squared() -> None:
    rng = np.random.default_rng(1)
    n = 1024
    pitch = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    z = np.column_stack([pitch, rng.standard_normal(n), rng.standard_normal(n)])
    out = compute_sap_from_arrays(
        z=z,
        factors={"pitch": pitch},
        factor_types={"pitch": "continuous"},
        random_state=0,
    )
    assert out["sap"] > 0.5


def test_sap_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        compute_sap_from_arrays(
            z=np.zeros((10, 3), dtype=np.float32),
            factors={"bad": np.zeros(9, dtype=np.int64)},
            random_state=0,
        )


def test_sap_returns_per_factor_gaps() -> None:
    rng = np.random.default_rng(2)
    n = 512
    v = rng.integers(0, 4, size=n)
    z = np.column_stack([v.astype(np.float32), rng.standard_normal(n)])
    out = compute_sap_from_arrays(
        z=z,
        factors={"v": v},
        random_state=0,
        svc_kwargs=_FAST,
    )
    assert "per_factor" in out and "v" in out["per_factor"]
    assert 0.0 <= out["per_factor"]["v"] <= 1.0
