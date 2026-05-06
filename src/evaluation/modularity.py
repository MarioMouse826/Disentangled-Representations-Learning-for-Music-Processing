"""Modularity — Ridgeway & Mozer 2018.

    theta[i, k] = normalized MI(z_i, v_k)                  (in [0, 1])
    Mod_i = 1 - ( sum_k theta[i, k]^2 - theta_max^2 ) / ( theta_max^2 * (K-1) )
    Modularity = mean_i Mod_i

Per-dim modularity equals 1 when latent i correlates with at most one factor
(one-hot theta row), 0 when it correlates uniformly with all factors.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mutual_info_score

from src.evaluation._common import (
    as_discrete as _as_discrete,
    entropy_nats as _entropy,
    equal_frequency_bin as _equal_frequency_bin,
)


def compute_modularity_from_arrays(
    *,
    z: np.ndarray,
    factors: dict[str, np.ndarray],
    n_bins: int = 20,
    min_theta_max: float = 0.05,
) -> dict[str, Any]:
    """Modularity from array inputs."""
    if z.ndim != 2:
        raise ValueError(f"z must be (N, D), got shape {z.shape}")
    if not factors:
        raise ValueError("factors must be non-empty")
    n, d = z.shape
    for name, v in factors.items():
        if v.shape[0] != n:
            raise ValueError(
                f"factor {name!r} length {v.shape[0]} != z length {n}"
            )

    z_disc = np.stack(
        [_equal_frequency_bin(z[:, j], n_bins) for j in range(d)], axis=1
    )

    names = list(factors.keys())
    v_discs: list[np.ndarray] = []
    v_entropies: list[float] = []
    for name in names:
        v_disc = _as_discrete(factors[name], n_bins)
        v_discs.append(v_disc)
        v_entropies.append(_entropy(v_disc))
    k = len(names)

    # theta[i, k] = MI(z_i, v_k) / H(v_k); skip zero-entropy factors by
    # leaving their column at 0 (no contribution to the sum).
    theta = np.zeros((d, k), dtype=np.float64)
    for col in range(k):
        h_v = v_entropies[col]
        if h_v <= 0.0:
            continue
        for j in range(d):
            theta[j, col] = float(mutual_info_score(z_disc[:, j], v_discs[col])) / h_v

    mod_per_dim = np.zeros(d, dtype=np.float64)
    if k == 1:
        # Nothing to compete against — per-dim modularity is trivially 1.
        mod_per_dim[:] = 1.0
    else:
        # Inactive dims (theta_max < min_theta_max) encode no factor; treat
        # them as fully modular so finite-sample MI noise doesn't suppress
        # the metric below its theoretical range.
        for i in range(d):
            row = theta[i, :]
            theta_max = float(row.max())
            if theta_max < min_theta_max:
                mod_per_dim[i] = 1.0
                continue
            theta_max_sq = theta_max ** 2
            excess = float((row ** 2).sum() - theta_max_sq)
            # Clamp: sklearn.mutual_info_score is a finite-sample estimator
            # and can return MI slightly > H(v_k), producing theta > 1 and
            # pushing this ratio above 1 (score below 0). Clip to [0, 1].
            mod_per_dim[i] = float(
                np.clip(1.0 - excess / (theta_max_sq * (k - 1)), 0.0, 1.0)
            )

    return {
        "modularity": float(mod_per_dim.mean()),
        "mod_per_dim": mod_per_dim,
        "theta_matrix": theta,
        "factor_names": names,
    }
