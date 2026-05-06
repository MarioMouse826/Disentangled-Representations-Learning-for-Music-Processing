"""Mutual Information Gap (MIG) disentanglement metric.

Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2018. "Isolating Sources of
Disentanglement in Variational Autoencoders." NeurIPS 2018. arXiv:1802.04942.

    MIG = (1/K) * sum_k [ I(z_{i*_k}; v_k) - I(z_{i**_k}; v_k) ] / H(v_k)

Latent means mu(z|x) are discretized per-dim into `n_bins` equal-frequency
bins and joint histogram MI is estimated empirically. Continuous factors are
binned the same way; integer factors are used as-is if few unique values.
"""
from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import torch

from src.evaluation._common import (
    LATENT_KEYS_DEFAULT,
    as_discrete as _as_discrete,
    entropy_nats as _entropy,
    equal_frequency_bin as _equal_frequency_bin,
    extract_factors as _extract_factors,
    extract_latents as _extract_latents,
    mi_nats as _mi,
)


# ---------------------------------------------------------------------------
# MIG on arrays


def compute_mig_from_arrays(
    *,
    z: np.ndarray,
    factors: dict[str, np.ndarray],
    n_bins: int = 20,
) -> dict[str, Any]:
    """Compute MIG given latent means and factor values as numpy arrays.

    Args:
        z: `(N, D)` float array of latent means.
        factors: mapping `name -> (N,)` array. Integer factors with few
            uniques are used directly; continuous factors are
            equal-frequency binned.
        n_bins: bin count for continuous latents and continuous factors.

    Returns:
        `{"mig": float, "per_factor": {name: gap}, "mi_matrix": (D, K)}`.
        Factors with zero entropy are skipped with a RuntimeWarning.
    """
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

    kept: list[tuple[str, np.ndarray, float]] = []
    for name, v in factors.items():
        v_disc = _as_discrete(v, n_bins)
        h = _entropy(v_disc)
        if h <= 0.0:
            warnings.warn(
                f"factor {name!r} is constant (H=0); skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        kept.append((name, v_disc, h))

    if not kept:
        raise ValueError("no factors with positive entropy to evaluate")

    k = len(kept)
    mi_matrix = np.zeros((d, k), dtype=np.float64)
    for col, (_, v_disc, _) in enumerate(kept):
        for j in range(d):
            mi_matrix[j, col] = _mi(z_disc[:, j], v_disc)

    per_factor: dict[str, float] = {}
    gaps: list[float] = []
    for col, (name, _, h_v) in enumerate(kept):
        col_mi = mi_matrix[:, col]
        order = np.argsort(col_mi)[::-1]
        top1 = col_mi[order[0]]
        top2 = col_mi[order[1]] if d >= 2 else 0.0
        gap = float((top1 - top2) / h_v)
        per_factor[name] = gap
        gaps.append(gap)

    return {
        "mig": float(np.mean(gaps)),
        "per_factor": per_factor,
        "mi_matrix": mi_matrix,
    }


# ---------------------------------------------------------------------------
# MIG on model + dataloader


@torch.no_grad()
def compute(
    *,
    model: torch.nn.Module,
    dataloader: Sequence[dict[str, Any]] | Any,
    factor_keys: Sequence[str],
    n_bins: int = 20,
    latent_keys: Sequence[str] = LATENT_KEYS_DEFAULT,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Compute MIG by running `model.encode` over `dataloader`.

    Collects latent means across all batches, stacks factor values, then
    delegates to :func:`compute_mig_from_arrays`.
    """
    if device is not None:
        model = model.to(device)
    # Switch module to inference mode (disables dropout/batchnorm updates).
    model.train(False)

    factor_keys = list(factor_keys)
    z_chunks: list[np.ndarray] = []
    factor_chunks: dict[str, list[np.ndarray]] = {k: [] for k in factor_keys}

    for batch in dataloader:
        x = batch["x"]
        if device is not None:
            x = x.to(device)
        enc = model.encode(x)
        z = _extract_latents(enc, latent_keys).detach().cpu().numpy()
        z_chunks.append(z)

        facs = _extract_factors(batch, factor_keys)
        for k, v in facs.items():
            factor_chunks[k].append(v.detach().cpu().numpy())

    z_all = np.concatenate(z_chunks, axis=0)
    factors = {k: np.concatenate(v, axis=0) for k, v in factor_chunks.items()}
    return compute_mig_from_arrays(z=z_all, factors=factors, n_bins=n_bins)
