"""Shared helpers for evaluation metrics — binning, MI, latent/factor
extraction. Kept internal (module-private) so public metric modules are the
stable API surface.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import mutual_info_score


# ---------------------------------------------------------------------------
# Binning + entropy


def equal_frequency_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize a 1-D array into `n_bins` equal-frequency bins.

    Duplicate quantile edges (common with heavy-tailed or integer-valued
    data) are deduplicated before `np.digitize` to avoid inconsistent
    bin assignment on ties.
    """
    if x.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {x.shape}")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, quantiles)
    # Deduplicate interior edges; tied quantiles collapse to fewer bins.
    interior = np.unique(edges[1:-1])
    labels = np.digitize(x, interior, right=False)
    max_label = max(interior.size, 0)  # labels in [0, interior.size]
    return np.clip(labels, 0, max_label).astype(np.int64)


def as_discrete(v: np.ndarray, n_bins: int) -> np.ndarray:
    """Integer labels for `v`. Integer arrays with <= n_bins uniques are
    dense-relabeled; other arrays are equal-frequency binned.
    """
    if v.ndim != 1:
        raise ValueError(f"factor must be 1-D, got shape {v.shape}")
    if np.issubdtype(v.dtype, np.integer):
        uniq = np.unique(v)
        if uniq.size <= n_bins:
            mapping = {u: i for i, u in enumerate(uniq.tolist())}
            return np.array([mapping[int(x)] for x in v], dtype=np.int64)
    return equal_frequency_bin(v.astype(np.float64), n_bins)


def mi_nats(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information in nats via empirical joint histogram."""
    return float(mutual_info_score(a, b))


def entropy_nats(labels: np.ndarray) -> float:
    """Empirical entropy (nats) of an integer label array."""
    _, counts = np.unique(labels, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Latent + factor extraction from model outputs


LATENT_KEYS_DEFAULT: tuple[str, ...] = ("mu", "mu_s", "mu_c")


def extract_latents(
    out: dict[str, torch.Tensor], keys: Sequence[str]
) -> torch.Tensor:
    """Concatenate latent-mean tensors from an encoder output dict.

    Accepts keys like `mu`, `mu_s`, `mu_c`; missing keys are skipped. Tensors
    with 3-D shape `(B, D, T)` are averaged over `T` to `(B, D)`.
    """
    pieces: list[torch.Tensor] = []
    for k in keys:
        t = out.get(k)
        if t is None:
            continue
        if t.ndim == 3:
            t = t.mean(dim=-1)
        elif t.ndim != 2:
            raise ValueError(
                f"latent {k!r} must be (B,D) or (B,D,T), got shape {tuple(t.shape)}"
            )
        pieces.append(t)
    if not pieces:
        raise ValueError(
            f"encoder output has none of expected latent keys {tuple(keys)}; "
            f"got keys {list(out.keys())}"
        )
    return torch.cat(pieces, dim=1)


def extract_factors(
    batch: dict[str, Any], factor_keys: Iterable[str]
) -> dict[str, torch.Tensor]:
    """Pull named factors from a batch's `labels` dict."""
    labels = batch.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("batch must contain a `labels` dict")
    out: dict[str, torch.Tensor] = {}
    for k in factor_keys:
        if k not in labels:
            raise KeyError(f"factor {k!r} missing from batch['labels']")
        v = labels[k]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        out[k] = v
    return out
