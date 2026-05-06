"""FactorVAE score — Kim & Mnih 2018. arXiv:1802.05983.

For each vote:
    1. Pick a discrete factor k and a value v of v_k uniformly.
    2. Sample a minibatch from the subset where v_k == v.
    3. Compute per-dim variance of z over the minibatch.
    4. Normalize by the global per-dim variance; record argmin as the "vote".

Train a majority-vote classifier on (argmin-dim -> factor-k) pairs and
report held-out classifier accuracy = FactorVAE score in [1/K, 1].

Only discrete factors are supported (the procedure fixes a factor value, so
the value must have a finite support with multiple occurrences).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def compute_factor_vae_from_arrays(
    *,
    z: np.ndarray,
    factors: dict[str, np.ndarray],
    n_votes: int = 800,
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """FactorVAE score from array inputs."""
    if z.ndim != 2:
        raise ValueError(f"z must be (N, D), got shape {z.shape}")
    if not factors:
        raise ValueError("factors must be non-empty")
    n, d = z.shape

    kept_factors: list[str] = []
    kept_arrays: list[np.ndarray] = []
    for name, v in factors.items():
        if v.shape[0] != n:
            raise ValueError(
                f"factor {name!r} length {v.shape[0]} != z length {n}"
            )
        if not np.issubdtype(v.dtype, np.integer):
            raise ValueError(
                f"factor {name!r} must be discrete (integer dtype); got {v.dtype}"
            )
        if np.unique(v).size < 2:
            warnings.warn(
                f"factor {name!r} is constant; skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        kept_factors.append(name)
        kept_arrays.append(v.astype(np.int64))
    if not kept_factors:
        raise ValueError("no factors with >= 2 unique values")

    k = len(kept_factors)
    # Normalize z per-dim by global std so dims with tiny native variance
    # don't dominate the argmin (Kim & Mnih 2018 §3.2).
    std = z.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    z_norm = (z - z.mean(axis=0)) / std

    rng = np.random.default_rng(random_state)
    votes = np.zeros(n_votes, dtype=np.int64)  # argmin dim
    labels = np.zeros(n_votes, dtype=np.int64)  # factor idx

    # Pre-index rows by (factor_idx, value) for fast minibatch sampling.
    factor_value_index: list[dict[int, np.ndarray]] = []
    for v in kept_arrays:
        mapping: dict[int, np.ndarray] = {}
        for val in np.unique(v).tolist():
            mapping[int(val)] = np.where(v == val)[0]
        factor_value_index.append(mapping)

    for t in range(n_votes):
        fi = int(rng.integers(0, k))
        value_map = factor_value_index[fi]
        # Skip values with too few samples; re-roll on degenerate picks.
        eligible = [val for val, idxs in value_map.items() if idxs.size >= 2]
        if not eligible:
            raise ValueError(
                f"factor {kept_factors[fi]!r} has no value with >= 2 samples"
            )
        val = eligible[int(rng.integers(0, len(eligible)))]
        idxs = value_map[val]
        take = min(batch_size, idxs.size)
        sel = rng.choice(idxs, size=take, replace=False)
        batch = z_norm[sel]
        batch_var = batch.var(axis=0)
        argmin = int(np.argmin(batch_var))
        votes[t] = argmin
        labels[t] = fi

    # Majority-vote classifier: per-argmin-dim, predict most-frequent factor.
    # Split into train/test; learn mapping on train, evaluate on test.
    perm = rng.permutation(n_votes)
    cut = int(n_votes * (1.0 - test_size))
    train_idx, test_idx = perm[:cut], perm[cut:]

    table = np.zeros((d, k), dtype=np.int64)  # (argmin-dim, factor) counts
    for t in train_idx:
        table[votes[t], labels[t]] += 1
    mapping = np.argmax(table, axis=1)  # argmin-dim -> predicted factor

    # Held-out accuracy; votes landing on an unseen dim fall back to the
    # global majority-factor.
    global_major = int(np.argmax(np.bincount(labels[train_idx], minlength=k)))
    seen_mask = table.sum(axis=1) > 0
    correct = 0
    for t in test_idx:
        pred = int(mapping[votes[t]]) if seen_mask[votes[t]] else global_major
        if pred == labels[t]:
            correct += 1
    score = float(correct / max(1, test_idx.size))

    return {
        "factor_vae_score": score,
        "classifier_table": table,
        "kept_factors": kept_factors,
    }
