"""DCI Disentanglement / Completeness / Informativeness metric.

Eastwood, C., Williams, C.K.I., 2018. "A Framework for the Quantitative
Evaluation of Disentangled Representations." ICLR 2018.

For each factor v_k, fit a Gradient Boosted Tree regressor (continuous) or
classifier (discrete) that predicts v_k from z. The per-dim importances
`R[i, k]` form the (D, K) importance matrix; D, C, I are derived from it:

    P_dim[i, k]  = R[i, k] / sum_k R[i, :]
    P_fact[i, k] = R[i, k] / sum_i R[:, k]
    D_i          = 1 - H_K(P_dim[i, :])         # one-dim disentanglement
    D            = sum_i rho_i * D_i,  rho_i = sum_k R[i, k] / R.sum()
    C_k          = 1 - H_D(P_fact[:, k])        # per-factor completeness
    I_k          = classif_acc  (discrete)  or  1 - NRMSE  (continuous)
                   on a held-out split

`H_K` is entropy normalized by log(K); H_D normalized by log(D). Both
collapse to 0 for one-hot importance rows/columns (perfect disentanglement
/ completeness) and to 1 for uniform importance.
"""
from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from src.evaluation._common import (
    LATENT_KEYS_DEFAULT,
    extract_factors as _extract_factors,
    extract_latents as _extract_latents,
)


FactorType = Literal["discrete", "continuous"]

_DEFAULT_GBT: dict[str, Any] = {"n_estimators": 500, "max_depth": 6}


# ---------------------------------------------------------------------------
# Entropy helpers


def _normalized_entropy(p: np.ndarray, base_size: int) -> float:
    """Entropy of discrete distribution `p` normalized so uniform -> 1."""
    if base_size <= 1:
        return 0.0
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = -np.sum(p * np.log(p))
    return float(h / np.log(base_size))


# ---------------------------------------------------------------------------
# Factor-type inference + validation


def _infer_factor_type(v: np.ndarray) -> FactorType:
    return "discrete" if np.issubdtype(v.dtype, np.integer) else "continuous"


def _resolve_factor_types(
    factors: dict[str, np.ndarray],
    factor_types: dict[str, str] | None,
) -> dict[str, FactorType]:
    if factor_types is None:
        return {name: _infer_factor_type(v) for name, v in factors.items()}
    resolved: dict[str, FactorType] = {}
    for name in factors:
        ft = factor_types.get(name)
        if ft is None:
            resolved[name] = _infer_factor_type(factors[name])
        elif ft not in ("discrete", "continuous"):
            raise ValueError(
                f"factor_types[{name!r}] must be 'discrete' or 'continuous', "
                f"got {ft!r}"
            )
        else:
            resolved[name] = ft  # type: ignore[assignment]
    return resolved


# ---------------------------------------------------------------------------
# Per-factor fit + importance


def _fit_and_score(
    z: np.ndarray,
    v: np.ndarray,
    factor_type: FactorType,
    gbt_kwargs: dict[str, Any],
    random_state: int | None,
    test_size: float,
) -> tuple[np.ndarray, float]:
    """Fit GBT on (z_train, v_train), return (importances, informativeness).

    Informativeness:
      - discrete: held-out classification accuracy in [0, 1]
      - continuous: 1 - NRMSE where NRMSE = RMSE / std(v_test), clipped to
        [0, 1]. Perfect prediction → 1; random guess → ~0.
    """
    z_train, z_test, v_train, v_test = train_test_split(
        z, v, test_size=test_size, random_state=random_state
    )
    if factor_type == "discrete":
        clf = GradientBoostingClassifier(random_state=random_state, **gbt_kwargs)
        clf.fit(z_train, v_train)
        pred = clf.predict(z_test)
        info = float(accuracy_score(v_test, pred))
        importances = clf.feature_importances_.astype(np.float64)
    else:
        reg = GradientBoostingRegressor(random_state=random_state, **gbt_kwargs)
        reg.fit(z_train, v_train)
        pred = reg.predict(z_test)
        rmse = float(np.sqrt(mean_squared_error(v_test, pred)))
        denom = float(np.std(v_test))
        nrmse = rmse / denom if denom > 0.0 else 0.0
        info = float(max(0.0, 1.0 - nrmse))
        importances = reg.feature_importances_.astype(np.float64)
    return importances, info


# ---------------------------------------------------------------------------
# DCI on arrays


def compute_dci_from_arrays(
    *,
    z: np.ndarray,
    factors: dict[str, np.ndarray],
    factor_types: dict[str, str] | None = None,
    gbt_kwargs: dict[str, Any] | None = None,
    random_state: int | None = 0,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """Compute DCI given latent means and factor values as numpy arrays.

    Args:
        z: `(N, D)` float latent-mean array.
        factors: mapping `name -> (N,)`.
        factor_types: optional mapping `name -> 'discrete'|'continuous'`.
            If omitted, integer arrays → discrete, floating → continuous.
        gbt_kwargs: overrides for GBT (default: plan's
            `n_estimators=500, max_depth=6`). Useful to shrink for tests.
        random_state: fixed seed for GBT + train/test split.
        test_size: held-out fraction for informativeness.

    Returns:
        {
          "disentanglement": float,
          "completeness": float,
          "informativeness": float,
          "importance_matrix": (D, K),
          "D_per_dim": (D,),
          "C_per_factor": (K,),
          "I_per_factor": (K,),
          "factor_names": [str],
        }
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

    resolved_types = _resolve_factor_types(factors, factor_types)
    kwargs = {**_DEFAULT_GBT, **(gbt_kwargs or {})}

    names = list(factors.keys())
    k = len(names)
    R = np.zeros((d, k), dtype=np.float64)
    info = np.zeros(k, dtype=np.float64)
    for col, name in enumerate(names):
        imp, inf = _fit_and_score(
            z=z,
            v=factors[name],
            factor_type=resolved_types[name],
            gbt_kwargs=kwargs,
            random_state=random_state,
            test_size=test_size,
        )
        R[:, col] = imp
        info[col] = inf

    # Disentanglement.
    row_sum = R.sum(axis=1, keepdims=True)  # (D, 1)
    P_dim = np.divide(
        R, row_sum, out=np.zeros_like(R), where=row_sum > 0
    )
    D_per_dim = np.array(
        [1.0 - _normalized_entropy(P_dim[i, :], k) for i in range(d)],
        dtype=np.float64,
    )
    total_mass = R.sum()
    rho = (row_sum.squeeze(-1) / total_mass) if total_mass > 0 else np.zeros(d)
    D_scalar = float(np.sum(rho * D_per_dim))

    # Completeness.
    col_sum = R.sum(axis=0, keepdims=True)  # (1, K)
    P_fact = np.divide(
        R, col_sum, out=np.zeros_like(R), where=col_sum > 0
    )
    C_per_factor = np.array(
        [1.0 - _normalized_entropy(P_fact[:, j], d) for j in range(k)],
        dtype=np.float64,
    )
    C_scalar = float(C_per_factor.mean())

    I_scalar = float(info.mean())

    return {
        "disentanglement": D_scalar,
        "completeness": C_scalar,
        "informativeness": I_scalar,
        "importance_matrix": R,
        "D_per_dim": D_per_dim,
        "C_per_factor": C_per_factor,
        "I_per_factor": info,
        "factor_names": names,
    }


# ---------------------------------------------------------------------------
# DCI on model + dataloader


@torch.no_grad()
def compute(
    *,
    model: torch.nn.Module,
    dataloader: Sequence[dict[str, Any]] | Any,
    factor_keys: Sequence[str],
    factor_types: dict[str, str] | None = None,
    gbt_kwargs: dict[str, Any] | None = None,
    random_state: int | None = 0,
    test_size: float = 0.2,
    latent_keys: Sequence[str] = LATENT_KEYS_DEFAULT,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Compute DCI by running `model.encode` over `dataloader`.

    Delegates to :func:`compute_dci_from_arrays` after collecting latents
    and factor values across batches.
    """
    if device is not None:
        model = model.to(device)
    model.train(False)

    factor_keys = list(factor_keys)
    z_chunks: list[np.ndarray] = []
    fac_chunks: dict[str, list[np.ndarray]] = {k: [] for k in factor_keys}

    for batch in dataloader:
        x = batch["x"]
        if device is not None:
            x = x.to(device)
        enc = model.encode(x)
        z_chunks.append(_extract_latents(enc, latent_keys).detach().cpu().numpy())
        for name, v in _extract_factors(batch, factor_keys).items():
            fac_chunks[name].append(v.detach().cpu().numpy())

    z_all = np.concatenate(z_chunks, axis=0)
    factors = {k: np.concatenate(v, axis=0) for k, v in fac_chunks.items()}
    return compute_dci_from_arrays(
        z=z_all,
        factors=factors,
        factor_types=factor_types,
        gbt_kwargs=gbt_kwargs,
        random_state=random_state,
        test_size=test_size,
    )
