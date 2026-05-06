"""SAP — Separated Attribute Predictability.

Kumar, A., Sattigeri, P., Balakrishnan, A., 2018. "Variational Inference of
Disentangled Latent Concepts from Unlabeled Observations." ICLR.

For each (latent dim i, factor v_k) train a 1-D predictor and record a score
S[i, k]:
    - discrete factor: held-out classification accuracy of a Linear SVC
    - continuous factor: R^2 of a LinearRegression

Per-factor SAP_k = S_{i*, k} - S_{i**, k} (top1 - top2 across dims).
Overall SAP = mean_k SAP_k.
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


FactorType = Literal["discrete", "continuous"]


def _infer_type(v: np.ndarray) -> FactorType:
    return "discrete" if np.issubdtype(v.dtype, np.integer) else "continuous"


def _score_dim_factor(
    z_col: np.ndarray,
    v: np.ndarray,
    factor_type: FactorType,
    svc_kwargs: dict[str, Any],
    random_state: int | None,
    test_size: float,
) -> float:
    x = z_col.reshape(-1, 1)
    z_train, z_test, v_train, v_test = train_test_split(
        x, v, test_size=test_size, random_state=random_state
    )
    if factor_type == "discrete":
        if np.unique(v_train).size < 2:
            return 0.0
        clf = LinearSVC(random_state=random_state, **svc_kwargs)
        clf.fit(z_train, v_train)
        return float(clf.score(z_test, v_test))
    reg = LinearRegression()
    reg.fit(z_train, v_train)
    return float(max(0.0, reg.score(z_test, v_test)))


def compute_sap_from_arrays(
    *,
    z: np.ndarray,
    factors: dict[str, np.ndarray],
    factor_types: dict[str, str] | None = None,
    svc_kwargs: dict[str, Any] | None = None,
    random_state: int | None = 0,
    test_size: float = 0.2,
) -> dict[str, Any]:
    """SAP from array inputs. See module docstring."""
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
    svc_kwargs = {**(svc_kwargs or {})}
    svc_kwargs.setdefault("dual", "auto")
    svc_kwargs.setdefault("max_iter", 1000)

    names = list(factors.keys())
    k = len(names)
    scores = np.zeros((d, k), dtype=np.float64)
    for col, name in enumerate(names):
        v = factors[name]
        ftype = (factor_types or {}).get(name) or _infer_type(v)
        if ftype not in ("discrete", "continuous"):
            raise ValueError(f"factor_types[{name!r}] invalid: {ftype!r}")
        for j in range(d):
            scores[j, col] = _score_dim_factor(
                z_col=z[:, j],
                v=v,
                factor_type=ftype,  # type: ignore[arg-type]
                svc_kwargs=svc_kwargs,
                random_state=random_state,
                test_size=test_size,
            )

    per_factor: dict[str, float] = {}
    gaps = np.zeros(k, dtype=np.float64)
    for col, name in enumerate(names):
        s = scores[:, col]
        if d >= 2:
            order = np.argsort(s)[::-1]
            gap = float(s[order[0]] - s[order[1]])
        else:
            gap = float(s[0])
        per_factor[name] = gap
        gaps[col] = gap

    return {
        "sap": float(gaps.mean()),
        "per_factor": per_factor,
        "score_matrix": scores,
        "factor_names": names,
    }
