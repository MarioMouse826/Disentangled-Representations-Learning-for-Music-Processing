"""Zero-shot OOD assessment (plan Task 6.5).

Two per-clip measurements on held-out ZeroShotBass audio:

    (a) identity_stability — per-dim variance of mu_s across sliding 4-s
        windows within a clip. Lower = better identity stability (timbre
        is supposed to be invariant to time/position within a clip).

    (b) equivariance_ratio — mean || mu_c(T_g x) - rho(g) mu_c(x) ||^2 /
        || mu_c(T_g x) ||^2 across shifts g, pooled across windows.

Aggregated mean across clips is reported vs baselines.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from src.evaluation.equivariance_err import compute_from_arrays as _equi_from_arrays


# ---------------------------------------------------------------------------
# Identity stability


def identity_variance_scalar(per_dim_variance: torch.Tensor) -> float:
    """Collapse per-dim style variance to a single scalar (mean)."""
    return float(per_dim_variance.mean().item())


@torch.no_grad()
def compute_identity_stability(
    *,
    model: torch.nn.Module,
    windows: torch.Tensor,
) -> dict[str, Any]:
    """Encode each window, return per-dim variance of mu_s across windows.

    Args:
        windows: `(W, *encoder_input_shape)` — W windows from the same clip.

    Returns:
        {"per_dim_variance": (d_s,) tensor, "scalar_variance": float}
    """
    if windows.ndim < 2:
        raise ValueError(
            f"windows must have a leading W dim, got shape {tuple(windows.shape)}"
        )
    model.train(False)
    enc = model.encode(windows)
    mu_s = enc["mu_s"]
    if mu_s.ndim != 2:
        raise ValueError(
            f"mu_s must be (W, d_s), got shape {tuple(mu_s.shape)}"
        )
    # Sample variance across the window dim (ddof=0 → population variance).
    if mu_s.shape[0] < 2:
        per_dim = torch.zeros(mu_s.shape[1], dtype=mu_s.dtype, device=mu_s.device)
    else:
        per_dim = mu_s.var(dim=0, unbiased=False)
    return {
        "per_dim_variance": per_dim.detach().cpu(),
        "scalar_variance": identity_variance_scalar(per_dim.detach().cpu()),
    }


# ---------------------------------------------------------------------------
# End-to-end zero-shot OOD


AugmentFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class _GroupRepLike:
    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor: ...


@torch.no_grad()
def compute_zero_shot_ood(
    *,
    model: torch.nn.Module,
    clip_windows: Iterable[torch.Tensor],
    g_cents_list: torch.Tensor,
    group_rep: _GroupRepLike,
    augment: AugmentFn,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Aggregate identity stability + equivariance ratio across clips.

    Args:
        clip_windows: iterable of `(W_i, *encoder_input_shape)` per-clip
            window tensors. Each entry is one held-out clip.
        g_cents_list: `(G,)` shift values swept per clip.

    Returns:
        {
            "per_clip": (N_clips, 2) — columns [variance, equivariance_ratio],
            "mean_variance": float,
            "mean_equivariance_ratio": float,
            "g_cents": (G,) ndarray,
        }
    """
    if device is not None:
        model = model.to(device)
    model.train(False)

    per_clip: list[tuple[float, float]] = []
    for windows in clip_windows:
        if device is not None:
            windows = windows.to(device)
        stab = compute_identity_stability(model=model, windows=windows)
        equi = _equi_from_arrays(
            model=model,
            x=windows,
            g_cents_list=g_cents_list,
            group_rep=group_rep,
            augment=augment,
        )
        per_clip.append((stab["scalar_variance"], equi["equivariance_ratio"]))

    if not per_clip:
        raise ValueError("clip_windows must yield at least one clip")
    arr = np.asarray(per_clip, dtype=np.float64)
    return {
        "per_clip": arr,
        "mean_variance": float(arr[:, 0].mean()),
        "mean_equivariance_ratio": float(arr[:, 1].mean()),
        "g_cents": g_cents_list.detach().cpu().numpy().astype(np.float64),
    }
