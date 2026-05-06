"""Invariance / equivariance diagnostics on style and content latents.

Invariance ratio (style):
    IR = mean_x mean_g  ||mu_s(x) - mu_s(T_g x)||^2 / (||mu_s(x)||^2 + eps)

Equivariance ratio (content):
    ER = mean_x mean_g  ||mu_c(T_g x) - rho(g) mu_c(x)||^2 / (||mu_c(T_g x)||^2 + eps)

Both ratios are non-negative; 0 means perfect invariance/equivariance.
Averaged over the assessment set and the list of g values (typically
{+/-100, +/-200, ..., +/-1200} cents).

Contract:
    `model.encode(x) -> {"mu_s": (B, d_s), "mu_c": (B, d_c) or (B, d_c, T)}`
    `group_rep.matrix(g_cents: (B,)) -> (B, d_c, d_c)`
    `augment(x, g_cents: (B,)) -> Tensor shaped like x`
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol

import numpy as np
import torch


class _GroupRepLike(Protocol):
    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor: ...


AugmentFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Core per-g computation


def _apply_rho(R: torch.Tensor, mu_c: torch.Tensor) -> torch.Tensor:
    """rho(g) * mu_c. Handles both (B, d_c) and (B, d_c, T) content latents."""
    if mu_c.ndim == 2:
        return torch.einsum("bij,bj->bi", R, mu_c)
    if mu_c.ndim == 3:
        return torch.einsum("bij,bjt->bit", R, mu_c)
    raise ValueError(
        f"mu_c must be (B, d_c) or (B, d_c, T), got shape {tuple(mu_c.shape)}"
    )


def _flat_norm_sq(t: torch.Tensor) -> torch.Tensor:
    """||.||^2 per batch element, flattening all non-batch dims."""
    return t.reshape(t.shape[0], -1).pow(2).sum(dim=1)


def _one_g_stats(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    g_cents_scalar: float,
    group_rep: _GroupRepLike,
    augment: AugmentFn,
    eps: float,
) -> tuple[float, float]:
    """Return (IR_g, ER_g) averaged across the batch `x`."""
    b = x.shape[0]
    g = torch.full(
        (b,), float(g_cents_scalar), device=x.device, dtype=x.dtype
    )
    enc_x = model.encode(x)
    mu_s = enc_x["mu_s"]
    mu_c = enc_x["mu_c"]

    x_g = augment(x, g)
    enc_g = model.encode(x_g)
    mu_s_g = enc_g["mu_s"]
    mu_c_g = enc_g["mu_c"]

    # Style invariance — same factor applied to mu_s of x and of T_g x.
    num_inv = _flat_norm_sq(mu_s - mu_s_g)
    den_inv = _flat_norm_sq(mu_s).clamp_min(eps)
    ir = (num_inv / den_inv).mean().item()

    # Content equivariance — compare mu_c(T_g x) vs rho(g) mu_c(x).
    R = group_rep.matrix(g)
    mu_c_pred = _apply_rho(R, mu_c)
    num_eq = _flat_norm_sq(mu_c_g - mu_c_pred)
    den_eq = _flat_norm_sq(mu_c_g).clamp_min(eps)
    er = (num_eq / den_eq).mean().item()
    return ir, er


# ---------------------------------------------------------------------------
# Array-level entry point


@torch.no_grad()
def compute_from_arrays(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    g_cents_list: torch.Tensor,
    group_rep: _GroupRepLike,
    augment: AugmentFn,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute invariance + equivariance ratios for a single batch `x`.

    Args:
        x: input batch, shape accepted by `model.encode` + `augment`.
        g_cents_list: `(G,)` float tensor of shift values to sweep.

    Returns:
        {
            "invariance_ratio": float,
            "equivariance_ratio": float,
            "per_g": (G, 2) ndarray with columns [IR_g, ER_g],
            "g_cents": (G,) ndarray of the shift values,
        }
    """
    if g_cents_list.ndim != 1:
        raise ValueError(
            f"g_cents_list must be 1-D, got shape {tuple(g_cents_list.shape)}"
        )
    if g_cents_list.numel() == 0:
        raise ValueError("g_cents_list must be non-empty")

    model.train(False)

    per_g = np.zeros((g_cents_list.shape[0], 2), dtype=np.float64)
    g_vals = g_cents_list.detach().cpu().tolist()
    for i, g_scalar in enumerate(g_vals):
        ir, er = _one_g_stats(
            model=model,
            x=x,
            g_cents_scalar=float(g_scalar),
            group_rep=group_rep,
            augment=augment,
            eps=eps,
        )
        per_g[i, 0] = ir
        per_g[i, 1] = er

    return {
        "invariance_ratio": float(per_g[:, 0].mean()),
        "equivariance_ratio": float(per_g[:, 1].mean()),
        "per_g": per_g,
        "g_cents": np.asarray(g_vals, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Dataloader wrapper


@torch.no_grad()
def compute(
    *,
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    g_cents_list: torch.Tensor,
    group_rep: _GroupRepLike,
    augment: AugmentFn,
    eps: float = 1e-8,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Sweep `dataloader` and aggregate invariance/equivariance ratios.

    Each batch contributes a (G, 2) matrix; results are averaged across
    batches weighted by batch size.
    """
    if g_cents_list.ndim != 1 or g_cents_list.numel() == 0:
        raise ValueError("g_cents_list must be non-empty 1-D tensor")

    if device is not None:
        model = model.to(device)
    model.train(False)

    g_count = g_cents_list.shape[0]
    per_g_sum = np.zeros((g_count, 2), dtype=np.float64)
    total_n = 0

    for batch in dataloader:
        x = batch["x"]
        if device is not None:
            x = x.to(device)
        b = x.shape[0]
        out = compute_from_arrays(
            model=model,
            x=x,
            g_cents_list=g_cents_list,
            group_rep=group_rep,
            augment=augment,
            eps=eps,
        )
        per_g_sum += out["per_g"] * b
        total_n += b

    per_g = per_g_sum / max(total_n, 1)
    return {
        "invariance_ratio": float(per_g[:, 0].mean()),
        "equivariance_ratio": float(per_g[:, 1].mean()),
        "per_g": per_g,
        "g_cents": g_cents_list.detach().cpu().numpy().astype(np.float64),
    }
