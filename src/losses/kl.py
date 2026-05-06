"""KL loss primitives for SC-VAE (plan Task 3.6).

Per-subspace KL with free-bits + cyclical β annealing:

    Composite loss (plan Task 3.5 update):
        L = L_recon + β_s · KL_s + β_c · KL_c + λ_inv · L_inv + λ_equi · L_equi + λ_swap · L_swap

Free-bits (Kingma 2016, IAF) prevents posterior collapse by clamping each
per-dim batch-averaged KL to a minimum `τ` — the optimizer stops getting
gradient pressure once a dim carries at least `τ` nats of information.

Cyclical β annealing (Fu et al. 2019, "Cyclical Annealing Schedule")
oscillates β from a low value up to the target multiple times during
training — combats the "KL-vanishing" failure mode where β·KL stays at 0
while recon dominates.
"""
from __future__ import annotations

import torch


def kl_subspace(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Per-subspace KL with optional per-example per-dim free-bits clamp.

    Free-bits ordering follows Kingma 2016 (IAF §2.3) exactly:
    **per-example clamp THEN batch-mean**. For each example `b` and
    latent dim `d`, raw `kl_{b,d}` is clamped at `free_bits` before the
    batch average. This keeps every example contributing at least
    `free_bits` nats per dim, preventing collapse on any single item.

    The alternative "batch-mean then clamp" ordering leaks gradient
    pressure on dims where half the batch is zero and half is above τ:
    the average passes the clamp and zero-KL examples contribute no
    signal, defeating the collapse-prevention purpose.

    Args:
        mu, logvar: `(B, D)` (pooled) or `(B, D, T)` (time-resolved).
        free_bits: per-dim minimum KL (nats). Canonical value 0.1
            (Kingma 2016).

    Returns:
        Scalar KL = `mean_B Σ_d max(kl_{b,d}, free_bits)`.
    """
    if mu.shape != logvar.shape:
        raise ValueError(
            f"shape mismatch: mu {tuple(mu.shape)} vs logvar {tuple(logvar.shape)}"
        )
    if mu.ndim not in (2, 3):
        raise ValueError(
            f"expected 2-D (B, D) or 3-D (B, D, T) tensors, got {tuple(mu.shape)}"
        )
    if free_bits < 0:
        raise ValueError(f"free_bits must be non-negative, got {free_bits}")

    raw = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    # For time-resolved input, sum over T so each (example, dim) carries
    # all its time contributions before the free-bits clamp.
    if raw.ndim == 3:
        raw = raw.sum(dim=-1)         # (B, D)
    if free_bits > 0.0:
        raw = raw.clamp(min=free_bits)  # per-example, per-dim (Kingma order)
    return raw.sum(dim=-1).mean()       # sum over D per example, mean over B


def kl_total(
    mu_s: torch.Tensor,
    logvar_s: torch.Tensor,
    mu_c: torch.Tensor,
    logvar_c: torch.Tensor,
    tau_s: float = 0.1,
    tau_c: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-subspace KL for SC-VAE.

    Returns `(kl_s, kl_c)` — scalars with independent free-bits floors.
    Phase-4 training step weights these with independent `β_s`, `β_c`.
    """
    kl_s = kl_subspace(mu_s, logvar_s, free_bits=tau_s)
    kl_c = kl_subspace(mu_c, logvar_c, free_bits=tau_c)
    return kl_s, kl_c


# -- cyclical β annealing --------------------------------------------------


def cyclical_beta(
    step: int,
    total_steps: int,
    n_cycles: int,
    ratio: float = 0.5,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
) -> float:
    """Fu et al. 2019 cyclical annealing schedule.

    Each cycle has two phases: linear ramp from `beta_min` to `beta_max`
    over the first `ratio` fraction of cycle steps, then hold at `beta_max`
    for the remainder. At cycle rollover, β snaps back to `beta_min` and
    the ramp restarts.

    Args:
        step: current global step (0-indexed).
        total_steps: total training steps.
        n_cycles: how many full cycles across `total_steps`.
        ratio: fraction of each cycle spent ramping (rest is hold phase).
        beta_min, beta_max: endpoints of the linear ramp.

    Returns:
        Scalar β for this step.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if n_cycles <= 0:
        raise ValueError(f"n_cycles must be positive, got {n_cycles}")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")
    if beta_min > beta_max:
        raise ValueError(
            f"beta_min ({beta_min}) must be <= beta_max ({beta_max})"
        )

    # Integer cycle length avoids float-modulo drift at non-divisible
    # (total_steps, n_cycles) pairs (e.g. 1001 / 4).
    cycle_len = max(1, total_steps // n_cycles)
    pos = (step % cycle_len) / cycle_len
    if pos < ratio:
        # Linear ramp.
        frac = pos / ratio
        return beta_min + frac * (beta_max - beta_min)
    # Hold phase.
    return float(beta_max)
