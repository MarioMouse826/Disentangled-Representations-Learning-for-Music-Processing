"""Symmetry losses for SC-VAE training (plan Task 3.5).

Three terms operating in latent + reconstruction space:

    L_inv   =  E_{x, g} [ ‖ μ_s(x) − μ_s(T_g x) ‖²_2 ]           (style invariance)
    L_equi  =  E_{x, g} [ ‖ μ_c(T_g x) − ρ(g) μ_c(x) ‖²_2 ]       (content equivariance)
    L_swap  =  E_{x, g} [ ‖ Dec(z_s(x), ρ(g) z_c(x)) − T_g x ‖_1 ] (swap consistency)

All three are pure functions on precomputed tensors — this file knows
nothing about encoders, `ρ(g)`, or decoders. Upstream is expected to
produce the argument tensors and pass them in.

Reduction conventions (match the plan's `‖Δ‖²_2` math — L2 norm over ALL
latent axes, then batch-mean):

- `invariance_loss`: sum over latent dim `d_s`, mean over batch.
- `equivariance_loss`: sum over BOTH channel dim `d_c` AND time `T`,
  mean over batch only. This keeps the effective scale symmetric with
  `invariance_loss`, so callers can set `λ_inv = λ_equi` and get equal
  weighting. (Prior `sum(d_c), mean(B, T)` form silently divided by T,
  shifting the effective weight.)
- `swap_consistency_loss`: `F.l1_loss` default mean reduction over the
  entire mel-spectrogram tensor.

Why squared L2 for the first two: the latent-space symmetry constraints
come from a metric-defined group action (Higgins et al. 2018, arXiv:1812.02230);
L2 is the natural distance for equivariant-representation work. Decoder
output uses L1 for perceptual match with the plan's main recon loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def invariance_loss(
    mu_s_x: torch.Tensor,
    mu_s_gx: torch.Tensor,
) -> torch.Tensor:
    """`L_inv = mean_b ‖ μ_s(x_b) − μ_s(T_g x_b) ‖²_2`.

    Args:
        mu_s_x, mu_s_gx: `(B, d_s)` style-posterior means for `x` and `T_g x`.
    """
    if mu_s_x.shape != mu_s_gx.shape:
        raise ValueError(
            f"shape mismatch: mu_s_x {tuple(mu_s_x.shape)} vs "
            f"mu_s_gx {tuple(mu_s_gx.shape)}"
        )
    return (mu_s_x - mu_s_gx).pow(2).sum(dim=-1).mean()


def equivariance_loss(
    mu_c_gx: torch.Tensor,
    rho_g_mu_c_x: torch.Tensor,
) -> torch.Tensor:
    """`L_equi = mean_b Σ_{j,t} (μ_c(T_g x)_{j,t} − (ρ(g) μ_c(x))_{j,t})²`.

    Sum over **both** the channel and time dims, mean over batch only.
    Matches the plan's `‖Δ‖²_2` formulation and keeps the scale
    symmetric with `invariance_loss` so `λ_inv = λ_equi` weights equally.

    Args:
        mu_c_gx: `(B, d_c, T')` content posterior mean of `T_g x`.
        rho_g_mu_c_x: `(B, d_c, T')` rotation-applied content posterior
            mean of `x` — upstream should compute this via
            `RotationRep(...)(mu_c_x, g_cents)` (Task 3.3).
    """
    if mu_c_gx.ndim != 3:
        raise ValueError(
            f"expected 3-D (B, d_c, T'), got {tuple(mu_c_gx.shape)}"
        )
    if mu_c_gx.shape != rho_g_mu_c_x.shape:
        raise ValueError(
            f"shape mismatch: mu_c_gx {tuple(mu_c_gx.shape)} vs "
            f"rho_g_mu_c_x {tuple(rho_g_mu_c_x.shape)}"
        )
    return (mu_c_gx - rho_g_mu_c_x).pow(2).sum(dim=(1, 2)).mean()


def swap_consistency_loss(
    decoded_swapped_mel: torch.Tensor,
    target_mel: torch.Tensor,
) -> torch.Tensor:
    """`L_swap = mean ‖ Dec(z_s(x), ρ(g) z_c(x)) − T_g x ‖_1`.

    Novel loss (plan §3.5 'reviewer-bait') constraining the decoder to be
    consistent with the group-theoretic latent structure: rotating the
    content latent by `ρ(g)` and decoding must match the pitch-shifted
    waveform's mel spectrogram.
    """
    if decoded_swapped_mel.shape != target_mel.shape:
        raise ValueError(
            f"shape mismatch: decoded {tuple(decoded_swapped_mel.shape)} vs "
            f"target {tuple(target_mel.shape)}"
        )
    return F.l1_loss(decoded_swapped_mel, target_mel, reduction="mean")
