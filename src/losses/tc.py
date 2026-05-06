"""Total-Correlation loss primitives.

Two estimators of the TC term `TC(z) = KL( q(z) || prod_j q(z_j) )`:

1. **Density-ratio via discriminator** (Kim & Mnih 2018, arXiv:1802.04942) —
   train an MLP to distinguish joint samples from per-dim-permuted samples;
   the log-ratio `log D(z)/(1 - D(z))` is a sample-wise TC estimate. Used by
   the `FactorVAE` baseline (Task 2.3).

2. **Minibatch-weighted sampling** (Chen et al. 2018, arXiv:1803.05428) —
   closed-form batch estimate of `log q(z)` without an auxiliary network.
   Used by the β-TCVAE baseline (Task 2.5). Decomposes the standard
   `KL(q(z|x) || p(z))` into three terms:

       KL = I_index + TC + dimwise_KL

   where `I_index = I(z; n)` is the index-code mutual information,
   `TC = KL(q(z) || prod_j q(z_j))` is Total Correlation, and
   `dimwise_KL = Σ_j KL(q(z_j) || p(z_j))` is the per-dim marginal KL.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def permute_latents(z: torch.Tensor) -> torch.Tensor:
    """Per-dim independent shuffle across the batch axis.

    Input `(B, D)`. Output has identical marginals per dim but independent
    dims — approximates a sample from `prod_j q(z_j)` given a sample from
    the joint `q(z)`.
    """
    if z.ndim != 2:
        raise ValueError(f"expected 2-D latent tensor (B, D), got shape {tuple(z.shape)}")
    B, D = z.shape
    # One permutation per dimension, stacked columnwise. `torch.gather` then
    # reindexes each column of z independently.
    perms = torch.stack(
        [torch.randperm(B, device=z.device) for _ in range(D)],
        dim=1,
    )  # (B, D)
    return torch.gather(z, 0, perms)


class TCDiscriminator(nn.Module):
    """MLP discriminator predicting joint vs factored latent distribution.

    Architecture from Kim & Mnih 2018 §A: 6 fully-connected layers of 1000
    units, LeakyReLU(0.2), output 2-class logits. We default to 5 hidden
    layers + 1 output layer to match the paper's effective depth; tests
    override `hidden` / `n_layers` for speed.
    """

    def __init__(
        self,
        d_z: int,
        hidden: int = 1000,
        n_layers: int = 5,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if d_z <= 0:
            raise ValueError(f"d_z must be positive, got {d_z}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")

        layers: list[nn.Module] = []
        c_in = d_z
        for _ in range(n_layers):
            layers.append(nn.Linear(c_in, hidden))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            c_in = hidden
        layers.append(nn.Linear(c_in, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (B, 2) — [joint-logit, factored-logit]


def tc_estimate_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Sample-mean TC estimate from discriminator logits.

    `logits[:, 0] - logits[:, 1]` equals `log D(z) - log(1 - D(z))` after
    softmax normalization, which is an unbiased estimate of `log q(z) /
    prod_j q(z_j)` when the discriminator is optimal. Mean over batch
    gives the TC scalar used in the VAE objective.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(
            f"expected logits of shape (B, 2), got {tuple(logits.shape)}"
        )
    return (logits[:, 0] - logits[:, 1]).mean()


def tc_discriminator_loss(
    logits_joint: torch.Tensor,
    logits_factored: torch.Tensor,
) -> torch.Tensor:
    """2-class cross-entropy training loss for the TC discriminator.

    Joint samples get class 0; factored (per-dim-permuted) samples get
    class 1. A perfectly-trained D hits ≈ 0 loss when q(z) is far from
    the factored distribution and ≈ log(2) when they are indistinguishable.
    """
    if logits_joint.shape != logits_factored.shape:
        raise ValueError(
            f"shape mismatch: joint {tuple(logits_joint.shape)} vs "
            f"factored {tuple(logits_factored.shape)}"
        )
    B = logits_joint.shape[0]
    zeros = torch.zeros(B, dtype=torch.long, device=logits_joint.device)
    ones = torch.ones(B, dtype=torch.long, device=logits_factored.device)
    all_logits = torch.cat([logits_joint, logits_factored], dim=0)
    all_labels = torch.cat([zeros, ones], dim=0)
    return F.cross_entropy(all_logits, all_labels)


# -- Minibatch-weighted decomposed KL (β-TCVAE, Task 2.5) --------------------


def _log_gaussian(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Per-dim log N(z; mu, diag exp(logvar)).

    Broadcasts `z:(N, 1, D)` against `mu, logvar:(1, M, D)` → returns shape
    `(N, M, D)` so that index `[n, m, j]` is `log N(z_n_j; mu_m_j, exp(logvar_m_j))`.
    """
    z = z.unsqueeze(1)
    mu = mu.unsqueeze(0)
    logvar = logvar.unsqueeze(0)
    return -0.5 * (
        math.log(2.0 * math.pi) + logvar + (z - mu).pow(2) * (-logvar).exp()
    )


def batch_tc_decomposition(
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minibatch-weighted sampling estimator (Chen et al. 2018, Eq. 4) of

        KL(q(z|x) ‖ p(z)) = I_index + TC + dimwise_KL

    operating purely on a batch of posterior samples + parameters — no
    auxiliary network required.

    Args:
        z: `(B, D)` latent samples `z_n ~ q(z | x_n)`.
        mu: `(B, D)` posterior means `μ_n`.
        logvar: `(B, D)` posterior log-variances.

    Returns:
        `(mi_index, tc, dimwise_kl)`, each a scalar tensor. All three are
        non-negative in expectation under correct input; for pathological
        small-batch inputs they can wobble slightly below zero due to the
        minibatch-weighted bias (ignore within ±0.05 nats in tests).

    Notes:
        We use the "minibatch-weighted sampling" (MWS) variant — simpler
        than "minibatch-stratified" (MSS) and commonly adopted in public
        β-TCVAE implementations. MWS drops a `log(N_dataset)` constant that
        would shift every term by the same amount (gradients unchanged).
    """
    if z.ndim != 2 or mu.shape != z.shape or logvar.shape != z.shape:
        raise ValueError(
            f"z, mu, logvar must all be (B, D); got {tuple(z.shape)}, "
            f"{tuple(mu.shape)}, {tuple(logvar.shape)}"
        )
    B, D = z.shape
    # B=1 degenerates: logsumexp over one element is the element itself, and
    # log(1)=0, so every decomposition term collapses to 0 with meaningless
    # gradient. Surface this explicitly instead of silently producing zeros.
    if B < 2:
        raise ValueError(
            f"batch_tc_decomposition requires B >= 2 for a meaningful MWS "
            f"estimate, got B={B}. Upstream: increase batch size or set "
            f"DataLoader(drop_last=True) for tail-batch edge cases."
        )

    # log p(z) under standard normal prior — per-sample scalar.
    log_p_z = (
        -0.5 * (math.log(2.0 * math.pi) + z.pow(2))
    ).sum(dim=1)  # (B,)

    # log q(z_n | x_n) — each sample under its own posterior.
    log_q_z_given_x = (
        -0.5 * (math.log(2.0 * math.pi) + logvar + (z - mu).pow(2) * (-logvar).exp())
    ).sum(dim=1)  # (B,)

    # log q(z_n | x_m) for every pair (n, m) — used to approximate both the
    # joint aggregate posterior and its product-of-marginals.
    log_q_zx_all = _log_gaussian(z, mu, logvar)  # (B, B, D)

    # log q(z_n) ≈ logsumexp_m [ Σ_j log q(z_nj | x_mj) ] − log(B)
    log_q_z = (
        torch.logsumexp(log_q_zx_all.sum(dim=2), dim=1) - math.log(B)
    )  # (B,)

    # log prod_j q(z_nj) ≈ Σ_j [ logsumexp_m log q(z_nj | x_mj) − log(B) ]
    log_q_z_prod = (
        torch.logsumexp(log_q_zx_all, dim=1) - math.log(B)
    ).sum(dim=1)  # (B,)

    mi_index = (log_q_z_given_x - log_q_z).mean()
    tc = (log_q_z - log_q_z_prod).mean()
    dimwise_kl = (log_q_z_prod - log_p_z).mean()
    return mi_index, tc, dimwise_kl
