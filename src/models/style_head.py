"""Style head — produces the invariance-constrained posterior `q(z_s | x)`.

Design (plan Task 3.1)
----------------------
Pool the shared encoder trunk output `h ∈ R^{C × T'}` over time with a
learnable single-query attention, then linear heads emit `(μ_s, log σ_s²)
∈ R^{d_s}`. The attention mechanism is:

    score_t = q^T · W · h_t                (learnable q ∈ R^{H}, W ∈ R^{H×C})
    α_t     = softmax_t(score_t)
    p       = Σ_t α_t · h_t                (pooled vector ∈ R^C)
    μ_s     = Linear_μ(p)
    logvar  = Linear_σ(p)   (clamped for numerical stability)

The architecture alone is *approximately* time-invariant because attention
pooling reduces T' → 1 before the Gaussian heads. The exact pitch-shift
invariance `z_s(x) = z_s(T_g x)` is enforced by the symmetry loss `L_inv`
(Task 3.5); attention pooling gives the gradient signal a well-posed
structure to learn that invariance (vs mean-pool which is rigid, or
concat-over-time which is maximally time-variant).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleHead(nn.Module):
    """Attention-pooled Gaussian head producing `q(z_s | x)`.

    Args:
        in_channels: feature-channel count `C` of the encoder trunk output.
        d_s: style-latent dimensionality.
        hidden_score: inner dim of the attention-score projection `W`.
        logvar_clamp: `(min, max)` clamp on the output log-variance. Keeps
            downstream reparameterization / KL numerics sane when the
            encoder briefly produces pathological activations.
    """

    def __init__(
        self,
        in_channels: int,
        d_s: int,
        hidden_score: int = 128,
        logvar_clamp: tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if d_s <= 0:
            raise ValueError(f"d_s must be positive, got {d_s}")
        if hidden_score <= 0:
            raise ValueError(f"hidden_score must be positive, got {hidden_score}")
        if logvar_clamp[0] >= logvar_clamp[1]:
            raise ValueError(
                f"logvar_clamp must be ascending, got {logvar_clamp}"
            )

        self._in_channels = in_channels
        self._d_s = d_s
        self._hidden_score = hidden_score
        self._logvar_clamp = logvar_clamp

        # Attention: project each time-step feature vector into the scoring
        # space, then inner-product with a learnable global query to produce
        # a scalar score per time step.
        self.score_proj = nn.Linear(in_channels, hidden_score)
        self.query = nn.Parameter(torch.randn(hidden_score) * 0.02)

        # Gaussian heads over the pooled vector.
        self.to_mu = nn.Linear(in_channels, d_s)
        self.to_logvar = nn.Linear(in_channels, d_s)

    @property
    def config(self) -> dict:
        return {
            "in_channels": self._in_channels,
            "d_s": self._d_s,
            "hidden_score": self._hidden_score,
            "logvar_clamp": self._logvar_clamp,
        }

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        if h.ndim != 3:
            raise ValueError(
                f"expected 3-D encoder output (B, C, T'), got shape {tuple(h.shape)}"
            )
        if h.shape[1] != self._in_channels:
            raise ValueError(
                f"expected in_channels={self._in_channels}, got {h.shape[1]}"
            )

        # Attention scores: (B, T', H) via per-time-step linear, then dot
        # with global query → (B, T'). Vaswani-style `1/sqrt(H)` scaling
        # keeps softmax from saturating to one-hot when query norm grows
        # during training (otherwise gradient collapses through the
        # softmax and style-head stops learning time-level discrimination).
        scores = torch.einsum(
            "bth,h->bt",
            self.score_proj(h.transpose(1, 2)),   # (B, T', H)
            self.query,
        ) / math.sqrt(self._hidden_score)
        attn = F.softmax(scores, dim=1)            # (B, T')

        # Pooled vector p = Σ_t α_t · h_t  →  (B, C).
        pooled = torch.einsum("bct,bt->bc", h, attn)

        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(*self._logvar_clamp)

        return {"mu_s": mu, "logvar_s": logvar, "attn_weights": attn}
