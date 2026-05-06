"""Autoregressive Hierarchical VAE baseline.

Reference: Sønderby et al. 2016 (Ladder VAE) + Dieleman/Engel hierarchical
audio VAE variants. 3-level hierarchy with an autoregressive Gaussian prior

    p(z_1, z_2, z_3) = p(z_3) · p(z_2 | z_3) · p(z_1 | z_2, z_3)

where `p(z_3) = N(0, I)` and the lower-level priors are learned neural
networks that emit `(μ_prior, logvar_prior)` given the upper latents.

Inference uses a *bottom-up* approximate posterior: each level's
`q(z_L | x) = N(μ_L^q, diag exp(logvar_L^q))` reads from the shared
`MelEncoder` trunk via a level-specific linear head. Because the priors
depend on upper-level latents (sampled from q), the ELBO reduces to:

    L = E_q[log p(x | z_{1:3})]
        − KL( q(z_3|x) ‖ N(0, I) )
        − E_{q(z_3|x)} [ KL( q(z_2|x) ‖ p(z_2 | z_3) ) ]
        − E_{q(z_{2:3}|x)} [ KL( q(z_1|x) ‖ p(z_1 | z_2, z_3) ) ]

All three KLs are analytic Gaussian KLs → non-negative by construction.

Decoder concatenates z_1, z_2, z_3 and feeds the joint latent into a
`BetaVAE`-style upsample stack.

Reuses `MelEncoder` and `_UpBlock` so every baseline shares the same
representation backbone.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines.beta_vae import _UpBlock
from src.models.encoder import MelEncoder


def _kl_gaussian(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """KL( N(μ_q, diag exp(logvar_q)) ‖ N(μ_p, diag exp(logvar_p)) ), per-batch.

    Standard closed-form diagonal-Gaussian KL. Sum over latent dim, mean
    over batch dim. Always ≥ 0 up to floating-point noise (caller should
    allow a small negative tolerance).
    """
    var_ratio = (logvar_q - logvar_p).exp()      # exp(logvar_q - logvar_p) = σ_q² / σ_p²
    mean_term = (mu_q - mu_p).pow(2) * (-logvar_p).exp()
    kl = 0.5 * (var_ratio + mean_term - 1.0 + (logvar_p - logvar_q))
    return kl.sum(dim=1).mean()


class _PriorNet(nn.Module):
    """Small 2-layer MLP producing (mu, logvar) of a conditional Gaussian prior."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2 * out_dim),
        )
        self._out_dim = out_dim

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(context)
        mu, logvar = h.chunk(2, dim=-1)
        # Clamp logvar so the prior variance never collapses to 0 or explodes.
        # Range [-10, 10] → σ in [e^-5, e^5] — plenty of room without ruining stability.
        return mu, logvar.clamp(-10.0, 10.0)


class AR_HVAE(nn.Module):
    """3-level Autoregressive Hierarchical VAE.

    Args:
        levels: per-level latent dims ordered bottom → top. Default (32, 16, 8).
        n_mels, n_time: expected input shape.
        channels, strides: encoder/decoder spatial schedule.
        decoder_base_freq: decoder bottleneck freq resolution.
    """

    def __init__(
        self,
        levels: Sequence[int] = (32, 16, 8),
        n_mels: int = 128,
        n_time: int = 401,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2, 2),
        decoder_base_freq: int = 4,
    ) -> None:
        super().__init__()
        levels = tuple(levels)
        if not levels:
            raise ValueError("levels must be non-empty")
        if any(d <= 0 for d in levels):
            raise ValueError(f"levels must be positive, got {levels}")
        if n_mels <= 0 or n_time <= 0:
            raise ValueError(f"n_mels and n_time must be positive")

        self._levels = levels            # (d_1, d_2, d_3) bottom → top
        self.n_mels = n_mels
        self.n_time = n_time
        self._channels = tuple(channels)
        self._strides = tuple(strides)
        self._decoder_base_freq = decoder_base_freq

        self.encoder = MelEncoder(channels=channels, strides=strides)
        self._t_enc = self._compute_encoder_time(n_time, self._strides)
        flat = self.encoder.out_channels * self._t_enc

        # Bottom-up q-heads — one linear layer per level emitting (mu, logvar).
        self.q_heads = nn.ModuleList([nn.Linear(flat, 2 * d) for d in levels])

        # Autoregressive priors:
        #   level L (top)   → p = N(0, I) (implicit, no net)
        #   level L-1 ... 1 → prior conditioned on all upper-level latents
        # `self.prior_nets[i]` produces the prior for `z_i` conditioned on
        # concat(z_{i+1}, ..., z_L).
        self.prior_nets = nn.ModuleList()
        for i in range(len(levels) - 1):
            context_dim = sum(levels[i + 1 :])
            self.prior_nets.append(_PriorNet(context_dim, levels[i]))

        # Decoder input is the concatenation of all sampled z_l.
        z_total = sum(levels)
        self._decoder_c0 = self.encoder.out_channels
        self.from_z = nn.Linear(
            z_total, self._decoder_c0 * decoder_base_freq * self._t_enc
        )

        rev = list(reversed(self._channels))
        self.up_blocks = nn.ModuleList()
        c_prev = rev[0]
        for c in rev[1:] + [rev[-1] // 2 if rev[-1] > 1 else 1]:
            self.up_blocks.append(_UpBlock(c_prev, c))
            c_prev = c
        self.final_conv = nn.Conv2d(c_prev, 1, kernel_size=3, padding=1)

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _compute_encoder_time(n_time: int, strides: Sequence[int]) -> int:
        t = n_time
        for s in strides:
            t = (t + s - 1) // s
        return t

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if not self.training:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * (0.5 * logvar).exp()

    # -- config ----------------------------------------------------------

    @property
    def config(self) -> dict:
        return {
            "levels": self._levels,
            "n_mels": self.n_mels,
            "n_time": self.n_time,
            "channels": self._channels,
            "strides": self._strides,
            "decoder_base_freq": self._decoder_base_freq,
            "encoder": self.encoder.config,
        }

    # -- forward + loss --------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.n_mels:
            raise ValueError(
                f"expected input shape (B, 1, {self.n_mels}, T), got {tuple(x.shape)}"
            )

        h = self.encoder(x).flatten(start_dim=1)

        # Per-level posteriors from the shared bottom-up trunk.
        mus_q: list[torch.Tensor] = []
        logvars_q: list[torch.Tensor] = []
        for head, d in zip(self.q_heads, self._levels):
            mu, logvar = head(h).chunk(2, dim=-1)
            logvar = logvar.clamp(-10.0, 10.0)
            mus_q.append(mu)
            logvars_q.append(logvar)

        # Sample top-down — top level first (so its sample can condition the
        # level-below prior), then cascade downward.
        L = len(self._levels)
        zs: list[torch.Tensor | None] = [None] * L
        # Per-level pre-allocation. Avoid `[x] * L` which aliases the same
        # tensor object across slots — benign today because the loop below
        # overwrites every slot, but one future `continue` turns that into
        # a silent correctness bug. List-comp with the correct per-level
        # shape is defensively safer.
        mus_p: list[torch.Tensor] = [torch.zeros_like(mus_q[i]) for i in range(L)]
        logvars_p: list[torch.Tensor] = [torch.zeros_like(logvars_q[i]) for i in range(L)]
        # Top level has standard normal prior (mu=0, logvar=0).
        for i in range(L - 1, -1, -1):
            if i == L - 1:
                mus_p[i] = torch.zeros_like(mus_q[i])
                logvars_p[i] = torch.zeros_like(logvars_q[i])
            else:
                # Concat all *above* z samples as the prior context.
                above = torch.cat([zs[j] for j in range(i + 1, L)], dim=-1)
                mu_p, logvar_p = self.prior_nets[i](above)
                mus_p[i] = mu_p
                logvars_p[i] = logvar_p
            zs[i] = self._reparameterize(mus_q[i], logvars_q[i])

        # Decode from concatenated z_{1:L}.
        z_cat = torch.cat([zs[i] for i in range(L)], dim=-1)
        g = self.from_z(z_cat)
        g = g.view(z_cat.shape[0], self._decoder_c0, self._decoder_base_freq, self._t_enc)
        for block in self.up_blocks:
            g = block(g)
        x_hat = self.final_conv(g)
        if x_hat.shape[-2:] != (self.n_mels, self.n_time):
            x_hat = F.interpolate(
                x_hat, size=(self.n_mels, self.n_time), mode="bilinear", align_corners=False
            )

        return {
            "x_hat": x_hat,
            "mu": mus_q,
            "logvar": logvars_q,
            "z": zs,
            "prior_mu": mus_p,
            "prior_logvar": logvars_p,
        }

    def loss(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = self(x)
        recon = F.l1_loss(out["x_hat"], x, reduction="mean")

        kl_terms: list[torch.Tensor] = []
        for i in range(len(self._levels)):
            kl_i = _kl_gaussian(
                out["mu"][i],
                out["logvar"][i],
                out["prior_mu"][i],
                out["prior_logvar"][i],
            )
            kl_terms.append(kl_i)
        kl_total = torch.stack(kl_terms).sum()

        total = recon + kl_total
        parts = {"recon": recon.detach(), "kl": kl_total.detach()}
        for i, kl_i in enumerate(kl_terms):
            parts[f"kl_{i}"] = kl_i.detach()
        return total, parts
