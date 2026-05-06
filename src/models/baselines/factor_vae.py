"""FactorVAE baseline.

Reference: Kim & Mnih 2018, "Disentangling by Factorising" (arXiv:1802.04942).
Extends β-VAE with a Total-Correlation (TC) penalty estimated adversarially
via a density-ratio discriminator:

    L_VAE(β, γ) = E_q[log p(x|z)]
                   − β · KL( q(z|x) ‖ p(z) )
                   − γ · TC_hat(z)          where TC_hat uses a frozen D

    L_disc       = CE( D(z_joint), 0 ) + CE( D(z_permuted), 1 )

The two losses are optimized in alternating half-batch steps (`z_joint` from
minibatch half A is used to train the VAE; half B's permuted latents train
D). We expose `loss_vae` and `loss_disc` as separate methods so Phase-4
Lightning module can orchestrate the alternation.

Reuses `BetaVAE` for the encoder/decoder stack — apples-to-apples
comparison with β-VAE and β-TCVAE over identical representations.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from src.losses.tc import (
    TCDiscriminator,
    permute_latents,
    tc_discriminator_loss,
    tc_estimate_from_logits,
)
from src.models.baselines.beta_vae import BetaVAE


class FactorVAE(BetaVAE):
    """β-VAE + density-ratio Total-Correlation penalty."""

    def __init__(
        self,
        d_z: int = 64,
        n_mels: int = 128,
        n_time: int = 401,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2, 2),
        decoder_base_freq: int = 4,
        disc_hidden: int = 1000,
        disc_layers: int = 5,
    ) -> None:
        super().__init__(
            d_z=d_z,
            n_mels=n_mels,
            n_time=n_time,
            channels=channels,
            strides=strides,
            decoder_base_freq=decoder_base_freq,
        )
        self._disc_hidden = disc_hidden
        self._disc_layers = disc_layers
        self.disc = TCDiscriminator(
            d_z=d_z,
            hidden=disc_hidden,
            n_layers=disc_layers,
        )

    # -- config ----------------------------------------------------------

    @property
    def config(self) -> dict:
        cfg = super().config
        cfg.update({"disc_hidden": self._disc_hidden, "disc_layers": self._disc_layers})
        return cfg

    # -- losses ----------------------------------------------------------

    def loss_vae(
        self,
        x: torch.Tensor,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """VAE update: recon + β·KL + γ·TC. Discriminator is frozen here.

        Returns:
            (total_loss, detached_parts_dict, z) — `z` is kept on the graph
            so Phase-4 can reuse it as the "joint" sample for the disc step
            without re-encoding.
        """
        out = self(x)
        recon = F.l1_loss(out["x_hat"], x, reduction="mean")
        kl = (
            -0.5 * (1.0 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
        ).sum(dim=1).mean()

        # TC estimate via discriminator. Block gradient into D parameters so
        # VAE optimizer cannot train the discriminator by accident — D is
        # trained in `loss_disc`.
        for p in self.disc.parameters():
            p.requires_grad_(False)
        try:
            logits = self.disc(out["z"])
            tc = tc_estimate_from_logits(logits)
        finally:
            for p in self.disc.parameters():
                p.requires_grad_(True)

        total = recon + beta * kl + gamma * tc
        parts = {"recon": recon.detach(), "kl": kl.detach(), "tc": tc.detach()}
        return total, parts, out["z"]

    def loss_disc(
        self,
        z_joint: torch.Tensor,
        z_factored: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Discriminator update: 2-class cross-entropy.

        Args:
            z_joint: latents from the VAE (treated as detached "real" joint
                samples). Paper: use the second half-batch's z.
            z_factored: optional per-dim-permuted latents. If None, derived
                from `z_joint` via `permute_latents` — still a valid approx
                when no separate half-batch is available (small-scale runs).
        """
        z_joint = z_joint.detach()
        if z_factored is None:
            z_factored = permute_latents(z_joint.clone())
        logits_j = self.disc(z_joint)
        logits_f = self.disc(z_factored)
        return tc_discriminator_loss(logits_j, logits_f)
