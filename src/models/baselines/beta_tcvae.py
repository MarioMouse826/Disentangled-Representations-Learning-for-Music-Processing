"""β-TCVAE baseline.

Reference: Chen et al. 2018, "Isolating Sources of Disentanglement in VAEs"
(arXiv:1803.05428). Decomposes the standard VAE KL term into three parts
and penalizes each with a separate coefficient:

    L = E_q[log p(x|z)]
        + α · I_index        (index-code mutual information)
        + β · TC             (Total Correlation — primary disentanglement term)
        + γ · dimwise_KL     (per-dim marginal KL)

With `α = γ = 1`, only TC is reweighted — this gives β-TCVAE its edge over
plain β-VAE: the extra-strong pressure on TC targets disentanglement without
the over-regularization of the index-code MI term that β-VAE conflates.

Reuses `BetaVAE` encoder/decoder — same shared representation as the other
baselines. The loss differs only in its KL-decomposition.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from src.losses.tc import batch_tc_decomposition
from src.models.baselines.beta_vae import BetaVAE


class BetaTCVAE(BetaVAE):
    """β-TCVAE with minibatch-weighted-sampling TC estimator."""

    def __init__(
        self,
        d_z: int = 64,
        n_mels: int = 128,
        n_time: int = 401,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2, 2),
        decoder_base_freq: int = 4,
        alpha: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(
            d_z=d_z,
            n_mels=n_mels,
            n_time=n_time,
            channels=channels,
            strides=strides,
            decoder_base_freq=decoder_base_freq,
        )
        if alpha < 0 or gamma < 0:
            raise ValueError(f"alpha and gamma must be non-negative, got {alpha}, {gamma}")
        self._alpha = float(alpha)
        self._gamma = float(gamma)

    @property
    def config(self) -> dict:
        cfg = super().config
        cfg.update({"alpha": self._alpha, "gamma": self._gamma})
        return cfg

    def loss(
        self,
        x: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """β-TCVAE objective.

        `loss = recon + α·MI + β·TC + γ·dimwise_KL`. MI and dimwise_KL use
        the parent's `α` and `γ` (set at construction); TC uses the
        call-site `beta` so Hydra sweeps can vary only the disentanglement
        pressure without touching α/γ.
        """
        out = self(x)
        recon = F.l1_loss(out["x_hat"], x, reduction="mean")

        # Sampled z carries the graph through encoder → bottleneck → decoder,
        # so the TC decomposition has gradient back to the encoder heads.
        mi, tc, dimwise_kl = batch_tc_decomposition(
            out["z"], out["mu"], out["logvar"]
        )

        total = recon + self._alpha * mi + beta * tc + self._gamma * dimwise_kl
        parts = {
            "recon": recon.detach(),
            "mi": mi.detach(),
            "tc": tc.detach(),
            "dimwise_kl": dimwise_kl.detach(),
            "kl": (mi + tc + dimwise_kl).detach(),
        }
        return total, parts
