"""PyTorch Lightning wrapper for SC-VAE + all baseline models (plan Task 4.1).

Exposes a single `VAELitModule` that dispatches to the right loss signature
based on the wrapped model's type:

- `BetaVAE`, `BetaTCVAE`, `AR_HVAE` → `model.loss(x, beta)` → `(total, parts)`
- `FactorVAE` → alternating VAE + discriminator steps (manual optimization)
- `SCVAE` → `model.loss(x, x_g, g_cents, β_s, β_c, τ_s, τ_c, λ_*)` → `(total, parts)`

Batch contract (from `PairedPitchShiftCollate`):
    {
        "x":        (B, 1, n_mels, n_time),
        "x_g":      (B, 1, n_mels, n_time),   # optional — SC-VAE only
        "g_cents":  (B,) long                # optional — SC-VAE only
    }

Simple baseline dataloaders can omit `x_g` / `g_cents` entirely; the wrapper
routes to the non-paired code path automatically.

Plan hyperparameters (fixed unless the Hydra sweep overrides):
- Optimizer: AdamW, lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-6
- Scheduler: 5000-step linear warmup → cosine decay over `total_steps`
- Per-subspace KL: β_s, β_c swept independently (plan Pass-2 update)
- Free bits: τ_s = τ_c = 0.1 (Kingma 2016)
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import pytorch_lightning as pl
import torch

from src.models.baselines.ar_hvae import AR_HVAE
from src.models.baselines.beta_tcvae import BetaTCVAE
from src.models.baselines.beta_vae import BetaVAE
from src.models.baselines.factor_vae import FactorVAE
from src.models.sc_vae import SCVAE


class VAELitModule(pl.LightningModule):
    """Model-agnostic Lightning wrapper for every VAE variant."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-6,
        warmup_steps: int = 5000,
        total_steps: int = 100000,
        beta_s: float = 1.0,
        beta_c: float = 1.0,
        tau_s: float = 0.1,
        tau_c: float = 0.1,
        lambda_inv: float = 1.0,
        lambda_equi: float = 1.0,
        lambda_swap: float = 0.0,
        grad_clip_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError(
                f"warmup_steps must be in [0, total_steps], "
                f"got warmup={warmup_steps}, total={total_steps}"
            )

        self.model = model

        self._lr = lr
        self._betas = betas
        self._weight_decay = weight_decay
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps

        self._beta_s = beta_s
        self._beta_c = beta_c
        self._tau_s = tau_s
        self._tau_c = tau_c
        self._lambda_inv = lambda_inv
        self._lambda_equi = lambda_equi
        self._lambda_swap = lambda_swap
        self._grad_clip_norm = grad_clip_norm

        # Lightning's hyperparameter logging — saves ctor args to checkpoint.
        self.save_hyperparameters(ignore=["model"])

    @property
    def config(self) -> dict:
        return {
            "lr": self._lr,
            "betas": self._betas,
            "weight_decay": self._weight_decay,
            "warmup_steps": self._warmup_steps,
            "total_steps": self._total_steps,
            "beta_s": self._beta_s,
            "beta_c": self._beta_c,
            "tau_s": self._tau_s,
            "tau_c": self._tau_c,
            "lambda_inv": self._lambda_inv,
            "lambda_equi": self._lambda_equi,
            "lambda_swap": self._lambda_swap,
            "grad_clip_norm": self._grad_clip_norm,
            "model": getattr(self.model, "config", None),
        }

    # -- loss dispatch ---------------------------------------------------

    def _compute_loss(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = batch["x"]
        x_g = batch.get("x_g")
        g_cents = batch.get("g_cents")

        if isinstance(self.model, SCVAE):
            # Paired inputs enable full symmetry losses. Omitted inputs
            # collapse to recon + KL only (matches β-VAE step).
            paired = x_g is not None and g_cents is not None
            if paired:
                return self.model.loss(
                    x,
                    x_g=x_g,
                    g_cents=g_cents,
                    beta_s=self._beta_s,
                    beta_c=self._beta_c,
                    tau_s=self._tau_s,
                    tau_c=self._tau_c,
                    lambda_inv=self._lambda_inv,
                    lambda_equi=self._lambda_equi,
                    lambda_swap=self._lambda_swap,
                )
            return self.model.loss(
                x,
                beta_s=self._beta_s,
                beta_c=self._beta_c,
                tau_s=self._tau_s,
                tau_c=self._tau_c,
            )

        if isinstance(self.model, AR_HVAE):
            return self.model.loss(x)

        if isinstance(self.model, FactorVAE):
            # FactorVAE canonically alternates VAE + disc steps with
            # manual optimization; for the basic Lightning path we call
            # only the VAE loss and leave disc step to a dedicated
            # subclass (out of Task 4.1 scope — Phase-4 follow-up).
            total, parts, _z = self.model.loss_vae(
                x, beta=self._beta_s, gamma=self._lambda_inv
            )
            return total, parts

        # β-VAE + β-TCVAE share the single-β loss signature.
        return self.model.loss(x, beta=self._beta_s)

    # -- train / val steps ----------------------------------------------

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor | None:
        total, parts = self._compute_loss(batch)
        if not torch.isfinite(total):
            self.log("train/nan_skipped", 1.0, on_step=True)
            return None
        self.log("train/loss", total, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in parts.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True)
        return total

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        total, parts = self._compute_loss(batch)
        self.log("val/loss", total, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in parts.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True)
        return total

    # -- optimizer + schedule -------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=self._betas,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self._warmup_cosine_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _warmup_cosine_factor(self, step: int) -> float:
        """Linear warmup → cosine decay to 0. Factor applied to base lr."""
        if step < self._warmup_steps:
            # Avoid div-by-zero when warmup_steps=0 — upper branch handles it.
            return float(step + 1) / float(max(1, self._warmup_steps))
        progress = (step - self._warmup_steps) / max(
            1, self._total_steps - self._warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
