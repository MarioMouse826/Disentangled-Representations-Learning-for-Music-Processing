"""Symmetry-Constrained VAE — full module (plan Task 3.8).

Wires every Phase-3 component into a single trainable model:

    x → MelEncoder → h
                ├── StyleHead    → q(z_s | x)   (invariance-constrained)
                └── ContentHead  → q(z_c | x)   (equivariance-constrained)
    z_s, z_c → SCDecoder → x_hat

Optional `ρ(g)` action (RotationRep | TranslationRep | IdentityRep) applied
to z_c during symmetry loss computation — never to z_s.

Composite loss:

    L  =  L_recon
         + β_s · KL_s
         + β_c · KL_c
         + λ_inv  · L_inv
         + λ_equi · L_equi
         + λ_swap · L_swap

Inference mode returns deterministic `z = μ` (reproducible eval passes).
Training mode samples via the reparameterization trick.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.losses.kl import kl_total
from src.losses.recon import l1_log_mel_loss
from src.losses.symmetry import (
    equivariance_loss,
    invariance_loss,
    swap_consistency_loss,
)
from src.models.content_head import ContentHead
from src.models.decoder import SCDecoder
from src.models.encoder import MelEncoder
from src.models.group_repr import IdentityRep, RotationRep, TranslationRep, _GroupRep
from src.models.style_head import StyleHead


_REP_TYPES = {"rotation", "translation", "identity"}


def _build_rep(rep_type: str, d_c: int, period_cents: float, freq_init: str) -> _GroupRep:
    if rep_type == "rotation":
        return RotationRep(d_c=d_c, period_cents=period_cents, freq_init=freq_init)
    if rep_type == "translation":
        return TranslationRep(d_c=d_c)
    if rep_type == "identity":
        return IdentityRep(d_c=d_c)
    raise ValueError(f"rep_type must be one of {sorted(_REP_TYPES)}, got {rep_type!r}")


class SCVAE(nn.Module):
    """Full Symmetry-Constrained VAE.

    Args:
        d_s: style (invariance) latent dim.
        d_c: content (equivariance) latent dim. Must be even when
            `rep_type="rotation"` (block-diagonal SO(2)).
        n_mels, n_time: expected log-mel input shape.
        channels, strides: encoder/decoder spatial schedule.
        decoder_base_freq: decoder bottleneck freq resolution.
        style_hidden_score: inner dim for StyleHead attention.
        rep_type: `"rotation"` (default), `"translation"`, or `"identity"`.
        period_cents: group period for RotationRep. 1200 = 1 octave.
        freq_init: `"octave"` or `"ones"` — RotationRep frequency init.
        logvar_clamp: per-head clamp on output log-variances.
    """

    def __init__(
        self,
        d_s: int = 32,
        d_c: int = 16,
        n_mels: int = 128,
        n_time: int = 401,
        channels: Sequence[int] = (32, 64, 128, 256, 512),
        strides: Sequence[int] = (2, 2, 2, 2, 2),
        decoder_base_freq: int = 4,
        style_hidden_score: int = 128,
        rep_type: str = "rotation",
        period_cents: float = 1200.0,
        freq_init: str = "octave",
        logvar_clamp: tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        super().__init__()
        if rep_type not in _REP_TYPES:
            raise ValueError(
                f"rep_type must be one of {sorted(_REP_TYPES)}, got {rep_type!r}"
            )

        self._d_s = d_s
        self._d_c = d_c
        self._n_mels = n_mels
        self._n_time = n_time
        self._rep_type = rep_type
        self._period_cents = period_cents
        self._freq_init = freq_init

        self.encoder = MelEncoder(channels=channels, strides=strides)
        c_out = self.encoder.out_channels
        self.t_enc = self._compute_encoder_time(n_time, tuple(strides))

        self.style_head = StyleHead(
            in_channels=c_out,
            d_s=d_s,
            hidden_score=style_hidden_score,
            logvar_clamp=logvar_clamp,
        )
        self.content_head = ContentHead(
            in_channels=c_out,
            d_c=d_c,
            logvar_clamp=logvar_clamp,
            require_even_d_c=(rep_type == "rotation"),
        )
        self.rep: _GroupRep = _build_rep(rep_type, d_c, period_cents, freq_init)
        self.decoder = SCDecoder(
            d_s=d_s,
            d_c=d_c,
            n_mels=n_mels,
            n_time=n_time,
            t_enc=self.t_enc,
            channels=channels,
            base_freq=decoder_base_freq,
        )

    # -- shape helpers ---------------------------------------------------

    @staticmethod
    def _compute_encoder_time(n_time: int, strides: tuple[int, ...]) -> int:
        t = n_time
        for s in strides:
            t = (t + s - 1) // s
        return t

    # -- config ----------------------------------------------------------

    @property
    def config(self) -> dict:
        return {
            "d_s": self._d_s,
            "d_c": self._d_c,
            "n_mels": self._n_mels,
            "n_time": self._n_time,
            "t_enc": self.t_enc,
            "rep_type": self._rep_type,
            "period_cents": self._period_cents,
            "freq_init": self._freq_init,
            "encoder": self.encoder.config,
            "style_head": self.style_head.config,
            "content_head": self.content_head.config,
            "decoder": self.decoder.config,
            "rep": self.rep.config,
        }

    # -- encode / reparameterize / decode -------------------------------

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)                              # (B, C, T_enc)
        style = self.style_head(h)                        # mu_s, logvar_s, attn_weights
        content = self.content_head(h)                    # mu_c, logvar_c
        return {**style, **content}

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * (0.5 * logvar).exp()

    def decode(self, z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_s, z_c)

    # -- forward ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self._n_mels:
            raise ValueError(
                f"expected input shape (B, 1, {self._n_mels}, T), "
                f"got {tuple(x.shape)}"
            )
        enc = self.encode(x)
        z_s = self._reparameterize(enc["mu_s"], enc["logvar_s"])
        z_c = self._reparameterize(enc["mu_c"], enc["logvar_c"])
        x_hat = self.decode(z_s, z_c)
        return {
            "x_hat": x_hat,
            "mu_s": enc["mu_s"],
            "logvar_s": enc["logvar_s"],
            "z_s": z_s,
            "mu_c": enc["mu_c"],
            "logvar_c": enc["logvar_c"],
            "z_c": z_c,
            "attn_weights": enc["attn_weights"],
        }

    # -- composite loss --------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        x_g: torch.Tensor | None = None,
        g_cents: torch.Tensor | None = None,
        *,
        beta_s: float = 1.0,
        beta_c: float = 1.0,
        tau_s: float = 0.1,
        tau_c: float = 0.1,
        lambda_inv: float = 1.0,
        lambda_equi: float = 1.0,
        lambda_swap: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Composite SC-VAE loss.

        Provide (x_g, g_cents) together for the symmetry terms; omit both
        for the no-pairing ablation (pure recon + KL, matches β-VAE).

        **L_swap floor note**: SCDecoder produces `(128, 416)` with the
        default stride schedule and bilinear-interpolates to `(128, 401)`
        on every forward pass (see `decoder.py`). This introduces a
        small, content-dependent interpolation bias that makes L_swap
        irreducibly positive even when the latent action is perfect. The
        gradient direction stays correct; the absolute floor does not
        reach zero. Adjust `decoder.base_freq` + stride schedule if a
        pixel-exact L_swap is required for a specific ablation.
        """
        has_symmetry = x_g is not None or g_cents is not None
        if has_symmetry and (x_g is None or g_cents is None):
            raise ValueError(
                "x_g and g_cents must be provided together (got one without the other)"
            )

        out_x = self(x)
        recon = l1_log_mel_loss(out_x["x_hat"], x)
        kl_s, kl_c = kl_total(
            out_x["mu_s"], out_x["logvar_s"],
            out_x["mu_c"], out_x["logvar_c"],
            tau_s=tau_s,
            tau_c=tau_c,
        )

        total = recon + beta_s * kl_s + beta_c * kl_c
        parts: dict[str, torch.Tensor] = {
            "recon": recon.detach(),
            "kl_s": kl_s.detach(),
            "kl_c": kl_c.detach(),
        }

        if has_symmetry:
            # Encode the paired T_g(x) — detach rho computation from the
            # symmetry losses to keep gradient paths clean.
            enc_g = self.encode(x_g)

            # L_inv: posterior-mean distance on z_s.
            l_inv = invariance_loss(out_x["mu_s"], enc_g["mu_s"])

            # L_equi: rotate content-mean and compare to paired content-mean.
            rho_mu_c = self.rep(out_x["mu_c"], g_cents)     # (B, d_c, T_enc)
            l_equi = equivariance_loss(enc_g["mu_c"], rho_mu_c)

            # L_swap: decode with rotated z_c, compare to T_g x.
            rho_z_c = self.rep(out_x["z_c"], g_cents)
            x_hat_swapped = self.decode(out_x["z_s"], rho_z_c)
            l_swap = swap_consistency_loss(x_hat_swapped, x_g)

            total = total + lambda_inv * l_inv + lambda_equi * l_equi + lambda_swap * l_swap
            parts["L_inv"] = l_inv.detach()
            parts["L_equi"] = l_equi.detach()
            parts["L_swap"] = l_swap.detach()

        return total, parts
