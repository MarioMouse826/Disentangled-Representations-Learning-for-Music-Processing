"""Content head — produces the equivariance-constrained posterior `q(z_c | x)`.

Design (plan Task 3.2)
----------------------
Preserve the time axis. `q(z_c | x)` is parameterized as
`(μ_c[t], log σ_c²[t]) ∈ R^{d_c × T'}`. Produced by **1×1 convolutions**
over the shared encoder output `(B, C, T')`.

Why 1×1:
- Trivially time-equivariant — rolling the input in time rolls the output
  by the same offset. This is the architectural prerequisite for pitch-
  shift equivariance `z_c(T_g x) = ρ(g) · z_c(x)` (Task 3.3 + Task 3.5).
- No temporal mixing across time-steps, so per-step posterior is a pure
  function of the encoder's receptive field at that step.
- Minimal parameter count, fast on GPU.

Any temporal context the content head needs already lives in the encoder
trunk's receptive field; the 1×1 head just projects channel-wise.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ContentHead(nn.Module):
    """1×1-conv Gaussian head producing time-resolved `q(z_c | x)`.

    Args:
        in_channels: encoder-trunk channel count `C`.
        d_c: content-latent dimensionality. Task 3.3 requires `d_c` to be
            even when used with the default block-diagonal SO(2) rotation
            representation. Set `require_even_d_c=True` to enforce at
            construction time; defaults to off so the head itself stays
            agnostic.
        logvar_clamp: `(min, max)` clamp on the output log-variance for
            stability.
        require_even_d_c: raise if `d_c` is odd. Off by default so alternate
            group representations (non-rotational) remain usable.
    """

    def __init__(
        self,
        in_channels: int,
        d_c: int,
        logvar_clamp: tuple[float, float] = (-10.0, 10.0),
        require_even_d_c: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if d_c <= 0:
            raise ValueError(f"d_c must be positive, got {d_c}")
        if require_even_d_c and d_c % 2 != 0:
            raise ValueError(
                f"d_c must be even for block-diagonal SO(2) rotation "
                f"(set require_even_d_c=False to override), got {d_c}"
            )
        if logvar_clamp[0] >= logvar_clamp[1]:
            raise ValueError(f"logvar_clamp must be ascending, got {logvar_clamp}")

        self._in_channels = in_channels
        self._d_c = d_c
        self._logvar_clamp = logvar_clamp

        # kernel_size=1 Conv1d ≡ per-time-step Linear on the channel axis.
        # Keeps the (B, C, T) layout that the encoder trunk uses, no
        # reshape/transpose overhead.
        self.to_mu = nn.Conv1d(in_channels, d_c, kernel_size=1)
        self.to_logvar = nn.Conv1d(in_channels, d_c, kernel_size=1)

    @property
    def config(self) -> dict:
        return {
            "in_channels": self._in_channels,
            "d_c": self._d_c,
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
        mu = self.to_mu(h)                                   # (B, d_c, T')
        logvar = self.to_logvar(h).clamp(*self._logvar_clamp)  # (B, d_c, T')
        return {"mu_c": mu, "logvar_c": logvar}
