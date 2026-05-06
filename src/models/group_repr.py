"""Group representation `ρ(g)` of the pitch-shift action on the content latent.

Reference: Higgins et al. 2018, "Towards a Definition of Disentangled
Representations" (arXiv:1812.02230). A disentangled representation
requires the group action on the latent space to be **linear** — hence
`ρ(g) · z_c`, not a non-linear `φ(g, z_c)`. The group is (Z, +) under
addition in cents; the representation on `R^{d_c}` is chosen to respect
the group homomorphism `ρ(g₁ + g₂) = ρ(g₁) ρ(g₂)`.

Three reps are provided:

1. `RotationRep` (primary) — block-diagonal direct sum of 2D rotations.
   `ρ(g) = diag(R(θ_k))` with `θ_k = ω_k · (2π g / period)` and learnable
   frequencies `ω_k > 0`. Faithful to the cyclic structure of pitch
   (octave equivalence) and exactly satisfies the homomorphism.

2. `TranslationRep` (ablation) — `ρ(g) z = z + g · v` for a learned
   direction `v ∈ R^{d_c}`. Technically an *affine* action — satisfies
   the homomorphism but is not orthogonal. Included per plan §Task 6.3
   ablation matrix.

3. `IdentityRep` (null-ablation) — `ρ(g) z = z`. Shows that ρ is what
   buys equivariance, not paired inputs alone.

Shape contract:
    matrix(g_cents: Tensor[B])        → Tensor[B, d_c, d_c]
    forward(z: Tensor[B, d_c, T], g)  → Tensor[B, d_c, T]
    forward(z: Tensor[B, d_c],    g)  → Tensor[B, d_c]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _GroupRep(nn.Module):
    """Common interface: `matrix(g) -> (B, d_c, d_c)` + `forward(z, g) -> z'`."""

    d_c: int

    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z: torch.Tensor, g_cents: torch.Tensor) -> torch.Tensor:
        # Route through `matrix` by default. Subclasses can override for
        # faster / more direct computation.
        if z.shape[0] != g_cents.shape[0]:
            raise ValueError(
                f"batch size mismatch: z has {z.shape[0]}, g has {g_cents.shape[0]}"
            )
        R = self.matrix(g_cents)  # (B, d_c, d_c)
        if z.ndim == 3:
            return torch.einsum("bij,bjt->bit", R, z)
        if z.ndim == 2:
            return torch.einsum("bij,bj->bi", R, z)
        raise ValueError(f"expected z of shape (B, d_c) or (B, d_c, T), got {tuple(z.shape)}")


class RotationRep(_GroupRep):
    """Block-diagonal SO(2) rotation representation.

    Args:
        d_c: content-latent dim. Must be even (pairs up into 2D blocks).
        period_cents: group period — shift by this many cents returns to
            identity when `ω_k = 1`. 1200 cents = 1 octave (Western tuning).
        freq_init: `"octave"` starts `ω_k = k` (so block k rotates k times
            per period); `"ones"` starts everything at 1 (no pre-assigned
            Fourier harmonics — pure data-driven).
    """

    def __init__(
        self,
        d_c: int,
        period_cents: float = 1200.0,
        freq_init: str = "octave",
    ) -> None:
        super().__init__()
        if d_c <= 0 or d_c % 2 != 0:
            raise ValueError(f"d_c must be positive and even, got {d_c}")
        if period_cents <= 0:
            raise ValueError(f"period_cents must be positive, got {period_cents}")
        if freq_init not in {"octave", "ones"}:
            raise ValueError(
                f"freq_init must be 'octave' or 'ones', got {freq_init!r}"
            )

        self.d_c: int = d_c
        self.n_blocks: int = d_c // 2
        self.period: float = float(period_cents)
        self._freq_init: str = freq_init

        if freq_init == "octave":
            init = torch.arange(1, self.n_blocks + 1, dtype=torch.float32)
        else:
            init = torch.ones(self.n_blocks)
        # log-parameterize ω so the exponential map keeps it strictly positive.
        self.log_omega = nn.Parameter(torch.log(init))

    @property
    def config(self) -> dict:
        return {
            "d_c": self.d_c,
            "n_blocks": self.n_blocks,
            "period_cents": self.period,
            "freq_init": self._freq_init,
        }

    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor:
        if g_cents.ndim != 1:
            raise ValueError(
                f"g_cents must be 1-D (B,), got shape {tuple(g_cents.shape)}"
            )
        B = g_cents.shape[0]
        omega = self.log_omega.exp()                             # (K,)
        # Anchor device/dtype on the parameter (omega) so CPU g_cents + CUDA
        # model still produces a GPU matrix. Cast g_cents explicitly.
        device = omega.device
        g = g_cents.to(device=device, dtype=omega.dtype)
        # θ_bk = (2π / period) · g_b · ω_k
        theta = (
            (2.0 * math.pi / self.period)
            * g[:, None]
            * omega[None, :]
        )                                                         # (B, K)
        c = torch.cos(theta)                                      # (B, K)
        s = torch.sin(theta)                                      # (B, K)

        # Vectorized block-diag scatter. `idx` indexes the pair positions
        # along the matrix diagonal; advanced indexing over (batch, rows,
        # cols) writes (B, K) values per entry.
        R = torch.zeros(B, self.d_c, self.d_c, device=device, dtype=omega.dtype)
        idx = torch.arange(self.n_blocks, device=device)
        R[:, 2 * idx, 2 * idx] = c
        R[:, 2 * idx, 2 * idx + 1] = -s
        R[:, 2 * idx + 1, 2 * idx] = s
        R[:, 2 * idx + 1, 2 * idx + 1] = c
        return R


class TranslationRep(_GroupRep):
    """Affine translation along a learned direction: `ρ(g) z = z + g · v`.

    This is an affine representation of (Z, +): it satisfies the
    homomorphism `ρ(g₁+g₂) z = ρ(g₁)(ρ(g₂) z) − g₁ · v = z + (g₁+g₂) v`
    only when ρ is interpreted as acting on an augmented vector `(z, 1)`.
    We implement the direct formulation so callers can treat the output
    as `ρ(g) · z` on a `(d_c+1)`-dim augmented space implicitly.
    """

    def __init__(self, d_c: int) -> None:
        super().__init__()
        if d_c <= 0:
            raise ValueError(f"d_c must be positive, got {d_c}")
        self.d_c = d_c
        # Small init so learning starts close to identity.
        self.direction = nn.Parameter(torch.randn(d_c) * 0.02)

    @property
    def config(self) -> dict:
        return {"d_c": self.d_c, "rep": "translation"}

    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor:
        """Returns identity — translation is affine, not linear.

        **Warning**: callers that introspect `ρ(g)` via `.matrix()` for
        analysis will see `I`, not the actual translation action. The
        real action lives in `forward()` below (`z + g · direction`).
        Only use `.matrix()` on `TranslationRep` if you want the *linear
        part* of the action (which is trivially identity).
        """
        B = g_cents.shape[0]
        I = torch.eye(self.d_c, device=g_cents.device, dtype=self.direction.dtype)
        return I.unsqueeze(0).expand(B, -1, -1).contiguous()

    def forward(self, z: torch.Tensor, g_cents: torch.Tensor) -> torch.Tensor:
        if z.shape[0] != g_cents.shape[0]:
            raise ValueError(
                f"batch size mismatch: z has {z.shape[0]}, g has {g_cents.shape[0]}"
            )
        if z.shape[-2 if z.ndim == 3 else -1] != self.d_c:
            raise ValueError(
                f"expected z channel dim = {self.d_c}, got shape {tuple(z.shape)}"
            )
        if z.ndim == 3:
            # (B, d_c, T) + (B,)·(d_c,) broadcast → (B, d_c, 1)
            shift = g_cents[:, None, None] * self.direction[None, :, None]
            return z + shift
        if z.ndim == 2:
            return z + g_cents[:, None] * self.direction[None, :]
        raise ValueError(f"expected z of shape (B, d_c) or (B, d_c, T), got {tuple(z.shape)}")


class IdentityRep(_GroupRep):
    """Null representation: `ρ(g) z = z` for every `g`.

    Ablation baseline — pins that the symmetry prior (not just paired
    inputs) drives equivariance. If SC-VAE with `IdentityRep` matches
    SC-VAE with `RotationRep`, the rotation isn't doing work.
    """

    def __init__(self, d_c: int) -> None:
        super().__init__()
        if d_c <= 0:
            raise ValueError(f"d_c must be positive, got {d_c}")
        self.d_c = d_c

    @property
    def config(self) -> dict:
        return {"d_c": self.d_c, "rep": "identity"}

    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor:
        B = g_cents.shape[0]
        # Match g_cents dtype when it is floating (respects autocast/fp16);
        # fall back to the default dtype when g_cents is integer-valued.
        dtype = g_cents.dtype if g_cents.is_floating_point() else torch.get_default_dtype()
        I = torch.eye(self.d_c, device=g_cents.device, dtype=dtype)
        return I.unsqueeze(0).expand(B, -1, -1).contiguous()

    def forward(self, z: torch.Tensor, g_cents: torch.Tensor) -> torch.Tensor:
        if z.shape[0] != g_cents.shape[0]:
            raise ValueError(
                f"batch size mismatch: z has {z.shape[0]}, g has {g_cents.shape[0]}"
            )
        return z
