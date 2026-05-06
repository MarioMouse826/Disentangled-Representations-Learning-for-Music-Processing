"""Latent traversal + identity-swap artifacts.

Three traversals (Engel et al. 2017, NSynth §4.2):

    (a) timbre_swap(x_a, x_b)    — z_s from x_b, z_c from x_a → swap timbre
                                    onto content of x_a.
    (b) pitch_traverse(x, g_list) — vary pitch via rho(g) z_c, hold z_s fixed.
    (c) slerp_timbre(x_a, x_b)    — spherical interpolation on z_s while
                                    holding z_c fixed. Reveals smoothness
                                    of timbre manifold.

`save_artifacts` renders mel PNG grids + per-step WAVs into `out_dir`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Slerp — spherical linear interpolation


def slerp(a: torch.Tensor, b: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Spherical linear interpolation between vectors `a` and `b`.

    Standard formula (Shoemake 1985):
        slerp(a, b, t) = sin((1-t)*omega)/sin(omega) * a + sin(t*omega)/sin(omega) * b
    where `omega` = angle between unit(a) and unit(b). Falls back to linear
    interpolation when sin(omega) is near zero (collinear inputs).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if a_flat.shape != b_flat.shape:
        raise ValueError(
            f"a and b must share shape, got {tuple(a.shape)} vs {tuple(b.shape)}"
        )

    a_norm = a_flat.norm().clamp_min(1e-12)
    b_norm = b_flat.norm().clamp_min(1e-12)
    a_unit = a_flat / a_norm
    b_unit = b_flat / b_norm
    cos_omega = (a_unit * b_unit).sum().clamp(-1.0, 1.0)
    omega = torch.arccos(cos_omega)
    sin_omega = torch.sin(omega)

    ts = torch.linspace(0.0, 1.0, n_steps, dtype=a_flat.dtype, device=a_flat.device)
    if sin_omega.abs() < 1e-6:
        # Collinear: linear interp on the original (non-normalized) vectors.
        out = (1.0 - ts)[:, None] * a_flat[None, :] + ts[:, None] * b_flat[None, :]
    else:
        # Interpolated norm so endpoints recover original magnitudes exactly.
        norms = (1.0 - ts) * a_norm + ts * b_norm
        wa = torch.sin((1.0 - ts) * omega) / sin_omega
        wb = torch.sin(ts * omega) / sin_omega
        unit_path = wa[:, None] * a_unit[None, :] + wb[:, None] * b_unit[None, :]
        path_norm = unit_path.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit_path = unit_path / path_norm
        out = norms[:, None] * unit_path
    return out.reshape((n_steps, *a.shape))


# ---------------------------------------------------------------------------
# Typed protocols — duck-type SC-VAE model and rotation rep


class _ModelLike(Protocol):
    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]: ...
    def decode(self, z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor: ...


class _GroupRepLike(Protocol):
    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Traversals


@torch.no_grad()
def timbre_swap(
    *, model: _ModelLike, x_a: torch.Tensor, x_b: torch.Tensor
) -> torch.Tensor:
    """Decode `(z_s(x_b), z_c(x_a))` — apply timbre of `x_b` to content of `x_a`."""
    enc_a = model.encode(x_a)
    enc_b = model.encode(x_b)
    return model.decode(enc_b["mu_s"], enc_a["mu_c"])


def _apply_rho(R: torch.Tensor, mu_c: torch.Tensor) -> torch.Tensor:
    if mu_c.ndim == 2:
        return torch.einsum("bij,bj->bi", R, mu_c)
    if mu_c.ndim == 3:
        return torch.einsum("bij,bjt->bit", R, mu_c)
    raise ValueError(
        f"mu_c must be (B, d_c) or (B, d_c, T), got {tuple(mu_c.shape)}"
    )


@torch.no_grad()
def pitch_traverse(
    *,
    model: _ModelLike,
    group_rep: _GroupRepLike,
    x: torch.Tensor,
    g_cents_list: torch.Tensor,
) -> torch.Tensor:
    """Render `decode(z_s, rho(g) z_c)` across a list of `g` values.

    Returns tensor shaped `(G, *decode_output_shape)`.
    """
    if g_cents_list.ndim != 1:
        raise ValueError(
            f"g_cents_list must be 1-D, got {tuple(g_cents_list.shape)}"
        )
    enc = model.encode(x)
    z_s = enc["mu_s"]
    z_c = enc["mu_c"]
    frames: list[torch.Tensor] = []
    for g_scalar in g_cents_list.tolist():
        g = torch.full((z_c.shape[0],), float(g_scalar), device=z_c.device, dtype=z_c.dtype)
        R = group_rep.matrix(g)
        z_c_g = _apply_rho(R, z_c)
        frames.append(model.decode(z_s, z_c_g))
    return torch.stack(frames, dim=0)


@torch.no_grad()
def slerp_timbre(
    *,
    model: _ModelLike,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    n_steps: int = 8,
) -> torch.Tensor:
    """Slerp on `z_s` between `x_a` and `x_b` while holding `z_c = z_c(x_a)` fixed.

    Expects batch size 1 on both inputs; the slerp path is always over a
    single pair. Returns `(n_steps, *decode_output_shape[1:])`, matching
    the convention of `timbre_swap` and `pitch_traverse`.
    """
    if x_a.shape[0] != 1 or x_b.shape[0] != 1:
        raise ValueError(
            f"slerp_timbre expects batch size 1 per input, got "
            f"x_a={x_a.shape[0]}, x_b={x_b.shape[0]}"
        )
    enc_a = model.encode(x_a)
    enc_b = model.encode(x_b)
    z_s_a = enc_a["mu_s"][0]
    z_s_b = enc_b["mu_s"][0]
    path = slerp(z_s_a, z_s_b, n_steps=n_steps)           # (n_steps, d_s)
    z_c_a = enc_a["mu_c"]                                  # (1, d_c[, T])
    z_c_a_rep = z_c_a.expand(n_steps, *z_c_a.shape[1:]).contiguous()
    # unsqueeze(1) keeps output rank consistent with pitch_traverse, which
    # also returns a 5-D (n_frames, B, C, n_mels, n_time) tensor. Indexing
    # patterns in `save_artifacts` depend on this alignment.
    return model.decode(path, z_c_a_rep).unsqueeze(1)


# ---------------------------------------------------------------------------
# Artifact writer — lightweight; no PIL hard dependency required


MelToWaveFn = Callable[[torch.Tensor], torch.Tensor]


def _mel_to_png(mel: torch.Tensor, path: Path) -> None:
    """Save a mel grid as a PNG using matplotlib. Mel shape: (..., n_mels, n_time).

    A traversal grid is stacked vertically: each row = one step.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = mel.detach().cpu().float().numpy()
    # Accept (G, 1, C, n_mels, n_time), (G, 1, n_mels, n_time), (1, 1, n_mels, n_time).
    while arr.ndim > 3:
        arr = arr.squeeze(1) if arr.shape[1] == 1 else arr[..., 0, :, :]
    if arr.ndim == 2:
        arr = arr[None, :, :]
    g, n_mels, n_time = arr.shape
    fig, ax = plt.subplots(g, 1, figsize=(6, 1.2 * g + 0.5), squeeze=False)
    try:
        for i in range(g):
            ax[i, 0].imshow(arr[i], origin="lower", aspect="auto")
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=100)
    finally:
        plt.close(fig)


def _write_wav(wave: np.ndarray, path: Path, sample_rate: int) -> None:
    """Write mono float32 WAV using scipy.io.wavfile. 16-bit PCM."""
    from scipy.io import wavfile

    wave = np.asarray(wave, dtype=np.float32)
    peak = float(np.max(np.abs(wave))) if wave.size > 0 else 0.0
    if peak > 0.0:
        wave = wave / peak
    pcm = (wave * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)


@torch.no_grad()
def save_artifacts(
    *,
    model: _ModelLike,
    group_rep: _GroupRepLike,
    x_a: torch.Tensor,
    x_b: torch.Tensor,
    g_cents_list: torch.Tensor,
    n_slerp_steps: int,
    out_dir: Path,
    mel_to_wave: MelToWaveFn,
    sample_rate: int,
) -> dict[str, Any]:
    """Render all three traversals to `out_dir` as mel PNGs + WAVs.

    Layout:
        out_dir/timbre_swap.png,  timbre_swap.wav
        out_dir/pitch_traverse.png, pitch_{i}.wav for i in range(G)
        out_dir/slerp.png, slerp_{i}.wav for i in range(n_slerp_steps)

    Returns the three grids for inspection in-memory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    swap = timbre_swap(model=model, x_a=x_a, x_b=x_b)
    pitch = pitch_traverse(
        model=model, group_rep=group_rep, x=x_a, g_cents_list=g_cents_list
    )
    slerp_grid = slerp_timbre(
        model=model, x_a=x_a, x_b=x_b, n_steps=n_slerp_steps
    )

    _mel_to_png(swap, out_dir / "timbre_swap.png")
    _mel_to_png(pitch, out_dir / "pitch_traverse.png")
    _mel_to_png(slerp_grid, out_dir / "slerp.png")

    _write_wav(mel_to_wave(swap[0]).detach().cpu().numpy(), out_dir / "timbre_swap.wav", sample_rate)
    for i in range(pitch.shape[0]):
        _write_wav(
            mel_to_wave(pitch[i][0]).detach().cpu().numpy(),
            out_dir / f"pitch_{i:02d}.wav",
            sample_rate,
        )
    for i in range(slerp_grid.shape[0]):
        _write_wav(
            mel_to_wave(slerp_grid[i][0]).detach().cpu().numpy(),
            out_dir / f"slerp_{i:02d}.wav",
            sample_rate,
        )

    return {"timbre_swap": swap, "pitch_traverse": pitch, "slerp": slerp_grid}
