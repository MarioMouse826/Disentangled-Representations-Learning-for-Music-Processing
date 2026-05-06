"""Tests for latent traversal artifacts — identity swap, pitch traversal,
timbre slerp."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.evaluation.latent_traversal import (
    pitch_traverse,
    save_artifacts,
    slerp,
    slerp_timbre,
    timbre_swap,
)


# -- mock model + group rep ----------------------------------------------


class _MockModel(torch.nn.Module):
    """SC-VAE-like oracle. `encode` returns `{mu_s, mu_c}`; `decode(z_s, z_c)`
    concatenates them along channel dim and reshapes to a mel-like (1, d, T).
    """

    def __init__(self, d_s: int = 4, d_c: int = 4, n_mels: int = 8, n_time: int = 16) -> None:
        super().__init__()
        self.d_s = d_s
        self.d_c = d_c
        self.n_mels = n_mels
        self.n_time = n_time

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b = x.shape[0]
        # Deterministic mapping: content of x in first n_mels rows.
        flat = x.reshape(b, -1)
        mu_s = flat[:, : self.d_s].contiguous()
        mu_c = flat[:, self.d_s : self.d_s + self.d_c].contiguous()
        return {"mu_s": mu_s, "mu_c": mu_c}

    def decode(self, z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        b = z_s.shape[0]
        # Pad or truncate to n_mels * n_time.
        total = self.n_mels * self.n_time
        combined = torch.cat([z_s, z_c], dim=-1)
        if combined.shape[-1] < total:
            pad = total - combined.shape[-1]
            combined = torch.cat([combined, torch.zeros(b, pad)], dim=-1)
        else:
            combined = combined[:, :total]
        return combined.reshape(b, 1, self.n_mels, self.n_time)


class _RotRep:
    def __init__(self, d_c: int, period: float = 1200.0) -> None:
        self.d_c = d_c
        self.period = period

    def matrix(self, g_cents: torch.Tensor) -> torch.Tensor:
        b = g_cents.shape[0]
        theta = (2.0 * math.pi / self.period) * g_cents
        c = torch.cos(theta)
        s = torch.sin(theta)
        R = torch.eye(self.d_c, dtype=g_cents.dtype).unsqueeze(0).expand(b, -1, -1).clone()
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        return R


# -- slerp -----------------------------------------------------------------


def test_slerp_endpoints_recover_inputs() -> None:
    torch.manual_seed(0)
    a = torch.randn(4)
    b = torch.randn(4)
    out = slerp(a, b, n_steps=5)
    assert out.shape == (5, 4)
    assert torch.allclose(out[0], a, atol=1e-5)
    assert torch.allclose(out[-1], b, atol=1e-5)


def test_slerp_preserves_norm_for_unit_vectors() -> None:
    a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0, 0.0])
    out = slerp(a, b, n_steps=9)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_slerp_handles_collinear_inputs() -> None:
    # When a and b point in the same direction, the formula's sin term
    # vanishes; fall back to linear interp.
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([2.0, 0.0])
    out = slerp(a, b, n_steps=5)
    assert torch.allclose(out[0], a)
    assert torch.allclose(out[-1], b)
    # Midpoint lies along the same ray.
    mid = out[2]
    assert mid[1].abs() < 1e-5


def test_slerp_rejects_nsteps_lt_2() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        slerp(torch.zeros(3), torch.ones(3), n_steps=1)


# -- timbre swap + pitch traverse -----------------------------------------


def test_timbre_swap_uses_zc_from_a_and_zs_from_b() -> None:
    model = _MockModel()
    torch.manual_seed(1)
    x_a = torch.randn(1, 1, 8, 16)
    x_b = torch.randn(1, 1, 8, 16)
    out = timbre_swap(model=model, x_a=x_a, x_b=x_b)
    # Output matches decode(z_s(b), z_c(a)).
    enc_a = model.encode(x_a)
    enc_b = model.encode(x_b)
    expected = model.decode(enc_b["mu_s"], enc_a["mu_c"])
    assert torch.allclose(out, expected, atol=1e-6)
    assert out.shape == x_a.shape


def test_pitch_traverse_returns_grid_over_g() -> None:
    model = _MockModel()
    rep = _RotRep(d_c=model.d_c)
    torch.manual_seed(2)
    x = torch.randn(1, 1, 8, 16)
    g_list = torch.tensor([-600.0, -100.0, 100.0, 600.0])
    out = pitch_traverse(model=model, group_rep=rep, x=x, g_cents_list=g_list)
    # (G, 1, C, n_mels, n_time) — grid along G axis.
    assert out.shape == (g_list.shape[0], 1, 1, 8, 16)


def test_pitch_traverse_at_g_zero_equals_reconstruction() -> None:
    model = _MockModel()
    rep = _RotRep(d_c=model.d_c)
    torch.manual_seed(3)
    x = torch.randn(1, 1, 8, 16)
    g_list = torch.tensor([0.0])
    out = pitch_traverse(model=model, group_rep=rep, x=x, g_cents_list=g_list)
    enc = model.encode(x)
    recon = model.decode(enc["mu_s"], enc["mu_c"])
    assert torch.allclose(out[0], recon, atol=1e-6)


def test_slerp_timbre_interpolates_n_steps() -> None:
    model = _MockModel()
    torch.manual_seed(4)
    x_a = torch.randn(1, 1, 8, 16)
    x_b = torch.randn(1, 1, 8, 16)
    out = slerp_timbre(model=model, x_a=x_a, x_b=x_b, n_steps=8)
    assert out.shape == (8, 1, 1, 8, 16)


# -- save_artifacts -------------------------------------------------------


def test_save_artifacts_writes_expected_files(tmp_path: Path) -> None:
    model = _MockModel()
    rep = _RotRep(d_c=model.d_c)
    torch.manual_seed(5)
    x_a = torch.randn(1, 1, 8, 16)
    x_b = torch.randn(1, 1, 8, 16)
    g_list = torch.tensor([-600.0, 0.0, 600.0])

    save_artifacts(
        model=model,
        group_rep=rep,
        x_a=x_a,
        x_b=x_b,
        g_cents_list=g_list,
        n_slerp_steps=4,
        out_dir=tmp_path,
        mel_to_wave=lambda mel: torch.randn(32),  # stub inverse
        sample_rate=8000,
    )
    # Expected layout: mel PNG grids + WAVs for swap + pitch + slerp.
    assert (tmp_path / "timbre_swap.png").exists()
    assert (tmp_path / "pitch_traverse.png").exists()
    assert (tmp_path / "slerp.png").exists()
    # One WAV per g in pitch, plus swap, plus n_slerp WAVs.
    pitch_wavs = sorted(tmp_path.glob("pitch_*.wav"))
    slerp_wavs = sorted(tmp_path.glob("slerp_*.wav"))
    assert len(pitch_wavs) == g_list.shape[0]
    assert len(slerp_wavs) == 4
    assert (tmp_path / "timbre_swap.wav").exists()
