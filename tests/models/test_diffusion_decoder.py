"""Tests for diffusion decoder arm (plan Task 7.2)."""
from __future__ import annotations

import math

import pytest
import torch

from src.models.diffusion_decoder import (
    DiffusionDecoder,
    cosine_beta_schedule,
    real_time_factor,
    sinusoidal_timestep_embedding,
)


# -- schedule + embedding --------------------------------------------------


def test_cosine_beta_schedule_monotone_and_in_range() -> None:
    betas = cosine_beta_schedule(n_timesteps=50)
    assert betas.shape == (50,)
    # Cosine schedule betas are monotonically non-decreasing.
    assert torch.all(betas[1:] >= betas[:-1] - 1e-6)
    # And strictly within (0, 1).
    assert (betas > 0).all() and (betas < 1).all()


def test_sinusoidal_timestep_embedding_shape_and_determinism() -> None:
    t = torch.tensor([0, 10, 99])
    emb = sinusoidal_timestep_embedding(t, dim=32)
    assert emb.shape == (3, 32)
    emb2 = sinusoidal_timestep_embedding(t, dim=32)
    assert torch.equal(emb, emb2)


# -- DiffusionDecoder shapes + gradient ------------------------------------


def _make_decoder(**kw: object) -> DiffusionDecoder:
    defaults: dict = dict(
        d_s=4, d_c=4, n_mels=16, n_time=32,
        base_channels=8, channel_mult=(1, 2, 4, 4),
        attn_resolutions=(4,),    # attention at (n_mels // 4) = 4
        n_timesteps=50,
    )
    defaults.update(kw)
    return DiffusionDecoder(**defaults)


def test_training_loss_is_scalar_and_finite() -> None:
    dec = _make_decoder()
    x0 = torch.randn(2, 1, 16, 32)
    z_s = torch.randn(2, 4)
    z_c = torch.randn(2, 4, 32)
    loss = dec.training_loss(x0=x0, z_s=z_s, z_c=z_c)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_training_loss_gradients_flow_into_all_conditioning_paths() -> None:
    dec = _make_decoder()
    x0 = torch.randn(2, 1, 16, 32)
    z_s = torch.randn(2, 4, requires_grad=True)
    z_c = torch.randn(2, 4, 32, requires_grad=True)
    loss = dec.training_loss(x0=x0, z_s=z_s, z_c=z_c)
    loss.backward()
    assert z_s.grad is not None and z_s.grad.abs().sum() > 0
    assert z_c.grad is not None and z_c.grad.abs().sum() > 0


def test_sample_ddim_output_shape() -> None:
    dec = _make_decoder()
    dec.train(False)
    z_s = torch.randn(1, 4)
    z_c = torch.randn(1, 4, 32)
    with torch.no_grad():
        sample = dec.sample_ddim(z_s=z_s, z_c=z_c, n_steps=5)
    assert sample.shape == (1, 1, 16, 32)
    assert torch.isfinite(sample).all()


def test_sample_ddim_accepts_guidance_scale() -> None:
    dec = _make_decoder()
    dec.train(False)
    z_s = torch.randn(1, 4)
    z_c = torch.randn(1, 4, 32)
    with torch.no_grad():
        s1 = dec.sample_ddim(z_s=z_s, z_c=z_c, n_steps=5, guidance_scale=1.0)
        s2 = dec.sample_ddim(z_s=z_s, z_c=z_c, n_steps=5, guidance_scale=3.0)
    # Guidance changes the trajectory → outputs differ.
    assert not torch.allclose(s1, s2)


def test_classifier_free_dropout_applied_during_training() -> None:
    torch.manual_seed(0)
    dec = _make_decoder(cond_dropout_p=1.0)  # force drop every time
    dec.train(True)
    x0 = torch.randn(2, 1, 16, 32)
    z_s = torch.randn(2, 4)
    z_c = torch.randn(2, 4, 32)
    # With p=1.0, every forward pass uses the null (zero) conditioning, so
    # two consecutive calls must agree up to the shared timestep/noise.
    # Verify by checking that the loss uses null-conditioning path —
    # internal flag exposed via dec.last_cond_kept.
    _ = dec.training_loss(x0=x0, z_s=z_s, z_c=z_c)
    assert dec.last_cond_kept.sum() == 0  # every sample dropped


# -- RTF -------------------------------------------------------------------


def test_real_time_factor_basic() -> None:
    assert math.isclose(real_time_factor(1.0, 2.0), 0.5)
    assert math.isclose(real_time_factor(10.0, 2.0), 5.0)


def test_real_time_factor_rejects_nonpositive_duration() -> None:
    with pytest.raises(ValueError, match="duration"):
        real_time_factor(wall_time_seconds=1.0, audio_seconds=0.0)


# -- toy overfit sanity check ---------------------------------------------


def test_diffusion_overfits_on_tiny_single_example() -> None:
    """Overfit check: 100 SGD steps on one example should drive training
    loss well below the initial value. Verifies gradient flow end-to-end."""
    torch.manual_seed(0)
    dec = _make_decoder(n_timesteps=50)
    opt = torch.optim.Adam(dec.parameters(), lr=3e-3)
    x0 = torch.randn(1, 1, 16, 32)
    z_s = torch.randn(1, 4)
    z_c = torch.randn(1, 4, 32)

    initial_losses = []
    final_losses = []
    for step in range(120):
        opt.zero_grad()
        loss = dec.training_loss(x0=x0, z_s=z_s, z_c=z_c)
        loss.backward()
        opt.step()
        if step < 10:
            initial_losses.append(loss.item())
        elif step >= 110:
            final_losses.append(loss.item())

    assert sum(final_losses) / len(final_losses) < sum(initial_losses) / len(initial_losses)
