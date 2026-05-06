"""Tests for Mutual Information Gap (MIG) — Chen et al. 2018.

Reference: Chen, R.T.Q., Li, X., Grosse, R., Duvenaud, D., 2018. "Isolating
Sources of Disentanglement in Variational Autoencoders." NeurIPS 2018.
arXiv:1802.04942.

MIG = (1/K) Σ_k [ I(z_{i*_k}; v_k) − I(z_{i**_k}; v_k) ] / H(v_k)

where i*_k is the top-MI latent for factor v_k and i**_k the runner-up.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.mig import compute, compute_mig_from_arrays


# -- core MI/MIG on arrays ------------------------------------------------


def test_mig_perfect_disentanglement_scores_near_one() -> None:
    # z_0 ≡ v_0, z_1 ≡ v_1, rest = pure noise → MIG should be ~1.0.
    rng = np.random.default_rng(0)
    n = 4096
    v0 = rng.integers(0, 10, size=n)
    v1 = rng.integers(0, 10, size=n)
    z = np.column_stack([
        v0.astype(np.float32),
        v1.astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
    ])
    out = compute_mig_from_arrays(
        z=z, factors={"v0": v0, "v1": v1}, n_bins=20
    )
    assert out["mig"] > 0.7, f"expected MIG > 0.7 under perfect id, got {out['mig']:.3f}"
    assert set(out["per_factor"].keys()) == {"v0", "v1"}
    # Mi matrix should show strongest link on the diagonal of the signal block.
    mi = out["mi_matrix"]  # (D=4, K=2)
    assert mi.shape == (4, 2)
    assert int(np.argmax(mi[:, 0])) == 0
    assert int(np.argmax(mi[:, 1])) == 1


def test_mig_entangled_factors_scores_low() -> None:
    # All latents carry both factors equally → MIG should be near zero because
    # top-2 MIs are similar.
    rng = np.random.default_rng(1)
    n = 4096
    v0 = rng.integers(0, 8, size=n)
    v1 = rng.integers(0, 8, size=n)
    mix = v0.astype(np.float32) + 0.5 * v1.astype(np.float32)
    z = np.column_stack([mix + rng.standard_normal(n) * 0.01 for _ in range(4)])
    out = compute_mig_from_arrays(z=z, factors={"v0": v0, "v1": v1}, n_bins=20)
    assert out["mig"] < 0.2, f"expected MIG < 0.2 under entanglement, got {out['mig']:.3f}"


def test_mig_handles_continuous_factor() -> None:
    # Continuous pitch (semitones as floats) should still be binnable.
    rng = np.random.default_rng(2)
    n = 2048
    pitch = rng.uniform(40, 80, size=n).astype(np.float32)
    z = np.column_stack([pitch, rng.standard_normal(n), rng.standard_normal(n)])
    out = compute_mig_from_arrays(z=z, factors={"pitch": pitch}, n_bins=20)
    assert out["mig"] > 0.4
    assert np.isfinite(out["mig"])


def test_mig_rejects_shape_mismatch() -> None:
    z = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="length"):
        compute_mig_from_arrays(
            z=z, factors={"bad": np.zeros(9, dtype=np.int64)}, n_bins=20
        )


def test_mig_rejects_empty_factors() -> None:
    with pytest.raises(ValueError, match="factors"):
        compute_mig_from_arrays(
            z=np.zeros((10, 3), dtype=np.float32), factors={}, n_bins=20
        )


def test_mig_constant_factor_has_zero_entropy_and_is_skipped() -> None:
    # A factor with only one unique value has H(v)=0; MI/H is undefined. The
    # implementation should skip it with a warning rather than divide by zero.
    rng = np.random.default_rng(3)
    n = 1024
    v0 = rng.integers(0, 10, size=n)
    v_const = np.zeros(n, dtype=np.int64)
    z = rng.standard_normal((n, 3)).astype(np.float32)
    z[:, 0] = v0.astype(np.float32)
    with pytest.warns(RuntimeWarning, match="constant"):
        out = compute_mig_from_arrays(
            z=z, factors={"v0": v0, "const": v_const}, n_bins=20
        )
    # v0 still contributes; result reflects only factors with entropy > 0.
    assert "v0" in out["per_factor"]
    assert "const" not in out["per_factor"]


# -- model + dataloader integration --------------------------------------


class _SyntheticFactorDataset(Dataset):
    def __init__(self, n: int = 512, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.pitch = rng.integers(0, 12, size=n).astype(np.int64)
        self.instrument = rng.integers(0, 5, size=n).astype(np.int64)
        # Latent-space "ground truth": x encodes pitch in dim 0 and instrument
        # in dim 1 deterministically.
        self._x = torch.stack([
            torch.from_numpy(self.pitch.astype(np.float32)),
            torch.from_numpy(self.instrument.astype(np.float32)),
            torch.randn(n),
            torch.randn(n),
        ], dim=1)

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            "x": self._x[i],
            "labels": {
                "pitch": int(self.pitch[i]),
                "instrument": int(self.instrument[i]),
            },
        }


def _collate(batch: list[dict]) -> dict:
    xs = torch.stack([b["x"] for b in batch], dim=0)
    labels = {
        "pitch": torch.tensor([b["labels"]["pitch"] for b in batch], dtype=torch.long),
        "instrument": torch.tensor(
            [b["labels"]["instrument"] for b in batch], dtype=torch.long
        ),
    }
    return {"x": xs, "labels": labels}


class _IdentityLatentModel(torch.nn.Module):
    """Fake model: `encode(x) -> {"mu": x}`. x already in latent-space form."""

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mu": x}


def test_compute_on_dataloader_extracts_mu_and_scores_high() -> None:
    ds = _SyntheticFactorDataset(n=2048, seed=4)
    loader = DataLoader(ds, batch_size=64, collate_fn=_collate)
    model = _IdentityLatentModel()
    out = compute(
        model=model,
        dataloader=loader,
        factor_keys=["pitch", "instrument"],
        n_bins=20,
    )
    assert out["mig"] > 0.5
    assert set(out["per_factor"].keys()) == {"pitch", "instrument"}


def test_compute_concatenates_mu_s_and_mu_c_for_scvae() -> None:
    # SC-VAE-style model returns mu_s + mu_c separately. Ensure both are
    # concatenated into the latent matrix used for MI estimation.
    class _TwoHeadModel(torch.nn.Module):
        def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"mu_s": x[:, :2], "mu_c": x[:, 2:], "logvar_s": torch.zeros_like(x[:, :2])}

    ds = _SyntheticFactorDataset(n=1024, seed=5)
    loader = DataLoader(ds, batch_size=32, collate_fn=_collate)
    model = _TwoHeadModel()
    out = compute(
        model=model,
        dataloader=loader,
        factor_keys=["pitch", "instrument"],
        n_bins=20,
    )
    # D = 2 (mu_s) + 2 (mu_c) = 4.
    assert out["mi_matrix"].shape == (4, 2)
    assert out["mig"] > 0.5
