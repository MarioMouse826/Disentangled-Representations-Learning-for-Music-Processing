"""Tests for DCI disentanglement metric — Eastwood & Williams 2018.

Reference: Eastwood, C., Williams, C.K.I., 2018. "A Framework for the
Quantitative Evaluation of Disentangled Representations." ICLR.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation.dci import compute, compute_dci_from_arrays


# Small/fast GBT config for tests — full-size run uses plan defaults
# (n_estimators=500, max_depth=6).
_FAST_GBT: dict[str, int] = {"n_estimators": 50, "max_depth": 4}


# -- core DCI on arrays ---------------------------------------------------


def test_dci_perfect_disentanglement_scores_high() -> None:
    # z_0 ≡ v_0, z_1 ≡ v_1, rest = noise → D, C, I all near 1.
    rng = np.random.default_rng(0)
    n = 2048
    v0 = rng.integers(0, 10, size=n)
    v1 = rng.integers(0, 10, size=n)
    z = np.column_stack([
        v0.astype(np.float32),
        v1.astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
    ])
    out = compute_dci_from_arrays(
        z=z,
        factors={"v0": v0, "v1": v1},
        factor_types={"v0": "discrete", "v1": "discrete"},
        gbt_kwargs=_FAST_GBT,
        random_state=0,
    )
    assert out["disentanglement"] > 0.8, out
    assert out["completeness"] > 0.8, out
    assert out["informativeness"] > 0.8, out  # classif accuracy averaged
    assert out["importance_matrix"].shape == (4, 2)
    # Signal dims (0,1) hold most importance mass.
    signal_mass = out["importance_matrix"][:2, :].sum()
    total_mass = out["importance_matrix"].sum()
    assert signal_mass / total_mass > 0.9


def test_dci_returns_per_dim_and_per_factor_scores() -> None:
    rng = np.random.default_rng(1)
    n = 1024
    v = rng.integers(0, 5, size=n)
    z = np.column_stack([v.astype(np.float32), rng.standard_normal(n)])
    out = compute_dci_from_arrays(
        z=z,
        factors={"v": v},
        factor_types={"v": "discrete"},
        gbt_kwargs=_FAST_GBT,
        random_state=0,
    )
    assert out["D_per_dim"].shape == (2,)
    assert out["C_per_factor"].shape == (1,)
    assert out["I_per_factor"].shape == (1,)
    # All scalar summaries within [0, 1].
    for key in ("disentanglement", "completeness", "informativeness"):
        assert 0.0 <= out[key] <= 1.0


def test_dci_continuous_factor_uses_regressor_rmse() -> None:
    rng = np.random.default_rng(2)
    n = 1024
    pitch = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    z = np.column_stack([pitch, rng.standard_normal(n), rng.standard_normal(n)])
    out = compute_dci_from_arrays(
        z=z,
        factors={"pitch": pitch},
        factor_types={"pitch": "continuous"},
        gbt_kwargs=_FAST_GBT,
        random_state=0,
    )
    # Informativeness for continuous is 1 - NRMSE; a model that perfectly
    # recovers pitch from z_0 should score high.
    assert out["informativeness"] > 0.8


def test_dci_rejects_shape_mismatch() -> None:
    z = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="length"):
        compute_dci_from_arrays(
            z=z,
            factors={"bad": np.zeros(9, dtype=np.int64)},
            factor_types={"bad": "discrete"},
            gbt_kwargs=_FAST_GBT,
            random_state=0,
        )


def test_dci_rejects_unknown_factor_type() -> None:
    z = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="factor_types"):
        compute_dci_from_arrays(
            z=z,
            factors={"v": np.zeros(10, dtype=np.int64)},
            factor_types={"v": "categorical"},
            gbt_kwargs=_FAST_GBT,
            random_state=0,
        )


def test_dci_defaults_to_discrete_for_integer_factors() -> None:
    # Omit `factor_types` entirely — integer factors inferred as discrete,
    # float factors inferred as continuous.
    rng = np.random.default_rng(3)
    n = 512
    v = rng.integers(0, 4, size=n)
    z = np.column_stack([v.astype(np.float32), rng.standard_normal(n)])
    out = compute_dci_from_arrays(
        z=z, factors={"v": v}, gbt_kwargs=_FAST_GBT, random_state=0
    )
    assert out["informativeness"] > 0.5


# -- model + dataloader integration --------------------------------------


class _SyntheticFactorDataset(Dataset):
    def __init__(self, n: int = 1024, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.pitch = rng.integers(0, 12, size=n).astype(np.int64)
        self.instrument = rng.integers(0, 5, size=n).astype(np.int64)
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
    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"mu": x}


def test_compute_on_dataloader_scores_high() -> None:
    ds = _SyntheticFactorDataset(n=1024, seed=4)
    loader = DataLoader(ds, batch_size=64, collate_fn=_collate)
    out = compute(
        model=_IdentityLatentModel(),
        dataloader=loader,
        factor_keys=["pitch", "instrument"],
        gbt_kwargs=_FAST_GBT,
        random_state=0,
    )
    assert out["disentanglement"] > 0.7
    assert out["completeness"] > 0.7
    assert out["informativeness"] > 0.7
    assert out["importance_matrix"].shape == (4, 2)
