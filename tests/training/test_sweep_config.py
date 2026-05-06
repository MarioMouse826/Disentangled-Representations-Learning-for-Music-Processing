"""Tests for sweep config validity (plan Task 6.2)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_HYDRA_SWEEP = Path("configs/train/sweep.yaml")
_WANDB_SWEEP = Path("configs/train/wandb_sweep.yaml")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_hydra_sweep_yaml_is_valid() -> None:
    cfg = _load(_HYDRA_SWEEP)
    assert cfg is not None
    assert "sweep" in cfg
    assert "model" in cfg and "lit" in cfg


@pytest.mark.parametrize(
    "axis,expected",
    [
        ("d_s", [16, 32, 64]),
        ("d_c", [16, 32, 64]),
        ("beta_s_const", [1, 4, 8]),
        ("beta_c_const", [1, 4, 8]),
        ("lambda_inv", [0, 0.1, 1, 10]),
        ("lambda_equi", [0, 0.1, 1, 10]),
        ("lambda_swap", [0, 0.5, 1]),
        ("rep_type", ["rotation", "translation", "identity"]),
        ("g_cents_dist", ["uniform_full", "uniform_half", "discrete_semitones"]),
        ("seeds", [0, 1, 2, 3, 4]),
    ],
)
def test_hydra_sweep_axes_match_plan(axis: str, expected: list) -> None:
    cfg = _load(_HYDRA_SWEEP)
    assert cfg["sweep"][axis] == expected


def test_hydra_sweep_d_c_values_are_even() -> None:
    cfg = _load(_HYDRA_SWEEP)
    for v in cfg["sweep"]["d_c"]:
        assert v % 2 == 0, f"d_c={v} must be even for RotationRep"


def test_wandb_sweep_yaml_is_valid() -> None:
    cfg = _load(_WANDB_SWEEP)
    assert cfg["method"] in {"bayes", "grid", "random"}
    assert cfg["metric"]["name"]
    assert "parameters" in cfg


def test_wandb_sweep_parameter_keys_align_with_hydra_paths() -> None:
    cfg = _load(_WANDB_SWEEP)
    params = cfg["parameters"]
    # All keys must be dotted Hydra override paths (e.g. "lit.beta_s").
    for key in params:
        if key == "seed":
            continue
        assert "." in key, f"W&B key {key!r} must be a dotted Hydra path"


def test_wandb_sweep_has_early_termination() -> None:
    cfg = _load(_WANDB_SWEEP)
    assert "early_terminate" in cfg
    assert cfg["early_terminate"]["type"] == "hyperband"
