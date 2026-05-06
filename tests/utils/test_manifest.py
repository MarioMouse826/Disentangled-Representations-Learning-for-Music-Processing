"""Tests for run manifest writer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.manifest import build_manifest, write_manifest


def test_manifest_contains_required_keys(tmp_path: Path) -> None:
    m = build_manifest(config={"foo": "bar"}, seed=0)
    # Required fields: git, config, seed, python/torch versions, cuda, hw.
    required = {
        "git_commit",
        "git_dirty",
        "config",
        "seed",
        "python_version",
        "torch_version",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "platform",
        "hostname",
        "timestamp_utc",
    }
    assert required.issubset(m.keys())
    assert m["config"] == {"foo": "bar"}
    assert m["seed"] == 0


def test_manifest_includes_pip_freeze_list() -> None:
    m = build_manifest(config={}, seed=1)
    assert isinstance(m["pip_freeze"], list)
    # At least one installed package should be in the freeze output.
    assert len(m["pip_freeze"]) > 0


def test_write_manifest_writes_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    write_manifest(config={"model": "beta_vae", "lr": 0.001}, seed=42, out_path=path)
    assert path.exists()
    content = json.loads(path.read_text())
    assert content["seed"] == 42
    assert content["config"]["model"] == "beta_vae"


def test_write_manifest_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "run_manifest.json"
    write_manifest(config={}, seed=0, out_path=path)
    assert path.exists()


def test_manifest_git_sha_is_40_hex_chars_or_none() -> None:
    m = build_manifest(config={}, seed=0)
    sha = m["git_commit"]
    # In a git repo SHA is 40 hex chars; outside one it's None.
    assert sha is None or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def test_manifest_embeds_compute_budget_when_provided() -> None:
    budget = {
        "wall_time_hours": 1.5,
        "gpu_hours": 3.0,
        "peak_vram_gb": 8.2,
        "flops_per_step": 10_000_000,
    }
    m = build_manifest(config={}, seed=0, budget=budget)
    assert m["compute_budget"] == budget


def test_manifest_compute_budget_defaults_to_empty() -> None:
    m = build_manifest(config={}, seed=0)
    assert m["compute_budget"] == {}


def test_build_manifest_is_json_serializable() -> None:
    m = build_manifest(config={"a": 1, "b": [1, 2, 3]}, seed=0)
    # Should round-trip without TypeError.
    dumped = json.dumps(m)
    assert '"seed": 0' in dumped
